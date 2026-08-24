import os

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
# 不强制 ROCBLAS_USE_HIPBLASLT：ROCm 默认会按 GEMM 形状自动选择
# hipBLASLt 或 Tensile。首次懒加载由训练前的多卡并行预热显式完成。

import argparse
import copy
import gc
import glob
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_model as load_safetensors_model

from config import TrainConfig
from dataset_gpt_full_sequence import GeometrizeFullSequenceDataset
from model_gpt import GeometrizeGPT
from pretrained import load_release_config
from token_layout import TOKEN_LAYOUT


RESET_OPTIMIZER_STATE = False

# 全局累计次数。三卡模式下 72 会分为每卡本地累计 24 次。
ACCUM_SCHEDULE_BY_LOSS = [[10, 60]]

LR_SCHEDULE_BY_EPOCH = [
    [1, 1e-10], [1.25,2e-4],[120, 2.0e-4], [140, 5e-5], [140, 5e-5],
]


class MMapBatchIterator:
    """用两个 pinned buffer 交替执行 mmap 预取与 GPU 消费。"""

    def __init__(
        self,
        dataset,
        start_index,
        batch_size,
        drop_last,
        pin_memory,
    ):
        self.dataset = dataset
        if not 0 <= start_index <= len(dataset):
            raise ValueError(
                f"start_index 必须位于 [0, {len(dataset)}]，收到 {start_index}"
            )
        self.next_index = start_index
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.last_wait_seconds = 0.0
        self.last_fill_seconds = 0.0
        self.tensors = [
            torch.empty(
                (batch_size, dataset.config.CONTEXT_TOKENS),
                dtype=torch.long,
                pin_memory=pin_memory,
            )
            for _ in range(2)
        ]
        self.numpy_views = [tensor.numpy() for tensor in self.tensors]
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mmap-batch-prefetch",
        )
        self.future = None
        self.future_buffer_index = None
        self.closed = False
        self._schedule(buffer_index=0)

    def _get_next_bounds(self):
        if self.next_index >= len(self.dataset):
            return None
        batch_end = min(
            self.next_index + self.batch_size, len(self.dataset)
        )
        item_count = batch_end - self.next_index
        if self.drop_last and item_count < self.batch_size:
            return None
        return self.next_index, batch_end, item_count

    def _fill_buffer(
        self, buffer_index, batch_start, batch_end, item_count
    ):
        fill_started_at = time.perf_counter()
        self.dataset.fill_numpy_batch(
            batch_start,
            batch_end,
            self.numpy_views[buffer_index][:item_count],
        )
        return (
            batch_end,
            item_count,
            time.perf_counter() - fill_started_at,
        )

    def _schedule(self, buffer_index):
        bounds = self._get_next_bounds()
        if bounds is None:
            self.future = None
            self.future_buffer_index = None
            return
        batch_start, batch_end, item_count = bounds
        self.future_buffer_index = buffer_index
        self.future = self.executor.submit(
            self._fill_buffer,
            buffer_index,
            batch_start,
            batch_end,
            item_count,
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.future is None:
            self.close()
            raise StopIteration

        wait_started_at = time.perf_counter()
        batch_end, item_count, self.last_fill_seconds = self.future.result()
        self.last_wait_seconds = time.perf_counter() - wait_started_at
        ready_buffer_index = self.future_buffer_index
        self.next_index = batch_end

        # 当前 buffer 交给训练线程后，只写另一个 buffer。下一次 __next__
        # 发生在当前 Batch 的 H2D/前反向完成后，届时再安全地交换回来。
        self._schedule(buffer_index=1 - ready_buffer_index)
        return self.tensors[ready_buffer_index][:item_count]

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass


def validate_data_dir(config):
    """在分配模型/GPU 资源前确认所有 epoch 版本的数据均可读取。"""
    data_dir = os.path.abspath(os.path.expanduser(os.fspath(config.DATA_DIR)))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"❌ 找不到训练数据根目录: {data_dir}\n"
            "可通过环境变量 ANIME_PAINTER_DATA_DIR 指定 output_256 目录。"
        )

    missing_versions = [
        f"v{version_idx}"
        for version_idx in range(1, config.VERSION_CYCLE_EPOCHS + 1)
        if not glob.glob(os.path.join(data_dir, f"v{version_idx}", "*.csv"))
    ]
    if missing_versions:
        raise FileNotFoundError(
            f"❌ 数据目录 {data_dir} 缺少可用 CSV 的版本: "
            f"{', '.join(missing_versions)}"
        )

    config.DATA_DIR = data_dir
    print(f"📁 [Dataset] 使用数据目录: {data_dir}")


class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        if device is not None:
            self.ema.to(device=device)
        for parameter in self.ema.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_parameter, model_parameter in zip(self.ema.parameters(), model.parameters()):
            ema_parameter.data.mul_(self.decay).add_(model_parameter.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


class UnusedPositionEmbeddingProtector:
    """保护当前上下文以外的位置行及其 AdamW/EMA 状态不被继续更新。"""

    def __init__(self, model, ema_model, optimizer, active_position_count):
        self.position_parameter = model.wpe.weight
        self.ema_position_parameter = ema_model.wpe.weight
        self.optimizer = optimizer
        self.tail_start = active_position_count

        if self.position_parameter.shape != self.ema_position_parameter.shape:
            raise RuntimeError("主模型与 EMA 的位置嵌入形状不一致。")
        if not 0 < self.tail_start <= self.position_parameter.size(0):
            raise ValueError(
                f"有效位置行数必须位于 (0, 位置表行数]，收到 {self.tail_start}，"
                f"位置表行数为 {self.position_parameter.size(0)}"
            )

        self.has_protected_tail = self.tail_start < self.position_parameter.size(0)
        self.model_tail_snapshot = None
        self.ema_tail_snapshot = None
        self.optimizer_tail_snapshots = {}
        if self.has_protected_tail:
            self.model_tail_snapshot = (
                self.position_parameter.detach()[self.tail_start:].clone()
            )
            self.ema_tail_snapshot = (
                self.ema_position_parameter.detach()[self.tail_start:].clone()
            )
            self._snapshot_existing_optimizer_states()

    def _iter_position_optimizer_tensor_states(self):
        state = self.optimizer.state.get(self.position_parameter, {})
        for state_name, state_value in state.items():
            if torch.is_tensor(state_value) and state_value.shape == self.position_parameter.shape:
                yield state_name, state_value

    @torch.no_grad()
    def _snapshot_existing_optimizer_states(self):
        for state_name, state_value in self._iter_position_optimizer_tensor_states():
            self.optimizer_tail_snapshots[state_name] = (
                state_value[self.tail_start:].detach().clone()
            )

    @torch.no_grad()
    def restore_after_optimizer_step(self):
        if not self.has_protected_tail:
            return
        self.position_parameter[self.tail_start:].copy_(self.model_tail_snapshot)
        for state_name, state_value in self._iter_position_optimizer_tensor_states():
            snapshot = self.optimizer_tail_snapshots.get(state_name)
            if snapshot is None:
                # 全新训练时 AdamW 状态在第一次 step 才创建；未使用行应从零状态开始。
                snapshot = torch.zeros_like(state_value[self.tail_start:])
                self.optimizer_tail_snapshots[state_name] = snapshot
            state_value[self.tail_start:].copy_(snapshot)

    @torch.no_grad()
    def restore_after_ema_update(self):
        if not self.has_protected_tail:
            return
        self.ema_position_parameter[self.tail_start:].copy_(self.ema_tail_snapshot)

    @torch.no_grad()
    def assert_unchanged(self):
        if not self.has_protected_tail:
            return
        if not torch.equal(
            self.position_parameter[self.tail_start:], self.model_tail_snapshot
        ):
            raise RuntimeError("未使用的主模型位置嵌入发生了变化。")
        if not torch.equal(
            self.ema_position_parameter[self.tail_start:], self.ema_tail_snapshot
        ):
            raise RuntimeError("未使用的 EMA 位置嵌入发生了变化。")
        for state_name, state_value in self._iter_position_optimizer_tensor_states():
            snapshot = self.optimizer_tail_snapshots.get(state_name)
            if snapshot is None or not torch.equal(
                state_value[self.tail_start:], snapshot
            ):
                raise RuntimeError(
                    f"未使用位置行的 optimizer 状态 {state_name!r} 发生了变化。"
                )


def get_current_lr(epoch, batch_idx, batches_per_epoch, global_step, schedule_array):
    current_fractional_epoch = epoch + (batch_idx / batches_per_epoch)
    first_epoch, first_lr = schedule_array[0]
    if current_fractional_epoch < first_epoch:
        return (current_fractional_epoch / first_epoch) * first_lr

    last_epoch, last_lr = schedule_array[-1]
    if current_fractional_epoch >= last_epoch:
        return last_lr

    for index in range(len(schedule_array) - 1):
        epoch_start, lr_start = schedule_array[index]
        epoch_end, lr_end = schedule_array[index + 1]
        if epoch_start <= current_fractional_epoch < epoch_end:
            progress = (current_fractional_epoch - epoch_start) / (epoch_end - epoch_start)
            return lr_start + progress * (lr_end - lr_start)
    return last_lr


def get_current_accum_steps(smoothed_loss):
    target_accumulation = 1
    for loss_threshold, accumulation in ACCUM_SCHEDULE_BY_LOSS:
        if smoothed_loss <= loss_threshold:
            target_accumulation = accumulation
    return target_accumulation


def print_model_parameter_count(model):
    """打印单个主模型的参数量；共享参数、EMA 和多卡副本不重复计数。"""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"模型参数量: {total_parameters:,} ({total_parameters / 1e6:.2f} M) | "
        f"可训练参数量: {trainable_parameters:,} "
        f"({trainable_parameters / 1e6:.2f} M)"
    )


def get_local_accumulation(global_accumulation, device_count):
    """将全局累计次数均分到各卡，返回 (每卡次数, 实际全局次数)。"""
    if device_count < 1:
        raise ValueError("device_count 必须至少为 1。")
    if device_count == 1:
        return global_accumulation, global_accumulation
    if global_accumulation < device_count:
        return 1, device_count
    if global_accumulation % device_count != 0:
        raise ValueError(
            f"全局累计次数 {global_accumulation} 不能被 {device_count} 张卡均分；"
            f"请将 ACCUM_SCHEDULE_BY_LOSS 中的次数设为 {device_count} 的倍数。"
        )
    return global_accumulation // device_count, global_accumulation


def get_ckpt_sort_key(file_path):
    match = re.search(r"epoch_(\d+)(?:_batch_(\d+))?\.pt", file_path)
    if match:
        epoch = int(match.group(1))
        batch = int(match.group(2)) if match.group(2) else float("inf")
        return epoch, batch
    return -1, -1


def strip_module_prefix(state_dict):
    """兼容 DataParallel 保存的 module. 前缀，以及未包装模型的普通 state_dict。"""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def validate_checkpoint_position_capacity(checkpoint, config):
    """验证 checkpoint 的词表与位置容量是否匹配当前训练配置。"""
    model_state = strip_module_prefix(checkpoint["model_state_dict"])
    token_weight = model_state.get("wte.weight")
    head_weight = model_state.get("lm_head.weight")
    expected_token_shape = (config.MODEL_VOCAB_SIZE, config.MODEL_D_MODEL)
    if token_weight is None or head_weight is None:
        raise KeyError("checkpoint 缺少 wte.weight 或 lm_head.weight。")
    if (
        tuple(token_weight.shape) != expected_token_shape
        or tuple(head_weight.shape) != expected_token_shape
    ):
        raise RuntimeError(
            "checkpoint 词表权重形状与当前配置不一致："
            f"wte={tuple(token_weight.shape)}, head={tuple(head_weight.shape)}, "
            f"期望={expected_token_shape}"
        )

    saved_vocab_size = checkpoint.get("vocab_size")
    if saved_vocab_size is not None and saved_vocab_size != config.MODEL_VOCAB_SIZE:
        raise RuntimeError(
            f"checkpoint 标记 vocab_size={saved_vocab_size}，"
            f"当前配置要求 {config.MODEL_VOCAB_SIZE}。"
        )
    saved_layout_version = checkpoint.get("token_layout_version")
    if (
        saved_layout_version is not None
        and saved_layout_version != config.TOKEN_LAYOUT_VERSION
    ):
        raise RuntimeError(
            f"checkpoint 词表布局为 {saved_layout_version}，"
            f"当前配置要求 {config.TOKEN_LAYOUT_VERSION}。"
        )

    position_weight = model_state.get("wpe.weight")
    if position_weight is None:
        raise KeyError("checkpoint 缺少 wpe.weight，无法验证位置嵌入容量。")

    position_tokens = position_weight.size(0)
    if position_tokens != config.POSITION_EMBEDDING_TOKENS:
        raise RuntimeError(
            f"checkpoint 位置表容量为 {position_tokens} tokens，当前配置要求 "
            f"{config.POSITION_EMBEDDING_TOKENS} tokens。"
        )

    saved_position_steps = checkpoint.get("position_embedding_steps")
    if saved_position_steps is None:
        if position_tokens % config.TOKENS_PER_STEP != 0:
            raise RuntimeError("旧 checkpoint 的位置表行数不能换算为完整图元步数。")
        saved_position_steps = position_tokens // config.TOKENS_PER_STEP
        print(
            "--> checkpoint 不含长度元数据；已从 wpe.weight 推断位置容量为 "
            f"{saved_position_steps} 步。"
        )
    if saved_position_steps != config.POSITION_EMBEDDING_STEPS:
        raise RuntimeError(
            f"checkpoint 标记的位置容量为 {saved_position_steps} 步，当前配置要求 "
            f"{config.POSITION_EMBEDDING_STEPS} 步。"
        )

    expected_sequence_metadata = {
        "prefix_steps": config.PREFIX_STEPS,
        "prediction_steps": config.PREDICTION_STEPS,
        "context_steps": config.CONTEXT_STEPS,
    }
    for metadata_name, expected_value in expected_sequence_metadata.items():
        saved_value = checkpoint.get(metadata_name)
        if saved_value is not None and saved_value != expected_value:
            raise RuntimeError(
                f"checkpoint 标记的 {metadata_name}={saved_value}，"
                f"当前配置要求 {expected_value}。"
            )


def validate_release_config_for_training(release_config, config):
    """Refuse a release package that does not exactly match this trainer."""
    architecture = release_config["architecture"]
    sequence = release_config["sequence"]
    expected_architecture = {
        "vocab_size": config.MODEL_VOCAB_SIZE,
        "d_model": config.MODEL_D_MODEL,
        "n_layer": config.MODEL_N_LAYER,
        "n_head": config.MODEL_N_HEAD,
    }
    for key, expected_value in expected_architecture.items():
        if architecture[key] != expected_value:
            raise RuntimeError(
                f"release config {key}={architecture[key]}，"
                f"当前训练器要求 {expected_value}。"
            )

    expected_sequence = {
        "prefix_steps": config.PREFIX_STEPS,
        "prediction_steps": config.PREDICTION_STEPS,
        "context_steps": config.CONTEXT_STEPS,
        "tokens_per_step": config.TOKENS_PER_STEP,
        "position_embedding_steps": config.POSITION_EMBEDDING_STEPS,
    }
    for key, expected_value in expected_sequence.items():
        if sequence[key] != expected_value:
            raise RuntimeError(
                f"release config {key}={sequence[key]}，"
                f"当前训练器要求 {expected_value}。"
            )
    if release_config["token_layout"]["version"] != config.TOKEN_LAYOUT_VERSION:
        raise RuntimeError("release token layout 与当前训练器不一致。")


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def iter_parameter_segment_slices(parameters, segment_start, segment_length):
    """将扁平同步分段映射到各参数内连续的切片。"""
    segment_end = segment_start + segment_length
    parameter_start = 0
    for parameter in parameters:
        parameter_end = parameter_start + parameter.numel()
        overlap_start = max(segment_start, parameter_start)
        overlap_end = min(segment_end, parameter_end)
        if overlap_start < overlap_end:
            yield (
                parameter,
                overlap_start - parameter_start,
                overlap_start - segment_start,
                overlap_end - overlap_start,
            )
        parameter_start = parameter_end


class ManualMultiGPUSynchronizer:
    """主卡更新，分段汇总任意数量副卡的梯度，并向所有副卡回传权重。"""

    def __init__(self, master_model, worker_models, mode, chunk_mb):
        self.master_parameters = list(master_model.parameters())
        self.worker_parameter_groups = [
            list(worker_model.parameters()) for worker_model in worker_models
        ]
        if not self.worker_parameter_groups:
            raise ValueError("手动多卡同步至少需要一个副卡模型。")
        for worker_parameters in self.worker_parameter_groups:
            if len(self.master_parameters) != len(worker_parameters):
                raise RuntimeError("主卡和副卡模型参数数量不一致。")
            if any(
                master.shape != worker.shape
                for master, worker in zip(
                    self.master_parameters, worker_parameters
                )
            ):
                raise RuntimeError("主卡和副卡模型参数形状不一致。")

        self.mode = mode
        self.total_numel = sum(parameter.numel() for parameter in self.master_parameters)
        self.cpu_buffer = None
        self.p2p_buffer = None
        buffer_bytes = chunk_mb * 1024 * 1024
        if mode == "manual_cpu_sync":
            buffer_numel = min(self.total_numel, buffer_bytes // 4)
            if buffer_numel <= 0:
                raise ValueError("CPU 同步缓冲区大小必须至少容纳一个 float32。")
            self.cpu_buffer = torch.empty(buffer_numel, dtype=torch.float32)
        elif mode == "manual_pcie_sync":
            buffer_dtype = self.master_parameters[0].dtype
            buffer_numel = min(
                self.total_numel,
                buffer_bytes // self.master_parameters[0].element_size(),
            )
            if buffer_numel <= 0:
                raise ValueError("P2P 同步缓冲区不能为零。")
            self.p2p_buffer = torch.empty(
                buffer_numel,
                dtype=buffer_dtype,
                device=self.master_parameters[0].device,
            )
        else:
            raise ValueError(f"未知的手动同步模式: {mode}")

    def iter_segments(self):
        if self.cpu_buffer is not None:
            buffer_numel = self.cpu_buffer.numel()
        elif self.p2p_buffer is not None:
            buffer_numel = self.p2p_buffer.numel()
        else:
            buffer_numel = self.total_numel
        for segment_start in range(0, self.total_numel, buffer_numel):
            yield segment_start, min(buffer_numel, self.total_numel - segment_start)

    @torch.no_grad()
    def synchronize_worker_parameters(self):
        """主卡更新后，把新权重同步到所有副卡。"""
        for segment_start, segment_length in self.iter_segments():
            master_slices = list(
                iter_parameter_segment_slices(
                    self.master_parameters, segment_start, segment_length
                )
            )
            if self.mode == "manual_cpu_sync":
                for parameter, parameter_offset, buffer_offset, length in master_slices:
                    source = parameter.detach().reshape(-1).narrow(
                        0, parameter_offset, length
                    )
                    self.cpu_buffer.narrow(0, buffer_offset, length).copy_(source)

            for worker_parameters in self.worker_parameter_groups:
                worker_slices = list(
                    iter_parameter_segment_slices(
                        worker_parameters, segment_start, segment_length
                    )
                )
                if self.mode == "manual_cpu_sync":
                    for parameter, parameter_offset, buffer_offset, length in worker_slices:
                        destination = parameter.detach().reshape(-1).narrow(
                            0, parameter_offset, length
                        )
                        source = self.cpu_buffer.narrow(
                            0, buffer_offset, length
                        ).to(device=destination.device, dtype=destination.dtype)
                        destination.copy_(source)
                else:
                    for master_slice, worker_slice in zip(
                        master_slices, worker_slices
                    ):
                        master, master_offset, _, length = master_slice
                        worker, worker_offset, _, _ = worker_slice
                        source = master.detach().reshape(-1).narrow(
                            0, master_offset, length
                        )
                        destination = worker.detach().reshape(-1).narrow(
                            0, worker_offset, length
                        )
                        # peer-access 可用时由 HIP 执行 GPU0 -> 副卡 PCIe DMA。
                        destination.copy_(source, non_blocking=True)

    @torch.no_grad()
    def average_worker_gradients_into_master(self):
        """所有卡的梯度等权平均；只在每轮本地累计完成后调用一次。"""
        model_count = 1 + len(self.worker_parameter_groups)
        for segment_start, segment_length in self.iter_segments():
            master_slices = list(
                iter_parameter_segment_slices(
                    self.master_parameters, segment_start, segment_length
                )
            )
            for parameter, _, _, _ in master_slices:
                if parameter.grad is None:
                    raise RuntimeError("主卡存在未参与反向传播的参数。")

            for worker_parameters in self.worker_parameter_groups:
                worker_slices = list(
                    iter_parameter_segment_slices(
                        worker_parameters, segment_start, segment_length
                    )
                )
                sync_buffer = (
                    self.cpu_buffer
                    if self.mode == "manual_cpu_sync"
                    else self.p2p_buffer
                )
                for parameter, parameter_offset, buffer_offset, length in worker_slices:
                    if parameter.grad is None:
                        raise RuntimeError("副卡存在未参与反向传播的参数。")
                    worker_grad = parameter.grad.detach().reshape(-1).narrow(
                        0, parameter_offset, length
                    )
                    sync_buffer.narrow(0, buffer_offset, length).copy_(
                        worker_grad,
                        non_blocking=self.mode == "manual_pcie_sync",
                    )
                for parameter, parameter_offset, buffer_offset, length in master_slices:
                    master_grad = parameter.grad.detach().reshape(-1).narrow(
                        0, parameter_offset, length
                    )
                    worker_grad = sync_buffer.narrow(
                        0, buffer_offset, length
                    ).to(device=master_grad.device, dtype=master_grad.dtype)
                    master_grad.add_(worker_grad)

            for parameter, parameter_offset, _, length in master_slices:
                parameter.grad.detach().reshape(-1).narrow(
                    0, parameter_offset, length
                ).div_(model_count)


def require_gpus(device_ids):
    if not torch.cuda.is_available():
        raise RuntimeError("多卡模式需要 CUDA/HIP GPU。")
    if torch.cuda.device_count() <= max(device_ids):
        raise RuntimeError(
            f"检测到 {torch.cuda.device_count()} 张 GPU，但设备编号 {device_ids} 不可用。"
        )


def peer_access_available(device_ids):
    if not hasattr(torch.cuda, "can_device_access_peer"):
        return False
    try:
        master_device_id = device_ids[0]
        return all(
            torch.cuda.can_device_access_peer(master_device_id, worker_device_id)
            and torch.cuda.can_device_access_peer(worker_device_id, master_device_id)
            for worker_device_id in device_ids[1:]
        )
    except RuntimeError:
        return False


def report_rocm_power_profile(device_ids):
    """提示会让 RDNA3 训练吞吐剧烈波动的 Linux 自动功耗档。"""
    if torch.version.hip is None or os.name == "nt":
        return
    level_paths = sorted(
        glob.glob(
            "/sys/class/drm/card*/device/"
            "power_dpm_force_performance_level"
        )
    )
    levels = []
    for level_path in level_paths:
        try:
            with open(level_path, "r", encoding="utf-8") as level_file:
                levels.append(level_file.read().strip())
        except OSError:
            continue
    if levels and any(level.lower() == "auto" for level in levels):
        device_id_arguments = " ".join(str(device_id) for device_id in device_ids)
        print(
            "⚠️ [ROCm Power] 检测到 AMDGPU 性能档仍为 AUTO。"
            "本机实测训练时 GFXCLK 会跌到约 100 MHz，造成窗口从 "
            "57s 波动到 60/66/99s。建议训练前执行：\n"
            "    sudo amd-smi set -P COMPUTE_MASK -l HIGH -g "
            f"{device_id_arguments}\n"
            "训练结束可恢复：\n"
            "    sudo amd-smi set -P BOOTUP_DEFAULT -l AUTO -g "
            f"{device_id_arguments}"
        )


def require_rocm_numa_balancing_disabled():
    """避免自动 NUMA 页迁移驱逐 KFD/HSA 计算队列。"""
    if torch.version.hip is None or os.name == "nt":
        return
    numa_balancing_path = "/proc/sys/kernel/numa_balancing"
    try:
        with open(
            numa_balancing_path, "r", encoding="utf-8"
        ) as balancing_file:
            balancing_enabled = balancing_file.read().strip() != "0"
    except OSError as error:
        print(
            "⚠️ [ROCm NUMA] 无法读取 "
            f"{numa_balancing_path}: {error}"
        )
        return
    if balancing_enabled:
        raise RuntimeError(
            "检测到 kernel.numa_balancing=1。当前训练进程已经出现多卡 "
            "KFD 队列被驱逐约 728 秒；自动 NUMA 页迁移会使 ROCm GPU "
            "在前向/反向中无响应，直到 userptr/SVM 恢复。\n"
            "请先执行：sudo sysctl kernel.numa_balancing=0\n"
            "确认命令：cat /proc/sys/kernel/numa_balancing  # 应输出 0"
        )


class KFDEvictionMonitor:
    """读取当前进程每个 KFD 节点累计的队列驱逐毫秒数。"""

    def __init__(self):
        self.previous = self._read()

    @staticmethod
    def _read():
        values = {}
        pattern = (
            f"/sys/class/kfd/kfd/proc/{os.getpid()}/"
            "stats_*/evicted_ms"
        )
        for evicted_path in glob.glob(pattern):
            node_id = os.path.basename(
                os.path.dirname(evicted_path)
            ).removeprefix("stats_")
            try:
                with open(
                    evicted_path, "r", encoding="utf-8"
                ) as evicted_file:
                    values[node_id] = int(evicted_file.read().strip())
            except (OSError, ValueError):
                continue
        return values

    def consume_deltas(self):
        current = self._read()
        deltas = {
            node_id: value - self.previous.get(node_id, value)
            for node_id, value in current.items()
            if value > self.previous.get(node_id, value)
        }
        self.previous = current
        return deltas


def compute_loss(model, input_ids, config, criterion):
    logits = model(input_ids)
    targets = input_ids.clone()
    targets[:, :config.PREFIX_TOKENS] = -100
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()
    return criterion(
        shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)
    )


def warm_up_training_backend(
    config,
    model,
    master_model,
    worker_models,
    primary_device,
    device_ids,
    dtype,
    criterion,
    autocast_enabled,
    is_manual_parallel,
    is_data_parallel,
):
    """在统计训练窗口前加载 Linear/SDPA 的前向与反向 GPU kernel。"""
    if primary_device.type != "cuda":
        return

    print(
        "🔥 [ROCm Warmup] 开始后端预热；仅执行前向/反向，不更新模型参数..."
    )
    for device_id in (
        device_ids if (is_manual_parallel or is_data_parallel) else [primary_device]
    ):
        torch.cuda.synchronize(device_id)
    warmup_started_at = time.perf_counter()

    def warm_up_one(target_model, device, batch_size):
        torch.cuda.set_device(device)
        input_ids = torch.zeros(
            (batch_size, config.CONTEXT_TOKENS),
            dtype=torch.long,
            device=device,
        )
        started_at = time.perf_counter()
        with torch.amp.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            loss = compute_loss(target_model, input_ids, config, criterion)
        loss.backward()
        torch.cuda.synchronize(device)
        target_model.zero_grad(set_to_none=True)
        return time.perf_counter() - started_at

    if is_manual_parallel:
        # 各卡的 hipBLASLt 上下文相互独立；并行预热可重叠代码对象加载。
        warmup_models = [master_model, *worker_models]
        with ThreadPoolExecutor(
            max_workers=len(warmup_models), thread_name_prefix="rocm-warmup"
        ) as executor:
            futures = [
                executor.submit(
                    warm_up_one,
                    target_model,
                    torch.device(f"cuda:{device_id}"),
                    config.MICRO_BATCH_SIZE,
                )
                for target_model, device_id in zip(
                    warmup_models, device_ids
                )
            ]
            device_times = [future.result() for future in futures]
    else:
        warmup_batch_size = config.MICRO_BATCH_SIZE * (
            len(device_ids) if is_data_parallel else 1
        )
        device_times = [
            warm_up_one(model, primary_device, warmup_batch_size)
        ]

    print(
        f"✅ [ROCm Warmup] 后端预热完成 | wall="
        f"{time.perf_counter() - warmup_started_at:.3f}s | "
        f"device_times={[round(value, 3) for value in device_times]} | "
        "参数未更新"
    )


def run_manual_gpu_microbatch(
    model,
    host_input_ids,
    device,
    config,
    criterion,
    dtype,
    autocast_enabled,
    local_accum_steps,
):
    """在固定 GPU 工作线程上完成一个 micro-batch 的前向和反向。"""
    torch.cuda.set_device(device)
    input_ids = host_input_ids.to(device, non_blocking=True)
    with torch.amp.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        loss = compute_loss(model, input_ids, config, criterion)
        scaled_loss = loss / local_accum_steps
    scaled_loss.backward()
    # 这只保证当前 pinned buffer 可安全复用，不传递梯度/参数。
    # 真正的跨卡同步仍只在完整 local_accum_steps 窗口末尾发生一次。
    return loss.detach().item()


def save_checkpoint(path, epoch, batch_idx, global_step, smoothed_loss, epoch_total_loss,
                    model, ema, optimizer, scaler, parallel_mode, config):
    torch.save(
        {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": global_step,
            "smoothed_loss": smoothed_loss,
            "epoch_total_loss": epoch_total_loss,
            "parallel_mode": parallel_mode,
            "prefix_steps": config.PREFIX_STEPS,
            "prediction_steps": config.PREDICTION_STEPS,
            "context_steps": config.CONTEXT_STEPS,
            "position_embedding_steps": config.POSITION_EMBEDDING_STEPS,
            "canvas_size": config.CANVAS_SIZE,
            "max_shape_size": config.MAX_SHAPE_SIZE,
            "vocab_size": config.MODEL_VOCAB_SIZE,
            "token_layout_version": config.TOKEN_LAYOUT_VERSION,
            "token_layout": TOKEN_LAYOUT.to_metadata(),
            "model_state_dict": unwrap_model(model).state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        path,
    )


def run_training(initial_model_dir=None):
    config = TrainConfig()
    config.validate_sequence_config()
    config.validate_model_config()
    config.validate_parallel_config()
    validate_data_dir(config)
    parallel_mode = config.PARALLEL_MODE
    is_manual_parallel = parallel_mode in {"manual_cpu_sync", "manual_pcie_sync"}
    is_data_parallel = parallel_mode == "data_parallel"
    is_multi_gpu = is_manual_parallel or is_data_parallel

    device_ids = list(config.PARALLEL_DEVICE_IDS) if is_multi_gpu else []
    if is_multi_gpu:
        require_gpus(device_ids)
        require_rocm_numa_balancing_disabled()
        report_rocm_power_profile(device_ids)
        if parallel_mode == "manual_pcie_sync" and not peer_access_available(device_ids):
            raise RuntimeError(
                "manual_pcie_sync 需要主卡与每张副卡双向互访；"
                "当前不可用，请改用 manual_cpu_sync。"
            )
        if parallel_mode == "manual_pcie_sync":
            print(
                "🔗 [P2P] 主卡与所有副卡的双向 GPU peer-access 已确认；"
                "梯度与参数通过 PCIe DMA 直传，CPU staging 已禁用。"
            )
        primary_device = torch.device(f"cuda:{device_ids[0]}")
        torch.cuda.set_device(primary_device)
    else:
        primary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"========== 启动全序列逐 Token GPT 预训练 | 并行模式: {parallel_mode} | "
        f"GPU 数: {torch.cuda.device_count()} | 使用设备: "
        f"{device_ids if is_multi_gpu else [primary_device]} =========="
    )
    print(
        f"序列配置: 前缀 {config.PREFIX_STEPS} 步 + 预测 {config.PREDICTION_STEPS} 步 "
        f"= 上下文 {config.CONTEXT_STEPS} 步 ({config.CONTEXT_TOKENS} tokens) | "
        f"位置嵌入容量 {config.POSITION_EMBEDDING_STEPS} 步 "
        f"({config.POSITION_EMBEDDING_TOKENS} tokens)"
    )
    if torch.version.hip is not None:
        print(
            f"ROCm 启动配置: MIOPEN_FIND_MODE="
            f"{os.environ.get('MIOPEN_FIND_MODE')} | "
            f"ROCBLAS_USE_HIPBLASLT="
            f"{os.environ.get('ROCBLAS_USE_HIPBLASLT', 'AUTO')} | "
            "本模型主体为 Linear/SDPA，主要后端是 rocBLAS/AOTriton。"
        )
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    epochs = 100000
    weight_decay = 2e-2
    start_epoch = 1
    resume_batch_idx = -1
    global_step = 0
    smoothed_loss = None
    resume_total_loss = 0.0

    dataset = GeometrizeFullSequenceDataset(config=config)
    master_model = GeometrizeGPT(
        vocab_size=config.MODEL_VOCAB_SIZE,
        d_model=config.MODEL_D_MODEL,
        n_layer=config.MODEL_N_LAYER,
        n_head=config.MODEL_N_HEAD,
        max_context_len=config.CONTEXT_TOKENS,
        max_position_embeddings=config.POSITION_EMBEDDING_TOKENS,
    )

    checkpoint = None
    ckpt_files = glob.glob(os.path.join(config.CKPT_DIR, "geometrize_gpt_pretrain_epoch_*.pt"))
    if ckpt_files and initial_model_dir is not None:
        raise RuntimeError(
            "发现本地训练断点时不能同时使用 --initial-model-dir。"
            "请删除/移动本地断点，或不传该参数以继续现有训练。"
        )
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=get_ckpt_sort_key)
        print(f"--> 发现历史存档！正在读取 {latest_ckpt}...")
        checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        validate_checkpoint_position_capacity(checkpoint, config)
        master_model.load_state_dict(
            strip_module_prefix(checkpoint["model_state_dict"]), strict=True
        )
    elif initial_model_dir is not None:
        initial_directory = Path(initial_model_dir).expanduser()
        if not initial_directory.is_dir():
            raise FileNotFoundError(
                f"--initial-model-dir 必须是本地模型包目录：{initial_directory}"
            )
        model_directory, release_config = load_release_config(initial_directory)
        validate_release_config_for_training(release_config, config)
        load_safetensors_model(
            master_model,
            str(model_directory / release_config["weights"]["file"]),
            strict=True,
            device="cpu",
        )
        print(f"--> 已载入 EMA 初始权重: {model_directory}")

    master_model.to(primary_device)

    worker_models = []
    synchronizer = None
    if is_manual_parallel:
        worker_models = [
            GeometrizeGPT(
                vocab_size=config.MODEL_VOCAB_SIZE,
                d_model=config.MODEL_D_MODEL,
                n_layer=config.MODEL_N_LAYER,
                n_head=config.MODEL_N_HEAD,
                max_context_len=config.CONTEXT_TOKENS,
                max_position_embeddings=config.POSITION_EMBEDDING_TOKENS,
            ).to(torch.device(f"cuda:{device_id}"))
            for device_id in device_ids[1:]
        ]
        synchronizer = ManualMultiGPUSynchronizer(
            master_model,
            worker_models,
            parallel_mode,
            config.PARALLEL_SYNC_BUFFER_CHUNK_MB,
        )
        synchronizer.synchronize_worker_parameters()
        # 初始化回传使用 non_blocking P2P copy；预热线程读取副本前必须完成交接。
        for device_id in device_ids:
            torch.cuda.synchronize(device_id)

    model = master_model
    if is_data_parallel:
        model = nn.DataParallel(
            master_model, device_ids=device_ids, output_device=device_ids[0]
        )

    ema = ModelEMA(master_model, decay=0.999, device=primary_device)
    optimizer = torch.optim.AdamW(
        master_model.parameters(),
        lr=1e-4,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(primary_device)
    autocast_enabled = primary_device.type == "cuda"
    dtype = torch.bfloat16 if autocast_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(
        primary_device.type, enabled=(autocast_enabled and dtype == torch.float16)
    )

    if checkpoint is not None:
        if "ema_state_dict" in checkpoint:
            ema.ema.load_state_dict(strip_module_prefix(checkpoint["ema_state_dict"]), strict=True)
        else:
            ema.update(master_model)
        start_epoch = checkpoint["epoch"]
        resume_batch_idx = checkpoint.get("batch_idx", -1)
        global_step = checkpoint.get("global_step", 0)
        smoothed_loss = checkpoint.get("smoothed_loss", 10.0)
        resume_total_loss = checkpoint.get("epoch_total_loss", 0.0)
        if resume_batch_idx == -1:
            start_epoch += 1
            resume_total_loss = 0.0
        if not RESET_OPTIMIZER_STATE and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(
            f"--> 将从 Epoch {start_epoch}, Batch {resume_batch_idx + 1} 继续。"
            f"历史平滑 Loss: {smoothed_loss:.4f}"
        )

    position_protector = UnusedPositionEmbeddingProtector(
        master_model,
        ema.ema,
        optimizer,
        active_position_count=config.CONTEXT_TOKENS,
    )
    if position_protector.has_protected_tail:
        protected_optimizer_states = sorted(
            position_protector.optimizer_tail_snapshots.keys()
        )
        print(
            f"🔒 位置嵌入保护已启用：训练行 0…{config.CONTEXT_TOKENS - 1}，"
            f"冻结行 {config.CONTEXT_TOKENS}…{config.POSITION_EMBEDDING_TOKENS - 1} | "
            f"已加载 optimizer 张量状态: {protected_optimizer_states or '尚未创建'}"
        )
    else:
        print(
            "🔓 当前上下文已覆盖完整位置嵌入表：全部位置行参与训练，"
            "无需冻结尾部。"
        )

    print_model_parameter_count(master_model)
    if checkpoint is not None:
        del checkpoint
        gc.collect()
        print("🧹 已释放 checkpoint 的 CPU 临时副本。")

    # 若当前 Epoch 尚无缓存，让多核 CSV 转换尽早与 GPU 后端预热重叠。
    dataset.start_next_epoch_cache_prefetch(start_epoch)
    warm_up_training_backend(
        config=config,
        model=model,
        master_model=master_model,
        worker_models=worker_models,
        primary_device=primary_device,
        device_ids=device_ids,
        dtype=dtype,
        criterion=criterion,
        autocast_enabled=autocast_enabled,
        is_manual_parallel=is_manual_parallel,
        is_data_parallel=is_data_parallel,
    )
    kfd_eviction_monitor = (
        KFDEvictionMonitor() if torch.version.hip is not None else None
    )
    print("\nStarting Training Loop...")
    first_optimizer_step_verified = False
    first_training_batch_verified = False
    device_count_for_accumulation = len(device_ids) if is_multi_gpu else 1
    loader_batch_size = config.MICRO_BATCH_SIZE * device_count_for_accumulation
    manual_step_executor = None
    if is_manual_parallel and not scaler.is_enabled():
        manual_step_executor = ThreadPoolExecutor(
            max_workers=device_count_for_accumulation,
            thread_name_prefix="manual-multi-step",
        )
        print(
            f"⚙️ [Parallel] {device_count_for_accumulation} 卡微批前向/反向"
            "由持久线程并发提交；"
            f"梯度同步方式仍为 {parallel_mode}。"
        )
        if synchronizer is not None and synchronizer.p2p_buffer is not None:
            print(
                "🔁 [P2P] 每张卡独立累计完整窗口后才跨卡同步一次 | "
                f"reusable_gpu_buffer="
                f"{synchronizer.p2p_buffer.numel() * synchronizer.p2p_buffer.element_size() / 2**20:.0f} MiB"
            )
    elif is_manual_parallel:
        print(
            "⚠️ [Parallel] FP16 GradScaler 已启用，为保证缩放状态安全，"
            "多卡微批沿用主线程顺序提交。"
        )
    manual_models = [master_model, *worker_models]
    manual_devices = [
        torch.device(f"cuda:{device_id}") for device_id in device_ids
    ] if is_manual_parallel else []
    if config.DATALOADER_NUM_WORKERS != 0:
        print(
            "⚠️ [DataPipeline] 当前 mmap 管线不使用 DataLoader 子进程；"
            f"忽略 DATALOADER_NUM_WORKERS={config.DATALOADER_NUM_WORKERS}。"
        )

    for epoch in range(start_epoch, epochs + 1):
        dataset.set_epoch(epoch)
        drop_last = is_multi_gpu
        if drop_last:
            batches_per_epoch = len(dataset) // loader_batch_size
        else:
            batches_per_epoch = (
                len(dataset) + loader_batch_size - 1
            ) // loader_batch_size

        epoch_start_batch = (
            resume_batch_idx + 1
            if epoch == start_epoch and resume_batch_idx >= 0
            else 0
        )
        if epoch_start_batch > batches_per_epoch:
            raise RuntimeError(
                f"checkpoint 要求从 Batch {epoch_start_batch} 恢复，"
                f"但当前 Epoch 只有 {batches_per_epoch} 个 Batch。"
            )

        start_sample_index = min(
            epoch_start_batch * loader_batch_size, len(dataset)
        )
        if epoch_start_batch > 0:
            print(
                f"⏩ [DataPipeline] 直接定位到 Batch {epoch_start_batch}/"
                f"{batches_per_epoch}（样本偏移 {start_sample_index}），"
                f"跳过构造前 {epoch_start_batch} 个已训练 Batch。"
            )

        remaining_sample_count = len(dataset) - start_sample_index
        if drop_last:
            remaining_batches = remaining_sample_count // loader_batch_size
        else:
            remaining_batches = (
                remaining_sample_count + loader_batch_size - 1
            ) // loader_batch_size
        expected_remaining_batches = batches_per_epoch - epoch_start_batch
        if remaining_batches != expected_remaining_batches:
            raise RuntimeError(
                "恢复后的批次数计算错误："
                f"实际 {remaining_batches}，期望 {expected_remaining_batches}。"
            )
        print(
            f"🚚 [DataPipeline] 已创建 | remaining_batches={remaining_batches} | "
            f"total_batches={batches_per_epoch} | "
            f"batch_size={loader_batch_size} | mmap_on_demand=True | "
            f"reusable_pinned_buffers=2 | async_ping_pong=True | "
            f"background_next_epoch_cache=True"
        )

        model.train()
        for worker_model in worker_models:
            worker_model.train()
        total_loss = resume_total_loss if epoch == start_epoch else 0.0
        optimizer.zero_grad(set_to_none=True)
        for worker_model in worker_models:
            worker_model.zero_grad(set_to_none=True)
        accum_counter = 0
        local_accum_steps = 1
        effective_global_accum = 1
        last_log_time = time.perf_counter()
        last_log_batch_idx = epoch_start_batch - 1
        window_data_wait_seconds = 0.0
        window_async_fill_seconds = 0.0
        window_update_seconds = 0.0
        window_update_count = 0
        next_epoch_cache_prefetch_started = False
        next_save_batch = config.SAVE_INTERVAL_BATCHES
        if epoch == start_epoch and resume_batch_idx >= 0:
            next_save_batch = (
                (resume_batch_idx // config.SAVE_INTERVAL_BATCHES) + 1
            ) * config.SAVE_INTERVAL_BATCHES

        first_batch_wait_started = time.perf_counter()
        first_batch_received = False
        batch_iterator = MMapBatchIterator(
            dataset,
            start_sample_index,
            loader_batch_size,
            drop_last,
            pin_memory=primary_device.type == "cuda",
        )
        for batch_idx, input_ids in enumerate(
            batch_iterator, start=epoch_start_batch
        ):
            window_data_wait_seconds += batch_iterator.last_wait_seconds
            window_async_fill_seconds += batch_iterator.last_fill_seconds
            if not first_batch_received:
                print(
                    f"✅ [DataPipeline] 首批数据已就绪 | shape={tuple(input_ids.shape)} | "
                    f"等待 {time.perf_counter() - first_batch_wait_started:.3f}s"
                )
                first_batch_received = True
            first_training_batch_started_at = (
                time.perf_counter()
                if not first_training_batch_verified
                else None
            )
            if input_ids.size(1) != config.CONTEXT_TOKENS:
                raise RuntimeError(
                    f"训练序列长度应为 {config.CONTEXT_TOKENS}，"
                    f"实际收到 {input_ids.size(1)}"
                )
            if is_multi_gpu and input_ids.size(0) != loader_batch_size:
                raise RuntimeError(
                    "多卡训练批次未被均分；请检查数据管线的 drop_last 设置。"
                )

            if accum_counter == 0:
                desired_global_accum = get_current_accum_steps(
                    smoothed_loss if smoothed_loss is not None else float("inf")
                )
                local_accum_steps, effective_global_accum = get_local_accumulation(
                    desired_global_accum, device_count_for_accumulation
                )
                local_accum_steps = min(
                    local_accum_steps, batches_per_epoch - batch_idx
                )
                effective_global_accum = (
                    local_accum_steps * device_count_for_accumulation
                )

            current_base_lr = get_current_lr(
                epoch,
                batch_idx,
                batches_per_epoch,
                global_step,
                LR_SCHEDULE_BY_EPOCH,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_base_lr

            if manual_step_executor is not None:
                microbatch_inputs = input_ids.split(
                    config.MICRO_BATCH_SIZE, dim=0
                )
                futures = [
                    manual_step_executor.submit(
                        run_manual_gpu_microbatch,
                        target_model,
                        host_input,
                        device,
                        config,
                        criterion,
                        dtype,
                        autocast_enabled,
                        local_accum_steps,
                    )
                    for target_model, host_input, device in zip(
                        manual_models,
                        microbatch_inputs,
                        manual_devices,
                    )
                ]
                microbatch_losses = [
                    future.result() for future in futures
                ]
                mean_loss = (
                    sum(microbatch_losses) / len(microbatch_losses)
                )
            else:
                with torch.amp.autocast(
                    device_type=primary_device.type,
                    dtype=dtype,
                    enabled=autocast_enabled,
                ):
                    if is_manual_parallel:
                        microbatch_inputs = input_ids.split(
                            config.MICRO_BATCH_SIZE, dim=0
                        )
                        microbatch_losses = []
                        scaled_microbatch_losses = []
                        for target_model, host_input, device in zip(
                            manual_models,
                            microbatch_inputs,
                            manual_devices,
                        ):
                            device_input = host_input.to(
                                device, non_blocking=True
                            )
                            # criterion 无参数和缓冲，可安全复用于所有 GPU。
                            microbatch_loss = compute_loss(
                                target_model, device_input, config, criterion
                            )
                            microbatch_losses.append(microbatch_loss)
                            scaled_microbatch_losses.append(
                                microbatch_loss / local_accum_steps
                            )
                        mean_loss = sum(
                            value.detach().item()
                            for value in microbatch_losses
                        ) / len(microbatch_losses)
                    else:
                        batch_input = input_ids.to(
                            primary_device, non_blocking=True
                        )
                        loss = compute_loss(model, batch_input, config, criterion)
                        scaled_loss = loss / local_accum_steps
                        mean_loss = loss.detach().item()

                if is_manual_parallel:
                    for scaled_microbatch_loss in scaled_microbatch_losses:
                        scaler.scale(scaled_microbatch_loss).backward()
                else:
                    scaler.scale(scaled_loss).backward()

            if not first_training_batch_verified:
                if primary_device.type == "cuda":
                    verification_device_ids = (
                        device_ids if is_multi_gpu else [primary_device]
                    )
                    for verification_device_id in verification_device_ids:
                        torch.cuda.synchronize(verification_device_id)
                print(
                    "✅ 首个实际训练 Batch 已完成前向与反向 | "
                    f"Batch={batch_idx}/{batches_per_epoch} | "
                    f"Accum(Local)={1}/{local_accum_steps} | "
                    f"耗时={time.perf_counter() - first_training_batch_started_at:.3f}s"
                )
                first_training_batch_verified = True

            smoothed_loss = (
                mean_loss
                if smoothed_loss is None
                else 0.99 * smoothed_loss + 0.01 * mean_loss
            )
            total_loss += mean_loss
            accum_counter += 1
            will_step = accum_counter >= local_accum_steps

            if will_step:
                update_started_at = time.perf_counter()
                if synchronizer is not None:
                    synchronizer.average_worker_gradients_into_master()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(master_model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                position_protector.restore_after_optimizer_step()
                ema.update(master_model)
                position_protector.restore_after_ema_update()
                if synchronizer is not None:
                    synchronizer.synchronize_worker_parameters()
                if primary_device.type == "cuda":
                    # 多线程模式下，副卡参数由主线程回传、下一批由固定
                    # worker 线程消费；必须在所有设备都完成后再跨线程交接。
                    update_device_ids = (
                        device_ids if is_multi_gpu else [primary_device]
                    )
                    for update_device_id in update_device_ids:
                        torch.cuda.synchronize(update_device_id)
                optimizer.zero_grad(set_to_none=True)
                for worker_model in worker_models:
                    worker_model.zero_grad(set_to_none=True)
                accum_counter = 0
                global_step += 1
                window_update_seconds += (
                    time.perf_counter() - update_started_at
                )
                window_update_count += 1

                if not first_optimizer_step_verified:
                    position_protector.assert_unchanged()
                    print(
                        "✅ 首个梯度累计窗口已完成 optimizer.step 验证 | "
                        f"每卡分别累计 {local_accum_steps} 个小批量；"
                        "micro-batch 间不传递梯度/参数，窗口末仅跨卡同步 1 次；"
                        "未使用的位置嵌入/EMA/AdamW 张量状态保持不变 | "
                        f"模式={parallel_mode} | 设备="
                        f"{device_ids if is_multi_gpu else [primary_device]}"
                    )
                    first_optimizer_step_verified = True

                if batch_idx >= next_save_batch:
                    ckpt_path = os.path.join(
                        config.CKPT_DIR,
                        f"geometrize_gpt_pretrain_epoch_{epoch}_batch_{batch_idx}.pt",
                    )
                    save_checkpoint(
                        ckpt_path, epoch, batch_idx, global_step, smoothed_loss,
                        total_loss, model, ema, optimizer, scaler, parallel_mode, config
                    )
                    print(f"--> 💾 [安全检查点] 进度存档已建立: {ckpt_path}\n")
                    next_save_batch += config.SAVE_INTERVAL_BATCHES

            if batch_idx % 80 == 0:
                if primary_device.type == "cuda":
                    timing_device_ids = (
                        device_ids if is_multi_gpu else [primary_device]
                    )
                    for timing_device_id in timing_device_ids:
                        torch.cuda.synchronize(timing_device_id)
                elapsed_seconds = time.perf_counter() - last_log_time
                window_batches = batch_idx - last_log_batch_idx
                train_compute_seconds = max(
                    0.0,
                    elapsed_seconds
                    - window_data_wait_seconds
                    - window_update_seconds,
                )
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Batch {batch_idx}/{batches_per_epoch} | "
                    f"Loss: {smoothed_loss:.4f} | "
                    f"Accum(Global/Local): {effective_global_accum}/{local_accum_steps} | "
                    f"Eq_BS: {effective_global_accum * config.MICRO_BATCH_SIZE} | "
                    f"LR_GPT: {current_base_lr:.2e} | "
                    f"Window: {window_batches} batches / {elapsed_seconds:.2f}s "
                    f"({elapsed_seconds / max(window_batches, 1):.3f}s per batch)"
                )
                print(
                    f"  ↳ Pipeline: data_wait={window_data_wait_seconds:.3f}s | "
                    f"async_mmap_fill={window_async_fill_seconds:.3f}s(overlapped) | "
                    f"train_compute={train_compute_seconds:.3f}s | "
                    f"optimizer_sync={window_update_seconds:.3f}s/"
                    f"{window_update_count} updates "
                    f"({window_update_seconds / max(window_update_count, 1):.3f}s/update) | "
                    "（墙钟分解，无逐 Batch GPU event 扰动）"
                )
                if primary_device.type == "cuda":
                    memory_status = []
                    for memory_device_id in timing_device_ids:
                        memory_status.append(
                            "GPU"
                            f"{memory_device_id}:"
                            f"alloc={torch.cuda.memory_allocated(memory_device_id) / 2**30:.2f}GiB,"
                            f"reserved={torch.cuda.memory_reserved(memory_device_id) / 2**30:.2f}GiB,"
                            f"peak={torch.cuda.max_memory_allocated(memory_device_id) / 2**30:.2f}GiB"
                        )
                    print(
                        "  ↳ Runtime: "
                        + " | ".join(memory_status)
                        + f" | gc_counts={gc.get_count()}"
                    )
                if kfd_eviction_monitor is not None:
                    eviction_deltas = (
                        kfd_eviction_monitor.consume_deltas()
                    )
                    if eviction_deltas:
                        formatted_deltas = ", ".join(
                            f"KFD{node_id}=+{delta_ms / 1000:.3f}s"
                            for node_id, delta_ms in sorted(
                                eviction_deltas.items()
                            )
                        )
                        print(
                            "  🚨 [KFD Queue Eviction] "
                            f"{formatted_deltas} | 这段时间 GPU 队列未执行；"
                            "不是模型计算或数据加载耗时。"
                        )
                if not next_epoch_cache_prefetch_started:
                    next_epoch_cache_prefetch_started = (
                        dataset.start_next_epoch_cache_prefetch(epoch + 1)
                    )
                last_log_time = time.perf_counter()
                last_log_batch_idx = batch_idx
                window_data_wait_seconds = 0.0
                window_async_fill_seconds = 0.0
                window_update_seconds = 0.0
                window_update_count = 0
        batch_iterator.close()

        if not next_epoch_cache_prefetch_started:
            dataset.start_next_epoch_cache_prefetch(epoch + 1)
        avg_loss = total_loss / max(batches_per_epoch, 1)
        print(f"==== Epoch {epoch} Completed | Avg Epoch Loss: {avg_loss:.4f} ====\n")
        ckpt_path = os.path.join(config.CKPT_DIR, f"geometrize_gpt_pretrain_epoch_{epoch}.pt")
        save_checkpoint(
            ckpt_path, epoch, -1, global_step, smoothed_loss, total_loss,
            model, ema, optimizer, scaler, parallel_mode, config
        )
        print(f"--> Saved checkpoint: {ckpt_path}\n")
        resume_batch_idx = -1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or continue training the 144-step Primitive Operation Painter model."
    )
    parser.add_argument(
        "--initial-model-dir",
        type=Path,
        default=None,
        help=(
            "Local release package containing config.json and model.safetensors. "
            "Used only when no local resumable checkpoint exists."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args().initial_model_dir)
