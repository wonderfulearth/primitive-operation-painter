# config.py
import os
from pathlib import Path

from token_layout import TOKEN_LAYOUT, TOKEN_LAYOUT_VERSION


def resolve_data_dir():
    """解析训练数据目录；公开版本不包含任何个人机器的绝对路径。"""
    env_data_dir = os.environ.get("ANIME_PAINTER_DATA_DIR")
    if env_data_dir:
        return os.path.abspath(os.path.expanduser(env_data_dir))

    project_data_dir = (
        Path(__file__).resolve().parent.parent
        / "High_reso_dataset"
        / "fast_shape_renderer"
        / "output_256"
    )
    return str(project_data_dir)


def resolve_data_cache_dir():
    """缓存放在项目目录内，也可通过环境变量迁移到其他高速磁盘。"""
    env_cache_dir = os.environ.get("ANIME_PAINTER_DATA_CACHE_DIR")
    if env_cache_dir:
        return os.path.abspath(os.path.expanduser(env_cache_dir))
    return str(
        Path(__file__).resolve().parent / "dataset_cache_gpt_full_sequence_256"
    )


class TrainConfig:
    # ==========================================
    # 📁 路径配置
    # ==========================================
    # 可用 ANIME_PAINTER_DATA_DIR 显式覆盖；Linux 默认使用项目旁的数据目录。
    DATA_DIR = resolve_data_dir()
    DATA_CACHE_DIR = resolve_data_cache_dir()
    # 公开的 EMA 权重使用 144-step context；本地续训断点保存在此目录。
    CKPT_DIR = "checkpoints_gpt_fullseq_144ctx_256reso"
    CANVAS_SIZE = 256
    MAX_SHAPE_SIZE = 128
    TOKEN_LAYOUT_VERSION = TOKEN_LAYOUT_VERSION

    # ==========================================
    # 📐 自回归上下文与位置嵌入容量
    # ==========================================
    # 每个图元固定由 9 个离散 token 表示。
    PREFIX_STEPS = 10
    PREDICTION_STEPS = 134
    CONTEXT_STEPS = 144
    TOKENS_PER_STEP = 9

    # 实际送入模型的训练/推理上下文：10 步前缀 + 134 步预测。
    CONTEXT_TOKENS = CONTEXT_STEPS * TOKENS_PER_STEP
    PREFIX_TOKENS = PREFIX_STEPS * TOKENS_PER_STEP

    # 保留原 256 步位置表以严格复用权重；训练器会冻结未使用的后 112 步。
    POSITION_EMBEDDING_STEPS = 256
    POSITION_EMBEDDING_TOKENS = POSITION_EMBEDDING_STEPS * TOKENS_PER_STEP
    # output_256 每张图包含 1 个背景步和 275 个绘制步。
    MAX_IMAGE_STEPS = 276

    def validate_sequence_config(self):
        if self.PREFIX_STEPS <= 0 or self.PREDICTION_STEPS <= 0:
            raise ValueError("PREFIX_STEPS 和 PREDICTION_STEPS 必须为正整数")
        if self.PREFIX_STEPS + self.PREDICTION_STEPS != self.CONTEXT_STEPS:
            raise ValueError(
                "PREFIX_STEPS + PREDICTION_STEPS 必须等于 CONTEXT_STEPS，"
                f"当前为 {self.PREFIX_STEPS} + {self.PREDICTION_STEPS} "
                f"!= {self.CONTEXT_STEPS}"
            )
        if self.CONTEXT_STEPS > self.POSITION_EMBEDDING_STEPS:
            raise ValueError(
                "CONTEXT_STEPS 不能超过 POSITION_EMBEDDING_STEPS，"
                f"当前为 {self.CONTEXT_STEPS} > {self.POSITION_EMBEDDING_STEPS}"
            )
        if self.CONTEXT_TOKENS != self.CONTEXT_STEPS * self.TOKENS_PER_STEP:
            raise ValueError("CONTEXT_TOKENS 与 CONTEXT_STEPS * TOKENS_PER_STEP 不一致")
        if self.PREFIX_TOKENS != self.PREFIX_STEPS * self.TOKENS_PER_STEP:
            raise ValueError("PREFIX_TOKENS 与 PREFIX_STEPS * TOKENS_PER_STEP 不一致")
        supervised_tokens = self.CONTEXT_TOKENS - self.PREFIX_TOKENS
        if supervised_tokens != self.PREDICTION_STEPS * self.TOKENS_PER_STEP:
            raise ValueError("预测区 token 数与 PREDICTION_STEPS 不一致")
        if self.POSITION_EMBEDDING_TOKENS != (
            self.POSITION_EMBEDDING_STEPS * self.TOKENS_PER_STEP
        ):
            raise ValueError(
                "POSITION_EMBEDDING_TOKENS 与 "
                "POSITION_EMBEDDING_STEPS * TOKENS_PER_STEP 不一致"
            )

    # ==========================================
    # 🧠 模型尺寸
    # ==========================================
    MODEL_VOCAB_SIZE = TOKEN_LAYOUT.vocab_size
    # Transformer 的隐藏宽度，也是 Q/K/V 投影后的总注意力维度。
    MODEL_D_MODEL = 1024
    MODEL_N_LAYER = 24
    MODEL_N_HEAD = 16

    def validate_model_config(self):
        if self.MODEL_D_MODEL <= 0 or self.MODEL_N_LAYER <= 0 or self.MODEL_N_HEAD <= 0:
            raise ValueError("MODEL_D_MODEL、MODEL_N_LAYER 和 MODEL_N_HEAD 必须为正整数")
        if self.MODEL_VOCAB_SIZE != TOKEN_LAYOUT.vocab_size:
            raise ValueError(
                f"MODEL_VOCAB_SIZE={self.MODEL_VOCAB_SIZE} 与集中式词表布局 "
                f"{TOKEN_LAYOUT.vocab_size} 不一致"
            )
        if self.MODEL_D_MODEL % self.MODEL_N_HEAD != 0:
            raise ValueError(
                f"MODEL_D_MODEL ({self.MODEL_D_MODEL}) 必须能被 "
                f"MODEL_N_HEAD ({self.MODEL_N_HEAD}) 整除"
            )
    # ==========================================
    # 🔄 数据平权与增强策略
    # ==========================================
    VERSION_CYCLE_EPOCHS = 10
    ENABLE_FLIP_AUGMENTATION = False
     # 🔴 [新增] 训练集序列极速截断限制。0表示使用全部数据，N表示只取前N张图。
    LIMIT_TRAIN_SEQUENCES = 0
    # ==========================================
    # 🚀 训练超参
    # ==========================================
    # 保持已验证的单卡 micro-batch 大小，续训时不改变批处理策略。
    MICRO_BATCH_SIZE = 11
    # 首次遇到某个 CSV 时，以分块方式并行转换为 uint16 磁盘缓存。
    # 当前机器有 24 个物理核；限制为最多 12 个进程，避免并发解析撑爆内存。
    DATA_CACHE_BUILD_WORKERS = min(12, max(1, (os.cpu_count() or 2) // 2))
    DATA_CACHE_CSV_CHUNK_ROWS = 250_000
    # ROCm 在主进程已经初始化 GPU 后再 fork DataLoader worker 可能永久阻塞。
    # Batch 直接由 mmap 填入循环复用的 pinned buffer，不创建 DataLoader 子进程。
    # 多核进程仅用于首次将 CSV 构建为分片缓存；下一 Epoch 缓存在后台提前构建。
    DATALOADER_NUM_WORKERS = 0

    # ==========================================
    # 🖥️ 并行训练策略
    # ==========================================
    # "single"：单卡训练，是公开版本的安全默认值。
    # "manual_cpu_sync"：单进程多模型，经 CPU 分段合并梯度。
    # "manual_pcie_sync"：同一手动累计方案，但直接在多张 GPU 间传递分段张量。
    # "data_parallel"：使用 nn.DataParallel，由 PyTorch 在单进程内管理多卡。
    # 完整保留 manual_cpu_sync 的计算、累计和数据流水，仅把累计窗口末尾的
    # 各副卡→GPU0 梯度合并及 GPU0→各副卡参数回传使用 PCIe P2P DMA。
    # 每张卡仍各自累计完整的 local_accum_steps，micro-batch 间不跨卡传输。
    PARALLEL_MODE = "single"
    PARALLEL_DEVICE_IDS = [0]
    # CPU staging 或 GPU P2P 的可复用同步缓冲区大小。
    PARALLEL_SYNC_BUFFER_CHUNK_MB = 48

    # ==========================================
    # 💾 断点保护机制 (随时可热更新)
    # ==========================================
    # 每隔多少个 Batch 执行一次强制保存？
    # 注意：底层会在达到该批次后，自动寻找最近的一个梯度更新点(accum==0)进行安全保存。
    SAVE_INTERVAL_BATCHES = 4000

    def validate_parallel_config(self):
        valid_modes = {"single", "manual_cpu_sync", "manual_pcie_sync", "data_parallel"}
        if self.PARALLEL_MODE not in valid_modes:
            raise ValueError(
                f"PARALLEL_MODE 必须是以下之一: {sorted(valid_modes)}"
            )
        if self.PARALLEL_MODE == "single":
            return
        if (
            not isinstance(self.PARALLEL_DEVICE_IDS, (list, tuple))
            or len(self.PARALLEL_DEVICE_IDS) < 2
            or any(
                not isinstance(device_id, int) or device_id < 0
                for device_id in self.PARALLEL_DEVICE_IDS
            )
            or len(set(self.PARALLEL_DEVICE_IDS)) != len(self.PARALLEL_DEVICE_IDS)
        ):
            raise ValueError(
                "多卡模式要求 PARALLEL_DEVICE_IDS 至少包含两个不重复的非负整数"
            )
        if self.PARALLEL_SYNC_BUFFER_CHUNK_MB <= 0:
            raise ValueError("PARALLEL_SYNC_BUFFER_CHUNK_MB 必须大于 0")
