import gc
import json
import multiprocessing as mp
import os
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import TrainConfig
from token_layout import (
    ANGLE_BINS_PER_DEGREE,
    SIZE_BINS_PER_PIXEL,
    TOKEN_LAYOUT,
    XY_BINS_PER_PIXEL,
)


CACHE_FORMAT_VERSION = 1
CACHE_DTYPE = np.dtype("<u2")
TOKENS_PER_STEP = 9

CSV_DTYPE = {
    "image_name": "string",
    "cx": "float32",
    "cy": "float32",
    "w": "float32",
    "h": "float32",
    "shape_type": "int8",
    "theta": "float32",
    "r": "uint8",
    "g": "uint8",
    "b": "uint8",
}


def _encode_complete_groups(
    frame, target_steps, apply_flip, canvas_size=256
):
    """把一批完整、连续的 image_name 分组编码成 [N, context_tokens]。"""
    if frame is None or frame.empty:
        return np.empty((0, target_steps * TOKENS_PER_STEP), dtype=CACHE_DTYPE)

    image_names = frame["image_name"].to_numpy(copy=False)
    group_starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(image_names[1:] != image_names[:-1]).astype(np.int64)
            + 1,
        )
    )
    group_ends = np.concatenate(
        (group_starts[1:], np.array([len(frame)], dtype=np.int64))
    )
    group_sizes = group_ends - group_starts
    valid_starts = group_starts[group_sizes >= target_steps]
    if valid_starts.size == 0:
        return np.empty((0, target_steps * TOKENS_PER_STEP), dtype=CACHE_DTYPE)

    selected_rows = (
        valid_starts[:, None] + np.arange(target_steps, dtype=np.int64)[None, :]
    ).reshape(-1)

    cx = frame["cx"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    cy = frame["cy"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    width = frame["w"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    height = frame["h"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    shape_type = frame["shape_type"].to_numpy(
        dtype=np.int32, copy=False
    )[selected_rows]
    theta = frame["theta"].to_numpy(dtype=np.float32, copy=False)[selected_rows]
    red = frame["r"].to_numpy(dtype=np.int32, copy=False)[selected_rows]
    green = frame["g"].to_numpy(dtype=np.int32, copy=False)[selected_rows]
    blue = frame["b"].to_numpy(dtype=np.int32, copy=False)[selected_rows]

    if apply_flip:
        # 像素中心坐标位于 [0, canvas_size - 1]。
        cx = (float(canvas_size) - 1.0) - cx
        theta = np.where(shape_type != -1, np.pi - theta, theta)
        theta = np.mod(theta, np.pi)

    theta_deg = np.mod(theta * 180.0 / np.pi, 180.0)
    swap_mask = theta_deg >= 90.0
    theta_deg = np.where(swap_mask, theta_deg - 90.0, theta_deg)
    canonical_width = np.where(swap_mask, height, width)
    canonical_height = np.where(swap_mask, width, height)

    tokens = np.stack(
        [
            np.clip(
                cx * XY_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.x_bins - 1
            ).astype(np.int32)
            + TOKEN_LAYOUT.x_offset,
            np.clip(
                cy * XY_BINS_PER_PIXEL, 0, TOKEN_LAYOUT.y_bins - 1
            ).astype(np.int32)
            + TOKEN_LAYOUT.y_offset,
            np.clip(
                theta_deg * ANGLE_BINS_PER_DEGREE,
                0,
                TOKEN_LAYOUT.angle_bins - 1,
            ).astype(np.int32)
            + TOKEN_LAYOUT.angle_offset,
            np.clip(
                canonical_width * SIZE_BINS_PER_PIXEL,
                0,
                TOKEN_LAYOUT.width_bins - 1,
            ).astype(np.int32)
            + TOKEN_LAYOUT.width_offset,
            np.clip(
                canonical_height * SIZE_BINS_PER_PIXEL,
                0,
                TOKEN_LAYOUT.height_bins - 1,
            ).astype(np.int32)
            + TOKEN_LAYOUT.height_offset,
            np.clip(
                shape_type + 1, 0, TOKEN_LAYOUT.shape_bins - 1
            ).astype(np.int32)
            + TOKEN_LAYOUT.shape_offset,
            np.clip(red // 2, 0, TOKEN_LAYOUT.color_bins - 1).astype(np.int32)
            + TOKEN_LAYOUT.red_offset,
            np.clip(green // 2, 0, TOKEN_LAYOUT.color_bins - 1).astype(np.int32)
            + TOKEN_LAYOUT.green_offset,
            np.clip(blue // 2, 0, TOKEN_LAYOUT.color_bins - 1).astype(np.int32)
            + TOKEN_LAYOUT.blue_offset,
        ],
        axis=1,
    ).astype(CACHE_DTYPE, copy=False)

    return tokens.reshape(valid_starts.size, target_steps * TOKENS_PER_STEP)


def _split_complete_prefix(frame):
    """留下最后一个可能跨 CSV chunk 的图像分组，其余分组可安全编码。"""
    if frame.empty:
        return frame, None
    image_names = frame["image_name"].to_numpy(copy=False)
    changes = np.flatnonzero(image_names[1:] != image_names[:-1]) + 1
    if changes.size == 0:
        return None, frame.copy()
    last_group_start = int(changes[-1])
    return frame.iloc[:last_group_start], frame.iloc[last_group_start:].copy()


def _build_cache_shard(task):
    """子进程入口：流式读取单个 CSV，并原子写入一个 token 分片。"""
    source_path = Path(task["source_path"])
    token_path = Path(task["token_path"])
    metadata_path = Path(task["metadata_path"])
    temporary_suffix = f".tmp.{os.getpid()}"
    temporary_token_path = Path(str(token_path) + temporary_suffix)
    temporary_metadata_path = Path(str(metadata_path) + temporary_suffix)
    started_at = time.perf_counter()
    sequence_count = 0
    chunk_count = 0
    carry_frame = None

    token_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_token_path.open("wb") as token_file:
            reader = pd.read_csv(
                source_path,
                dtype=CSV_DTYPE,
                engine="c",
                chunksize=task["chunk_rows"],
            )
            for chunk_count, chunk in enumerate(reader, start=1):
                if carry_frame is not None:
                    chunk = pd.concat(
                        (carry_frame, chunk), ignore_index=True, copy=False
                    )
                    carry_frame = None

                complete_frame, carry_frame = _split_complete_prefix(chunk)
                encoded = _encode_complete_groups(
                    complete_frame,
                    task["target_steps"],
                    task["apply_flip"],
                    task["canvas_size"],
                )
                if encoded.size:
                    encoded.tofile(token_file)
                    sequence_count += encoded.shape[0]

                del chunk, complete_frame, encoded
                if chunk_count % 10 == 0:
                    print(
                        f"⚙️ [Cache:{source_path.name}] chunks={chunk_count} | "
                        f"sequences={sequence_count}",
                        flush=True,
                    )

            encoded = _encode_complete_groups(
                carry_frame,
                task["target_steps"],
                task["apply_flip"],
                task["canvas_size"],
            )
            if encoded.size:
                encoded.tofile(token_file)
                sequence_count += encoded.shape[0]
            token_file.flush()
            os.fsync(token_file.fileno())

        source_stat = source_path.stat()
        if (
            source_stat.st_size != task["source_size"]
            or source_stat.st_mtime_ns != task["source_mtime_ns"]
        ):
            raise RuntimeError(f"缓存构建期间源 CSV 发生变化: {source_path}")

        expected_bytes = (
            sequence_count * task["context_tokens"] * CACHE_DTYPE.itemsize
        )
        actual_bytes = temporary_token_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"缓存大小不一致: {temporary_token_path}，"
                f"实际 {actual_bytes}，期望 {expected_bytes}"
            )

        metadata = {
            **task["signature"],
            "source_path": str(source_path.resolve()),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "sequence_count": sequence_count,
            "context_tokens": task["context_tokens"],
            "dtype": CACHE_DTYPE.str,
            "token_file": token_path.name,
        }
        with temporary_metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
            metadata_file.flush()
            os.fsync(metadata_file.fileno())

        os.replace(temporary_token_path, token_path)
        os.replace(temporary_metadata_path, metadata_path)
        return {
            "source_path": str(source_path),
            "sequence_count": sequence_count,
            "chunk_count": chunk_count,
            "elapsed_seconds": time.perf_counter() - started_at,
        }
    except BaseException:
        temporary_token_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)
        raise


class GeometrizeFullSequenceDataset(Dataset):
    """固定长度序列数据集；CSV 仅首次流式转换，训练时由 mmap 按需读取。"""

    def __init__(self, config=None):
        self.config = config if config is not None else TrainConfig()
        self.config.validate_sequence_config()
        if self.config.MAX_IMAGE_STEPS < self.config.CONTEXT_STEPS:
            raise ValueError(
                "MAX_IMAGE_STEPS 必须不小于 CONTEXT_STEPS，"
                f"当前为 {self.config.MAX_IMAGE_STEPS} < {self.config.CONTEXT_STEPS}"
            )
        if self.config.CONTEXT_TOKENS != (
            self.config.CONTEXT_STEPS * TOKENS_PER_STEP
        ):
            raise ValueError("缓存数据格式要求每个图元恰好包含 9 个 token。")

        self.sequence_indices = np.empty(0, dtype=np.int32)
        self.token_shards = []
        self.shard_starts = np.empty(0, dtype=np.int64)
        self.shard_ends = np.empty(0, dtype=np.int64)
        self.shard_metadata = []
        self.cache_prefetch_thread = None
        self.cache_prefetch_epoch = None
        self.cache_prefetch_error = None

    def _version_for_epoch(self, epoch):
        version_idx = ((epoch - 1) % self.config.VERSION_CYCLE_EPOCHS) + 1
        apply_flip = (
            self.config.ENABLE_FLIP_AUGMENTATION
            and epoch > self.config.VERSION_CYCLE_EPOCHS
        )
        return version_idx, apply_flip

    def set_epoch(self, epoch):
        version_idx, apply_flip = self._version_for_epoch(epoch)
        if (
            self.cache_prefetch_thread is not None
            and self.cache_prefetch_epoch == epoch
        ):
            print(f"⏳ [Cache] 等待 Epoch {epoch} 的后台缓存任务收尾...")
            self.cache_prefetch_thread.join()
            if self.cache_prefetch_error is not None:
                raise RuntimeError(
                    f"Epoch {epoch} 的后台缓存构建失败"
                ) from self.cache_prefetch_error
        print(
            f"\n🔄 [Dataset] 准备 Epoch {epoch} -> 锁定版本: "
            f"[v{version_idx}] | 水平翻转: [{apply_flip}]"
        )
        self._open_version_cache(version_idx, apply_flip, epoch)

    def start_next_epoch_cache_prefetch(self, next_epoch):
        """训练当前 epoch 时，在后台用多核构建下一版本的磁盘分片。"""
        if (
            self.cache_prefetch_thread is not None
            and self.cache_prefetch_thread.is_alive()
        ):
            return False

        version_idx, apply_flip = self._version_for_epoch(next_epoch)
        self.cache_prefetch_epoch = next_epoch
        self.cache_prefetch_error = None

        def prepare():
            try:
                print(
                    f"🔮 [Cache] 后台准备 Epoch {next_epoch} "
                    f"(v{version_idx}, flip={apply_flip})...",
                    flush=True,
                )
                self._ensure_version_cache(version_idx, apply_flip)
                print(
                    f"✅ [Cache] Epoch {next_epoch} 的磁盘分片已提前就绪。",
                    flush=True,
                )
            except BaseException as error:
                self.cache_prefetch_error = error
                print(
                    f"❌ [Cache] Epoch {next_epoch} 后台准备失败: {error}",
                    flush=True,
                )

        self.cache_prefetch_thread = threading.Thread(
            target=prepare,
            name=f"gpt-cache-epoch-{next_epoch}",
            daemon=True,
        )
        self.cache_prefetch_thread.start()
        return True

    def _cache_signature(self, apply_flip):
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "token_layout_version": self.config.TOKEN_LAYOUT_VERSION,
            "context_steps": self.config.CONTEXT_STEPS,
            "context_tokens": self.config.CONTEXT_TOKENS,
            "apply_flip": bool(apply_flip),
            "canvas_size": self.config.CANVAS_SIZE,
        }

    def _cache_variant_dir(self, version_idx, apply_flip):
        safe_layout_version = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", self.config.TOKEN_LAYOUT_VERSION
        )
        variant_name = (
            f"{safe_layout_version}_steps{self.config.CONTEXT_STEPS}_"
            f"flip{int(apply_flip)}_cache{CACHE_FORMAT_VERSION}"
        )
        return (
            Path(self.config.DATA_CACHE_DIR)
            / f"v{version_idx}"
            / variant_name
        )

    @staticmethod
    def _cache_paths(cache_dir, source_path):
        base_name = source_path.name
        return (
            cache_dir / f"{base_name}.tokens.bin",
            cache_dir / f"{base_name}.metadata.json",
        )

    def _read_valid_metadata(
        self, source_path, token_path, metadata_path, signature
    ):
        if not token_path.is_file() or not metadata_path.is_file():
            return None
        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            source_stat = source_path.stat()
            for key, expected_value in signature.items():
                if metadata.get(key) != expected_value:
                    return None
            if (
                metadata.get("source_size") != source_stat.st_size
                or metadata.get("source_mtime_ns") != source_stat.st_mtime_ns
                or metadata.get("context_tokens") != self.config.CONTEXT_TOKENS
                or metadata.get("dtype") != CACHE_DTYPE.str
            ):
                return None
            sequence_count = int(metadata["sequence_count"])
            expected_bytes = (
                sequence_count
                * self.config.CONTEXT_TOKENS
                * CACHE_DTYPE.itemsize
            )
            if token_path.stat().st_size != expected_bytes:
                return None
            metadata["token_path"] = str(token_path)
            metadata["metadata_path"] = str(metadata_path)
            return metadata
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _ensure_version_cache(self, version_idx, apply_flip):
        version_dir = Path(self.config.DATA_DIR) / f"v{version_idx}"
        csv_files = sorted(version_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"❌ 找不到数据目录或 CSV: {version_dir}")

        cache_dir = self._cache_variant_dir(version_idx, apply_flip)
        signature = self._cache_signature(apply_flip)
        valid_metadata_by_source = {}
        build_tasks = []

        for source_path in csv_files:
            token_path, metadata_path = self._cache_paths(
                cache_dir, source_path
            )
            metadata = self._read_valid_metadata(
                source_path, token_path, metadata_path, signature
            )
            if metadata is not None:
                valid_metadata_by_source[str(source_path)] = metadata
                continue

            source_stat = source_path.stat()
            build_tasks.append(
                {
                    "source_path": str(source_path),
                    "token_path": str(token_path),
                    "metadata_path": str(metadata_path),
                    "source_size": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "target_steps": self.config.CONTEXT_STEPS,
                    "context_tokens": self.config.CONTEXT_TOKENS,
                    "apply_flip": bool(apply_flip),
                    "canvas_size": self.config.CANVAS_SIZE,
                    "chunk_rows": self.config.DATA_CACHE_CSV_CHUNK_ROWS,
                    "signature": signature,
                }
            )

        if build_tasks:
            worker_count = min(
                len(build_tasks), self.config.DATA_CACHE_BUILD_WORKERS
            )
            print(
                f"🏗️ [Cache] 首次构建 v{version_idx} 的 {len(build_tasks)} 个分片 | "
                f"workers={worker_count}/{self.config.DATA_CACHE_BUILD_WORKERS} | "
                f"chunk_rows={self.config.DATA_CACHE_CSV_CHUNK_ROWS:,} | "
                f"目录={cache_dir}",
                flush=True,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            if worker_count == 1:
                results = [_build_cache_shard(build_tasks[0])]
            else:
                spawn_context = mp.get_context("spawn")
                results = []
                with ProcessPoolExecutor(
                    max_workers=worker_count, mp_context=spawn_context
                ) as executor:
                    futures = {
                        executor.submit(_build_cache_shard, task): task
                        for task in build_tasks
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        print(
                            f"✅ [Cache] {Path(result['source_path']).name} 完成 | "
                            f"sequences={result['sequence_count']} | "
                            f"chunks={result['chunk_count']} | "
                            f"耗时={result['elapsed_seconds']:.2f}s",
                            flush=True,
                        )

            if worker_count == 1:
                result = results[0]
                print(
                    f"✅ [Cache] {Path(result['source_path']).name} 完成 | "
                    f"sequences={result['sequence_count']} | "
                    f"chunks={result['chunk_count']} | "
                    f"耗时={result['elapsed_seconds']:.2f}s",
                    flush=True,
                )

        metadata_in_source_order = []
        for source_path in csv_files:
            token_path, metadata_path = self._cache_paths(
                cache_dir, source_path
            )
            metadata = self._read_valid_metadata(
                source_path, token_path, metadata_path, signature
            )
            if metadata is None:
                raise RuntimeError(f"缓存构建后校验失败: {source_path}")
            metadata_in_source_order.append(metadata)
        return metadata_in_source_order

    def _open_version_cache(self, version_idx, apply_flip, epoch):
        metadata_list = self._ensure_version_cache(version_idx, apply_flip)

        self.token_shards.clear()
        self.shard_metadata = metadata_list
        gc.collect()

        sequence_counts = np.array(
            [int(metadata["sequence_count"]) for metadata in metadata_list],
            dtype=np.int64,
        )
        self.shard_starts = np.zeros(len(sequence_counts), dtype=np.int64)
        if len(sequence_counts) > 1:
            np.cumsum(sequence_counts[:-1], out=self.shard_starts[1:])
        self.shard_ends = self.shard_starts + sequence_counts
        total_sequence_count = int(sequence_counts.sum())

        for metadata, sequence_count in zip(
            metadata_list, sequence_counts.tolist()
        ):
            token_shard = np.memmap(
                metadata["token_path"],
                dtype=CACHE_DTYPE,
                mode="r",
                shape=(sequence_count, self.config.CONTEXT_TOKENS),
            )
            self.token_shards.append(token_shard)

        visible_sequence_count = total_sequence_count
        if self.config.LIMIT_TRAIN_SEQUENCES > 0:
            visible_sequence_count = min(
                visible_sequence_count,
                self.config.LIMIT_TRAIN_SEQUENCES,
            )
            print(
                f"🛑 [Dataset] 应用全局序列数量限制: "
                f"{visible_sequence_count}/{total_sequence_count}"
            )

        self.sequence_indices = np.arange(
            visible_sequence_count, dtype=np.int32
        )
        np.random.RandomState(epoch * 42).shuffle(self.sequence_indices)

        cache_bytes = sum(
            Path(metadata["token_path"]).stat().st_size
            for metadata in metadata_list
        )
        resident_index_mb = (
            self.sequence_indices.nbytes
            + self.shard_starts.nbytes
            + self.shard_ends.nbytes
        ) / (1024 * 1024)
        print(
            f"📥 [Dataset] mmap 分片已就绪 | 有效图像={visible_sequence_count} | "
            f"分片={len(self.token_shards)} | 磁盘缓存={cache_bytes / 2**30:.2f} GiB"
        )
        print(
            f"⚡ [Dataset] 常驻索引内存约 {resident_index_mb:.2f} MiB；"
            "token 页由操作系统按需读取和回收。"
        )

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        sequence_index = int(self.sequence_indices[idx])
        shard_index = int(
            np.searchsorted(self.shard_ends, sequence_index, side="right")
        )
        if shard_index >= len(self.token_shards):
            raise IndexError(
                f"序列索引 {sequence_index} 超出缓存总量 "
                f"{int(self.shard_ends[-1]) if len(self.shard_ends) else 0}"
            )
        local_index = sequence_index - int(self.shard_starts[shard_index])
        tokens = self.token_shards[shard_index][local_index]
        return torch.tensor(tokens, dtype=torch.long)

    def get_numpy_batch(self, start_index, end_index):
        """纯 NumPy/mmap 批量读取；可安全地在后台线程运行。"""
        batch = np.empty(
            (end_index - start_index, self.config.CONTEXT_TOKENS),
            dtype=np.int64,
        )
        self.fill_numpy_batch(start_index, end_index, batch)
        return batch

    def fill_numpy_batch(self, start_index, end_index, output):
        """将一个 batch 直接写入调用方提供的 NumPy/pinned-memory 视图。"""
        selected_sequence_indices = self.sequence_indices[
            start_index:end_index
        ].astype(np.int64, copy=False)
        expected_shape = (
            len(selected_sequence_indices),
            self.config.CONTEXT_TOKENS,
        )
        if output.shape != expected_shape or output.dtype != np.int64:
            raise ValueError(
                f"output 应为 int64 {expected_shape}，"
                f"实际为 {output.dtype} {output.shape}"
            )
        if len(selected_sequence_indices) == 0:
            return

        shard_indices = np.searchsorted(
            self.shard_ends, selected_sequence_indices, side="right"
        )
        for shard_index in np.unique(shard_indices):
            batch_positions = np.flatnonzero(shard_indices == shard_index)
            local_indices = (
                selected_sequence_indices[batch_positions]
                - self.shard_starts[shard_index]
            )
            output[batch_positions] = self.token_shards[shard_index][
                local_indices
            ]

    def iter_numpy_batches(self, start_index, batch_size, drop_last):
        """按 dataset 索引顺序产生 NumPy batch，不调用 torch。"""
        if not 0 <= start_index <= len(self):
            raise ValueError(
                f"start_index 必须位于 [0, {len(self)}]，收到 {start_index}"
            )
        for batch_start in range(start_index, len(self), batch_size):
            batch_end = min(batch_start + batch_size, len(self))
            if drop_last and batch_end - batch_start < batch_size:
                break
            yield self.get_numpy_batch(batch_start, batch_end)
