mod engine;
mod types;

use engine::GpuEngine;
use image::imageops::FilterType;
use rand::Rng;
use rayon::prelude::*;
use std::fs;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;
use types::{Config, GpuShape, ImageState};
use wgpu::util::DeviceExt;

// ==========================================
// ⚙️ 全局生产力配置
// ==========================================
const W: u32 = 256;
const H: u32 = 256;
const ALPHA: f32 = 0.5;
const PENALTY_FACTOR: f32 = 1.8;

const MAX_ITERATIONS: usize = 275;

// ==========================================
// 🧬 GPU 搜索与变异配置
// 修改这里的值后重新执行 cargo run --release。
// 程序启动时会把这些值生成为 WGSL const，每张 GPU 只编译一次。
// ==========================================
/// 每一步先随机生成多少个图形参加海选；计算量近似随该值线性增加。
const INITIAL_CANDIDATE_COUNT: u32 = 50;
/// 海选后保留多少个候选继续爬山；必须不大于 INITIAL_CANDIDATE_COUNT。
const SURVIVOR_COUNT: u32 = 5;
/// 连续多少次变异没有改良后，采纳当前候选迄今找到的最佳参数。
const MUTATION_STAGNATION_LIMIT: u32 = 100;
/// 单个候选允许尝试的变异硬上限。
const MAX_MUTATIONS_PER_SURVIVOR: u32 = 10000;

const POSITION_MUTATION_RANGE: f32 = 28.0;
const WIDTH_MUTATION_RANGE: f32 = 28.0;
const HEIGHT_MUTATION_RANGE: f32 = 28.0;
/// 供人编辑时使用角度制；生成 WGSL 时转换成弧度。
const ANGLE_MUTATION_RANGE_DEGREES: f32 = 16.0;
const MIN_HALF_SIZE: f32 = 1.0;
const MAX_HALF_SIZE: f32 = 128.0;

// 五类变异的相对权重。全部为 1 时与旧版 hash % 5 完全一致。
const POSITION_MUTATION_WEIGHT: u32 = 1;
const WIDTH_MUTATION_WEIGHT: u32 = 1;
const HEIGHT_MUTATION_WEIGHT: u32 = 1;
const ANGLE_MUTATION_WEIGHT: u32 = 1;
const SHAPE_TYPE_MUTATION_WEIGHT: u32 = 1;

// 256px + 最大半尺寸 128 会显著增加单次 shader 工作量。
// 限制每张 GPU 的分片，避免超大资源和过长的单次 dispatch。
const DEFAULT_IMAGES_PER_GPU_BATCH: usize = 1000;
// 限制未完成命令的积压；等待发生在 GPU 已有工作可执行时，不会降低计算并行度。
const GPU_QUEUE_SYNC_INTERVAL: usize = 25;
// 历史始终完整保存在显存中，只在最终输出时按图片块映射到 CPU。
const DEFAULT_HISTORY_READBACK_IMAGES: usize = 128;

const NUM_VERSIONS: usize = 10;
const MAX_CSV_SIZE_BYTES: usize = 1024 * 1024 * 1024;

// 🛑 早停机制配置
const HISTORY_WINDOW: usize = 1000;
const EARLY_STOP_THRESHOLD: f32 = 0.0005;

// ==========================================
// 🧩 辅助结构体
// ==========================================
#[derive(Copy, Clone, Debug, PartialEq)]
struct ShaderConstants {
    canvas_width: u32,
    canvas_height: u32,
    shape_alpha: f32,
    penalty_factor: f32,
    max_iterations: usize,
    history_window: usize,
    early_stop_threshold: f32,
    initial_candidate_count: u32,
    survivor_count: u32,
    mutation_stagnation_limit: u32,
    max_mutations_per_survivor: u32,
    position_mutation_range: f32,
    width_mutation_range: f32,
    height_mutation_range: f32,
    angle_mutation_range_degrees: f32,
    min_half_size: f32,
    max_half_size: f32,
    mutation_weights: [u32; 5],
}

struct HostImgData {
    target_pixels: Vec<u32>,
    bg_color: (u8, u8, u8),
    bg_packed: u32,
    initial_ssd: f64,
}

struct CsvState {
    file_idx: usize,
    current_size: usize,
}

fn shader_constants() -> ShaderConstants {
    ShaderConstants {
        canvas_width: W,
        canvas_height: H,
        shape_alpha: ALPHA,
        penalty_factor: PENALTY_FACTOR,
        max_iterations: MAX_ITERATIONS,
        history_window: HISTORY_WINDOW,
        early_stop_threshold: EARLY_STOP_THRESHOLD,
        initial_candidate_count: INITIAL_CANDIDATE_COUNT,
        survivor_count: SURVIVOR_COUNT,
        mutation_stagnation_limit: MUTATION_STAGNATION_LIMIT,
        max_mutations_per_survivor: MAX_MUTATIONS_PER_SURVIVOR,
        position_mutation_range: POSITION_MUTATION_RANGE,
        width_mutation_range: WIDTH_MUTATION_RANGE,
        height_mutation_range: HEIGHT_MUTATION_RANGE,
        angle_mutation_range_degrees: ANGLE_MUTATION_RANGE_DEGREES,
        min_half_size: MIN_HALF_SIZE,
        max_half_size: MAX_HALF_SIZE,
        mutation_weights: [
            POSITION_MUTATION_WEIGHT,
            WIDTH_MUTATION_WEIGHT,
            HEIGHT_MUTATION_WEIGHT,
            ANGLE_MUTATION_WEIGHT,
            SHAPE_TYPE_MUTATION_WEIGHT,
        ],
    }
}

fn validate_shader_constants(constants: &ShaderConstants) -> Result<u32, String> {
    if constants.canvas_width == 0 {
        return Err("❌ 参数 W=0 无效：画布宽度必须大于 0".to_string());
    }
    if constants.canvas_height == 0 {
        return Err("❌ 参数 H=0 无效：画布高度必须大于 0".to_string());
    }
    if !constants.shape_alpha.is_finite()
        || constants.shape_alpha <= 0.0
        || constants.shape_alpha > 1.0
    {
        return Err(format!(
            "❌ 参数 ALPHA={} 无效：必须是 (0.0, 1.0] 内的有限数",
            constants.shape_alpha
        ));
    }
    if !constants.penalty_factor.is_finite() || constants.penalty_factor <= 0.0 {
        return Err(format!(
            "❌ 参数 PENALTY_FACTOR={} 无效：必须是大于 0 的有限数",
            constants.penalty_factor
        ));
    }
    if constants.max_iterations == 0 || constants.max_iterations > u32::MAX as usize {
        return Err(format!(
            "❌ 参数 MAX_ITERATIONS={} 无效：合法范围为 1..={}",
            constants.max_iterations,
            u32::MAX
        ));
    }
    if constants.history_window == 0 || constants.history_window > u32::MAX as usize {
        return Err(format!(
            "❌ 参数 HISTORY_WINDOW={} 无效：合法范围为 1..={}",
            constants.history_window,
            u32::MAX
        ));
    }
    if !constants.early_stop_threshold.is_finite() || constants.early_stop_threshold < 0.0 {
        return Err(format!(
            "❌ 参数 EARLY_STOP_THRESHOLD={} 无效：必须是大于等于 0 的有限数",
            constants.early_stop_threshold
        ));
    }
    if constants.initial_candidate_count == 0 {
        return Err("❌ 参数 INITIAL_CANDIDATE_COUNT=0 无效：初始候选数必须大于 0".to_string());
    }
    if constants.survivor_count == 0 || constants.survivor_count > constants.initial_candidate_count
    {
        return Err(format!(
            "❌ 参数 SURVIVOR_COUNT={} 无效：合法范围为 1..=INITIAL_CANDIDATE_COUNT({})",
            constants.survivor_count, constants.initial_candidate_count
        ));
    }
    if constants.mutation_stagnation_limit == 0 {
        return Err(
            "❌ 参数 MUTATION_STAGNATION_LIMIT=0 无效：无改良截止次数必须大于 0".to_string(),
        );
    }
    if constants.max_mutations_per_survivor == 0 {
        return Err("❌ 参数 MAX_MUTATIONS_PER_SURVIVOR=0 无效：变异硬上限必须大于 0".to_string());
    }
    if constants.mutation_stagnation_limit > constants.max_mutations_per_survivor {
        return Err(format!(
            "❌ 参数 MUTATION_STAGNATION_LIMIT={} 无效：不得大于 MAX_MUTATIONS_PER_SURVIVOR({})",
            constants.mutation_stagnation_limit, constants.max_mutations_per_survivor
        ));
    }
    if !constants.min_half_size.is_finite()
        || !constants.max_half_size.is_finite()
        || constants.min_half_size < 1.0
        || constants.min_half_size > constants.max_half_size
        || constants.max_half_size > 128.0
    {
        return Err(format!(
            "❌ 参数半尺寸范围 [{}, {}] 无效：必须满足 1.0 <= MIN_HALF_SIZE <= MAX_HALF_SIZE <= 128.0",
            constants.min_half_size, constants.max_half_size
        ));
    }

    for (name, value) in [
        ("POSITION_MUTATION_RANGE", constants.position_mutation_range),
        ("WIDTH_MUTATION_RANGE", constants.width_mutation_range),
        ("HEIGHT_MUTATION_RANGE", constants.height_mutation_range),
        (
            "ANGLE_MUTATION_RANGE_DEGREES",
            constants.angle_mutation_range_degrees,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "❌ 参数 {name}={value} 无效：必须是大于等于 0 的有限数"
            ));
        }
    }

    let total_weight = constants
        .mutation_weights
        .iter()
        .try_fold(0u32, |total, &weight| total.checked_add(weight))
        .ok_or_else(|| {
            format!(
                "❌ 五类变异权重 {:?} 的总和超过 u32 上限 {}",
                constants.mutation_weights,
                u32::MAX
            )
        })?;
    if total_weight == 0 {
        return Err(format!(
            "❌ 五类变异权重 {:?} 无效：至少一个权重必须大于 0",
            constants.mutation_weights
        ));
    }

    Ok(total_weight)
}

fn wgsl_f32_literal(value: f32) -> String {
    let mut literal = format!("{value:?}");
    if !literal.contains('.') && !literal.contains('e') && !literal.contains('E') {
        literal.push_str(".0");
    }
    literal
}

fn build_shader_source(constants: &ShaderConstants) -> Result<String, String> {
    let total_mutation_weight = validate_shader_constants(constants)?;
    let angle_radians = constants.angle_mutation_range_degrees.to_radians();
    let [position_weight, width_weight, height_weight, angle_weight, shape_type_weight] =
        constants.mutation_weights;

    let prefix = format!(
        "// 此区域由 main.rs 的启动参数自动生成，请勿直接编辑。\n\
const CANVAS_WIDTH: u32 = {}u;\n\
const CANVAS_HEIGHT: u32 = {}u;\n\
const SHAPE_ALPHA: f32 = {};\n\
const PENALTY_FACTOR: f32 = {};\n\
const MAX_ITERATIONS: u32 = {}u;\n\
const HISTORY_WINDOW: u32 = {}u;\n\
const EARLY_STOP_THRESHOLD: f32 = {};\n\
const INITIAL_CANDIDATE_COUNT: u32 = {}u;\n\
const SURVIVOR_COUNT: u32 = {}u;\n\
const MUTATION_STAGNATION_LIMIT: u32 = {}u;\n\
const MAX_MUTATIONS_PER_SURVIVOR: u32 = {}u;\n\
const POSITION_MUTATION_RANGE: f32 = {};\n\
const WIDTH_MUTATION_RANGE: f32 = {};\n\
const HEIGHT_MUTATION_RANGE: f32 = {};\n\
const ANGLE_MUTATION_RANGE_RADIANS: f32 = {};\n\
const MIN_HALF_SIZE: f32 = {};\n\
const MAX_HALF_SIZE: f32 = {};\n\
const POSITION_MUTATION_WEIGHT: u32 = {}u;\n\
const WIDTH_MUTATION_WEIGHT: u32 = {}u;\n\
const HEIGHT_MUTATION_WEIGHT: u32 = {}u;\n\
const ANGLE_MUTATION_WEIGHT: u32 = {}u;\n\
const SHAPE_TYPE_MUTATION_WEIGHT: u32 = {}u;\n\
const TOTAL_MUTATION_WEIGHT: u32 = {}u;\n\n",
        constants.canvas_width,
        constants.canvas_height,
        wgsl_f32_literal(constants.shape_alpha),
        wgsl_f32_literal(constants.penalty_factor),
        constants.max_iterations,
        constants.history_window,
        wgsl_f32_literal(constants.early_stop_threshold),
        constants.initial_candidate_count,
        constants.survivor_count,
        constants.mutation_stagnation_limit,
        constants.max_mutations_per_survivor,
        wgsl_f32_literal(constants.position_mutation_range),
        wgsl_f32_literal(constants.width_mutation_range),
        wgsl_f32_literal(constants.height_mutation_range),
        wgsl_f32_literal(angle_radians),
        wgsl_f32_literal(constants.min_half_size),
        wgsl_f32_literal(constants.max_half_size),
        position_weight,
        width_weight,
        height_weight,
        angle_weight,
        shape_type_weight,
        total_mutation_weight,
    );

    Ok(prefix + include_str!("shader.wgsl"))
}

fn print_shader_constants(constants: &ShaderConstants) {
    println!(
        "🧬 WGSL 固定参数 | 画布: {}x{} | 步数: {} | alpha: {} | 惩罚: {}",
        constants.canvas_width,
        constants.canvas_height,
        constants.max_iterations,
        constants.shape_alpha,
        constants.penalty_factor
    );
    println!(
        "🧬 搜索参数 | 初筛: {} | 保留: {} | 无改良截止: {} | 单候选硬上限: {}",
        constants.initial_candidate_count,
        constants.survivor_count,
        constants.mutation_stagnation_limit,
        constants.max_mutations_per_survivor
    );
    println!(
        "🧬 变异参数 | 位置/宽/高: ±{}/±{}/±{} px | 角度: ±{}° | 半尺寸: {}..={} | 权重: {:?}",
        constants.position_mutation_range,
        constants.width_mutation_range,
        constants.height_mutation_range,
        constants.angle_mutation_range_degrees,
        constants.min_half_size,
        constants.max_half_size,
        constants.mutation_weights
    );
    println!(
        "🛑 GPU 早停 | 窗口: {} | 阈值: {}",
        constants.history_window, constants.early_stop_threshold
    );
}

/// 一个批次中属于某张 GPU 的连续图片分片及其常驻目标缓冲区。
struct GpuBatchShard {
    range: Range<usize>,
    target_buf: wgpu::Buffer,
    canvas_base: Vec<u32>,
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp"
            )
        })
        .unwrap_or(false)
}

/// 递归发现图片并排序；不会把数据集中的进度文本误当成图片。
fn collect_image_paths(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut pending_dirs = vec![root.to_path_buf()];
    let mut paths = Vec::new();

    while let Some(dir) = pending_dirs.pop() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let path = entry.path();

            if file_type.is_dir() {
                pending_dirs.push(path);
            } else if file_type.is_file() && is_supported_image(&path) {
                paths.push(path);
            }
        }
    }

    paths.sort_unstable();
    Ok(paths)
}

/// 将 total 个元素尽可能均匀地切成不超过 worker_count 个非空连续区间。
fn balanced_ranges(total: usize, worker_count: usize) -> Vec<Range<usize>> {
    if total == 0 || worker_count == 0 {
        return Vec::new();
    }

    let active_workers = worker_count.min(total);
    let base_size = total / active_workers;
    let remainder = total % active_workers;
    let mut start = 0;

    (0..active_workers)
        .map(|index| {
            let size = base_size + usize::from(index < remainder);
            let range = start..start + size;
            start += size;
            range
        })
        .collect()
}

/// 将连续数据切成固定上限的小块，供 staging buffer 分块回读使用。
fn chunked_ranges(total: usize, max_chunk_size: usize) -> Vec<Range<usize>> {
    if total == 0 || max_chunk_size == 0 {
        return Vec::new();
    }

    (0..total)
        .step_by(max_chunk_size)
        .map(|start| start..(start + max_chunk_size).min(total))
        .collect()
}

fn images_per_gpu_batch() -> usize {
    positive_usize_env("SHAPE_RENDERER_IMAGES_PER_GPU_BATCH")
        .unwrap_or(DEFAULT_IMAGES_PER_GPU_BATCH)
}

fn history_readback_images() -> usize {
    positive_usize_env("SHAPE_RENDERER_HISTORY_READBACK_IMAGES")
        .unwrap_or(DEFAULT_HISTORY_READBACK_IMAGES)
}

fn positive_usize_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

/// Optionally limit each emitted image sequence to a fixed number of rows.
/// The count includes the first background row.  No setting preserves the
/// historical behavior of writing every accepted primitive.
fn output_step_limit() -> Result<Option<usize>, String> {
    let Some(raw_value) = std::env::var_os("SHAPE_RENDERER_MAX_OUTPUT_STEPS") else {
        return Ok(None);
    };
    if raw_value.is_empty() {
        return Err(
            "❌ SHAPE_RENDERER_MAX_OUTPUT_STEPS 不能为空；移除该变量即可输出完整序列。".to_string(),
        );
    }

    let value = raw_value.to_string_lossy().parse::<usize>().map_err(|_| {
        "❌ SHAPE_RENDERER_MAX_OUTPUT_STEPS 必须是正整数（包含背景步骤）。".to_string()
    })?;
    if value == 0 || value > MAX_ITERATIONS + 1 {
        return Err(format!(
            "❌ SHAPE_RENDERER_MAX_OUTPUT_STEPS={} 无效；合法范围为 1..={}（包含背景步骤）。",
            value,
            MAX_ITERATIONS + 1
        ));
    }
    Ok(Some(value))
}

fn input_dir() -> Result<PathBuf, String> {
    std::env::var_os("SHAPE_RENDERER_INPUT_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            "❌ 未设置输入图片目录。请设置环境变量 SHAPE_RENDERER_INPUT_DIR。".to_string()
        })
}

fn output_dir() -> PathBuf {
    std::env::var_os("SHAPE_RENDERER_OUTPUT_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("fast_shape_render must have a project parent")
                .join("data")
                .join("output_256")
        })
}

fn preprocess_image(path: &Path) -> HostImgData {
    let img = image::open(path)
        .unwrap_or_else(|err| panic!("❌ 无法读取图片 {}: {err}", path.display()))
        .resize_exact(W, H, FilterType::Lanczos3)
        .into_rgb8();
    let mut target_pixels = vec![0u32; (W * H) as usize];
    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;

    for (i, pixel) in img.pixels().enumerate() {
        let r = pixel[0] as u32;
        let g = pixel[1] as u32;
        let b = pixel[2] as u32;
        target_pixels[i] = (r << 16) | (g << 8) | b;
        sum_r += r as u64;
        sum_g += g as u64;
        sum_b += b as u64;
    }

    let count = (W * H) as u64;
    let bg_r = (sum_r / count) as u8;
    let bg_g = (sum_g / count) as u8;
    let bg_b = (sum_b / count) as u8;
    let bg_packed = ((bg_r as u32) << 16) | ((bg_g as u32) << 8) | bg_b as u32;

    let mut initial_ssd = 0.0f64;
    for target in &target_pixels {
        let tr = ((target >> 16) & 255) as f64;
        let tg = ((target >> 8) & 255) as f64;
        let tb = (target & 255) as f64;
        initial_ssd +=
            (tr - bg_r as f64).powi(2) + (tg - bg_g as f64).powi(2) + (tb - bg_b as f64).powi(2);
    }
    if initial_ssd == 0.0 {
        initial_ssd = 1.0;
    }

    HostImgData {
        target_pixels,
        bg_color: (bg_r, bg_g, bg_b),
        bg_packed,
        initial_ssd,
    }
}

/// Convert one image's accepted GPU history into the CSV row group consumed
/// by the Python data pipeline.  A fixed limit includes the background row;
/// when requested it must be met exactly so callers never receive a silently
/// shortened sequence.
fn format_image_sequence(
    image_name: &str,
    background: (u8, u8, u8),
    history: &[GpuShape],
    max_output_steps: Option<usize>,
) -> Result<String, String> {
    let requested_shape_count = max_output_steps.map(|step_count| step_count - 1);
    let accepted_shapes: Vec<&GpuShape> = history
        .iter()
        .filter(|shape| shape.delta_err < 0.0)
        .collect();

    if let Some(required) = requested_shape_count {
        if accepted_shapes.len() < required {
            return Err(format!(
                "图片 {image_name} 仅产生 {} 个有效图形，无法满足要求的 {} 个序列步骤（背景 + {} 个图形）。",
                accepted_shapes.len(),
                required + 1,
                required
            ));
        }
    }

    let shape_count = requested_shape_count.unwrap_or(accepted_shapes.len());
    let mut text = format!(
        "{image_name},0.0,0.0,0.0,0.0,-1,0.0,{},{},{}\n",
        background.0, background.1, background.2
    );
    for shape in accepted_shapes.into_iter().take(shape_count) {
        text.push_str(&format!(
            "{image_name},{},{},{},{},{},{},{},{},{}\n",
            shape.cx,
            shape.cy,
            shape.hw * 2.0,
            shape.hh * 2.0,
            shape.shape_type,
            shape.theta,
            shape.r,
            shape.g,
            shape.b
        ));
    }
    Ok(text)
}

fn create_gpu_shards(engines: &[GpuEngine], host_data: &[HostImgData]) -> Vec<GpuBatchShard> {
    balanced_ranges(host_data.len(), engines.len())
        .into_iter()
        .enumerate()
        .map(|(engine_index, range)| {
            let num_images = range.end - range.start;
            let pixel_count = num_images * (W * H) as usize;
            let mut flat_targets = Vec::with_capacity(pixel_count);
            let mut canvas_base = Vec::with_capacity(pixel_count);

            for data in &host_data[range.clone()] {
                flat_targets.extend_from_slice(&data.target_pixels);
                canvas_base.extend(std::iter::repeat_n(data.bg_packed, (W * H) as usize));
            }

            let target_buf = engines[engine_index].device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Targets"),
                    contents: bytemuck::cast_slice(&flat_targets),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            );

            println!(
                "  🛠️ GPU {} ({}) 分配图片区间 [{}..{})，共 {} 张",
                engines[engine_index].gpu_index,
                engines[engine_index].name,
                range.start,
                range.end,
                num_images
            );

            GpuBatchShard {
                range,
                target_buf,
                canvas_base,
            }
        })
        .collect()
}

async fn process_version_on_gpu(
    engine: &GpuEngine,
    shard: &GpuBatchShard,
    batch_paths: &[PathBuf],
    host_data: &[HostImgData],
    version: usize,
    steps_to_run: usize,
    max_output_steps: Option<usize>,
) -> Result<Vec<String>, String> {
    if steps_to_run == 0 || steps_to_run > MAX_ITERATIONS {
        return Err(format!(
            "GPU {} 的执行步数 {} 无效，合法范围为 1..={MAX_ITERATIONS}",
            engine.gpu_index, steps_to_run
        ));
    }

    let paths = &batch_paths[shard.range.clone()];
    let image_data = &host_data[shard.range.clone()];
    let num_images = paths.len();
    let mut rng = rand::thread_rng();

    println!(
        "  🔄 GPU {} ({}) 开始生成 v{}，图片数: {}",
        engine.gpu_index,
        engine.name,
        version + 1,
        num_images
    );

    let initial_states: Vec<ImageState> = image_data
        .iter()
        .map(|data| ImageState {
            current_ssd: data.initial_ssd as f32,
            initial_ssd: data.initial_ssd as f32,
            last_delta_err: 0.0,
            is_done: 0,
        })
        .collect();

    let mut config_data = Config {
        seed: 0,
        step: 0,
        num_images: num_images as u32,
        pad: 0,
    };

    let config_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config"),
            contents: bytemuck::bytes_of(&config_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let canvas_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Canvas"),
            contents: bytemuck::cast_slice(&shard.canvas_base),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let states_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("States"),
            contents: bytemuck::cast_slice(&initial_states),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let history_byte_size = (num_images * MAX_ITERATIONS * std::mem::size_of::<GpuShape>()) as u64;
    let states_byte_size = (num_images * std::mem::size_of::<ImageState>()) as u64;

    let history_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("History"),
        size: history_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let error_history_byte_size = (num_images * HISTORY_WINDOW * std::mem::size_of::<f32>()) as u64;
    let error_history_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Early Stop Error History"),
        size: error_history_byte_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mega Bind Group"),
        layout: &engine.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: shard.target_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: canvas_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: states_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: history_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: error_history_buf.as_entire_binding(),
            },
        ],
    });

    for step in 0..steps_to_run {
        config_data.seed = rng.gen();
        config_data.step = step as u32;
        engine
            .queue
            .write_buffer(&config_buf, 0, bytemuck::bytes_of(&config_data));
        engine.dispatch_step(&bind_group, num_images as u32);

        if (step + 1) % GPU_QUEUE_SYNC_INTERVAL == 0 || step + 1 == steps_to_run {
            engine.wait_for_gpu(&format!(
                "v{} 第 {}/{} 步队列同步",
                version + 1,
                step + 1,
                steps_to_run
            ))?;
            println!(
                "  ⏳ GPU {} v{}: 已完成 {}/{} 步",
                engine.gpu_index,
                version + 1,
                step + 1,
                steps_to_run
            );
        }
    }

    let final_states = engine
        .read_buffer_range::<ImageState>(&states_buf, 0, states_byte_size, "States Readback")
        .await?;
    let stopped_images = final_states
        .iter()
        .filter(|state| state.is_done == 1)
        .count();
    println!(
        "  🛑 GPU {} v{}: GPU 早停 {}/{} 张",
        engine.gpu_index,
        version + 1,
        stopped_images,
        num_images
    );

    let history_items_per_image = MAX_ITERATIONS;
    let history_bytes_per_image =
        (history_items_per_image * std::mem::size_of::<GpuShape>()) as u64;
    let readback_images = history_readback_images().min(num_images).max(1);
    let mut all_history = Vec::with_capacity(num_images * history_items_per_image);
    let readback_ranges = chunked_ranges(num_images, readback_images);
    let total_chunks = readback_ranges.len();
    println!(
        "  📥 GPU {} v{}: 历史保留在显存，按每块最多 {} 张图回读（共 {} 块）",
        engine.gpu_index,
        version + 1,
        readback_images,
        total_chunks
    );

    for (chunk_index, image_range) in readback_ranges.into_iter().enumerate() {
        let image_count = image_range.end - image_range.start;
        let source_offset = image_range.start as u64 * history_bytes_per_image;
        let chunk_byte_size = image_count as u64 * history_bytes_per_image;
        let label = format!(
            "History Readback GPU {} v{} chunk {}/{}",
            engine.gpu_index,
            version + 1,
            chunk_index + 1,
            total_chunks
        );
        let mut chunk = engine
            .read_buffer_range::<GpuShape>(&history_buf, source_offset, chunk_byte_size, &label)
            .await?;
        all_history.append(&mut chunk);
    }
    debug_assert_eq!(
        all_history.len(),
        history_byte_size as usize / std::mem::size_of::<GpuShape>()
    );

    let image_strings: Result<Vec<String>, String> = paths
        .par_iter()
        .enumerate()
        .map(|(local_index, path)| {
            let file_name = path
                .file_stem()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("❌ 图片文件名无效：{}", path.display()))?;
            let bg = image_data[local_index].bg_color;
            let history_slice =
                &all_history[local_index * MAX_ITERATIONS..(local_index + 1) * MAX_ITERATIONS];
            format_image_sequence(file_name, bg, history_slice, max_output_steps)
        })
        .collect();
    let image_strings = image_strings?;

    println!(
        "  ✅ GPU {} ({}) 已完成 v{}",
        engine.gpu_index,
        engine.name,
        version + 1
    );
    Ok(image_strings)
}

fn write_version_csv(
    output_dir: &Path,
    version: usize,
    image_groups: Vec<Vec<String>>,
    state: &mut CsvState,
    header: &str,
) -> std::io::Result<()> {
    let version_dir = output_dir.join(format!("v{}", version + 1));
    let open_part = |index| {
        let path = version_dir.join(format!("data_part_{index}.csv"));
        fs::OpenOptions::new().create(true).append(true).open(path)
    };

    let mut writer = BufWriter::new(open_part(state.file_idx)?);
    if state.current_size == 0 {
        writer.write_all(header.as_bytes())?;
        state.current_size += header.len();
    }

    for image_text in image_groups.into_iter().flatten() {
        let image_bytes = image_text.len();
        if state.current_size + image_bytes > MAX_CSV_SIZE_BYTES {
            writer.flush()?;
            state.file_idx += 1;
            writer = BufWriter::new(open_part(state.file_idx)?);
            writer.write_all(header.as_bytes())?;
            state.current_size = header.len();
        }

        writer.write_all(image_text.as_bytes())?;
        state.current_size += image_bytes;
    }
    writer.flush()
}

// ==========================================
// 🚀 主执行流程
// ==========================================
async fn run() -> Result<(), String> {
    // 在访问数据集和创建 GPU 前完成参数校验与 WGSL 生成。
    let shader_constants = shader_constants();
    let shader_source = build_shader_source(&shader_constants)?;
    print_shader_constants(&shader_constants);

    let input_dir = input_dir()?;
    let output_dir = output_dir();
    let num_versions = positive_usize_env("SHAPE_RENDERER_NUM_VERSIONS").unwrap_or(NUM_VERSIONS);
    let max_output_steps = output_step_limit()?;

    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("❌ 无法创建输出目录 {}: {err}", output_dir.display()))?;
    for version in 0..num_versions {
        fs::create_dir_all(output_dir.join(format!("v{}", version + 1)))
            .map_err(|err| format!("❌ 无法创建版本输出目录: {err}"))?;
    }

    println!("🔎 正在递归扫描输入目录: {}", input_dir.display());
    let paths = collect_image_paths(&input_dir)
        .map_err(|err| format!("❌ 无法扫描输入目录 {}: {err}", input_dir.display()))?;
    if paths.is_empty() {
        println!("⚠️ 没有检测到输入图片。");
        return Ok(());
    }
    println!("🖼️ 共发现 {} 张输入图片", paths.len());

    let engines = GpuEngine::discover_all(&shader_source).await?;
    println!(
        "🧬 计划生成版本数: {} | 画布: {}x{} | GPU 最大步数: {} | 输出步骤上限: {} | 单个 CSV 上限: {} MB",
        num_versions,
        W,
        H,
        MAX_ITERATIONS,
        max_output_steps
            .map(|value| value.to_string())
            .unwrap_or_else(|| "不限制".to_string()),
        MAX_CSV_SIZE_BYTES / 1024 / 1024
    );
    println!("📂 输出目录: {}", output_dir.display());

    let mut csv_states: Vec<CsvState> = (0..num_versions)
        .map(|_| CsvState {
            file_idx: 1,
            current_size: 0,
        })
        .collect();
    let header = "image_name,cx,cy,w,h,shape_type,theta,r,g,b\n";
    let per_gpu_batch_size = images_per_gpu_batch();
    let gpu_batch_size = per_gpu_batch_size * engines.len();
    println!(
        "📦 每批最多 {} 张 | 每张 GPU 约 {} 张",
        gpu_batch_size, per_gpu_batch_size
    );

    for (batch_index, batch_paths) in paths.chunks(gpu_batch_size).enumerate() {
        let num_images = batch_paths.len();
        println!("📦 ====================================================");
        println!(
            "📦 开始处理第 {} 批次 | 包含 {} 张图片",
            batch_index + 1,
            num_images
        );
        let batch_start = Instant::now();

        let host_data: Vec<HostImgData> = batch_paths
            .par_iter()
            .map(|path| preprocess_image(path))
            .collect();
        let gpu_shards = create_gpu_shards(&engines, &host_data);
        let active_engines = &engines[..gpu_shards.len()];

        for version in 0..num_versions {
            println!("  🔄 并行生成变体 v{} / {}", version + 1, num_versions);

            // IndexedParallelIterator::collect 保留 GPU/分片顺序，因此 CSV 顺序稳定且无并发写入。
            let image_groups: Result<Vec<Vec<String>>, String> = active_engines
                .par_iter()
                .zip(gpu_shards.par_iter())
                .map(|(engine, shard)| {
                    pollster::block_on(process_version_on_gpu(
                        engine,
                        shard,
                        batch_paths,
                        &host_data,
                        version,
                        MAX_ITERATIONS,
                        max_output_steps,
                    ))
                })
                .collect();
            let image_groups = image_groups.map_err(|err| {
                format!(
                    "❌ v{} GPU 计算或回读失败，未写入该版本 CSV: {}",
                    version + 1,
                    err
                )
            })?;

            write_version_csv(
                &output_dir,
                version,
                image_groups,
                &mut csv_states[version],
                header,
            )
            .map_err(|err| format!("❌ 写入 v{} CSV 失败: {err}", version + 1))?;
            println!("  📝 v{} CSV 写入完成", version + 1);
        }

        println!(
            "✅ 第 {} 批次的 {} 个版本已全部完成！批次耗时: {:?}",
            batch_index + 1,
            num_versions,
            batch_start.elapsed()
        );
    }

    println!("🎉 所有图片及其变体版本均已生成完毕！");
    Ok(())
}

#[cfg(windows)]
fn configure_utf8_console() {
    const CP_UTF8: u32 = 65001;

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn SetConsoleCP(code_page: u32) -> i32;
        fn SetConsoleOutputCP(code_page: u32) -> i32;
    }

    // 设置失败时仍继续运行；也可以在终端中手动执行 `chcp 65001`。
    unsafe {
        let _ = SetConsoleCP(CP_UTF8);
        let _ = SetConsoleOutputCP(CP_UTF8);
    }
}

#[cfg(not(windows))]
fn configure_utf8_console() {}

fn main() {
    configure_utf8_console();
    if let Err(err) = pollster::block_on(run()) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn splits_work_evenly_without_empty_ranges() {
        assert_eq!(balanced_ranges(10, 3), vec![0..4, 4..7, 7..10]);
        assert_eq!(balanced_ranges(2, 4), vec![0..1, 1..2]);
        assert!(balanced_ranges(0, 2).is_empty());
        assert!(balanced_ranges(5, 0).is_empty());
    }

    #[test]
    fn chunks_large_history_readback_without_gaps_or_overlaps() {
        let ranges = chunked_ranges(1000, 128);
        assert_eq!(ranges.len(), 8);
        assert_eq!(ranges.first(), Some(&(0..128)));
        assert_eq!(ranges.last(), Some(&(896..1000)));

        let flattened: Vec<usize> = ranges.into_iter().flatten().collect();
        assert_eq!(flattened, (0..1000).collect::<Vec<_>>());
        assert!(chunked_ranges(0, 128).is_empty());
        assert!(chunked_ranges(1000, 0).is_empty());
    }

    #[test]
    fn recognizes_supported_image_extensions_case_insensitively() {
        assert!(is_supported_image(Path::new("face.JPG")));
        assert!(is_supported_image(Path::new("face.jpeg")));
        assert!(is_supported_image(Path::new("face.png")));
        assert!(is_supported_image(Path::new("face.WEBP")));
        assert!(!is_supported_image(Path::new(
            ".face_cutter_progress_gpu0.txt"
        )));
        assert!(!is_supported_image(Path::new("no_extension")));
    }

    #[test]
    fn fixed_output_sequence_keeps_background_and_first_accepted_shapes() {
        let history = [
            GpuShape {
                delta_err: 0.0,
                ..GpuShape::zeroed()
            },
            GpuShape {
                cx: 1.0,
                delta_err: -1.0,
                ..GpuShape::zeroed()
            },
            GpuShape {
                cx: 2.0,
                delta_err: -2.0,
                ..GpuShape::zeroed()
            },
            GpuShape {
                cx: 3.0,
                delta_err: -3.0,
                ..GpuShape::zeroed()
            },
        ];

        let text = format_image_sequence("demo", (4, 5, 6), &history, Some(3)).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "demo,0.0,0.0,0.0,0.0,-1,0.0,4,5,6");
        assert!(lines[1].starts_with("demo,1,"));
        assert!(lines[2].starts_with("demo,2,"));
    }

    #[test]
    fn fixed_output_sequence_rejects_insufficient_accepted_shapes() {
        let history = [GpuShape {
            delta_err: -1.0,
            ..GpuShape::zeroed()
        }];
        let error = format_image_sequence("demo", (0, 0, 0), &history, Some(3)).unwrap_err();
        assert!(error.contains("demo"));
        assert!(error.contains("仅产生 1 个有效图形"));
    }

    #[test]
    fn recursively_collects_only_images() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("shape_renderer_scan_test_{unique}"));
        let nested = root.join("0007");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join(".progress.txt"), b"progress").unwrap();
        fs::write(root.join("root.png"), b"test").unwrap();
        fs::write(nested.join("face.JPG"), b"test").unwrap();

        let paths = collect_image_paths(&root).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|path| is_supported_image(path)));
        assert!(paths.iter().any(|path| path.ends_with("root.png")));
        assert!(paths.iter().any(|path| path.ends_with("face.JPG")));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gpu_struct_layouts_match_wgsl() {
        assert_eq!(std::mem::size_of::<Config>(), 16);
        assert_eq!(std::mem::size_of::<ImageState>(), 16);
        assert_eq!(std::mem::size_of::<GpuShape>(), 48);
    }

    #[test]
    fn default_shader_constants_match_current_production_configuration() {
        let constants = shader_constants();
        assert_eq!(constants.canvas_width, 256);
        assert_eq!(constants.canvas_height, 256);
        assert_eq!(constants.shape_alpha, 0.5);
        assert_eq!(constants.penalty_factor, 1.8);
        assert_eq!(constants.max_iterations, 275);
        assert_eq!(constants.history_window, 1000);
        assert_eq!(constants.early_stop_threshold, 0.0005);
        assert_eq!(constants.initial_candidate_count, 50);
        assert_eq!(constants.survivor_count, 5);
        assert_eq!(constants.mutation_stagnation_limit, 100);
        assert_eq!(constants.max_mutations_per_survivor, 10000);
        assert_eq!(constants.position_mutation_range, 28.0);
        assert_eq!(constants.width_mutation_range, 28.0);
        assert_eq!(constants.height_mutation_range, 28.0);
        assert_eq!(constants.angle_mutation_range_degrees, 16.0);
        assert_eq!(constants.min_half_size, 1.0);
        assert_eq!(constants.max_half_size, 128.0);
        assert_eq!(constants.mutation_weights, [1, 1, 1, 1, 1]);
    }

    #[test]
    fn generated_shader_contains_constants_and_no_template_markers() {
        let source = build_shader_source(&shader_constants()).unwrap();
        assert!(source.contains("const CANVAS_WIDTH: u32 = 256u;"));
        assert!(source.contains("const INITIAL_CANDIDATE_COUNT: u32 = 50u;"));
        assert!(source.contains("const SURVIVOR_COUNT: u32 = 5u;"));
        assert!(source.contains("const MUTATION_STAGNATION_LIMIT: u32 = 100u;"));
        assert!(source.contains("const TOTAL_MUTATION_WEIGHT: u32 = 5u;"));
        assert!(source.contains("array<GpuShape, SURVIVOR_COUNT>"));
        assert!(!source.contains("__"));
        assert!(!source.contains("config.w"));
        assert!(!source.contains("config.max_iterations"));
    }

    #[test]
    fn rejects_invalid_shader_constants_before_gpu_initialization() {
        let mut invalid = shader_constants();
        invalid.survivor_count = invalid.initial_candidate_count + 1;
        assert!(validate_shader_constants(&invalid)
            .unwrap_err()
            .contains("SURVIVOR_COUNT"));

        invalid = shader_constants();
        invalid.mutation_stagnation_limit = 0;
        assert!(validate_shader_constants(&invalid)
            .unwrap_err()
            .contains("MUTATION_STAGNATION_LIMIT"));

        invalid = shader_constants();
        invalid.mutation_stagnation_limit = invalid.max_mutations_per_survivor + 1;
        assert!(validate_shader_constants(&invalid)
            .unwrap_err()
            .contains("MAX_MUTATIONS_PER_SURVIVOR"));

        invalid = shader_constants();
        invalid.max_half_size = 129.0;
        assert!(validate_shader_constants(&invalid)
            .unwrap_err()
            .contains("MAX_HALF_SIZE"));

        invalid = shader_constants();
        invalid.position_mutation_range = -1.0;
        assert!(validate_shader_constants(&invalid)
            .unwrap_err()
            .contains("POSITION_MUTATION_RANGE"));

        invalid = shader_constants();
        invalid.mutation_weights = [0; 5];
        assert!(validate_shader_constants(&invalid)
            .unwrap_err()
            .contains("至少一个权重"));

        invalid = shader_constants();
        invalid.mutation_weights = [u32::MAX, 1, 0, 0, 0];
        assert!(validate_shader_constants(&invalid)
            .unwrap_err()
            .contains("超过 u32 上限"));
    }

    #[test]
    fn equal_mutation_weights_match_legacy_hash_modulo_mapping() {
        fn weighted_choice(hash: u32, weights: [u32; 5]) -> usize {
            let choice = hash % weights.iter().sum::<u32>();
            let mut upper_bound = 0u32;
            for (index, weight) in weights.into_iter().enumerate() {
                upper_bound += weight;
                if choice < upper_bound {
                    return index;
                }
            }
            unreachable!()
        }

        for hash in 0..10_000 {
            assert_eq!(weighted_choice(hash, [1; 5]), (hash % 5) as usize);
        }
    }

    #[test]
    fn sufficient_statistics_match_direct_error_evaluation() {
        let alpha = 0.5f64;
        let samples = [
            (241.0, 18.0, 1.0),
            (72.0, 150.0, 0.2),
            (129.0, 44.0, 0.75),
            (5.0, 220.0, 0.43),
            (200.0, 190.0, 0.01),
        ];

        let weight_sum: f64 = samples.iter().map(|&(_, _, coverage)| coverage).sum();
        let opt_sum: f64 = samples
            .iter()
            .map(|&(target, canvas, coverage)| (((target - canvas) / alpha) + canvas) * coverage)
            .sum();
        let opt = (opt_sum / weight_sum).clamp(0.0, 255.0).trunc();

        let direct_delta: f64 = samples
            .iter()
            .map(|&(target, canvas, coverage)| {
                let k = alpha * coverage;
                let old_residual = target - canvas;
                let new_residual = old_residual - k * (opt - canvas);
                new_residual * new_residual - old_residual * old_residual
            })
            .sum();

        let mut a = 0.0;
        let mut b = 0.0;
        let mut c = 0.0;
        for &(target, canvas, coverage) in &samples {
            let d = target - canvas;
            let k = alpha * coverage;
            let k2 = k * k;
            a += k2;
            b += -2.0 * k * d - 2.0 * k2 * canvas;
            c += 2.0 * k * d * canvas + k2 * canvas * canvas;
        }
        let stats_delta = a * opt * opt + b * opt + c;

        assert!((direct_delta - stats_delta).abs() < 1e-8);
    }

    fn cpu_coverage(
        shape_type: u32,
        hw: f64,
        hh: f64,
        dx: f64,
        dy: f64,
        cos_t: f64,
        sin_t: f64,
    ) -> f64 {
        let lx = dx * cos_t + dy * sin_t;
        let ly = -dx * sin_t + dy * cos_t;
        let d = if shape_type == 0 {
            let qx = lx.abs() - hw;
            let qy = ly.abs() - hh;
            (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0)
        } else {
            let inv_hw2 = 1.0 / (hw * hw).max(0.0001);
            let inv_hh2 = 1.0 / (hh * hh).max(0.0001);
            let f = lx * lx * inv_hw2 + ly * ly * inv_hh2 - 1.0;
            let gx = 2.0 * lx * inv_hw2;
            let gy = 2.0 * ly * inv_hh2;
            let gradient = (gx * gx + gy * gy).sqrt();
            if gradient > 1e-5 {
                f / gradient
            } else {
                f
            }
        };
        let t = (d + 0.5).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    fn optimized_bounds(
        shape_type: u32,
        cx: f64,
        cy: f64,
        hw: f64,
        hh: f64,
        theta: f64,
    ) -> (i32, i32, i32, i32) {
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let (extent_x, extent_y) = if shape_type == 0 {
            (
                cos_t.abs() * hw + sin_t.abs() * hh,
                sin_t.abs() * hw + cos_t.abs() * hh,
            )
        } else {
            (
                ((hw * cos_t).powi(2) + (hh * sin_t).powi(2)).sqrt(),
                ((hw * sin_t).powi(2) + (hh * cos_t).powi(2)).sqrt(),
            )
        };
        let margin = 1.0;
        (
            (cx - extent_x - margin).floor().max(0.0) as i32,
            (cx + extent_x + margin).ceil().min((W - 1) as f64) as i32,
            (cy - extent_y - margin).floor().max(0.0) as i32,
            (cy + extent_y + margin).ceil().min((H - 1) as f64) as i32,
        )
    }

    #[test]
    fn optimized_bounds_contain_every_covered_pixel() {
        let shapes: [(u32, f64, f64, f64, f64, f64); 4] = [
            (0, 96.0, 96.0, 80.0, 5.0, 0.73),
            (0, 8.0, 180.0, 45.0, 17.0, 2.41),
            (1, 96.0, 96.0, 100.0, 8.0, 1.13),
            (1, 184.0, 12.0, 72.0, 31.0, 2.77),
        ];

        for (shape_type, cx, cy, hw, hh, theta) in shapes {
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let (min_x, max_x, min_y, max_y) = optimized_bounds(shape_type, cx, cy, hw, hh, theta);

            for y in 0..H as i32 {
                for x in 0..W as i32 {
                    let coverage = cpu_coverage(
                        shape_type,
                        hw,
                        hh,
                        x as f64 - cx,
                        y as f64 - cy,
                        cos_t,
                        sin_t,
                    );
                    if coverage > 0.0 {
                        assert!(
                            x >= min_x && x <= max_x && y >= min_y && y <= max_y,
                            "covered pixel ({x}, {y}) escaped bounds for shape {shape_type}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ring_history_matches_original_early_stop_window() {
        let window = 100usize;
        let threshold = 0.0005f32;
        let mut history = vec![0.0f32; window];
        let errors = vec![1.0f32; window + 1];
        let stop_step = errors.iter().enumerate().find_map(|(step, &current)| {
            let slot = step % window;
            let should_stop = step >= window && history[slot] - current <= threshold;
            history[slot] = current;
            should_stop.then_some(step)
        });
        assert_eq!(stop_step, Some(100));
    }

    /// 端到端 GPU 单步测试：验证 6 个 binding、单遍统计、GPU 状态和历史回读。
    #[test]
    #[ignore = "requires local hardware GPU and runs one optimized compute step"]
    fn optimized_shader_computes_one_step_on_all_gpus() {
        configure_utf8_console();
        let shader_source = build_shader_source(&shader_constants()).unwrap();
        let engines = pollster::block_on(GpuEngine::discover_all(&shader_source)).unwrap();
        let pixel_count = (W * H) as usize;
        let paths: Vec<PathBuf> = (0..engines.len())
            .map(|index| PathBuf::from(format!("fast_smoke_gpu_{index}.png")))
            .collect();
        let host_data: Vec<HostImgData> = (0..engines.len())
            .map(|index| {
                let mut target_pixels = Vec::with_capacity(pixel_count);
                for pixel in 0..pixel_count {
                    let value = ((pixel + index * 31) % 256) as u32;
                    target_pixels.push((value << 16) | ((255 - value) << 8) | (value / 2));
                }
                HostImgData {
                    target_pixels,
                    bg_color: (127, 127, 63),
                    bg_packed: (127 << 16) | (127 << 8) | 63,
                    initial_ssd: 1_000_000_000.0,
                }
            })
            .collect();
        let shards = create_gpu_shards(&engines, &host_data);

        let groups: Vec<Vec<String>> = engines
            .par_iter()
            .zip(shards.par_iter())
            .map(|(engine, shard)| {
                pollster::block_on(process_version_on_gpu(
                    engine, shard, &paths, &host_data, 0, 1, None,
                ))
                .expect("optimized GPU step and readback should succeed")
            })
            .collect();

        assert_eq!(groups.iter().map(Vec::len).sum::<usize>(), engines.len());
        assert!(groups.iter().flatten().all(|text| !text.is_empty()));
    }
}
