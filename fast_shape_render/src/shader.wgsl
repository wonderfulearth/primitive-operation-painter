struct Config {
    seed: u32, step: u32, num_images: u32, pad: u32,
};

struct GpuShape {
    cx: f32, cy: f32, hw: f32, hh: f32, theta: f32, shape_type: u32,
    r: u32, g: u32, b: u32, delta_err: f32, pad1: u32, pad2: u32,
};

struct ImageState {
    current_ssd: f32, initial_ssd: f32, last_delta_err: f32, is_done: u32,
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> target_tex: array<u32>;
@group(0) @binding(2) var<storage, read_write> canvas_tex: array<u32>;
@group(0) @binding(3) var<storage, read_write> states: array<ImageState>;
@group(0) @binding(4) var<storage, read_write> history: array<GpuShape>;
@group(0) @binding(5) var<storage, read_write> error_history: array<f32>;

var<private> rng_state: u32;
fn pcg_hash() -> u32 {
    rng_state = rng_state * 747796405u + 2891336453u;
    let word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn rand_f32() -> f32 { return f32(pcg_hash()) / 4294967296.0; }
fn rand_range(min_v: f32, max_v: f32) -> f32 { return min_v + rand_f32() * (max_v - min_v); }

fn gen_random_shape() -> GpuShape {
    var s: GpuShape;
    s.cx = rand_range(0.0, f32(CANVAS_WIDTH - 1u));
    s.cy = rand_range(0.0, f32(CANVAS_HEIGHT - 1u));
    s.hw = rand_range(MIN_HALF_SIZE, MAX_HALF_SIZE);
    s.hh = rand_range(MIN_HALF_SIZE, MAX_HALF_SIZE);
    s.theta = rand_range(0.0, 3.1415926535);
    s.shape_type = pcg_hash() % 2u;
    return s;
}

fn mutate_shape(base: GpuShape) -> GpuShape {
    var clone = base;
    let choice = pcg_hash() % TOTAL_MUTATION_WEIGHT;
    let width_start = POSITION_MUTATION_WEIGHT;
    let height_start = width_start + WIDTH_MUTATION_WEIGHT;
    let angle_start = height_start + HEIGHT_MUTATION_WEIGHT;
    let shape_type_start = angle_start + ANGLE_MUTATION_WEIGHT;
    if (choice < width_start) {
        clone.cx = clamp(clone.cx + rand_range(-POSITION_MUTATION_RANGE, POSITION_MUTATION_RANGE), 0.0, f32(CANVAS_WIDTH - 1u));
        clone.cy = clamp(clone.cy + rand_range(-POSITION_MUTATION_RANGE, POSITION_MUTATION_RANGE), 0.0, f32(CANVAS_HEIGHT - 1u));
    } else if (choice < height_start) {
        clone.hw = clamp(clone.hw + rand_range(-WIDTH_MUTATION_RANGE, WIDTH_MUTATION_RANGE), MIN_HALF_SIZE, MAX_HALF_SIZE);
    } else if (choice < angle_start) {
        clone.hh = clamp(clone.hh + rand_range(-HEIGHT_MUTATION_RANGE, HEIGHT_MUTATION_RANGE), MIN_HALF_SIZE, MAX_HALF_SIZE);
    } else if (choice < shape_type_start) {
        let pi = 3.1415926535;
        let mut_theta = clone.theta + rand_range(-ANGLE_MUTATION_RANGE_RADIANS, ANGLE_MUTATION_RANGE_RADIANS);

        // 🔄 核心修复：环形折叠 (Wrap-around)
        // 将角度完美地限制在 [0.0, PI) 的区间内。
        // 如果 mut_theta 是 -0.1，算出来会自动变成 3.04 (PI - 0.1)
        // 如果 mut_theta 是 3.24，算出来会自动变成 0.1 (3.24 - PI)
        clone.theta = mut_theta - pi * floor(mut_theta / pi);
    } else {
        clone.shape_type = 1u - clone.shape_type;
    }
    return clone;
}

// 🚀 核心黑魔法：SDF (符号距离场) 覆盖率计算
// 它会根据像素中心点到图形边缘的几何距离，计算出 0.0 到 1.0 的完美平滑覆盖率
fn get_coverage(
    shape_type: u32,
    hw: f32,
    hh: f32,
    dx: f32,
    dy: f32,
    cos_t: f32,
    sin_t: f32,
    inv_hw2: f32,
    inv_hh2: f32,
) -> f32 {
    let lx = dx * cos_t + dy * sin_t;
    let ly = -dx * sin_t + dy * cos_t;
    var d: f32 = 0.0;

    if (shape_type == 0u) {
        // 矩形的精确解析距离场
        let d_vec = abs(vec2<f32>(lx, ly)) - vec2<f32>(hw, hh);
        d = length(max(d_vec, vec2<f32>(0.0))) + min(max(d_vec.x, d_vec.y), 0.0);
    } else {
        // 椭圆的梯度近似解析距离场（利用一阶泰勒展开，极其高效）
        let f = (lx * lx) * inv_hw2 + (ly * ly) * inv_hh2 - 1.0;
        let gx = 2.0 * lx * inv_hw2;
        let gy = 2.0 * ly * inv_hh2;
        let g_len = length(vec2<f32>(gx, gy));

        if (g_len > 1e-5) {
            d = f / g_len;
        } else {
            d = f; // 中心点特殊处理
        }
    }

    // smoothstep 将边界 -0.5(内部) 到 0.5(外部) 的距离映射为 0.0 到 1.0
    // 1.0 - smoothstep 即反转为：内部为1，外部为0，边缘平滑渐变
    return 1.0 - smoothstep(-0.5, 0.5, d);
}

const WG_SIZE = 256u;
var<workgroup> sum_opt_r: array<f32, WG_SIZE>;
var<workgroup> sum_opt_g: array<f32, WG_SIZE>;
var<workgroup> sum_opt_b: array<f32, WG_SIZE>;
var<workgroup> sum_weight: array<f32, WG_SIZE>;
var<workgroup> sum_k2: array<f32, WG_SIZE>;
var<workgroup> sum_b_r: array<f32, WG_SIZE>;
var<workgroup> sum_b_g: array<f32, WG_SIZE>;
var<workgroup> sum_b_b: array<f32, WG_SIZE>;
var<workgroup> sum_c_r: array<f32, WG_SIZE>;
var<workgroup> sum_c_g: array<f32, WG_SIZE>;
var<workgroup> sum_c_b: array<f32, WG_SIZE>;

var<workgroup> shared_opt_r: f32;
var<workgroup> shared_opt_g: f32;
var<workgroup> shared_opt_b: f32;
var<workgroup> out_err: f32;
var<workgroup> current_shape: GpuShape;
var<workgroup> shared_cos: f32;
var<workgroup> shared_sin: f32;
var<workgroup> shared_inv_hw2: f32;
var<workgroup> shared_inv_hh2: f32;
var<workgroup> shared_min_x: i32;
var<workgroup> shared_max_x: i32;
var<workgroup> shared_min_y: i32;
var<workgroup> shared_max_y: i32;
var<workgroup> shared_box_w: u32;
var<workgroup> shared_box_h: u32;
var<workgroup> wg_age: u32;
var<workgroup> wg_apply_flag: u32;
var<workgroup> wg_skip_flag: u32;

// 每个候选只由线程 0 计算一次三角函数、尺寸倒数和精确旋转包围盒。
// 1 像素余量覆盖 SDF 的抗锯齿过渡区。
fn prepare_current_shape(tid: u32) {
    if (tid == 0u) {
        shared_cos = cos(current_shape.theta);
        shared_sin = sin(current_shape.theta);
        shared_inv_hw2 = 1.0 / max(current_shape.hw * current_shape.hw, 0.0001);
        shared_inv_hh2 = 1.0 / max(current_shape.hh * current_shape.hh, 0.0001);

        let abs_cos = abs(shared_cos);
        let abs_sin = abs(shared_sin);
        var extent_x: f32;
        var extent_y: f32;

        if (current_shape.shape_type == 0u) {
            // 旋转矩形的精确轴对齐包围盒。
            extent_x = abs_cos * current_shape.hw + abs_sin * current_shape.hh;
            extent_y = abs_sin * current_shape.hw + abs_cos * current_shape.hh;
        } else {
            // 旋转椭圆的精确轴对齐包围盒。
            let hw_cos = current_shape.hw * shared_cos;
            let hh_sin = current_shape.hh * shared_sin;
            let hw_sin = current_shape.hw * shared_sin;
            let hh_cos = current_shape.hh * shared_cos;
            extent_x = sqrt(hw_cos * hw_cos + hh_sin * hh_sin);
            extent_y = sqrt(hw_sin * hw_sin + hh_cos * hh_cos);
        }

        let aa_margin = 1.0;
        shared_min_x = max(0, i32(floor(current_shape.cx - extent_x - aa_margin)));
        shared_max_x = min(
            i32(CANVAS_WIDTH - 1u),
            i32(ceil(current_shape.cx + extent_x + aa_margin)),
        );
        shared_min_y = max(0, i32(floor(current_shape.cy - extent_y - aa_margin)));
        shared_max_y = min(
            i32(CANVAS_HEIGHT - 1u),
            i32(ceil(current_shape.cy + extent_y + aa_margin)),
        );
        shared_box_w = u32(max(0, shared_max_x - shared_min_x + 1));
        shared_box_h = u32(max(0, shared_max_y - shared_min_y + 1));
    }
    workgroupBarrier();
}

fn eval_current_shape(img_offset: u32, tid: u32) {
    sum_opt_r[tid] = 0.0;
    sum_opt_g[tid] = 0.0;
    sum_opt_b[tid] = 0.0;
    sum_weight[tid] = 0.0;
    sum_k2[tid] = 0.0;
    sum_b_r[tid] = 0.0;
    sum_b_g[tid] = 0.0;
    sum_b_b[tid] = 0.0;
    sum_c_r[tid] = 0.0;
    sum_c_g[tid] = 0.0;
    sum_c_b[tid] = 0.0;
    prepare_current_shape(tid);

    // ==========================================
    // 单遍扫描：同时聚合当前颜色解与误差二次式的充分统计量。
    //
    // 对每个通道，k = alpha * coverage，d = target - canvas：
    // delta = (d - k * (opt - canvas))^2 - d^2
    //       = A * opt^2 + B * opt + C
    // 扫描结束后将当前算法的 opt 代入即可，无需第二次计算 SDF。
    // ==========================================
    var idx = tid;
    let total_pixels = shared_box_w * shared_box_h;
    let inv_alpha = 1.0 / SHAPE_ALPHA;
    while (idx < total_pixels) {
        let x = u32(shared_min_x) + (idx % shared_box_w);
        let y = u32(shared_min_y) + (idx / shared_box_w);
        let px_idx = img_offset + y * CANVAS_WIDTH + x;
        let dx = f32(x) - current_shape.cx;
        let dy = f32(y) - current_shape.cy;

        let coverage = get_coverage(
            current_shape.shape_type,
            current_shape.hw,
            current_shape.hh,
            dx,
            dy,
            shared_cos,
            shared_sin,
            shared_inv_hw2,
            shared_inv_hh2,
        );

        if (coverage > 0.0) {
            let trgb = target_tex[px_idx];
            let crgb = canvas_tex[px_idx];
            let tr = f32((trgb >> 16u) & 255u);
            let tg = f32((trgb >> 8u) & 255u);
            let tb = f32(trgb & 255u);
            let cr = f32((crgb >> 16u) & 255u);
            let cg = f32((crgb >> 8u) & 255u);
            let cb = f32(crgb & 255u);
            let dr = tr - cr;
            let dg = tg - cg;
            let db = tb - cb;
            let k = SHAPE_ALPHA * coverage;
            let k2 = k * k;

            sum_opt_r[tid] += (dr * inv_alpha + cr) * coverage;
            sum_opt_g[tid] += (dg * inv_alpha + cg) * coverage;
            sum_opt_b[tid] += (db * inv_alpha + cb) * coverage;
            sum_weight[tid] += coverage;
            sum_k2[tid] += k2;

            sum_b_r[tid] += -2.0 * k * dr - 2.0 * k2 * cr;
            sum_b_g[tid] += -2.0 * k * dg - 2.0 * k2 * cg;
            sum_b_b[tid] += -2.0 * k * db - 2.0 * k2 * cb;
            sum_c_r[tid] += 2.0 * k * dr * cr + k2 * cr * cr;
            sum_c_g[tid] += 2.0 * k * dg * cg + k2 * cg * cg;
            sum_c_b[tid] += 2.0 * k * db * cb + k2 * cb * cb;
        }
        idx += WG_SIZE;
    }
    workgroupBarrier();

    for (var s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sum_opt_r[tid] += sum_opt_r[tid + s];
            sum_opt_g[tid] += sum_opt_g[tid + s];
            sum_opt_b[tid] += sum_opt_b[tid + s];
            sum_weight[tid] += sum_weight[tid + s];
            sum_k2[tid] += sum_k2[tid + s];
            sum_b_r[tid] += sum_b_r[tid + s];
            sum_b_g[tid] += sum_b_g[tid + s];
            sum_b_b[tid] += sum_b_b[tid + s];
            sum_c_r[tid] += sum_c_r[tid + s];
            sum_c_g[tid] += sum_c_g[tid + s];
            sum_c_b[tid] += sum_c_b[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        if (sum_weight[0] > 0.0) {
            shared_opt_r = f32(u32(clamp(sum_opt_r[0] / sum_weight[0], 0.0, 255.0)));
            shared_opt_g = f32(u32(clamp(sum_opt_g[0] / sum_weight[0], 0.0, 255.0)));
            shared_opt_b = f32(u32(clamp(sum_opt_b[0] / sum_weight[0], 0.0, 255.0)));
        } else {
            shared_opt_r = 0.0;
            shared_opt_g = 0.0;
            shared_opt_b = 0.0;
        }

        var raw_err =
            sum_k2[0] * (
                shared_opt_r * shared_opt_r +
                shared_opt_g * shared_opt_g +
                shared_opt_b * shared_opt_b
            ) +
            sum_b_r[0] * shared_opt_r +
            sum_b_g[0] * shared_opt_g +
            sum_b_b[0] * shared_opt_b +
            sum_c_r[0] +
            sum_c_g[0] +
            sum_c_b[0];

        // 与原版相同的越界惩罚。
        let inside_area = sum_weight[0];
        var total_area = 0.0;
        if (current_shape.shape_type == 0u) {
            total_area = 4.0 * current_shape.hw * current_shape.hh;
        } else {
            total_area = 3.1415926535 * current_shape.hw * current_shape.hh;
        }

        let base_inside_ratio = inside_area / total_area;
        var oob_ratio = 1.0 - base_inside_ratio;
        if (oob_ratio < 0.0) { oob_ratio = 0.0; }

        let eff_oob = oob_ratio * PENALTY_FACTOR;
        var final_ratio = 1.0 - eff_oob;
        if (final_ratio < 0.001) { final_ratio = 0.001; }
        if (final_ratio > 1.0) { final_ratio = 1.0; }

        if (raw_err < 0.0) { out_err = raw_err * final_ratio; } else { out_err = raw_err / final_ratio; }
    }
    workgroupBarrier();
}

var<workgroup> top_shapes: array<GpuShape, SURVIVOR_COUNT>;
var<workgroup> top_errs: array<f32, SURVIVOR_COUNT>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let img_idx = group_id.y * 256u + group_id.x;

    // 只有线程 0 读取边界/早停状态，再通过 workgroupUniformLoad 将结果
    // 声明为整个工作组一致。这样提前退出不会让部分线程跳过后续 barrier。
    if (tid == 0u) {
        if (img_idx >= config.num_images) {
            wg_skip_flag = 1u;
        } else {
            wg_skip_flag = states[img_idx].is_done;
        }
    }
    let skip_workgroup = workgroupUniformLoad(&wg_skip_flag);
    if (skip_workgroup == 1u) { return; }

    let img_offset = img_idx * CANVAS_WIDTH * CANVAS_HEIGHT;

    if (tid == 0u) {
        rng_state = img_idx ^ config.seed;
        for (var i=0u; i<SURVIVOR_COUNT; i++) { top_errs[i] = 99999999.0; }
        wg_apply_flag = 0u;
    }
    workgroupBarrier();

    // 🌊 海选
    for (var i = 0u; i < INITIAL_CANDIDATE_COUNT; i++) {
        if (tid == 0u) { current_shape = gen_random_shape(); }
        workgroupBarrier();
        eval_current_shape(img_offset, tid);

        if (tid == 0u) {
            var e = out_err; var s = current_shape;
            s.r = u32(shared_opt_r); s.g = u32(shared_opt_g); s.b = u32(shared_opt_b); s.delta_err = e;
            for (var j = 0u; j < SURVIVOR_COUNT; j++) {
                if (e < top_errs[j]) {
                    for (var k = SURVIVOR_COUNT - 1u; k > j; k--) { top_errs[k] = top_errs[k-1]; top_shapes[k] = top_shapes[k-1]; }
                    top_errs[j] = e; top_shapes[j] = s;
                    break;
                }
            }
        }
        workgroupBarrier();
    }

    // 🧬 爬山
    var global_best_shape: GpuShape;
    var global_best_err = 99999999.0;

    for (var survivor = 0u; survivor < SURVIVOR_COUNT; survivor++) {
        var local_best_shape = top_shapes[survivor];
        var local_best_err = top_errs[survivor];

        if (tid == 0u) { wg_age = 0u; }
        workgroupBarrier();

        var iters = 0u;
        while (iters < MAX_MUTATIONS_PER_SURVIVOR) {
            // wg_age 只由线程 0 更新；显式进行 uniform load，保证所有线程
            // 一起退出循环，不会出现部分线程越过循环内 barrier 的情况。
            let uniform_age = workgroupUniformLoad(&wg_age);
            if (uniform_age >= MUTATION_STAGNATION_LIMIT) { break; }

            if (tid == 0u) { current_shape = mutate_shape(local_best_shape); }
            workgroupBarrier();
            eval_current_shape(img_offset, tid);

            if (tid == 0u) {
                if (out_err < local_best_err) {
                    local_best_err = out_err;
                    local_best_shape = current_shape;
                    local_best_shape.r = u32(shared_opt_r); local_best_shape.g = u32(shared_opt_g); local_best_shape.b = u32(shared_opt_b);
                    local_best_shape.delta_err = out_err;
                    wg_age = 0u;
                } else {
                    wg_age += 1u;
                }
            }
            workgroupBarrier();
            iters += 1u;
        }

        if (tid == 0u) {
            if (local_best_err < global_best_err) {
                global_best_err = local_best_err;
                global_best_shape = local_best_shape;
            }
        }
        workgroupBarrier();
    }

    // 🎨 判定结果并记录
    if (tid == 0u) {
        let hist_idx = img_idx * MAX_ITERATIONS + config.step;
        var step_delta = 0.0;
        if (global_best_err < 0.0) {
            wg_apply_flag = 1u;
            current_shape = global_best_shape;
            history[hist_idx] = global_best_shape;
            step_delta = global_best_err;
        } else {
            var dummy: GpuShape;
            dummy.delta_err = 9999.0;
            history[hist_idx] = dummy;
        }

        // GPU 端维护与旧 CPU VecDeque 相同的 100 步环形误差历史。
        states[img_idx].last_delta_err = step_delta;
        states[img_idx].current_ssd += step_delta;
        let current_err = states[img_idx].current_ssd / states[img_idx].initial_ssd;
        if (HISTORY_WINDOW > 0u) {
            let history_slot = config.step % HISTORY_WINDOW;
            let error_idx = img_idx * HISTORY_WINDOW + history_slot;

            if (config.step >= HISTORY_WINDOW) {
                let old_err = error_history[error_idx];
                if (old_err - current_err <= EARLY_STOP_THRESHOLD) {
                    states[img_idx].is_done = 1u;
                }
            }
            error_history[error_idx] = current_err;
        }
    }
    workgroupBarrier();

    // ==========================================
    // SDF 平滑上色涂抹
    // ==========================================
    let apply_flag = workgroupUniformLoad(&wg_apply_flag);
    if (apply_flag == 1u) {
        prepare_current_shape(tid);

        var idx = tid;
        while (idx < shared_box_w * shared_box_h) {
            let x = u32(shared_min_x) + (idx % shared_box_w);
            let y = u32(shared_min_y) + (idx / shared_box_w);
            let px_idx = img_offset + y * CANVAS_WIDTH + x;
            let dx = f32(x) - current_shape.cx;
            let dy = f32(y) - current_shape.cy;

            let coverage = get_coverage(
                current_shape.shape_type,
                current_shape.hw,
                current_shape.hh,
                dx,
                dy,
                shared_cos,
                shared_sin,
                shared_inv_hw2,
                shared_inv_hh2,
            );

            if (coverage > 0.0) {
                let crgb = canvas_tex[px_idx];
                let cr = f32((crgb >> 16u) & 255u); let cg = f32((crgb >> 8u) & 255u); let cb = f32(crgb & 255u);

                let eff_alpha = SHAPE_ALPHA * coverage;
                let nr = u32(f32(current_shape.r) * eff_alpha + cr * (1.0 - eff_alpha));
                let ng = u32(f32(current_shape.g) * eff_alpha + cg * (1.0 - eff_alpha));
                let nb = u32(f32(current_shape.b) * eff_alpha + cb * (1.0 - eff_alpha));

                canvas_tex[px_idx] = (nr << 16u) | (ng << 8u) | nb;
            }
            idx += WG_SIZE;
        }
    }
}
