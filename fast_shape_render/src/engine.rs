// src/engine.rs
use std::borrow::Cow;
use std::sync::{Arc, Mutex};

const READBACK_ATTEMPTS: usize = 3;

#[cfg(target_os = "windows")]
fn platform_backend() -> (wgpu::Backends, &'static str) {
    (wgpu::Backends::DX12, "DX12")
}

#[cfg(target_os = "linux")]
fn platform_backend() -> (wgpu::Backends, &'static str) {
    (wgpu::Backends::VULKAN, "Vulkan")
}

#[cfg(target_os = "macos")]
fn platform_backend() -> (wgpu::Backends, &'static str) {
    (wgpu::Backends::METAL, "Metal")
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn platform_backend() -> (wgpu::Backends, &'static str) {
    (wgpu::Backends::PRIMARY, "平台主后端")
}

pub struct GpuEngine {
    pub gpu_index: usize,
    pub name: String,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    device_lost_message: Arc<Mutex<Option<String>>>,
}

impl GpuEngine {
    /// 通过当前平台的原生后端枚举并初始化全部硬件 GPU。
    ///
    /// 只枚举单一后端，避免同一块物理显卡通过多个图形 API 出现而被重复使用。
    pub async fn discover_all(shader_source: &str) -> Result<Vec<Self>, String> {
        let (backends, backend_name) = platform_backend();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapters: Vec<_> = instance
            .enumerate_adapters(backends)
            .into_iter()
            .filter(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
            .collect();

        if adapters.is_empty() {
            return Err(format!(
                "❌ 没有检测到可用的 {backend_name} 硬件 GPU（当前平台: {}）",
                std::env::consts::OS
            ));
        }

        println!(
            "🔍 通过 {backend_name} 检测到 {} 张硬件 GPU",
            adapters.len()
        );
        let mut engines = Vec::with_capacity(adapters.len());

        for (gpu_index, adapter) in adapters.into_iter().enumerate() {
            let info = adapter.get_info();
            println!(
                "  🎮 GPU {}: {} | {:?} | {:?}",
                gpu_index, info.name, info.device_type, info.backend
            );

            match Self::new(gpu_index, adapter, shader_source).await {
                Ok(engine) => engines.push(engine),
                Err(err) => eprintln!(
                    "  ⚠️ GPU {} ({}) 初始化失败，将继续使用其余显卡: {}",
                    gpu_index, info.name, err
                ),
            }
        }

        if engines.is_empty() {
            Err("❌ 检测到了 GPU，但没有任何 GPU 能成功创建设备".to_string())
        } else {
            println!("✅ 已成功初始化 {} 张 GPU", engines.len());
            Ok(engines)
        }
    }

    pub async fn new(
        gpu_index: usize,
        adapter: wgpu::Adapter,
        shader_source: &str,
    ) -> Result<Self, String> {
        let info = adapter.get_info();

        // 🚀 核心优化：解除显卡的默认显存块限制，允许单次提交高达数 GB 的历史记录阵列
        let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter.limits().max_buffer_size;
        limits.max_storage_buffers_per_shader_stage =
            adapter.limits().max_storage_buffers_per_shader_stage;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Mega-Kernel Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await
            .map_err(|err| format!("无法创建逻辑设备: {err}"))?;

        let device_lost_message = Arc::new(Mutex::new(None));
        let callback_message = Arc::clone(&device_lost_message);
        let callback_name = info.name.clone();
        device.set_device_lost_callback(move |reason, message| {
            if message.trim().eq_ignore_ascii_case("Device dropped.") {
                return;
            }
            match reason {
                wgpu::DeviceLostReason::Destroyed
                | wgpu::DeviceLostReason::Dropped
                | wgpu::DeviceLostReason::ReplacedCallback => return,
                wgpu::DeviceLostReason::Unknown => {}
            }

            let diagnostic = format!(
                "GPU {} 设备丢失，原因: {:?}，驱动信息: {}",
                callback_name, reason, message
            );
            eprintln!("  ❌ {diagnostic}");
            if let Ok(mut slot) = callback_message.lock() {
                *slot = Some(diagnostic);
            }
        });

        // 每张 GPU 在初始化时编译一次由 main.rs 常量生成的完整 WGSL。
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mega-Kernel Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        // 🏗️ 建立计算资源布局：严格映射 WGSL 中的 6 个 binding
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mega-Pool Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // binding(0) - Config (尺寸, 随机种子, 步数)
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // binding(1) - Target Tex (只读目标图)
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // binding(2) - Canvas Tex (可读写画布)
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // binding(3) - States (早停状态与误差回传)
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // binding(4) - History (历史轨迹)
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // binding(5) - GPU 端早停误差环形历史
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main", // 对应 shader.wgsl 里的 @compute fn main
        });

        Ok(Self {
            gpu_index,
            name: info.name,
            device,
            queue,
            pipeline,
            bind_group_layout,
            device_lost_message,
        })
    }

    /// 执行一笔：让显卡并行处理 num_images 张图，内部跑完海选和 100 步变异
    // src/engine.rs (部分替换)

    pub fn dispatch_step(&self, bind_group: &wgpu::BindGroup, num_images: u32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);

            // 🚀 核心突破：将 1 维过长的阵列折叠成 X=256 的 2D 阵列
            let group_x = 256;
            let group_y = (num_images + group_x - 1) / group_x; // 向上取整

            cpass.dispatch_workgroups(group_x, group_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn device_lost_error(&self) -> Option<String> {
        self.device_lost_message
            .lock()
            .ok()
            .and_then(|message| message.clone())
    }

    /// 等待此前提交的命令完成，为长批次提供背压并及时暴露设备丢失。
    pub fn wait_for_gpu(&self, context: &str) -> Result<(), String> {
        self.device.poll(wgpu::Maintain::Wait);
        if let Some(message) = self.device_lost_error() {
            Err(format!(
                "GPU {} ({}) 在 {} 时不可用: {}",
                self.gpu_index, self.name, context, message
            ))
        } else {
            Ok(())
        }
    }

    /// 将显存中的一段数据分块拉回 CPU。
    ///
    /// 每次尝试都创建新的 CPU 可见 staging buffer。这样即使 GPU 驱动偶发
    /// map_async 失败，也不会在已失败的映射资源上继续工作。
    pub async fn read_buffer_range<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        source_offset: wgpu::BufferAddress,
        size: wgpu::BufferAddress,
        label: &str,
    ) -> Result<Vec<T>, String> {
        let item_size = std::mem::size_of::<T>() as u64;
        if item_size == 0 || size == 0 || size % item_size != 0 {
            return Err(format!(
                "GPU {} ({}) 的 {} 回读大小无效: offset={}, size={}, item_size={}",
                self.gpu_index, self.name, label, source_offset, size, item_size
            ));
        }

        let mut last_error = None;
        for attempt in 1..=READBACK_ATTEMPTS {
            if let Some(message) = self.device_lost_error() {
                return Err(format!(
                    "GPU {} ({}) 无法回读 {}: {}",
                    self.gpu_index, self.name, label, message
                ));
            }

            let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Readback Copy Encoder"),
                });
            encoder.copy_buffer_to_buffer(buffer, source_offset, &staging_buf, 0, size);
            self.queue.submit(Some(encoder.finish()));

            let buffer_slice = staging_buf.slice(..);
            let (sender, receiver) = flume::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.device.poll(wgpu::Maintain::Wait);

            let map_result = receiver.recv_async().await.map_err(|err| {
                format!(
                    "GPU {} ({}) 的 {} 映射回调通道断开: {}",
                    self.gpu_index, self.name, label, err
                )
            })?;

            match map_result {
                Ok(()) => {
                    let data = buffer_slice.get_mapped_range();
                    let result = bytemuck::cast_slice(&data).to_vec();
                    drop(data);
                    staging_buf.unmap();
                    return Ok(result);
                }
                Err(err) => {
                    let message = format!(
                        "GPU {} ({}) 的 {} 映射失败（第 {}/{} 次，offset={}，size={}）: {}",
                        self.gpu_index,
                        self.name,
                        label,
                        attempt,
                        READBACK_ATTEMPTS,
                        source_offset,
                        size,
                        err
                    );
                    eprintln!("  ⚠️ {message}");
                    last_error = Some(message);
                    if self.device_lost_error().is_some() {
                        break;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            format!(
                "GPU {} ({}) 的 {} 回读失败",
                self.gpu_index, self.name, label
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手动烟雾测试：只枚举并初始化 GPU/计算管线，不读取或处理数据集。
    #[test]
    #[ignore = "requires local hardware GPU"]
    fn initializes_all_platform_hardware_gpus() {
        let shader_source = crate::build_shader_source(&crate::shader_constants())
            .expect("default shader constants should be valid");
        let engines = pollster::block_on(GpuEngine::discover_all(&shader_source))
            .expect("at least one platform hardware GPU should initialize");
        assert!(!engines.is_empty());
    }
}
