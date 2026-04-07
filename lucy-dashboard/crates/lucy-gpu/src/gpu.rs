//! GPU compute engine — standalone versie voor remote service

use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use serde::{Serialize, Deserialize};

/// Volledige veldstaat voor GPU upload/download — gedeeld met gpu_client
#[derive(Serialize, Deserialize)]
pub struct FieldSnapshot {
    pub phi: Vec<f32>,
    pub prediction: Vec<f32>,
    pub memory: Vec<f32>,
    pub tension: Vec<f32>,
    pub tension_ema: Vec<f32>,
    pub energy: Vec<f32>,
    pub eta: Vec<f32>,
    pub curiosity: Vec<f32>,
    pub satisfaction: Vec<f32>,
    pub frustration: Vec<f32>,
    pub coupling_offsets: Vec<u32>,
    pub coupling_targets: Vec<u32>,
    pub coupling_weights: Vec<f32>,
    pub coupling_momentum: Vec<f32>,
    pub n_points: u32,
    pub lambda: f32,
    pub beta: f32,
    pub alpha: f32,
    pub rho: f32,
    pub mu: f32,
    pub kappa: f32,
    pub omega: f32,
    pub ema_decay: f32,
    pub growth_threshold: f32,
}

const DIM: usize = 64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    lambda: f32,
    beta: f32,
    alpha: f32,
    rho: f32,
    mu: f32,
    kappa: f32,
    omega: f32,
    ema_decay: f32,
    growth_threshold: f32,
    n_points: u32,
    _pad1: u32,
    _pad2: u32,
}

#[allow(dead_code)]
pub struct GpuCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,

    phi_pipeline: wgpu::ComputePipeline,
    memory_pipeline: wgpu::ComputePipeline,
    tension_pipeline: wgpu::ComputePipeline,
    weights_pipeline: wgpu::ComputePipeline,

    // Group 0: punt data
    phi_buf: wgpu::Buffer,
    prediction_buf: wgpu::Buffer,
    memory_buf: wgpu::Buffer,
    tension_buf: wgpu::Buffer,
    tension_ema_buf: wgpu::Buffer,
    energy_buf: wgpu::Buffer,
    eta_buf: wgpu::Buffer,
    curiosity_buf: wgpu::Buffer,
    satisfaction_buf: wgpu::Buffer,
    frustration_buf: wgpu::Buffer,

    // Group 1: koppelingen + clamps
    coupling_offsets_buf: wgpu::Buffer,
    coupling_targets_buf: wgpu::Buffer,
    coupling_weights_buf: wgpu::Buffer,
    coupling_momentum_buf: wgpu::Buffer,
    clamp_mask_buf: wgpu::Buffer,
    clamp_values_buf: wgpu::Buffer,

    // Group 2: params
    params_buf: wgpu::Buffer,

    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    bind_group_2: wgpu::BindGroup,

    n_points: u32,
    total_couplings: u32,
}

impl GpuCompute {
    pub fn from_snapshot(snap: &FieldSnapshot) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Geen GPU gevonden");

        let adapter_name = adapter.get_info().name.clone();
        tracing::info!(name = %adapter_name, backend = ?adapter.get_info().backend, "GPU gevonden");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Lucy GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("GPU device creation failed");

        let n_points = snap.n_points;
        let usage_rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let usage_r = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let mk = |label: &str, data: &[f32], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data), usage,
            })
        };
        let mk_u32 = |label: &str, data: &[u32], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::cast_slice(data), usage,
            })
        };

        let phi_buf = mk("phi", &snap.phi, usage_rw);
        let prediction_buf = mk("prediction", &snap.prediction, usage_rw);
        let memory_buf = mk("memory", &snap.memory, usage_rw);
        let tension_buf = mk("tension", &snap.tension, usage_rw);
        let tension_ema_buf = mk("tension_ema", &snap.tension_ema, usage_rw);
        let energy_buf = mk("energy", &snap.energy, usage_rw);
        let eta_buf = mk("eta", &snap.eta, usage_rw);
        let curiosity_buf = mk("curiosity", &snap.curiosity, usage_rw);
        let satisfaction_buf = mk("satisfaction", &snap.satisfaction, usage_rw);
        let frustration_buf = mk("frustration", &snap.frustration, usage_rw);

        let total_couplings = snap.coupling_targets.len() as u32;

        // wgpu vereist minstens 1 element in storage buffers
        let safe_targets = if snap.coupling_targets.is_empty() { vec![0u32] } else { snap.coupling_targets.clone() };
        let safe_weights = if snap.coupling_weights.is_empty() { vec![0.0f32] } else { snap.coupling_weights.clone() };
        let safe_momentum = if snap.coupling_momentum.is_empty() { vec![0.0f32] } else { snap.coupling_momentum.clone() };

        let coupling_offsets_buf = mk_u32("offsets", &snap.coupling_offsets, usage_r);
        let coupling_targets_buf = mk_u32("targets", &safe_targets, usage_r);
        let coupling_weights_buf = mk("weights", &safe_weights, usage_rw);
        let coupling_momentum_buf = mk("momentum", &safe_momentum, usage_rw);

        let clamp_mask = vec![0u32; n_points as usize];
        let clamp_values = vec![0.0f32; n_points as usize * DIM];
        let clamp_mask_buf = mk_u32("clamp_mask", &clamp_mask, usage_r);
        let clamp_values_buf = mk("clamp_values", &clamp_values, usage_r);

        let gpu_params = GpuParams {
            lambda: snap.lambda, beta: snap.beta, alpha: snap.alpha,
            rho: snap.rho, mu: snap.mu, kappa: snap.kappa, omega: snap.omega,
            ema_decay: snap.ema_decay, growth_threshold: snap.growth_threshold,
            n_points, _pad1: 0, _pad2: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Shader
        let shader_source = include_str!("gpu_compute.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lucy compute shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Bind group layouts
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("group 0"),
            entries: &(0..10).map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }).collect::<Vec<_>>(),
        });

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("group 1"),
            entries: &[
                storage_entry(0, true), storage_entry(1, true),
                storage_entry(2, false), storage_entry(3, false),
                storage_entry(4, true), storage_entry(5, true),
            ],
        });

        let bgl_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("group 2"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lucy pipeline layout"),
            bind_group_layouts: &[&bgl_0, &bgl_1, &bgl_2],
            push_constant_ranges: &[],
        });

        let create_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let phi_pipeline = create_pipeline("update_phi");
        let memory_pipeline = create_pipeline("update_memory");
        let tension_pipeline = create_pipeline("compute_tension_kernel");
        let weights_pipeline = create_pipeline("update_weights_kernel");

        // Bind groups
        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group 0"), layout: &bgl_0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: phi_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: prediction_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: memory_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tension_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tension_ema_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: energy_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: eta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: curiosity_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: satisfaction_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: frustration_buf.as_entire_binding() },
            ],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group 1"), layout: &bgl_1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: coupling_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: coupling_targets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: coupling_weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: coupling_momentum_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: clamp_mask_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: clamp_values_buf.as_entire_binding() },
            ],
        });

        let bind_group_2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group 2"), layout: &bgl_2,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            ],
        });

        tracing::info!(n_points, "GPU compute initialized from snapshot");

        Self {
            device, queue, adapter_name,
            phi_pipeline, memory_pipeline, tension_pipeline, weights_pipeline,
            phi_buf, prediction_buf, memory_buf, tension_buf, tension_ema_buf,
            energy_buf, eta_buf, curiosity_buf, satisfaction_buf, frustration_buf,
            coupling_offsets_buf, coupling_targets_buf, coupling_weights_buf,
            coupling_momentum_buf, clamp_mask_buf, clamp_values_buf,
            params_buf,
            bind_group_0, bind_group_1, bind_group_2,
            n_points, total_couplings,
        }
    }

    pub fn gpu_name(&self) -> String { self.adapter_name.clone() }
    pub fn n_points(&self) -> u32 { self.n_points }

    pub fn tick(&self) {
        let wg = (self.n_points + 63) / 64;
        let mut encoder = self.device.create_command_encoder(&Default::default());

        for pipeline in [&self.phi_pipeline, &self.memory_pipeline, &self.tension_pipeline] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group_0, &[]);
            pass.set_bind_group(1, &self.bind_group_1, &[]);
            pass.set_bind_group(2, &self.bind_group_2, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn update_weights(&self) {
        let wg = (self.n_points + 63) / 64;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.weights_pipeline);
            pass.set_bind_group(0, &self.bind_group_0, &[]);
            pass.set_bind_group(1, &self.bind_group_1, &[]);
            pass.set_bind_group(2, &self.bind_group_2, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn set_clamps_raw(&self, mask: &[u32], values: &[f32]) {
        self.queue.write_buffer(&self.clamp_mask_buf, 0, bytemuck::cast_slice(mask));
        self.queue.write_buffer(&self.clamp_values_buf, 0, bytemuck::cast_slice(values));
    }

    pub fn sync_params_raw(
        &self, lambda: f32, beta: f32, alpha: f32, rho: f32, mu: f32,
        kappa: f32, omega: f32, ema_decay: f32, growth_threshold: f32,
    ) {
        let p = GpuParams {
            lambda, beta, alpha, rho, mu, kappa, omega, ema_decay, growth_threshold,
            n_points: self.n_points, _pad1: 0, _pad2: 0,
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
    }

    pub fn download_snapshot(&self) -> FieldSnapshot {
        let n = self.n_points as usize;
        let phi = self.read_f32(&self.phi_buf, n * DIM);
        let prediction = self.read_f32(&self.prediction_buf, n * DIM);
        let memory = self.read_f32(&self.memory_buf, n * DIM);
        let tension = self.read_f32(&self.tension_buf, n);
        let tension_ema = self.read_f32(&self.tension_ema_buf, n);
        let energy = self.read_f32(&self.energy_buf, n);
        let eta = self.read_f32(&self.eta_buf, n);
        let curiosity = self.read_f32(&self.curiosity_buf, n);
        let satisfaction = self.read_f32(&self.satisfaction_buf, n);
        let frustration = self.read_f32(&self.frustration_buf, n);
        let tc = self.total_couplings.max(1) as usize;
        let coupling_weights = self.read_f32(&self.coupling_weights_buf, tc);
        let coupling_momentum = self.read_f32(&self.coupling_momentum_buf, tc);

        // We geven een partial snapshot terug — offsets/targets veranderen niet
        FieldSnapshot {
            phi, prediction, memory, tension, tension_ema,
            energy, eta, curiosity, satisfaction, frustration,
            coupling_offsets: vec![], // onveranderd, hoeft niet terug
            coupling_targets: vec![], // onveranderd
            coupling_weights,
            coupling_momentum,
            n_points: self.n_points,
            lambda: 0.0, beta: 0.0, alpha: 0.0, rho: 0.0, mu: 0.0,
            kappa: 0.0, omega: 0.0, ema_decay: 0.0, growth_threshold: 0.0,
        }
    }

    fn read_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        if count == 0 { return vec![]; }
        let size = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"), size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}
