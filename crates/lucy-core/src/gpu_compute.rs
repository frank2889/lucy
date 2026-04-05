//! # gpu_compute — Lucy's Veld op de GPU
//!
//! Alle 2176+ veldpunten worden PARALLEL geüpdatet op de GPU.
//! CPU doet: coordinatie, I/O, locking.
//! GPU doet: phi update, weights, tension, emoties — alles tegelijk.
//!
//! Speedup: 100-1000× t.o.v. single-threaded CPU.

use lucy_state::*;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// GPU hyperparameters — moet matchen met de WGSL shader struct
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

/// Lucy's GPU compute engine
#[allow(dead_code)]
pub struct GpuCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Pipelines
    phi_pipeline: wgpu::ComputePipeline,
    memory_pipeline: wgpu::ComputePipeline,
    tension_pipeline: wgpu::ComputePipeline,
    weights_pipeline: wgpu::ComputePipeline,

    // Buffers — group 0: punt data
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

    // Buffers — group 1: koppelingen + clamps
    coupling_offsets_buf: wgpu::Buffer,
    coupling_targets_buf: wgpu::Buffer,
    coupling_weights_buf: wgpu::Buffer,
    coupling_momentum_buf: wgpu::Buffer,
    clamp_mask_buf: wgpu::Buffer,
    clamp_values_buf: wgpu::Buffer,

    // Uniform buffer — group 2: hyperparameters
    params_buf: wgpu::Buffer,

    // Bind groups
    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    bind_group_2: wgpu::BindGroup,

    // Dimensies
    n_points: u32,
    _total_couplings: u32,
}

impl GpuCompute {
    /// Initialiseer de GPU met de huidige veldstaat
    pub fn new(field: &Field) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Geen GPU gevonden");

        tracing::info!(
            name = adapter.get_info().name,
            backend = ?adapter.get_info().backend,
            "GPU gevonden"
        );

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

        let n_points = field.points.len() as u32;

        // === Flatten field data naar GPU-formaat ===
        let (phi_data, pred_data, mem_data, tension_data, tension_ema_data,
             energy_data, eta_data, curiosity_data, satisfaction_data, frustration_data) =
            Self::flatten_points(&field.points);

        let (offsets, targets, weights, momentum) = Self::flatten_couplings(&field.points);
        let total_couplings = targets.len() as u32;

        // Clamps (initieel leeg)
        let clamp_mask = vec![0u32; n_points as usize];
        let clamp_values = vec![0.0f32; n_points as usize * DIM];

        // === Maak GPU buffers ===
        let usage_rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let usage_r = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let phi_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("phi"), contents: bytemuck::cast_slice(&phi_data), usage: usage_rw });
        let prediction_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prediction"), contents: bytemuck::cast_slice(&pred_data), usage: usage_rw });
        let memory_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("memory"), contents: bytemuck::cast_slice(&mem_data), usage: usage_rw });
        let tension_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tension"), contents: bytemuck::cast_slice(&tension_data), usage: usage_rw });
        let tension_ema_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tension_ema"), contents: bytemuck::cast_slice(&tension_ema_data), usage: usage_rw });
        let energy_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("energy"), contents: bytemuck::cast_slice(&energy_data), usage: usage_rw });
        let eta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("eta"), contents: bytemuck::cast_slice(&eta_data), usage: usage_rw });
        let curiosity_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("curiosity"), contents: bytemuck::cast_slice(&curiosity_data), usage: usage_rw });
        let satisfaction_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("satisfaction"), contents: bytemuck::cast_slice(&satisfaction_data), usage: usage_rw });
        let frustration_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frustration"), contents: bytemuck::cast_slice(&frustration_data), usage: usage_rw });

        let coupling_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offsets"), contents: bytemuck::cast_slice(&offsets), usage: usage_r });
        let coupling_targets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("targets"), contents: bytemuck::cast_slice(&targets), usage: usage_r });
        let coupling_weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("weights"), contents: bytemuck::cast_slice(&weights), usage: usage_rw });
        let coupling_momentum_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("momentum"), contents: bytemuck::cast_slice(&momentum), usage: usage_rw });
        let clamp_mask_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("clamp_mask"), contents: bytemuck::cast_slice(&clamp_mask), usage: usage_r });
        let clamp_values_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("clamp_values"), contents: bytemuck::cast_slice(&clamp_values), usage: usage_r });

        let gpu_params = GpuParams {
            lambda: field.lambda,
            beta: field.beta,
            alpha: field.alpha,
            rho: field.rho,
            mu: field.mu,
            kappa: field.kappa,
            omega: field.omega,
            ema_decay: field.ema_decay,
            growth_threshold: field.growth_threshold,
            n_points,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === Shader laden ===
        let shader_source = include_str!("gpu_compute.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lucy compute shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // === Bind group layouts ===
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

        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("group 1"),
            entries: &[
                Self::storage_entry(0, true),  // offsets (read only)
                Self::storage_entry(1, true),  // targets (read only)
                Self::storage_entry(2, false), // weights
                Self::storage_entry(3, false), // momentum
                Self::storage_entry(4, true),  // clamp_mask (read only)
                Self::storage_entry(5, true),  // clamp_values (read only)
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

        // === Pipelines ===
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

        // === Bind groups ===
        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group 0"),
            layout: &bgl_0,
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
            label: Some("group 1"),
            layout: &bgl_1,
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
            label: Some("group 2"),
            layout: &bgl_2,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            ],
        });

        tracing::info!(n_points, total_couplings, "GPU compute initialized");

        Self {
            device, queue,
            phi_pipeline, memory_pipeline, tension_pipeline, weights_pipeline,
            phi_buf, prediction_buf, memory_buf, tension_buf, tension_ema_buf,
            energy_buf, eta_buf, curiosity_buf, satisfaction_buf, frustration_buf,
            coupling_offsets_buf, coupling_targets_buf, coupling_weights_buf,
            coupling_momentum_buf, clamp_mask_buf, clamp_values_buf,
            params_buf,
            bind_group_0, bind_group_1, bind_group_2,
            n_points, _total_couplings: total_couplings,
        }
    }

    /// Run één GPU tick: phi update + memory + tension/emoties
    pub fn tick(&self) {
        let workgroups = (self.n_points + 63) / 64;

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Pass 1: update phi
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.phi_pipeline);
            pass.set_bind_group(0, &self.bind_group_0, &[]);
            pass.set_bind_group(1, &self.bind_group_1, &[]);
            pass.set_bind_group(2, &self.bind_group_2, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: update memory
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.memory_pipeline);
            pass.set_bind_group(0, &self.bind_group_0, &[]);
            pass.set_bind_group(1, &self.bind_group_1, &[]);
            pass.set_bind_group(2, &self.bind_group_2, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 3: compute tension + emoties
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.tension_pipeline);
            pass.set_bind_group(0, &self.bind_group_0, &[]);
            pass.set_bind_group(1, &self.bind_group_1, &[]);
            pass.set_bind_group(2, &self.bind_group_2, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Run weight update op GPU
    pub fn update_weights(&self) {
        let workgroups = (self.n_points + 63) / 64;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.weights_pipeline);
            pass.set_bind_group(0, &self.bind_group_0, &[]);
            pass.set_bind_group(1, &self.bind_group_1, &[]);
            pass.set_bind_group(2, &self.bind_group_2, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Upload clamps naar GPU
    pub fn set_clamps(&self, field: &Field) {
        let mut mask = vec![0u32; self.n_points as usize];
        let mut values = vec![0.0f32; self.n_points as usize * DIM];

        for (&id, clamp) in &field.clamps {
            let idx = id as usize;
            if idx < self.n_points as usize {
                mask[idx] = 1;
                values[idx * DIM..(idx + 1) * DIM].copy_from_slice(clamp);
            }
        }

        self.queue.write_buffer(&self.clamp_mask_buf, 0, bytemuck::cast_slice(&mask));
        self.queue.write_buffer(&self.clamp_values_buf, 0, bytemuck::cast_slice(&values));
    }

    /// Download veldstaat van GPU terug naar CPU Field
    pub fn download_to_field(&self, field: &mut Field) {
        let n = self.n_points as usize;

        let phi = self.read_buffer_f32(&self.phi_buf, n * DIM);
        let tension = self.read_buffer_f32(&self.tension_buf, n);
        let tension_ema = self.read_buffer_f32(&self.tension_ema_buf, n);
        let curiosity = self.read_buffer_f32(&self.curiosity_buf, n);
        let satisfaction = self.read_buffer_f32(&self.satisfaction_buf, n);
        let frustration = self.read_buffer_f32(&self.frustration_buf, n);

        for i in 0..n.min(field.points.len()) {
            field.points[i].phi.copy_from_slice(&phi[i * DIM..(i + 1) * DIM]);
            field.points[i].tension = tension[i];
            field.points[i].tension_ema = tension_ema[i];
            field.points[i].curiosity = curiosity[i];
            field.points[i].satisfaction = satisfaction[i];
            field.points[i].frustration = frustration[i];
        }
    }

    /// Sync hyperparameters naar GPU
    pub fn sync_params(&self, field: &Field) {
        let params = GpuParams {
            lambda: field.lambda,
            beta: field.beta,
            alpha: field.alpha,
            rho: field.rho,
            mu: field.mu,
            kappa: field.kappa,
            omega: field.omega,
            ema_decay: field.ema_decay,
            growth_threshold: field.growth_threshold,
            n_points: self.n_points,
            _pad1: 0,
            _pad2: 0,
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    // === Helper functions ===

    fn flatten_points(points: &[FieldPoint]) -> (
        Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>,
        Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>,
    ) {
        let n = points.len();
        let mut phi = Vec::with_capacity(n * DIM);
        let mut pred = Vec::with_capacity(n * DIM);
        let mut mem = Vec::with_capacity(n * DIM);
        let mut tension = Vec::with_capacity(n);
        let mut tension_ema = Vec::with_capacity(n);
        let mut energy = Vec::with_capacity(n);
        let mut eta = Vec::with_capacity(n);
        let mut curiosity = Vec::with_capacity(n);
        let mut satisfaction = Vec::with_capacity(n);
        let mut frustration = Vec::with_capacity(n);

        for p in points {
            phi.extend_from_slice(&p.phi);
            pred.extend_from_slice(&p.prediction);
            mem.extend_from_slice(&p.memory);
            tension.push(p.tension);
            tension_ema.push(p.tension_ema);
            energy.push(p.energy);
            eta.push(p.eta);
            curiosity.push(p.curiosity);
            satisfaction.push(p.satisfaction);
            frustration.push(p.frustration);
        }

        (phi, pred, mem, tension, tension_ema, energy, eta, curiosity, satisfaction, frustration)
    }

    fn flatten_couplings(points: &[FieldPoint]) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
        let mut offsets = Vec::with_capacity(points.len() + 1);
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        let mut momentum = Vec::new();

        let mut offset = 0u32;
        for p in points {
            offsets.push(offset);
            for c in &p.couplings {
                targets.push(c.target);
                weights.push(c.weight);
                momentum.push(c.momentum);
            }
            offset += p.couplings.len() as u32;
        }
        offsets.push(offset);

        // wgpu vereist minstens 1 element in storage buffers
        if targets.is_empty() {
            targets.push(0);
            weights.push(0.0);
            momentum.push(0.0);
        }

        (offsets, targets, weights, momentum)
    }

    fn read_buffer_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let size = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        result
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}
