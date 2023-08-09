//! Bitonic sort implementation.
//!
//! This is a WebGPU port of <https://github.com/microsoft/DirectX-Graphics-Samples/blob/5ca41579b6837b3064c8b7333071859425c5c4de/MiniEngine/Core/Shaders/Bitonic32PreSortCS.hlsl>.
//! It is not currently used (anymore) and serves mostly as an example. Maybe it will be useful to
//! someone else.

#![allow(dead_code)]

use std::{iter, sync::Arc};

use wgpu::*;

use crate::{
    dynamic::{DynamicBindGroup, DynamicBuffer},
    Gpu,
};

/// Number of elements presorted per workgroup. Also the final size of all presorted blocks.
const PRESORT_ELEMS: u32 = 2048;
/// Number of elements sorted by one `bitonic_sort` workgroup.
const SORT_ELEMS: u32 = 2048;
/// Elements sorted by a "small j sort" shader workgroup.
const SMALL_SORT_ELEMS: u32 = 2048;
/// Highest value of `j` supported by the "small j sort" shader.
const SMALL_SORT_MAX_J: u32 = 1024;

pub struct Sorter {
    gpu: Arc<Gpu>,
    sort_bind_group: DynamicBindGroup,
    sort_pipeline: ComputePipeline,
    small_sort_pipeline: ComputePipeline,
    presort_pipeline: ComputePipeline,
}

impl Sorter {
    pub fn new(gpu: Arc<Gpu>) -> Self {
        let sort_shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sort_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sort.wgsl").into()),
        });

        let sort_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("sort_bind_group_layout"),
                    entries: &[
                        // `elements`
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let sort_pipeline_layout = gpu
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("sort_pipeline_layout"),
                bind_group_layouts: &[&sort_bind_group_layout],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..12,
                }],
            });
        let sort_pipeline = gpu
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("sort_pipeline"),
                layout: Some(&sort_pipeline_layout),
                module: &sort_shader,
                entry_point: "bitonic_sort",
            });
        let small_sort_pipeline = gpu
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("small_sort_pipeline"),
                layout: Some(&sort_pipeline_layout),
                module: &sort_shader,
                entry_point: "sort_small_j",
            });
        let presort_pipeline = gpu
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("presort_pipeline"),
                layout: Some(&sort_pipeline_layout),
                module: &sort_shader,
                entry_point: "presort",
            });

        Self {
            sort_bind_group: DynamicBindGroup::new(
                gpu.clone(),
                sort_bind_group_layout.into(),
                "sort_bind_group",
            ),
            gpu,
            sort_pipeline,
            small_sort_pipeline,
            presort_pipeline,
        }
    }

    pub fn sort(&mut self, data: &mut DynamicBuffer, num_elems: u32) {
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        self.sort_with_encoder(data, num_elems, &mut enc);
        self.gpu.queue.submit([enc.finish()]);
    }

    fn sort_with_encoder(
        &mut self,
        data: &mut DynamicBuffer,
        num_elems: u32,
        enc: &mut CommandEncoder,
    ) {
        if num_elems == 0 {
            return;
        }

        // Bitonic sort only works when the number of elements is a power of two.
        // We can pad any input sequence to a power of two by appending `u32::MAX`.
        // We perform the write using the `pad` shader entry point, since that doesn't suffer
        // from alignment requirements.
        // If the buffer doesn't have enough space for the padding elements, we have to
        // reallocate it.
        let num_elems_padded = num_elems.next_power_of_two();
        //let padding_elems = num_elems_padded - num_elems;
        data.reserve(u64::from(num_elems_padded) * 4);

        let bind_group = self.sort_bind_group.bind_group(&[data.as_resource()]);

        let mut compute = enc.begin_compute_pass(&ComputePassDescriptor::default());
        compute.set_bind_group(0, bind_group, &[]);

        compute.set_pipeline(&self.presort_pipeline);
        compute.set_push_constants(0, bytemuck::bytes_of(&[num_elems, 0, 0]));
        let workgroups = (num_elems_padded + PRESORT_ELEMS - 1) / PRESORT_ELEMS;
        log::trace!("presort: num_elems={num_elems}, num_elems_padded={num_elems_padded}, workgroups={workgroups}");
        compute.dispatch_workgroups(workgroups, 1, 1);

        if workgroups == 1 {
            // For few enough elements, we're already done.
            return;
        }

        // We process some elements per workgroup and need to process N per iteration.
        let num_workgroups_per_iteration = (num_elems_padded + SORT_ELEMS - 1) / SORT_ELEMS;
        assert!(num_workgroups_per_iteration > 0);
        let num_workgroups_per_small_iteration =
            (num_elems_padded + SMALL_SORT_ELEMS - 1) / SMALL_SORT_ELEMS;
        assert!(num_workgroups_per_small_iteration > 0);

        let mut normal_passes = 0;
        let mut small_passes = 0;
        for k in iter::successors(Some(PRESORT_ELEMS * 2), |k| Some(k * 2))
            .take_while(|&k| k <= num_elems_padded)
        {
            for j in iter::successors(Some(k >> 1), |k| Some(k >> 1)).take_while(|&j| j > 0) {
                if j == SMALL_SORT_MAX_J {
                    log::trace!(
                        "scheduling small sorting pass num_elems_padded={num_elems_padded} k={k} j={j} workgroups={}",
                        num_workgroups_per_small_iteration,
                    );
                    compute.set_pipeline(&self.small_sort_pipeline);
                    compute.set_push_constants(0, bytemuck::bytes_of(&[num_elems_padded, k, j]));
                    compute.dispatch_workgroups(num_workgroups_per_small_iteration, 1, 1);
                    small_passes += 1;
                    // A single pass does all the remaining `j` iterations.
                    break;
                } else {
                    log::trace!(
                        "scheduling sorting pass num_elems_padded={num_elems_padded} k={k} j={j} workgroups={}",
                        num_workgroups_per_iteration,
                    );
                    compute.set_pipeline(&self.sort_pipeline);
                    compute.set_push_constants(0, bytemuck::bytes_of(&[num_elems_padded, k, j]));
                    compute.dispatch_workgroups(num_workgroups_per_iteration, 1, 1);
                    normal_passes += 1;
                }
            }
        }
        log::trace!("total passes: {normal_passes} + {small_passes}");
    }
}

#[cfg(test)]
mod tests {
    use crate::dynamic::DownloadBuffer;

    use super::*;

    fn device_descriptor() -> DeviceDescriptor<'static> {
        DeviceDescriptor {
            features: Features::PUSH_CONSTANTS,
            limits: Limits {
                max_push_constant_size: 16,
                max_compute_workgroup_size_x: 1024,
                max_compute_invocations_per_workgroup: 1024,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    async fn open() -> crate::Result<Gpu> {
        let instance = Instance::new(InstanceDescriptor {
            // The OpenGL backend panics spuriously, so don't enable it.
            backends: Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .ok_or_else(|| crate::Error::from("no supported graphics adapter found"))?;
        let (device, queue) = adapter
            .request_device(&device_descriptor(), None)
            .await
            .map_err(|_| crate::Error::from("no supported graphics device found"))?;

        Gpu::from_wgpu(device.into(), queue.into())
    }

    #[track_caller]
    fn check(unsorted: &[u32]) {
        env_logger::builder()
            .filter_module(env!("CARGO_PKG_NAME"), log::LevelFilter::Trace)
            .parse_default_env()
            .try_init()
            .ok();

        let gpu = Arc::new(pollster::block_on(open()).unwrap());
        let mut sorter = Sorter::new(gpu.clone());
        let mut dbuf = DynamicBuffer::new(
            gpu.clone(),
            "sort_input",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        dbuf.write(unsorted);

        let mut enc = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        sorter.sort_with_encoder(&mut dbuf, unsorted.len().try_into().unwrap(), &mut enc);

        let mut dl = DownloadBuffer::new(gpu.clone());
        dl.download(&dbuf, &mut enc);

        gpu.queue.submit([enc.finish()]);
        dl.map();
        gpu.device.poll(MaintainBase::Wait);

        let view = dl.mapped();
        let view: &[u32] = if view.is_empty() {
            &[]
        } else {
            &bytemuck::cast_slice(&*view)[..unsorted.len()]
        };
        assert_eq!(view.len(), unsorted.len(), "length mismatch");

        let mut expected = unsorted.to_vec();
        expected.sort_unstable();

        assert_eq!(view, &*expected);
    }

    #[test]
    fn noop() {
        check(&[]);
        check(&[0]);
        check(&[0, 1]);
        check(&[0, 1, 2]);
        check(&[0, 1, 2, 3]);
        check(&[42; 1023]);
        check(&[42; 5000]);
    }

    #[test]
    fn reverse() {
        check(&[3, 2, 1, 0]);

        let v = (0..64).rev().collect::<Vec<_>>();
        check(&v);

        let v = (0..63).rev().collect::<Vec<_>>();
        check(&v);

        let v = (0..127).rev().collect::<Vec<_>>();
        check(&v);

        let v = (0..16000).rev().collect::<Vec<_>>();
        check(&v);
    }

    #[test]
    fn random() {
        let mut rng = fastrand::Rng::new();
        rng.seed(0x6456346456734555);
        rng.f64();
        rng.f64();

        for _ in 0..16 {
            let v: Vec<_> = (0..rng.usize(10000..=50000))
                .map(|_| fastrand::u32(..))
                .collect();
            check(&v);
        }
    }
}
