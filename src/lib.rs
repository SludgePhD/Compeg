mod dynamic;
mod error;
mod file;
mod huffman;
mod metadata;
mod sort;

use std::{
    borrow::Cow,
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use bytemuck::Zeroable;
use dynamic::{DownloadBuffer, DynamicBindGroup};
use error::{Error, Result};
use file::{JpegParser, SegmentKind};
use sort::Sorter;
use wgpu::*;

use crate::{
    dynamic::{DynamicBuffer, DynamicTexture},
    file::SofMarker,
    huffman::{HuffmanTables, TableData},
    metadata::QTable,
};

const OUTPUT_FORMAT: TextureFormat = TextureFormat::Rgba8Uint;
const WORKGROUP_SIZE: u32 = 64;

/// An open handle to a GPU.
///
/// This stores all static data (shaders, pipelines) needed for JPEG decoding.
pub struct Gpu {
    device: Arc<Device>,
    queue: Arc<Queue>,
    shared_bind_group_layout: Arc<BindGroupLayout>,
    compute_start_positions_pipeline: ComputePipeline,
    huffman_decode_pipeline: ComputePipeline,
}

impl Gpu {
    pub fn device_descriptor() -> DeviceDescriptor<'static> {
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

    pub async fn open() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor {
            // The OpenGL backend panics spuriously, so don't enable it.
            backends: Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .ok_or_else(|| Error::from("no supported graphics adapter found"))?;
        let (device, queue) = adapter
            .request_device(&Self::device_descriptor(), None)
            .await
            .map_err(|_| Error::from("no supported graphics device found"))?;

        Self::from_wgpu(device.into(), queue.into())
    }

    pub fn from_wgpu(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self> {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("main_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let shared_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("shared_bind_group_layout"),
                entries: &[
                    // `metadata`
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
                    // `scan_data`
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // `scan_positions`
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // `out`
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: OUTPUT_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // `debug`
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // `huffman_luts`
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let shared_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shared_pipeline_layout"),
            bind_group_layouts: &[&shared_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_start_positions_pipeline =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("compute_start_positions_pipeline"),
                layout: Some(&shared_pipeline_layout),
                module: &shader,
                entry_point: "compute_start_positions",
            });
        let huffman_decode_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("huffman_decode_pipeline"),
            layout: Some(&shared_pipeline_layout),
            module: &shader,
            entry_point: "huffman_decode",
        });

        Ok(Self {
            device,
            queue,
            shared_bind_group_layout: Arc::new(shared_bind_group_layout),
            compute_start_positions_pipeline,
            huffman_decode_pipeline,
        })
    }
}

/// A GPU JPEG decode context.
///
/// Holds all on-GPU buffers and textures needed for JPEG decoding.
pub struct Decoder {
    gpu: Arc<Gpu>,
    sorter: Sorter,
    metadata: Buffer,
    metadata_staging: Buffer,
    huffman_tables: Buffer,
    debug: Buffer,
    debug_staging: Buffer,
    /// Holds all the scan data of the JPEG (including all embedded RST markers). This constitutes
    /// the main input data to the shader pipeline.
    scan_data: DynamicBuffer,
    start_positions_buffer: DynamicBuffer,
    start_positions_staging_buffer: DownloadBuffer,
    output: DynamicTexture,
    shared_bind_group: DynamicBindGroup,
    start_positions: Vec<u32>,
}

impl Decoder {
    pub fn new(gpu: Arc<Gpu>) -> Self {
        let metadata = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("metadata"),
            size: mem::size_of::<metadata::Metadata>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let metadata_staging = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("metadata_staging"),
            size: metadata.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let huffman_tables = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("huffman_tables"),
            size: HuffmanTables::TOTAL_SIZE as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let debug = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("debug"),
            size: 1024,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let debug_staging = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("debug_staging"),
            size: debug.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scan_data = DynamicBuffer::new(
            gpu.clone(),
            "scan_data",
            BufferUsages::COPY_DST | BufferUsages::STORAGE,
        );
        let mut start_positions_buffer = DynamicBuffer::new(
            gpu.clone(),
            "start_positions",
            BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
        );
        let start_positions_staging_buffer = DownloadBuffer::new(gpu.clone());

        let output = DynamicTexture::new(
            gpu.clone(),
            "output",
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
            OUTPUT_FORMAT,
        );

        let shared_bind_group = DynamicBindGroup::new(
            gpu.clone(),
            gpu.shared_bind_group_layout.clone(),
            "shared_bind_group",
        );

        start_positions_buffer.reserve(500000);

        Self {
            sorter: Sorter::new(gpu.clone()),
            gpu,
            metadata,
            metadata_staging,
            huffman_tables,
            debug,
            debug_staging,
            scan_data,
            start_positions_buffer,
            start_positions_staging_buffer,
            output,
            shared_bind_group,
            start_positions: Vec::new(),
        }
    }

    pub fn start_decode(&mut self, data: &ImageData<'_>) {
        self.output
            .reserve(data.metadata.width, data.metadata.height);

        let t_compute_positions_cpu =
            time(|| compute_start_positions(&mut self.start_positions, data.scan_data()));

        let t_enqueue_writes = time(|| {
            self.gpu
                .queue
                .write_buffer(&self.metadata, 0, bytemuck::bytes_of(&data.metadata));
            self.scan_data.write(data.scan_data());

            // Offset 0 is always a start position.
            self.start_positions_buffer.write(&[0u32]);
        });

        self.gpu.queue.write_buffer(
            &self.huffman_tables,
            0,
            bytemuck::bytes_of(data.huffman_tables.raw()),
        );

        let start = Instant::now();
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let mut compute = enc.begin_compute_pass(&ComputePassDescriptor::default());

        let bind_group = self.shared_bind_group.bind_group(&[
            self.metadata.as_entire_binding().into(),
            self.scan_data.as_resource(),
            self.start_positions_buffer.as_resource(),
            self.output.as_resource(),
            self.debug.as_entire_binding().into(),
            self.huffman_tables.as_entire_binding().into(),
        ]);
        compute.set_bind_group(0, bind_group, &[]);

        // For 32 Bytes per invocation, and 64 invocations per workgroup, each workgroup will process
        // 32*64 bytes of scan data.
        const BYTES_PER_INVOC: usize = 32;
        let invocs = (data.scan_data().len() + BYTES_PER_INVOC - 1) / BYTES_PER_INVOC;
        let workgroups = (invocs + WORKGROUP_SIZE as usize - 1) / WORKGROUP_SIZE as usize;
        let workgroups = workgroups.try_into().unwrap();
        compute.set_pipeline(&self.gpu.compute_start_positions_pipeline);
        compute.dispatch_workgroups(workgroups, 1, 1);
        drop(compute);

        log::trace!(
            "compute start positions: {} workgroups ({} shader invocations; {} scan data bytes)",
            workgroups,
            workgroups * WORKGROUP_SIZE,
            data.scan_data().len(),
        );
        enc.copy_buffer_to_buffer(
            &self.metadata,
            0,
            &self.metadata_staging,
            0,
            self.metadata.size(),
        );
        self.gpu.queue.submit([enc.finish()]);
        self.metadata_staging
            .slice(..)
            .map_async(MapMode::Read, Result::unwrap);
        self.gpu.device.poll(MaintainBase::Wait);
        let view = self.metadata_staging.slice(..).get_mapped_range();
        let &meta: &metadata::Metadata = bytemuck::from_bytes(&view);
        let count = meta.start_position_count;
        drop(view);
        self.metadata_staging.unmap();
        assert_eq!(count as usize, self.start_positions.len());
        let t_compute_positions_gpu = start.elapsed();

        let start = Instant::now();
        self.sorter.sort(&mut self.start_positions_buffer, count);
        self.gpu.device.poll(MaintainBase::Wait);
        let t_sort = start.elapsed();

        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut compute = enc.begin_compute_pass(&ComputePassDescriptor::default());
        let invocs: u32 = self.start_positions.len().try_into().unwrap();
        let workgroups = (invocs + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        if workgroups > 65535 {
            panic!("restart interval count {invocs} exceeds maximum of 65535*{WORKGROUP_SIZE}");
        }
        compute.set_bind_group(0, bind_group, &[]);
        compute.set_pipeline(&self.gpu.huffman_decode_pipeline);
        compute.dispatch_workgroups(workgroups.into(), 1, 1);
        drop(compute);

        log::trace!(
            "dispatching {} workgroups ({} shader invocations; {} restart intervals)",
            workgroups,
            workgroups * WORKGROUP_SIZE,
            self.start_positions.len(),
        );

        self.start_positions_staging_buffer
            .download(&self.start_positions_buffer, &mut enc);
        enc.copy_buffer_to_buffer(&self.debug, 0, &self.debug_staging, 0, self.debug.size());

        let buffer = enc.finish();
        let t_submit = time(|| self.gpu.queue.submit([buffer]));

        self.start_positions_staging_buffer.map();
        self.debug_staging
            .slice(..)
            .map_async(MapMode::Read, Result::unwrap);
        let t_poll = time(|| self.gpu.device.poll(MaintainBase::Wait));

        let range = self.debug_staging.slice(..).get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&range);
        log::debug!("debug = {:08x?}", words);
        drop(range);
        self.debug_staging.unmap();

        let range = self.start_positions_staging_buffer.mapped();
        let words: &[u32] = &bytemuck::cast_slice(&range)[..count as usize];
        log::debug!("start positions: {}", count);
        for (i, (actual, expected)) in words.iter().zip(&self.start_positions).enumerate() {
            if actual != expected {
                panic!("{actual} <-> {expected} at index {i}");
            }
        }
        drop(range);

        log::trace!(
            "t_compute_positions_cpu={t_compute_positions_cpu:?}, \
            t_compute_positions_gpu={t_compute_positions_gpu:?}, \
            t_sort={t_sort:?}, \
            t_enqueue_writes={t_enqueue_writes:?}, \
            t_submit={t_submit:?}, \
            t_poll={t_poll:?}"
        );
    }

    #[inline]
    pub fn output(&self) -> &Texture {
        self.output.texture()
    }
}

fn compute_start_positions(start_positions: &mut Vec<u32>, scan_data: &[u8]) {
    start_positions.clear();
    start_positions.push(0);

    for (i, pair) in scan_data.windows(2).enumerate() {
        let &[first, next] = pair else { unreachable!() };

        if first == 0xff && next != 0x00 {
            // Next chunk starts after `next`.
            start_positions.push(i as u32 + 2);
        }
    }
}

fn time<R>(f: impl FnOnce() -> R) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

/// A parsed JPEG image, containing all data needed for on-GPU decoding.
pub struct ImageData<'a> {
    metadata: metadata::Metadata,
    huffman_tables: HuffmanTables,
    jpeg: Cow<'a, [u8]>,
    scan_data_offset: usize,
    scan_data_len: usize,
}

impl<'a> ImageData<'a> {
    /// Reads [`ImageData`] from an in-memory JPEG file.
    ///
    /// If this returns an error, it either means that the JPEG file is malformed, or that it uses
    /// features this library does not support. In either case, the application should fall back to
    /// a more fully-featured software decoder.
    pub fn new(jpeg: impl Into<Cow<'a, [u8]>>) -> Result<Self> {
        Self::new_impl(jpeg.into())
    }

    fn new_impl(jpeg: Cow<'a, [u8]>) -> Result<Self> {
        macro_rules! bail {
            ($($args:tt)*) => {
                return Err(Error::from(format!(
                    $($args)*
                )))
            };
        }

        let mut size = None;
        let mut ri = None;
        let mut huffman_tables = HuffmanTables::new();
        let mut qtables = [QTable::zeroed(); 4];
        let mut scan_data = None;
        let mut components = None;
        let mut component_indices = None;
        let mut component_dchuff = [0; 3];
        let mut component_achuff = [0; 3];

        let mut parser = JpegParser::new(&jpeg);
        while let Some(segment) = parser.next_segment()? {
            match segment.kind {
                SegmentKind::Sof(sof) => {
                    if sof.sof() != SofMarker::SOF0 {
                        bail!("not a baseline JPEG (SOF={:?})", sof.sof());
                    }

                    if sof.P() != 8 {
                        bail!("sample precision of {} bits is not supported", sof.P());
                    }

                    if component_indices.is_some() {
                        bail!("encountered multiple SOF markers");
                    }

                    match sof.components() {
                        [y, u, v] => {
                            log::trace!("frame components:");
                            log::trace!("{:?}", y);
                            log::trace!("{:?}", u);
                            log::trace!("{:?}", v);

                            if y.Tqi() > 3 || u.Tqi() > 3 || v.Tqi() > 3 {
                                bail!("invalid quantization table selection [{},{},{}] (only tables 0-3 are valid)", y.Tqi(), u.Tqi(), v.Tqi());
                            }
                            if y.Hi() != 2 || y.Vi() != 1 {
                                bail!(
                                    "invalid sampling factors {}x{} for Y component (expected 2x1)",
                                    y.Hi(),
                                    y.Vi(),
                                );
                            }
                            if u.Hi() != v.Hi() || u.Vi() != v.Vi() || u.Hi() != 1 || u.Vi() != 1 {
                                bail!(
                                    "invalid U/V sampling factors {}x{} and {}x{}",
                                    u.Hi(),
                                    u.Vi(),
                                    v.Hi(),
                                    v.Vi(),
                                );
                            }

                            component_indices = Some([y.Ci(), u.Ci(), v.Ci()]);

                            components = Some([y, u, v]);
                        }
                        _ => {
                            bail!("frame with {} components not supported (only 3 components are supported)", sof.components().len());
                        }
                    }

                    size = Some((sof.X(), sof.Y()));
                }
                SegmentKind::Dqt(dqt) => {
                    for table in dqt.tables() {
                        if table.Pq() != 0 {
                            bail!(
                                "invalid quantization table precision Pq={} (only 0 is allowed)",
                                table.Pq()
                            );
                        }
                        if table.Tq() > 3 {
                            bail!(
                                "invalid quantization table destination Tq={} (0-3 are allowed)",
                                table.Tq()
                            );
                        }

                        for (dest, src) in qtables[usize::from(table.Tq())]
                            .values
                            .iter_mut()
                            .zip(table.Qk())
                        {
                            *dest = u32::from(*src);
                        }
                    }
                }
                SegmentKind::Dht(dht) => {
                    for table in dht.tables() {
                        let index = table.Th();
                        if index > 1 {
                            bail!(
                                "DHT Th={}, only 0 and 1 are allowed for baseline JPEGs",
                                table.Th()
                            );
                        }

                        let class = match table.Tc() {
                            class @ (0 | 1) => class,
                            err => bail!("invalid table class Tc={err} (only 0 and 1 are valid)"),
                        };

                        let index = (index << 1) | class;
                        let data = TableData::build(table.Li(), table.Vij());
                        huffman_tables.set(index, &data);
                    }
                }
                SegmentKind::Dri(dri) => {
                    // FIXME: add some checks here, we probably should have a maximum Ri value?
                    ri = Some(dri.Ri() as u32);
                }
                SegmentKind::Sos(sos) => {
                    if sos.Ss() != 0 || sos.Se() != 63 || sos.Ah() != 0 || sos.Al() != 0 {
                        bail!("non-baseline scan header");
                    }

                    let Some(component_indices) = component_indices else {
                        bail!("SOS not preceded by SOF header");
                    };

                    match sos.components() {
                        [y,u,v] => {
                            log::trace!("scan components:");
                            log::trace!("{:?}", y);
                            log::trace!("{:?}", u);
                            log::trace!("{:?}", v);

                            let scan_indices = [y.Csj(), u.Csj(), v.Csj()];
                            if component_indices != scan_indices {
                                bail!("scan component index mismatch (expected component order {:?}, got {:?})", component_indices, scan_indices);
                            }

                            component_dchuff = [y.Tdj(), u.Tdj(), v.Tdj()];
                            component_achuff = [y.Taj(), u.Taj(), v.Taj()];
                        }
                        _ => bail!("scan with {} components not supported (only 3 components are supported)", sos.components().len()),
                    }

                    scan_data = Some((sos.data_offset(), sos.data().len()));
                }
                SegmentKind::Eoi => break,
                _ => {}
            }
        }

        let (Some((width, height)), Some(components), Some(ri), Some((scan_data_offset, scan_data_len))) = (size, components, ri, scan_data) else {
            bail!("missing DRI/SOS/SOI marker");
        };

        let metadata = metadata::Metadata {
            width: width.into(),
            height: height.into(),
            restart_interval: ri,
            qtables,
            components: [0, 1, 2].map(|i| metadata::Component {
                hsample: components[i].Hi().into(),
                vsample: components[i].Vi().into(),
                qtable: components[i].Tqi().into(),
                dchuff: component_dchuff[i].into(),
                achuff: component_achuff[i].into(),
            }),
            start_position_count: 1, // first start pos is always 0
        };

        Ok(Self {
            metadata,
            huffman_tables,
            jpeg,
            scan_data_offset,
            scan_data_len,
        })
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.metadata.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.metadata.height
    }

    fn scan_data(&self) -> &[u8] {
        &self.jpeg[self.scan_data_offset..][..self.scan_data_len]
    }
}
