mod bits;
mod dynamic;
mod error;
mod file;
mod huffman;
mod metadata;
mod scan;
mod sort;

use std::{
    borrow::Cow,
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use bytemuck::Zeroable;
use dynamic::DynamicBindGroup;
use error::{Error, Result};
use file::{JpegParser, SegmentKind};
use scan::ScanBuffer;
use wgpu::*;

use crate::{
    dynamic::{DynamicBuffer, DynamicTexture},
    file::SofMarker,
    huffman::{HuffmanTables, TableData},
    metadata::{QTable, UNZIGZAG},
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
    jpeg_decode_pipeline: ComputePipeline,
}

impl Gpu {
    pub fn device_descriptor() -> DeviceDescriptor<'static> {
        // We're back to vanilla WebGPU for now (since the sort shader removal).
        DeviceDescriptor::default()
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
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // `huffman_l1`
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
                    // `huffman_l2`
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // `scan_data`
                    BindGroupLayoutEntry {
                        binding: 3,
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
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // `out`
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: OUTPUT_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shared_pipeline_layout"),
            bind_group_layouts: &[&shared_bind_group_layout],
            push_constant_ranges: &[],
        });
        let jpeg_decode_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("jpeg_decode_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "jpeg_decode",
        });

        Ok(Self {
            device,
            queue,
            shared_bind_group_layout: Arc::new(shared_bind_group_layout),
            jpeg_decode_pipeline,
        })
    }
}

/// A GPU JPEG decode context.
///
/// Holds all on-GPU buffers and textures needed for JPEG decoding.
pub struct Decoder {
    gpu: Arc<Gpu>,
    metadata: Buffer,
    huffman_l1: Buffer,
    huffman_l2: DynamicBuffer,
    /// Holds all the scan data of the JPEG (including all embedded RST markers). This constitutes
    /// the main input data to the shader pipeline.
    scan_data: DynamicBuffer,
    start_positions_buffer: DynamicBuffer,
    output: DynamicTexture,
    shared_bind_group: DynamicBindGroup,
    scan_buffer: ScanBuffer,
}

impl Decoder {
    pub fn new(gpu: Arc<Gpu>) -> Self {
        let metadata = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("metadata"),
            size: mem::size_of::<metadata::Metadata>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let huffman_l1 = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("huffman_l1"),
            size: HuffmanTables::TOTAL_L1_SIZE as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let huffman_l2 = DynamicBuffer::new(
            gpu.clone(),
            "huffman_l2",
            BufferUsages::COPY_DST | BufferUsages::STORAGE,
        );

        let scan_data = DynamicBuffer::new(
            gpu.clone(),
            "scan_data",
            BufferUsages::COPY_DST | BufferUsages::STORAGE,
        );
        let start_positions_buffer = DynamicBuffer::new(
            gpu.clone(),
            "start_positions",
            BufferUsages::COPY_DST | BufferUsages::STORAGE,
        );

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

        Self {
            gpu,
            metadata,
            huffman_l1,
            huffman_l2,
            scan_data,
            start_positions_buffer,
            output,
            shared_bind_group,
            scan_buffer: ScanBuffer::new(),
        }
    }

    pub fn start_decode(&mut self, data: &ImageData<'_>) {
        self.output.reserve(data.width(), data.height());

        let total_restart_intervals = data.metadata.total_restart_intervals;
        let t_preprocess = time(|| {
            self.scan_buffer
                .process(data.scan_data(), total_restart_intervals)
        });

        let t_enqueue_writes = time(|| {
            self.gpu
                .queue
                .write_buffer(&self.metadata, 0, bytemuck::bytes_of(&data.metadata));
            self.scan_data.write(self.scan_buffer.processed_scan_data());
            self.start_positions_buffer
                .write(self.scan_buffer.start_positions());

            self.gpu
                .queue
                .write_buffer(&self.huffman_l1, 0, data.huffman_tables.l1_data());
            self.huffman_l2.write(data.huffman_tables.l2_data());
        });

        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let bind_group = self.shared_bind_group.bind_group(&[
            self.metadata.as_entire_binding().into(),
            self.huffman_l1.as_entire_binding().into(),
            self.huffman_l2.as_resource(),
            self.scan_data.as_resource(),
            self.start_positions_buffer.as_resource(),
            self.output.as_resource(),
        ]);

        let mut compute = enc.begin_compute_pass(&ComputePassDescriptor::default());
        let workgroups = (total_restart_intervals + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        if workgroups > 65535 {
            panic!("restart interval count {total_restart_intervals} exceeds maximum of 65535*{WORKGROUP_SIZE}");
        }
        compute.set_bind_group(0, bind_group, &[]);
        compute.set_pipeline(&self.gpu.jpeg_decode_pipeline);
        compute.dispatch_workgroups(workgroups, 1, 1);
        drop(compute);

        log::trace!(
            "dispatching {} workgroups ({} shader invocations; {} restart intervals)",
            workgroups,
            workgroups * WORKGROUP_SIZE,
            total_restart_intervals,
        );

        let buffer = enc.finish();
        let t_submit = time(|| self.gpu.queue.submit([buffer]));
        let t_poll = time(|| self.gpu.device.poll(MaintainBase::Wait));

        log::trace!(
            "t_preprocess={t_preprocess:?}, \
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

fn time<R>(f: impl FnOnce() -> R) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

/// A parsed JPEG image, containing all data needed for on-GPU decoding.
pub struct ImageData<'a> {
    metadata: metadata::Metadata,
    width: u16,
    height: u16,
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
        let mut huffman_tables = [
            TableData::default_luminance_dc(),
            TableData::default_luminance_ac(),
            TableData::default_chrominance_dc(),
            TableData::default_chrominance_ac(),
        ];
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
                        huffman_tables[usize::from(index)] = data;
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

        let max_hsample = components.iter().map(|c| c.Hi()).max().unwrap().into();
        let max_vsample = components.iter().map(|c| c.Vi()).max().unwrap().into();
        let width_dus = u32::from((width + 7) / 8);
        let height_dus = u32::from((height + 7) / 8);
        let width_mcus = width_dus / max_hsample;
        let height_mcus = height_dus / max_vsample;

        let total_restart_intervals = height_mcus * width_mcus / ri;

        let metadata = metadata::Metadata {
            restart_interval: ri,
            qtables,
            components: [0, 1, 2].map(|i| metadata::Component {
                hsample: components[i].Hi().into(),
                vsample: components[i].Vi().into(),
                qtable: components[i].Tqi().into(),
                dchuff: u32::from(component_dchuff[i] << 1),
                achuff: u32::from((component_achuff[i] << 1) | 1),
            }),
            total_restart_intervals,
            width_mcus,
            max_hsample,
            max_vsample,
            unzigzag: UNZIGZAG,
        };

        let huffman_tables = HuffmanTables::new(huffman_tables);

        Ok(Self {
            metadata,
            width,
            height,
            huffman_tables,
            jpeg,
            scan_data_offset,
            scan_data_len,
        })
    }

    /// Returns the width of the image in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width.into()
    }

    /// Returns the height of the image in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height.into()
    }

    fn scan_data(&self) -> &[u8] {
        &self.jpeg[self.scan_data_offset..][..self.scan_data_len]
    }
}
