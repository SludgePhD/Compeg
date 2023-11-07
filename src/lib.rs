//! WebGPU compute shader JPEG decoder.
//!
//! Usage:
//!
//! - Create a [`Gpu`] context (either automatically via [`Gpu::open`] or from an existing [`wgpu`] context via [`Gpu::from_wgpu`]).
//! - Create a [`Decoder`] (or multiple) via [`Decoder::new`].
//! - For each JPEG image you want to decode, create an [`ImageData`] object and pass it to [`Decoder::start_decode`].
//!   - The [`Decoder`] will automatically resize buffers and textures when they are too small for the passed [`ImageData`].
//! - Access the output [`Texture`] via [`DecodeOp::texture`].
//!   - [`wgpu`] will automatically ensure that the proper barriers are in place when this
//!     [`Texture`] is used in a GPU operation.

mod bits;
mod dynamic;
mod error;
mod file;
mod huffman;
mod metadata;
mod scan;

#[cfg(test)]
mod tests;

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
use wgpu::*;

use crate::{
    dynamic::{DynamicBuffer, DynamicTexture},
    file::SofMarker,
    huffman::{HuffmanTables, TableData},
    metadata::QTable,
};

/// **Not** part of the public API. Used for benchmarks only.
#[doc(hidden)]
pub use scan::ScanBuffer;

const OUTPUT_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

const HUFFMAN_WORKGROUP_SIZE: u32 = 64;

const DCT_WORKGROUP_SIZE: u32 = 256;
const THREADS_PER_DCT: u32 = 8;
const DCTS_PER_WORKGROUP: u32 = DCT_WORKGROUP_SIZE / THREADS_PER_DCT;

const FINALIZE_WORKGROUP_SIZE: u32 = 256;
const MCU_HEIGHT: u32 = 8;
const THREADS_PER_MCU: u32 = MCU_HEIGHT;
const MCUS_PER_WORKGROUP: u32 = FINALIZE_WORKGROUP_SIZE / THREADS_PER_MCU;

/// An open handle to a GPU.
///
/// This stores all static data (shaders, pipelines) needed for JPEG decoding.
pub struct Gpu {
    device: Arc<Device>,
    queue: Arc<Queue>,
    metadata_bgl: Arc<BindGroupLayout>,
    huffman_bgl: Arc<BindGroupLayout>,
    coefficients_bgl: Arc<BindGroupLayout>,
    output_bgl: Arc<BindGroupLayout>,
    huffman_decode_pipeline: ComputePipeline,
    dct_pipeline: ComputePipeline,
    finalize_pipeline: ComputePipeline,
}

impl Gpu {
    /// Opens a suitable default GPU.
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
            .request_device(&Default::default(), None)
            .await
            .map_err(|_| Error::from("no supported graphics device found"))?;

        let info = adapter.get_info();
        log::info!(
            "opened {:?} adapter {} ({})",
            info.backend,
            info.name,
            info.driver
        );

        Self::from_wgpu(device.into(), queue.into())
    }

    /// Creates a [`Gpu`] handle from an existing [`wgpu`] [`Device`] and [`Queue`].
    pub fn from_wgpu(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self> {
        let shared = include_str!("shared.wgsl");
        let huffman = include_str!("huffman.wgsl");
        let dct = include_str!("dct.wgsl");
        let huffman = format!("{shared}\n\n{huffman}");
        let dct = format!("{shared}\n\n{dct}");
        let huffman = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("huffman"),
            source: wgpu::ShaderSource::Wgsl(huffman.into()),
        });
        let dct = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("dct"),
            source: wgpu::ShaderSource::Wgsl(dct.into()),
        });

        let metadata_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("metadata_bgl"),
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
            ],
        });
        let huffman_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("huffman_bgl"),
            entries: &[
                // `huffman_l1`
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
                // `huffman_l2`
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
                // `scan_data`
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
                // `scan_positions`
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
            ],
        });
        let coefficients_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("coefficients_bgl"),
            entries: &[
                // `coefficients`
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
        let output_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("output_bgl"),
            entries: &[
                // `out`
                BindGroupLayoutEntry {
                    binding: 0,
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
        let huffman_decode_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("huffman_decode_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&metadata_bgl, &huffman_bgl, &coefficients_bgl],
                push_constant_ranges: &[],
            })),
            module: &huffman,
            entry_point: "huffman",
        });
        let dct_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("dct_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&metadata_bgl, &coefficients_bgl, &output_bgl],
                push_constant_ranges: &[],
            })),
            module: &dct,
            entry_point: "dct",
        });
        let finalize_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("finalize_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&metadata_bgl, &coefficients_bgl, &output_bgl],
                push_constant_ranges: &[],
            })),
            module: &dct,
            entry_point: "finalize",
        });

        Ok(Self {
            device,
            queue,
            metadata_bgl: Arc::new(metadata_bgl),
            huffman_bgl: Arc::new(huffman_bgl),
            coefficients_bgl: Arc::new(coefficients_bgl),
            output_bgl: Arc::new(output_bgl),
            huffman_decode_pipeline,
            dct_pipeline,
            finalize_pipeline,
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
    coefficients: DynamicBuffer,
    output: DynamicTexture,
    metadata_bg: DynamicBindGroup,
    huffman_bg: DynamicBindGroup,
    coefficients_bg: DynamicBindGroup,
    output_bg: DynamicBindGroup,
    scan_buffer: ScanBuffer,
}

impl Decoder {
    /// [`wgpu`] only guarantees that it is able to dispatch 65535 workgroups at once, so this is
    /// the maximum number of shader invocations we can run (and thus the max. number of restart
    /// intervals we can process).
    const MAX_RESTART_INTERVALS: u32 = HUFFMAN_WORKGROUP_SIZE * 65535;
    // FIXME: fix this to use MCUs/DUs as the limiting factor?

    /// Creates a new JPEG decoding context on the given [`Gpu`].
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
        let coefficients = DynamicBuffer::new(gpu.clone(), "coefficients", BufferUsages::STORAGE);

        let output = DynamicTexture::new(
            gpu.clone(),
            "output",
            TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            OUTPUT_FORMAT,
        );

        let metadata_bg =
            DynamicBindGroup::new(gpu.clone(), gpu.metadata_bgl.clone(), "metadata_bg");
        let huffman_bg = DynamicBindGroup::new(gpu.clone(), gpu.huffman_bgl.clone(), "huffman_bg");
        let coefficients_bg =
            DynamicBindGroup::new(gpu.clone(), gpu.coefficients_bgl.clone(), "coefficients_bg");
        let output_bg = DynamicBindGroup::new(gpu.clone(), gpu.output_bgl.clone(), "output_bg");

        Self {
            gpu,
            metadata,
            huffman_l1,
            huffman_l2,
            scan_data,
            start_positions_buffer,
            coefficients,
            output,
            metadata_bg,
            huffman_bg,
            coefficients_bg,
            output_bg,
            scan_buffer: ScanBuffer::new(),
        }
    }

    /// Preprocesses and uploads a JPEG image, and dispatches the decoding operation on the GPU.
    ///
    /// Returns a [`DecodeOp`] with information about the decode operation.
    pub fn start_decode(&mut self, data: &ImageData<'_>) -> DecodeOp<'_> {
        let texture_changed = self.output.reserve(data.width(), data.height());

        let total_restart_intervals = data.metadata.total_restart_intervals;
        let total_mcus = total_restart_intervals * data.metadata.restart_interval;
        let total_dus = total_mcus * data.metadata.dus_per_mcu;
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

            // Reserve space for the decoded coefficients. There are 64 32-bit values per data unit.
            self.coefficients
                .reserve(4 * u64::from(data.metadata.retained_coefficients) * u64::from(total_dus));
        });

        let metadata_bg = self
            .metadata_bg
            .bind_group(&[self.metadata.as_entire_binding().into()]);
        let huffman_bg = self.huffman_bg.bind_group(&[
            self.huffman_l1.as_entire_binding().into(),
            self.huffman_l2.as_resource(),
            self.scan_data.as_resource(),
            self.start_positions_buffer.as_resource(),
        ]);
        let coefficients_bg = self
            .coefficients_bg
            .bind_group(&[self.coefficients.as_resource()]);
        let output_bg = self.output_bg.bind_group(&[self.output.as_resource()]);

        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        enc.clear_buffer(self.coefficients.buffer(), 0, None);

        let mut compute = enc.begin_compute_pass(&ComputePassDescriptor::default());
        let huffman_workgroups =
            (total_restart_intervals + HUFFMAN_WORKGROUP_SIZE - 1) / HUFFMAN_WORKGROUP_SIZE;
        let dct_workgroups = (total_dus + DCTS_PER_WORKGROUP - 1) / DCTS_PER_WORKGROUP;
        let finalize_workgroups = (total_mcus + MCUS_PER_WORKGROUP - 1) / MCUS_PER_WORKGROUP;

        compute.set_bind_group(0, metadata_bg, &[]);
        compute.set_bind_group(1, huffman_bg, &[]);
        compute.set_bind_group(2, coefficients_bg, &[]);
        compute.set_pipeline(&self.gpu.huffman_decode_pipeline);
        compute.dispatch_workgroups(huffman_workgroups, 1, 1);

        compute.set_bind_group(0, metadata_bg, &[]);
        compute.set_bind_group(1, coefficients_bg, &[]);
        compute.set_bind_group(2, output_bg, &[]);
        compute.set_pipeline(&self.gpu.dct_pipeline);
        compute.dispatch_workgroups(dct_workgroups, 1, 1);
        compute.set_pipeline(&self.gpu.finalize_pipeline);
        compute.dispatch_workgroups(finalize_workgroups, 1, 1);

        drop(compute);

        log::trace!(
            "dispatching {} workgroups for huffman decoding ({} shader invocations; {} restart intervals)",
            huffman_workgroups,
            huffman_workgroups * HUFFMAN_WORKGROUP_SIZE,
            total_restart_intervals,
        );
        log::trace!(
            "dispatching {} workgroups for IDCT ({} shader invocations; {} MCUs; {} DUs)",
            dct_workgroups,
            dct_workgroups * DCT_WORKGROUP_SIZE,
            total_mcus,
            total_dus,
        );
        log::trace!(
            "dispatching {} workgroups for compositing ({} shader invocations; {} MCUs)",
            finalize_workgroups,
            finalize_workgroups * FINALIZE_WORKGROUP_SIZE,
            total_mcus,
        );

        let buffer = enc.finish();
        let submission = self.gpu.queue.submit([buffer]);

        log::trace!(
            "t_preprocess={t_preprocess:?}, \
            t_enqueue_writes={t_enqueue_writes:?}"
        );

        DecodeOp {
            submission,
            texture: self.output.texture(),
            texture_changed,
        }
    }

    /// Performs a blocking decode operation.
    ///
    /// This method works identically to [`Decoder::start_decode`], but will wait until the
    /// operation on the GPU is finished.
    ///
    /// Note that it is not typically necessary to use this method, since [`wgpu`] will
    /// automatically insert barriers before the target texture is accessed.
    pub fn decode_blocking(&mut self, data: &ImageData<'_>) -> DecodeOp<'_> {
        // FIXME: destructuring and recreation is annoyingly needed because the `DecodeOp` will
        // borrow `self` *mutably*, even though an immutable borrow would suffice.
        let DecodeOp {
            submission,
            texture: _,
            texture_changed,
        } = self.start_decode(data);
        let t_poll = time(|| {
            self.gpu
                .device
                .poll(MaintainBase::WaitForSubmissionIndex(submission.clone()))
        });

        log::trace!("t_poll={:?}", t_poll);

        DecodeOp {
            submission,
            texture: self.output.texture(),
            texture_changed,
        }
    }
}

fn time<R>(f: impl FnOnce() -> R) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

/// Information about an ongoing JPEG decode operation.
///
/// Returned by [`Decoder::start_decode`].
pub struct DecodeOp<'a> {
    submission: SubmissionIndex,
    texture: &'a Texture,
    texture_changed: bool,
}

impl<'a> DecodeOp<'a> {
    /// Returns the [`SubmissionIndex`] associated with the compute shader dispatch.
    #[inline]
    pub fn submission(&self) -> &SubmissionIndex {
        &self.submission
    }

    /// Returns a reference to the target [`Texture`] that the JPEG decode operation is writing to.
    ///
    /// Note that, when using the [`Decoder`] with JPEG images of varying sizes, not the entire
    /// target texture will be written to. The caller has to ensure to only use the area of the
    /// [`Texture`] indicated by [`ImageData::width`] and [`ImageData::height`].
    #[inline]
    pub fn texture(&self) -> &Texture {
        self.texture
    }

    /// Returns a [`bool`] indicating whether the target [`Texture`] has been reallocated since the
    /// last decode operation on the same [`Decoder`] was started.
    ///
    /// If this is the first decode operation, this method will return `true`. The return value of
    /// this method can be used to determine whether any bind groups referencing the target
    /// [`Texture`] need to be recreated.
    #[inline]
    pub fn texture_changed(&self) -> bool {
        self.texture_changed
    }
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

        let mut parser = JpegParser::new(&jpeg)?;
        while let Some(segment) = parser.next_segment()? {
            let Some(kind) = segment.as_segment_kind() else {
                continue;
            };
            match kind {
                SegmentKind::SOF(sof) => {
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
                            log::trace!("- {:?}", y);
                            log::trace!("- {:?}", u);
                            log::trace!("- {:?}", v);

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
                                    "invalid U/V sampling factors {}x{} and {}x{} (expected 1x1)",
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
                SegmentKind::DQT(dqt) => {
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
                SegmentKind::DHT(dht) => {
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
                SegmentKind::DRI(dri) => {
                    // FIXME: add some checks here, we probably should have a maximum Ri value?
                    ri = Some(dri.Ri() as u32);
                }
                SegmentKind::SOS(sos) => {
                    if sos.Ss() != 0 || sos.Se() != 63 || sos.Ah() != 0 || sos.Al() != 0 {
                        bail!("non-baseline scan header");
                    }

                    let Some(component_indices) = component_indices else {
                        bail!("SOS not preceded by SOF header");
                    };

                    match sos.components() {
                        [y,u,v] => {
                            log::trace!("scan components:");
                            log::trace!("- {:?}", y);
                            log::trace!("- {:?}", u);
                            log::trace!("- {:?}", v);

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
                _ => {}
            }
        }

        #[rustfmt::skip]
        let (
            Some((width, height)),
            Some(components),
            Some((scan_data_offset, scan_data_len)),
        ) = (size, components, scan_data) else {
            bail!("missing SOS/SOI marker");
        };

        let dus_per_mcu = components
            .iter()
            .map(|c| c.Hi() * c.Vi())
            .sum::<u8>()
            .into();

        let max_hsample = components.iter().map(|c| c.Hi()).max().unwrap().into();
        let max_vsample = components.iter().map(|c| c.Vi()).max().unwrap().into();
        let width_dus = u32::from((width + 7) / 8);
        let height_dus = u32::from((height + 7) / 8);
        let width_mcus = (width_dus + max_hsample - 1) / max_hsample; // (round up)
        let height_mcus = (height_dus + max_vsample - 1) / max_vsample; // (round up)

        log::trace!("max Hi={} Vi={}", max_hsample, max_vsample);
        log::trace!("width={width} height={height} width_dus={width_dus} height_dus={height_dus} width_mcus={width_mcus} height_mcus={height_mcus}");

        let ri = ri.unwrap_or(height_mcus * width_mcus);
        let total_restart_intervals = height_mcus * width_mcus / ri;

        if total_restart_intervals > Decoder::MAX_RESTART_INTERVALS {
            bail!(
                "number of restart intervals exceeds limit ({} > {})",
                total_restart_intervals,
                Decoder::MAX_RESTART_INTERVALS,
            );
        }

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
            dus_per_mcu,
            retained_coefficients: metadata::DEFAULT_RETAINED_COEFFICIENTS,
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

    /// Returns the total parallelism this JPEG permits.
    ///
    /// This number indicates how many parts of the image can be processed in parallel. It is
    /// crucial for performance that this number is as high as possible. If it is below 10000, it is
    /// likely faster to use a CPU-based decoder instead.
    #[inline]
    pub fn parallelism(&self) -> u32 {
        self.metadata.total_restart_intervals
    }

    fn scan_data(&self) -> &[u8] {
        &self.jpeg[self.scan_data_offset..][..self.scan_data_len]
    }
}
