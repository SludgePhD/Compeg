use std::{
    env, fs,
    path::Path,
    process,
    sync::{Arc, Mutex},
    thread,
};

use anyhow::bail;
use compeg::{Decoder, Gpu, ImageData};
use linuxvideo::format::{PixFormat, PixelFormat};
use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutEntry, ColorTargetState,
    ColorWrites, InstanceDescriptor, PipelineLayoutDescriptor, RenderPipelineDescriptor,
    ShaderStages, TextureFormat, TextureViewDescriptor, VertexState,
};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

struct Frame {
    jpeg: Mutex<Vec<u8>>,
}

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_module(env!("CARGO_PKG_NAME"), log::LevelFilter::Trace)
        .filter_module(env!("CARGO_CRATE_NAME"), log::LevelFilter::Trace)
        .parse_default_env()
        .init();

    let ev = EventLoop::new()?;
    let proxy = ev.create_proxy();

    let (width, height);
    let frame = match &*env::args().skip(1).collect::<Vec<_>>() {
        [path] => {
            if let Ok(device) = linuxvideo::Device::open(Path::new(path)) {
                let cap = device.video_capture(PixFormat::new(8000, 8000, PixelFormat::MJPG))?;
                width = cap.format().width();
                height = cap.format().height();
                match cap.format().pixel_format() {
                    PixelFormat::JPEG | PixelFormat::MJPG => {}
                    fmt => bail!("unsupported pixel format {fmt}"),
                }
                let mut stream = cap.into_stream()?;
                let mut jpeg = Vec::new();
                stream.dequeue(|view| {
                    jpeg.extend_from_slice(&view);
                    Ok(())
                })?;
                let frame = Arc::new(Frame {
                    jpeg: Mutex::new(jpeg),
                });

                let f = frame.clone();
                thread::spawn(move || loop {
                    let mut jpeg = Vec::new();
                    stream
                        .dequeue(|view| {
                            jpeg.extend_from_slice(&view);
                            Ok(())
                        })
                        .unwrap();
                    *f.jpeg.lock().unwrap() = jpeg;
                    proxy.send_event(()).unwrap();
                });

                frame
            } else {
                let jpeg = fs::read(&path)?;
                let image = ImageData::new(&jpeg)?;
                width = image.width();
                height = image.height();
                let frame = Arc::new(Frame {
                    jpeg: Mutex::new(jpeg),
                });
                frame
            }
        }
        _ => {
            eprintln!("usage: viewer <file.jpg>");
            process::exit(1);
        }
    };

    let win = Arc::new(
        WindowBuilder::new()
            .with_inner_size(PhysicalSize::new(width, height))
            .build(&ev)?,
    );

    let instance = wgpu::Instance::new(InstanceDescriptor {
        // The OpenGL backend panics spuriously, and segfaults when dropping the `Device`, so don't
        // enable it.
        backends: Backends::PRIMARY,
        ..Default::default()
    });
    let surface = instance.create_surface(win.clone())?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .ok_or_else(|| anyhow::anyhow!("no compatible graphics adapter found"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None))?;
    let (device, queue) = (Arc::new(device), Arc::new(queue));

    let mut surface_conf = surface
        .get_default_config(&adapter, width, height)
        .expect("incompatible surface, despite requiring one");
    surface_conf.usage |= wgpu::TextureUsages::COPY_DST;
    surface.configure(&device, &surface_conf);

    let gpu = Gpu::from_wgpu(device.clone(), queue.clone())?;
    let mut decoder = Decoder::new(Arc::new(gpu));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0)
var in_texture: texture_2d<f32>;
@group(0) @binding(1)
var in_sampler: sampler;

struct VertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32
) -> VertexOutput {
    // Logic copied from bevy's fullscreen quad shader
    var out: VertexOutput;
    out.uv = vec2<f32>(f32(vertex_index >> 1u), f32(vertex_index & 1u)) * 2.0;
    out.position = vec4<f32>(out.uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(in_texture, in_sampler, in.uv);
}
"#
            .into(),
        ),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let sampler = device.create_sampler(&Default::default());
    let pipe_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipe = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipe_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: "vertex",
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fragment",
            targets: &[Some(ColorTargetState {
                format: surface_conf.format,
                blend: None,
                write_mask: ColorWrites::all(),
            })],
        }),
        multiview: None,
    });

    let mut win_width = win.inner_size().width;
    let mut win_height = win.inner_size().height;
    let mut bindgroup = None;
    ev.run(move |event, target| match event {
        Event::UserEvent(()) => win.request_redraw(),
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::RedrawRequested => {
                let st = loop {
                    match surface.get_current_texture() {
                        Ok(tex) => break tex,
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let surface_conf = surface
                                .get_default_config(&adapter, win_width, win_height)
                                .expect("incompatible surface, despite requesting one");
                            log::info!("reconfiguring surface: {win_width}x{win_height}");
                            surface.configure(&device, &surface_conf);
                        }
                        Err(e) => {
                            eprintln!("fatal error: {e}");
                            process::exit(1);
                        }
                    }
                };

                let jpeg = frame.jpeg.lock().unwrap();
                let image = match ImageData::new(&*jpeg) {
                    Ok(image) => image,
                    Err(e) => {
                        eprintln!("failed to parse JPEG image: {e}");
                        process::exit(1);
                    }
                };

                let mut enc =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                let texture_changed = decoder.enqueue(&image, &mut enc);

                if texture_changed {
                    // Recreate bind group.
                    bindgroup = Some(device.create_bind_group(&BindGroupDescriptor {
                        label: None,
                        layout: &bgl,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &decoder.texture().create_view(&TextureViewDescriptor {
                                        format: Some(TextureFormat::Rgba8UnormSrgb),
                                        ..Default::default()
                                    }),
                                ),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                        ],
                    }))
                }

                let view = st.texture.create_view(&Default::default());
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, bindgroup.as_ref().unwrap(), &[]);
                pass.draw(0..3, 0..1);
                drop(pass);

                queue.submit([enc.finish()]);

                st.present();
            }
            WindowEvent::MouseWheel { .. } => win.request_redraw(),
            WindowEvent::CloseRequested => target.exit(),
            WindowEvent::Resized(size) => {
                win_width = size.width;
                win_height = size.height;

                let surface_conf = surface
                    .get_default_config(&adapter, win_width, win_height)
                    .expect("incompatible surface, despite requiring one");
                log::info!("reconfiguring surface after window resize: {win_width}x{win_height}");
                surface.configure(&device, &surface_conf);
            }
            _ => {}
        },
        _ => {}
    })?;
    Ok(())
}
