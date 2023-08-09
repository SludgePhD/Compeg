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
use wgpu::InstanceDescriptor;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

struct Frame {
    jpeg: Mutex<Vec<u8>>,
}

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_module(env!("CARGO_PKG_NAME"), log::LevelFilter::Trace)
        .parse_default_env()
        .init();

    let ev = EventLoop::new();
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
            eprintln!("usage: viewer <file.jpeg>");
            process::exit(1);
        }
    };

    let win = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&ev)?;

    let instance = wgpu::Instance::new(InstanceDescriptor::default());
    let surface = unsafe { instance.create_surface(&win)? };
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .ok_or_else(|| anyhow::anyhow!("no compatible graphics adapter found"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None))?;
    let (device, queue) = (Arc::new(device), Arc::new(queue));

    let mut conf = surface
        .get_default_config(&adapter, width, height)
        .expect("incompatible surface, despite requiring one");
    conf.usage |= wgpu::TextureUsages::COPY_DST;
    surface.configure(&device, &conf);

    let gpu = Gpu::from_wgpu(device.clone(), queue.clone())?;
    let mut decoder = Decoder::new(Arc::new(gpu));

    let stride = width * 4; // RGBA texture
    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch"),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        size: u64::from(stride) * u64::from(height),
        mapped_at_creation: false,
    });

    ev.run(move |event, _target, flow| match event {
        Event::RedrawRequested(_) => {
            let st = loop {
                match surface.get_current_texture() {
                    Ok(tex) => break tex,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        surface.configure(&device, &conf);
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
            let decode_op = decoder.decode_blocking(&image);

            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let copy_size = wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            };
            let icb = wgpu::ImageCopyBuffer {
                buffer: &scratch,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: None,
                },
            };
            enc.copy_texture_to_buffer(decode_op.texture().as_image_copy(), icb, copy_size);
            enc.copy_buffer_to_texture(icb, st.texture.as_image_copy(), copy_size);
            queue.submit([enc.finish()]);

            st.present();
        }
        Event::UserEvent(()) => win.request_redraw(),
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => *flow = ControlFlow::Exit,
        _ => {}
    });
}
