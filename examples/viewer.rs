use std::{env, fs, process, sync::Arc};

use compeg::{Decoder, Gpu, ImageData};
use wgpu::InstanceDescriptor;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_module(env!("CARGO_PKG_NAME"), log::LevelFilter::Trace)
        .parse_default_env()
        .init();

    let jpeg = match &*env::args().skip(1).collect::<Vec<_>>() {
        [path] => fs::read(&path)?,
        _ => {
            eprintln!("usage: viewer <file.jpeg>");
            process::exit(1);
        }
    };

    let image = ImageData::new(jpeg)?;

    let ev = EventLoop::new();
    let win = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(image.width(), image.height()))
        .build(&ev)?;

    let instance = wgpu::Instance::new(InstanceDescriptor::default());
    let surface = unsafe { instance.create_surface(&win)? };
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .ok_or_else(|| anyhow::anyhow!("no compatible graphics adapter found"))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Gpu::device_descriptor(), None))?;
    let (device, queue) = (Arc::new(device), Arc::new(queue));

    let mut conf = surface
        .get_default_config(&adapter, image.width(), image.height())
        .expect("incompatible surface, despite requiring one");
    conf.usage |= wgpu::TextureUsages::COPY_DST;
    surface.configure(&device, &conf);

    let gpu = Gpu::from_wgpu(device.clone(), queue.clone())?;
    let mut decoder = Decoder::new(Arc::new(gpu));

    let stride = image.width() * 4; // RGBA texture
    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scratch"),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        size: u64::from(stride) * u64::from(image.height()),
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

            decoder.start_decode(&image);

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
            enc.copy_texture_to_buffer(decoder.output().as_image_copy(), icb, copy_size);
            enc.copy_buffer_to_texture(icb, st.texture.as_image_copy(), copy_size);
            queue.submit([enc.finish()]);

            st.present();
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => *flow = ControlFlow::Exit,
        _ => {}
    });
}
