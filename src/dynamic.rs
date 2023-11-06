//! Utilities for dynamically reallocated GPU resources.

use std::{borrow::Cow, cmp, ops::Deref, sync::Arc};

use bytemuck::NoUninit;
use wgpu::*;

use crate::Gpu;

/// A GPU buffer that is automatically enlarged when too small.
pub struct DynamicBuffer {
    gpu: Arc<Gpu>,
    name: Cow<'static, str>,
    buffer: Buffer,
    /// The generation counter starts at 0 and is incremented every time the underlying [`Buffer`]
    /// is reallocated to make more space. It is used for change detection, in order to recreate
    /// the [`BindGroup`] this buffer is used in.
    generation: u64,
}

impl DynamicBuffer {
    pub fn new(gpu: Arc<Gpu>, name: impl Into<Cow<'static, str>>, usages: BufferUsages) -> Self {
        let name = name.into();

        Self {
            buffer: gpu.device.create_buffer(&BufferDescriptor {
                label: Some(&name),
                size: 0,
                usage: usages | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            generation: 0,
            gpu,
            name,
        }
    }

    pub fn reserve(&mut self, capacity: BufferAddress) {
        if self.buffer.size() < capacity {
            log::debug!(
                "recreating DynamicBuffer '{}' at {} bytes",
                self.name,
                capacity
            );
            let new_buffer = self.gpu.device.create_buffer(&BufferDescriptor {
                label: Some(&self.name),
                size: capacity,
                usage: self.buffer.usage() | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Schedule a copy from the old to the new buffer.
            // TODO: this might be unneeded
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, self.buffer.size());
            self.gpu.queue.submit([enc.finish()]);

            self.buffer = new_buffer;
            self.generation += 1;
        }
    }

    pub fn write<T: NoUninit>(&mut self, data: &[T]) {
        self.write_bytes(bytemuck::cast_slice(data))
    }

    fn write_bytes(&mut self, data: &[u8]) {
        self.reserve(data.len() as u64);
        self.gpu.queue.write_buffer(&self.buffer, 0, data);
    }

    pub fn as_resource(&self) -> DynamicBindingResource<'_> {
        DynamicBindingResource::Buffer(self)
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// A buffer for downloading data from a [`DynamicBuffer`].
///
/// This is useful for debugging what happens on the GPU, but is otherwise unused.
#[allow(dead_code)]
pub struct DownloadBuffer {
    gpu: Arc<Gpu>,
    buffer: Buffer,
    generation: u64,
}

#[allow(dead_code)]
impl DownloadBuffer {
    pub fn new(gpu: Arc<Gpu>) -> Self {
        Self {
            buffer: gpu.device.create_buffer(&BufferDescriptor {
                label: None,
                mapped_at_creation: false,
                size: 0,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            }),
            generation: 0,
            gpu,
        }
    }

    /// Returns a mappable buffer at least as large as `buffer`.
    fn buffer(&mut self, buffer: &DynamicBuffer) -> &Buffer {
        if self.buffer.size() < buffer.buffer.size() {
            self.buffer = self.gpu.device.create_buffer(&BufferDescriptor {
                label: Some(&format!("{} (staging)", buffer.name)),
                mapped_at_creation: false,
                size: buffer.buffer.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            });
            self.generation += 1;
        }

        &self.buffer
    }

    /// Schedules a copy from `src` into this [`DownloadBuffer`] with `enc`, and requests the
    /// [`DownloadBuffer`] to be mapped.
    pub fn download(&mut self, src: &DynamicBuffer, enc: &mut CommandEncoder) {
        let buffer = self.buffer(src);
        let copy_size = cmp::min(buffer.size(), src.buffer.size());
        enc.copy_buffer_to_buffer(&src.buffer, 0, buffer, 0, copy_size);
    }

    pub fn map(&self) {
        self.buffer
            .slice(..)
            .map_async(MapMode::Read, Result::unwrap);
    }

    /// Obtains a mapped view of the buffer.
    pub fn mapped(&self) -> Mapped<'_> {
        let view = self.buffer.slice(..).get_mapped_range();
        Mapped {
            buffer: &self.buffer,
            view: Some(view),
        }
    }
}

pub struct Mapped<'a> {
    buffer: &'a Buffer,
    view: Option<BufferView<'a>>,
}

impl<'a> Deref for Mapped<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.view.as_ref().unwrap()
    }
}

impl<'a> Drop for Mapped<'a> {
    fn drop(&mut self) {
        self.view.take();
        self.buffer.unmap();
    }
}

/// An on-GPU [`Texture`] that is automatically reallocated when its dimensions are too small.
pub struct DynamicTexture {
    gpu: Arc<Gpu>,
    name: Cow<'static, str>,
    texture: Texture,
    view: TextureView,
    /// The generation counter starts at 0 and is incremented every time the underlying [`Texture`]
    /// is reallocated to make more space. It is used for change detection, in order to recreate
    /// the [`BindGroup`] this texture is used in.
    generation: u64,
}

impl DynamicTexture {
    pub fn new(
        gpu: Arc<Gpu>,
        name: impl Into<Cow<'static, str>>,
        usage: TextureUsages,
        format: TextureFormat,
    ) -> Self {
        let name = name.into();

        let texture = gpu.device.create_texture(&TextureDescriptor {
            label: Some(&name),
            size: Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage,
            view_formats: &[format.add_srgb_suffix(), format.remove_srgb_suffix()],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some(&name),
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        Self {
            texture,
            view,
            generation: 0,
            gpu,
            name,
        }
    }

    /// Ensures that the texture can store at least `width x height` texels.
    pub fn reserve(&mut self, width: u32, height: u32) -> bool {
        if width > self.texture.size().width || height > self.texture.size().height {
            log::debug!(
                "recreating DynamicTexture '{}' at {}x{}",
                self.name,
                width,
                height
            );
            self.texture = self.gpu.device.create_texture(&TextureDescriptor {
                label: Some(&self.name),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: self.texture.format(),
                usage: self.texture.usage(),
                view_formats: &[
                    self.texture.format().add_srgb_suffix(),
                    self.texture.format().remove_srgb_suffix(),
                ],
            });
            self.view = self.texture.create_view(&TextureViewDescriptor {
                label: Some(&self.name),
                ..Default::default()
            });
            self.generation += 1;
            true
        } else {
            false
        }
    }

    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    pub fn as_resource(&self) -> DynamicBindingResource<'_> {
        DynamicBindingResource::Texture(self)
    }
}

/// A [`BindGroup`] that is automatically recreated when its contents change.
pub struct DynamicBindGroup {
    gpu: Arc<Gpu>,
    layout: Arc<BindGroupLayout>,
    name: Cow<'static, str>,
    bind_group: Option<BindGroup>,
    /// Stores generation counters for the resources used in the `bind_group` above.
    generations: Vec<u64>,
}

impl DynamicBindGroup {
    pub fn new(
        gpu: Arc<Gpu>,
        layout: Arc<BindGroupLayout>,
        name: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            gpu,
            layout,
            name: name.into(),
            bind_group: None,
            generations: Vec::new(),
        }
    }

    /// Returns the cached [`BindGroup`], or recreates it if necessary.
    ///
    /// The `resources` array passed to this method must refer to the same resources on every call.
    pub fn bind_group(&mut self, resources: &[DynamicBindingResource<'_>]) -> &BindGroup {
        self.generations.resize(resources.len(), !0);
        let any_changed = self
            .generations
            .iter()
            .zip(resources)
            .any(|(gen, res)| *gen != res.generation());
        if any_changed || self.bind_group.is_none() {
            for (gen, res) in self.generations.iter_mut().zip(resources) {
                *gen = res.generation();
            }

            log::debug!(
                "recreating DynamicBindGroup '{}' with {} resources",
                self.name,
                resources.len()
            );
            let entries: Vec<_> = resources
                .iter()
                .enumerate()
                .map(|(i, res)| BindGroupEntry {
                    binding: i.try_into().unwrap(),
                    resource: res.resource(),
                })
                .collect();
            self.bind_group = Some(self.gpu.device.create_bind_group(&BindGroupDescriptor {
                label: Some(&self.name),
                layout: &self.layout,
                entries: &entries,
            }));
        }

        self.bind_group.as_ref().unwrap()
    }
}

/// A bind group resource that can be used in a [`DynamicBindGroup`].
pub enum DynamicBindingResource<'a> {
    Buffer(&'a DynamicBuffer),
    Texture(&'a DynamicTexture),
    /// A static [`BindingResource`] that does *not* support dynamic reallocation.
    ///
    /// If this is used, the underlying resource *must* be the same for every call to
    /// [`DynamicBindGroup::bind_group`].
    Static(BindingResource<'a>),
}

impl<'a> DynamicBindingResource<'a> {
    fn generation(&self) -> u64 {
        match self {
            DynamicBindingResource::Buffer(b) => b.generation,
            DynamicBindingResource::Texture(t) => t.generation,
            DynamicBindingResource::Static(_) => 0,
        }
    }

    fn resource(&self) -> BindingResource<'_> {
        match self {
            DynamicBindingResource::Buffer(b) => b.buffer.as_entire_binding(),
            DynamicBindingResource::Texture(t) => BindingResource::TextureView(&t.view),
            DynamicBindingResource::Static(s) => s.clone(),
        }
    }
}

impl<'a> From<BindingResource<'a>> for DynamicBindingResource<'a> {
    fn from(value: BindingResource<'a>) -> Self {
        Self::Static(value)
    }
}
