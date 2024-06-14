// NB: these test don't match *exactly* what other decoders output, in
// particular around the 8x8 block boundaries. This only seems to happen with
// subsampled chroma, so it might be that we aren't using the correct sampling
// locations?
// The impact of this is not terribly big, and it makes the decoded image match
// the reference *exactly*, so it should be fine for now.

use std::{fs::File, sync::Arc};

use anyhow::{bail, ensure, Context};
use wgpu::{
    BufferDescriptor, BufferUsages, Extent3d, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout,
};

use crate::{Gpu, ImageData};

const ABS_TOLERANCE: u8 = 3;

fn check_impl(jpg: &str, png_reference: &str) -> anyhow::Result<()> {
    let jpg = format!("src/refs/{jpg}");
    let png_reference = format!("src/refs/{png_reference}");

    let infile = File::open(png_reference)?;
    let png = png::Decoder::new(infile);
    let mut reader = png.read_info()?;

    ensure!(reader.info().color_type == png::ColorType::Rgb);
    ensure!(reader.info().bit_depth == png::BitDepth::Eight);

    let width = reader.info().width;
    let height = reader.info().height;
    let mut buf = vec![0; width as usize * height as usize * 3];
    reader.next_frame(&mut buf)?;

    let jpeg = std::fs::read(jpg)?;
    let data = ImageData::new(&jpeg)?;
    ensure!(data.width() == width);
    ensure!(data.height() == height);

    let gpu = Arc::new(pollster::block_on(Gpu::open())?);
    let mut decoder = crate::Decoder::new(gpu.clone());
    let op = decoder.decode_blocking(&data);

    let buffer = gpu.device.create_buffer(&BufferDescriptor {
        label: None,
        size: u64::from(width) * u64::from(height) * 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        ImageCopyTexture {
            texture: op.texture(),
            mip_level: 0,
            origin: Default::default(),
            aspect: Default::default(),
        },
        ImageCopyBuffer {
            buffer: &buffer,
            layout: ImageDataLayout {
                bytes_per_row: Some(width * 4),
                ..Default::default()
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let index = gpu.queue.submit([enc.finish()]);

    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, Result::unwrap);

    gpu.device
        .poll(wgpu::MaintainBase::WaitForSubmissionIndex(index));

    let map = buffer.slice(..).get_mapped_range();

    compare(&buf, &map, width as usize, height as usize)?;

    drop(map);
    buffer.unmap();

    Ok(())
}

fn compare(rgb: &[u8], rgba: &[u8], width: usize, height: usize) -> anyhow::Result<()> {
    for y in 0..height {
        for x in 0..width {
            let mut ref_pixel = [0; 3];
            ref_pixel.copy_from_slice(&rgb[y * width * 3 + x * 3..][..3]);

            let mut actual = [0; 3];
            actual.copy_from_slice(&rgba[y * width * 4 + x * 4..][..3]);

            let max_diff = ref_pixel
                .iter()
                .copied()
                .zip(actual)
                .map(|(a, b)| u8::abs_diff(a, b))
                .max()
                .unwrap();

            if max_diff > ABS_TOLERANCE {
                bail!(
                    "image mismatch at {},{}: expected approx {:x?} got {:x?}",
                    x,
                    y,
                    ref_pixel,
                    actual,
                );
            }
        }
    }
    Ok(())
}

fn check(jpg: &str, png: &str) {
    check_impl(jpg, png)
        .context(format!("test image {jpg}, reference {png}"))
        .unwrap();
}

#[test]
fn reftests_4_2_2() {
    check("64x8-Ri-1.jpg", "64x8.png");
    check("64x8-Ri-2.jpg", "64x8.png");
}

#[test]
#[ignore = "non-4:2:2 subsampling is not yet implemented"]
fn reftests_4_4_4() {
    // FIXME: this test passes when uncommenting some checks in lib.rs, but the functionality is not implemented yet!
    check("64x8-Hi1-Vi1.jpg", "64x8.png");
}

#[test]
fn test_compare() {
    compare(&[0, 0, 0], &[0, 0, 0, 0], 1, 1).unwrap();
    compare(&[0, 0, 0], &[ABS_TOLERANCE, 0, 0, 0], 1, 1).unwrap();
    compare(&[ABS_TOLERANCE, 0, 0], &[0, 0, 0, 0], 1, 1).unwrap();
    compare(&[ABS_TOLERANCE, 0, 0], &[ABS_TOLERANCE, 0, 0, 0], 1, 1).unwrap();
    compare(&[ABS_TOLERANCE + 1, 0, 0], &[0, 0, 0, 0], 1, 1).unwrap_err();
    compare(&[0, 0, 0], &[ABS_TOLERANCE + 1, 0, 0, 0], 1, 1).unwrap_err();

    // Alpha is ignored
    compare(&[0, 0, 0], &[0, 0, 0, ABS_TOLERANCE + 1], 1, 1).unwrap();
}
