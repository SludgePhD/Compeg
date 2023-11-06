use std::{env, fs::File};

use anyhow::{bail, ensure};
use jpeg_encoder::{Encoder, SamplingFactor};

fn main() -> anyhow::Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();

    let [infile, outfile, rest @ ..] = &*args else {
        bail!("usage: enc <infile.png> <outfile.jpg>")
    };
    let ri = match rest {
        [] => 0,
        [ri] => ri.parse::<u16>()?,
        _ => bail!("usage: enc <infile.png> <outfile.jpg>"),
    };

    let infile = File::open(infile)?;
    let png = png::Decoder::new(infile);
    let mut reader = png.read_info()?;

    ensure!(reader.info().color_type == png::ColorType::Rgb);
    ensure!(reader.info().bit_depth == png::BitDepth::Eight);

    let width = reader.info().width;
    let height = reader.info().height;
    let mut buf = vec![0; width as usize * height as usize * 3];
    reader.next_frame(&mut buf)?;

    let outfile = File::create(outfile)?;
    let mut enc = Encoder::new(outfile, 100);
    enc.set_sampling_factor(SamplingFactor::R_4_2_2);
    enc.set_restart_interval(ri);

    enc.encode(
        &buf,
        width.try_into().unwrap(),
        height.try_into().unwrap(),
        jpeg_encoder::ColorType::Rgb,
    )?;

    Ok(())
}
