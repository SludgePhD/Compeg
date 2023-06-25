//! JPEG/JFIF file parser.

#![allow(non_snake_case)]

#[cfg(test)]
mod tests;

use std::{fmt, mem};

use bytemuck::{AnyBitPattern, Pod, Zeroable};

use crate::error::{Error, Result};

pub struct JpegParser<'a> {
    reader: Reader<'a>,
}

impl<'a> JpegParser<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            reader: Reader { buf, position: 0 },
        }
    }

    pub fn next_segment(&mut self) -> Result<Option<Segment<'a>>> {
        if self.reader.remaining().is_empty() {
            return Ok(None);
        }

        while self.reader.read_u8()? != 0xff {}

        let position = self.reader.position - 1;
        let marker = self.reader.read_u8()?;

        let kind = match marker {
            0x00 => return Err(Error::from("invalid ff 00 marker".to_string())),
            0xD8 => SegmentKind::Soi,
            0xD9 => SegmentKind::Eoi,
            0xDB => SegmentKind::Dqt(self.read_dqt()?),
            0xC4 => SegmentKind::Dht(self.read_dht()?),
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                SegmentKind::Sof(self.read_sof(marker)?)
            }
            0xDA => SegmentKind::Sos(self.read_sos()?),
            0xDD => SegmentKind::Dri(self.read_dri()?),
            _ => SegmentKind::Other {
                marker,
                data: self.reader.read_segment()?.remaining(),
            },
        };

        Ok(Some(Segment {
            pos: position,
            kind,
        }))
    }

    fn read_dqt(&mut self) -> Result<Dqt<'a>> {
        let mut seg = self.reader.read_segment()?;
        let inner = seg.read_remaining_objs::<QuantizationTable>()?;
        Ok(Dqt(inner))
    }

    fn read_dht(&mut self) -> Result<Dht<'a>> {
        const MIN_DHT_LEN: usize = 18; // Tc+Th + 16 length bytes + at least one symbol-length assignment

        let mut seg = self.reader.read_segment()?;
        let mut tables = Vec::new();

        while seg.remaining().len() >= MIN_DHT_LEN {
            let header: &DhtHeader = seg.read_obj()?;
            let values = seg.read_slice(header.num_values())?;
            tables.push(HuffmanTable {
                header,
                Vij: values,
            });
        }

        Ok(Dht { tables })
    }

    fn read_sof(&mut self, sof: u8) -> Result<Sof<'a>> {
        let mut seg = self.reader.read_segment()?;
        let P = seg.read_u8()?;
        let Y = seg.read_u16()?;
        let X = seg.read_u16()?;
        let num_components = seg.read_u8()?;
        let components = seg.read_objs::<FrameComponent>(num_components.into())?;
        Ok(Sof {
            sof: SofMarker(sof),
            P,
            Y,
            X,
            components,
        })
    }

    fn read_sos(&mut self) -> Result<Sos<'a>> {
        let mut seg = self.reader.read_segment()?;
        let num_components = seg.read_u8()?;
        let components = seg.read_objs(num_components.into())?;
        let Ss = seg.read_u8()?;
        let Se = seg.read_u8()?;
        let AhAl = seg.read_u8()?;

        // The scan itself can contain `RST` markers. We skip them and include them in the scan
        // data.
        let data_start = self.reader.position;
        loop {
            while self.reader.peek_u8(0)? != 0xff {
                self.reader.position += 1;
            }

            let mut offset = 1;
            let mut byte = self.reader.peek_u8(offset)?;
            while byte == 0xff {
                offset += 1;
                byte = self.reader.peek_u8(offset)?;
            }

            match byte {
                0x00 | 0xD0..=0xD7 => {
                    // Include all RST markers in the scan data, since that's what VA-API expects.
                    self.reader.position += offset + 1;
                }
                _ => {
                    self.reader.position += offset - 1;
                    break;
                }
            }
        }

        let data_end = self.reader.position;

        Ok(Sos {
            components,
            Ss,
            Se,
            AhAl,
            data_offset: data_start,
            data: &self.reader.buf[data_start..data_end],
        })
    }

    fn read_dri(&mut self) -> Result<Dri> {
        let mut seg = self.reader.read_segment()?;
        let Ri = seg.read_u16()?;
        Ok(Dri { Ri })
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> &'a [u8] {
        &self.buf[self.position..]
    }

    fn peek_u8(&self, offset: usize) -> Result<u8> {
        if self.position + offset >= self.buf.len() {
            Err(Error::from(
                "reached end of data while decoding JPEG stream".to_string(),
            ))
        } else {
            let byte = self.buf[self.position + offset];
            Ok(byte)
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        let res = self.peek_u8(0);
        if res.is_ok() {
            self.position += 1;
        }
        res
    }

    fn read_u16(&mut self) -> Result<u16> {
        let b = [self.read_u8()?, self.read_u8()?];
        Ok(u16::from_be_bytes(b))
    }

    fn read_slice(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.remaining().len() < count {
            Err(Error::from(
                "reached end of data while decoding JPEG stream".to_string(),
            ))
        } else {
            let slice = &self.remaining()[..count];
            self.position += count;
            Ok(slice)
        }
    }

    fn read_obj<T: AnyBitPattern>(&mut self) -> Result<&'a T> {
        assert_eq!(mem::align_of::<T>(), 1);

        if self.remaining().len() < mem::size_of::<T>() {
            return Err(Error::from(
                "reached end of data while decoding JPEG stream".to_string(),
            ));
        }

        let object = bytemuck::from_bytes(&self.remaining()[..mem::size_of::<T>()]);

        self.position += mem::size_of::<T>();
        Ok(object)
    }

    fn read_remaining_objs<T: AnyBitPattern>(&mut self) -> Result<&'a [T]> {
        let count = self.remaining().len() / mem::size_of::<T>();
        self.read_objs(count)
    }

    fn read_objs<T: AnyBitPattern>(&mut self, count: usize) -> Result<&'a [T]> {
        assert_eq!(mem::align_of::<T>(), 1);

        let byte_count = count * mem::size_of::<T>();
        let slice = bytemuck::cast_slice(&self.remaining()[..byte_count]);
        self.position += byte_count;
        Ok(slice)
    }

    fn read_length(&mut self) -> Result<u16> {
        let len = self.read_u16()?;
        if len < 2 {
            return Err(Error::from(format!("invalid segment length {len}")));
        }
        Ok(len)
    }

    fn read_segment(&mut self) -> Result<Reader<'a>> {
        let len = usize::from(self.read_length()?) - 2;
        if self.remaining().len() < len {
            return Err(Error::from(
                "reached end of data while decoding JPEG stream".to_string(),
            ));
        }

        let r = Reader {
            buf: &self.remaining()[..len],
            position: 0,
        };
        self.position += len;
        Ok(r)
    }
}

#[derive(Debug)]
pub struct Segment<'a> {
    /// Offset of the segment's marker in the input buffer.
    pub pos: usize,
    pub kind: SegmentKind<'a>,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum SegmentKind<'a> {
    Dqt(Dqt<'a>),
    Dht(Dht<'a>),
    Dri(Dri),
    Sof(Sof<'a>),
    Sos(Sos<'a>),
    Soi,
    Eoi,
    Other { marker: u8, data: &'a [u8] },
}

#[derive(Copy, Clone, AnyBitPattern)]
#[repr(C)]
pub struct QuantizationTable {
    PqTq: u8,
    Qk: [u8; 64],
}

impl QuantizationTable {
    /// Returns the quantization table element precision.
    ///
    /// - 0: 8-bit `Qk` values
    /// - 1: 16-bit `Qk` values
    ///
    /// Must be 0 when the sample precision `P` is 8 bits.
    #[inline]
    pub fn Pq(&self) -> u8 {
        self.PqTq >> 4
    }

    /// Returns the destination identifier (0-3).
    #[inline]
    pub fn Tq(&self) -> u8 {
        self.PqTq & 0xf
    }

    /// Returns the quantization table elements.
    #[inline]
    pub fn Qk(&self) -> &[u8; 64] {
        &self.Qk
    }
}

impl fmt::Debug for QuantizationTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuantizationTable")
            .field("Pq", &self.Pq())
            .field("Tq", &self.Tq())
            .field("Qk", &self.Qk)
            .finish()
    }
}

/// **DQT** Define Quantization Table – defines one or more [`QuantizationTable`]s.
#[derive(Debug)]
pub struct Dqt<'a>(&'a [QuantizationTable]);

impl<'a> Dqt<'a> {
    pub fn tables(&self) -> impl Iterator<Item = &QuantizationTable> {
        self.0.iter()
    }
}

#[derive(Clone, Copy, AnyBitPattern)]
#[repr(C)]
struct DhtHeader {
    TcTh: u8,
    Li: [u8; 16],
}

impl DhtHeader {
    fn num_values(&self) -> usize {
        self.Li.iter().map(|l| *l as usize).sum()
    }
}

pub struct HuffmanTable<'a> {
    header: &'a DhtHeader,
    Vij: &'a [u8],
}

impl<'a> HuffmanTable<'a> {
    /// Returns the table class (0 = DC, 1 = AC).
    #[inline]
    pub fn Tc(&self) -> u8 {
        self.header.TcTh >> 4
    }

    /// Returns the table destination identifier (0-3).
    #[inline]
    pub fn Th(&self) -> u8 {
        self.header.TcTh & 0xf
    }

    /// Returns an array containing the number of codes of each length.
    #[inline]
    pub fn Li(&self) -> &[u8; 16] {
        &self.header.Li
    }

    /// Returns the values associated with each huffman code.
    #[inline]
    pub fn Vij(&self) -> &[u8] {
        &self.Vij
    }
}

impl<'a> fmt::Debug for HuffmanTable<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HuffmanTable")
            .field("Tc", &self.Tc())
            .field("Th", &self.Th())
            .field("Li", &self.Li())
            .field("Vij", &self.Vij)
            .finish()
    }
}

/// **DHT** Define Huffman Tables – defines one or more [`HuffmanTable`]s.
#[derive(Debug)]
pub struct Dht<'a> {
    tables: Vec<HuffmanTable<'a>>,
}

impl<'a> Dht<'a> {
    pub fn tables(&self) -> impl Iterator<Item = &HuffmanTable<'a>> {
        self.tables.iter()
    }
}

/// **DRI** Define Restart Interval.
#[derive(Debug, Clone, Copy, AnyBitPattern)]
pub struct Dri {
    Ri: u16,
}

impl Dri {
    /// Returns the number of MCUs contained in each restart interval.
    #[inline]
    pub fn Ri(&self) -> u16 {
        self.Ri
    }
}

/// **SOF** Start Of Frame
#[derive(Debug)]
pub struct Sof<'a> {
    /// The SOF marker.
    sof: SofMarker,
    /// Sample precision in bits.
    P: u8,
    Y: u16,
    X: u16,
    components: &'a [FrameComponent],
}

impl<'a> Sof<'a> {
    #[inline]
    pub fn sof(&self) -> SofMarker {
        self.sof
    }

    /// Returns the sample precision in bits.
    #[inline]
    pub fn P(&self) -> u8 {
        self.P
    }

    /// Returns the number of lines in the image (the height of the frame).
    #[inline]
    pub fn Y(&self) -> u16 {
        self.Y
    }

    /// Returns the number of samples per line (the width of the frame).
    #[inline]
    pub fn X(&self) -> u16 {
        self.X
    }

    #[inline]
    pub fn components(&self) -> &'a [FrameComponent] {
        self.components
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SofMarker(u8);

impl fmt::Debug for SofMarker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::SOF0 => f.write_str("SOF0"),
            Self::SOF1 => f.write_str("SOF1"),
            Self::SOF2 => f.write_str("SOF2"),
            _ => f
                .debug_tuple("SofMarker")
                .field(&format_args!("{:02x}", self.0))
                .finish(),
        }
    }
}

#[allow(dead_code)]
impl SofMarker {
    /// Baseline DCT.
    ///
    /// This is the *only* type of image that we support.
    pub const SOF0: Self = Self(0xC0);
    /// Extended Sequential DCT.
    pub const SOF1: Self = Self(0xC1);
    /// Progressive DCT.
    pub const SOF2: Self = Self(0xC2);
    /// Lossless sequential.
    pub const SOF3: Self = Self(0xC3);
    /// Differential sequential DCT.
    pub const SOF5: Self = Self(0xC5);
    /// Differential progressive DCT.
    pub const SOF6: Self = Self(0xC6);
    /// Differential lossless (sequential).
    pub const SOF7: Self = Self(0xC7);
    /// Reserved for JPEG extensions.
    pub const JPG: Self = Self(0xC8);
    /// Extended sequential DCT.
    pub const SOF9: Self = Self(0xC9);
    /// Progressive DCT.
    pub const SOF10: Self = Self(0xCA);
    /// Lossless (sequential).
    pub const SOF11: Self = Self(0xCB);
    /// Differential sequential DCT.
    pub const SOF13: Self = Self(0xCD);
    /// Differential progressive DCT.
    pub const SOF14: Self = Self(0xCE);
    /// Differential lossless (sequential).
    pub const SOF15: Self = Self(0xCF);
}

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct FrameComponent {
    Ci: u8,
    HiVi: u8,
    Tqi: u8,
}

impl FrameComponent {
    /// Returns this component's component identifier.
    ///
    /// The component identifier is an arbitrary 8-bit value that uniquely identifies each
    /// component. The scan header ([`Sos`]) refers to components using their identifier.
    #[inline]
    pub fn Ci(&self) -> u8 {
        self.Ci
    }

    /// Returns the horizontal subsampling factor for this component.
    ///
    /// This number also specifies the number of "horizontal data units" encoded in each MCU.
    #[inline]
    pub fn Hi(&self) -> u8 {
        self.HiVi >> 4
    }

    /// Returns the vertical subsampling factor for this component.
    ///
    /// This number also specifies the number of "vertical data units" encoded in each MCU.
    #[inline]
    pub fn Vi(&self) -> u8 {
        self.HiVi & 0xf
    }

    /// Returns the index of the quantization table to use for this component.
    ///
    /// Valid values are 0-3.
    #[inline]
    pub fn Tqi(&self) -> u8 {
        self.Tqi
    }
}

impl fmt::Debug for FrameComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameComponent")
            .field("Ci", &self.Ci)
            .field("Hi", &self.Hi())
            .field("Vi", &self.Vi())
            .field("Tqi", &self.Tqi)
            .finish()
    }
}

/// **SOS** Start Of Scan – a scan header, followed by entropy-coded scan data.
pub struct Sos<'a> {
    components: &'a [ScanComponent],
    Ss: u8,
    Se: u8,
    AhAl: u8,
    data_offset: usize,
    data: &'a [u8],
}

impl<'a> Sos<'a> {
    #[inline]
    pub fn components(&self) -> &[ScanComponent] {
        self.components
    }

    #[inline]
    pub fn Ss(&self) -> u8 {
        self.Ss
    }

    #[inline]
    pub fn Se(&self) -> u8 {
        self.Se
    }

    #[inline]
    pub fn Ah(&self) -> u8 {
        self.AhAl >> 4
    }

    #[inline]
    pub fn Al(&self) -> u8 {
        self.AhAl & 0xf
    }

    /// Returns the offset of the scan data in the original JPEG stream.
    #[inline]
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Returns the data in this scan, including any contained `RST` markers.
    #[inline]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
}

impl<'a> fmt::Debug for Sos<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sos")
            .field("components", &self.components)
            .field("Ss", &self.Ss)
            .field("Se", &self.Se)
            .field("Ah", &self.Ah())
            .field("Al", &self.Al())
            .field("data", &self.data)
            .finish()
    }
}

#[derive(Clone, Copy, AnyBitPattern)]
#[repr(C)]
pub struct ScanComponent {
    Csj: u8,
    TdjTaj: u8,
}

impl ScanComponent {
    /// Returns the scan component selector.
    #[inline]
    pub fn Csj(&self) -> u8 {
        self.Csj
    }

    /// Returns the DC entropy coding table destination selector.
    #[inline]
    pub fn Tdj(&self) -> u8 {
        self.TdjTaj >> 4
    }

    /// Returns the AC entropy coding table destination selector.
    #[inline]
    pub fn Taj(&self) -> u8 {
        self.TdjTaj & 0xf
    }
}

impl fmt::Debug for ScanComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanComponent")
            .field("Csj", &self.Csj)
            .field("Tdj", &self.Tdj())
            .field("Taj", &self.Taj())
            .finish()
    }
}
