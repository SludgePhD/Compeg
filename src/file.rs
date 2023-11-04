//! JPEG/JFIF file parser.

#![allow(non_snake_case, dead_code)]

#[cfg(test)]
mod tests;

use std::{fmt, marker::PhantomData, mem};

use bytemuck::{AnyBitPattern, Pod, Zeroable};

use crate::error::{Error, Result};

pub struct JpegParser<'a> {
    reader: Reader<'a>,
}

impl<'a> JpegParser<'a> {
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        let mut reader = Reader { buf, position: 0 };
        if reader.read_u8()? != 0xFF || reader.read_u8()? != 0xD8 {
            return Err(Error::from(
                "JPEG image does not start with SOI marker".to_string(),
            ));
        }
        Ok(Self { reader })
    }

    /// Reads the next [`Segment`] from the JPEG data.
    ///
    /// This will not return `SOI`/`EOI` markers, since those are handled internally by the parser,
    /// and are not the start of any marker segment.
    ///
    /// Returns `Ok(None)` when the EOI marker is encountered, signaling the end of the image. There
    /// may be data stored after the EOI marker, which can be retrieved by calling
    /// [`JpegParser::remaining`].
    pub fn next_segment(&mut self) -> Result<Option<Segment<'a>>> {
        while self.reader.read_u8()? != 0xff {}

        let segment_offset = self.reader.position - 1;
        let marker = self.reader.read_u8()?;

        if marker == 0x00 {
            return Err(Error::from("invalid ff 00 marker".to_string()));
        }

        if marker == 0xD9 {
            // EOI marker
            if !self.reader.remaining().is_empty() {
                log::warn!(
                    "ignoring {} trailing bytes after EOI",
                    self.reader.remaining().len()
                );
            }

            return Ok(None);
        }

        // The standalone markers are SOI, EOI, TEM, and RSTn.
        // SOI is read in `new`, EOI is handled above, RSTn is invalid outside of the scan data
        // (which is emitted as part of the SOS segment), and TEM is invalid when encountered.
        // Every remaining marker (even an unknown one) is followed by the segment length.

        let length = usize::from(self.reader.read_length()?);
        let expected_end = self.reader.position + length;
        let mut reader = Reader {
            buf: &self.reader.buf[..expected_end],
            position: self.reader.position,
        };
        let kind = match marker {
            0xDB => Some(SegmentKind::Dqt(self.read_dqt(&mut reader)?)),
            0xC4 => Some(SegmentKind::Dht(self.read_dht(&mut reader)?)),
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                Some(SegmentKind::Sof(self.read_sof(marker, &mut reader)?))
            }
            0xDA => Some(SegmentKind::Sos(self.read_sos(&mut reader)?)),
            0xDD => Some(SegmentKind::Dri(self.read_dri(&mut reader)?)),
            0xE0..=0xEF => Some(SegmentKind::App(self.read_app(marker, &mut reader)?)),
            0xFE => Some(SegmentKind::Com(self.read_com(&mut reader)?)),
            _ => {
                self.reader.position = expected_end;
                None
            }
        };

        // The segment specified a bigger length than what we ended up reading. Skip the remaining
        // bytes and log a warning.
        if reader.position < expected_end {
            let remaining = expected_end - reader.position;
            log::warn!(
                "ff {:02x} segment specified a length of {} bytes, but {} remain after decoding",
                marker,
                length,
                remaining,
            );
            self.reader.position = expected_end;
        }

        Ok(Some(Segment {
            marker,
            raw_bytes: &self.reader.buf[segment_offset + 4..][..length],
            offset: segment_offset,
            kind,
        }))
    }

    /// Returns the remaining (unparsed) bytes of the input data.
    ///
    /// After retrieving a segment via [`JpegParser::next_segment`], the result of this method is
    /// the data immediately following that segment.
    pub fn remaining(&self) -> &'a [u8] {
        self.reader.remaining()
    }

    fn read_dqt(&mut self, reader: &mut Reader<'a>) -> Result<Dqt<'a>> {
        // The size of the DQT segment tells us how many quantization tables there are.
        // FIXME: does not support 16-bit qtables
        let count = reader.remaining().len() / mem::size_of::<QuantizationTable>();
        if count * mem::size_of::<QuantizationTable>() != reader.remaining().len() {
            log::warn!(
                "DQT segment with {} bytes should have been a multiple of {} bytes",
                reader.remaining().len(),
                mem::size_of::<QuantizationTable>()
            );
        }
        let qts = reader.read_objs(count)?;
        Ok(Dqt(qts))
    }

    fn read_dht(&mut self, reader: &mut Reader<'a>) -> Result<Dht<'a>> {
        const MIN_DHT_LEN: usize = 18; // Tc+Th + 16 length bytes + at least one symbol-length assignment

        let mut tables = Vec::new();

        while reader.remaining().len() >= MIN_DHT_LEN {
            let header: &DhtHeader = reader.read_obj()?;
            let values = reader.read_slice(header.num_values())?;
            tables.push(HuffmanTable {
                header,
                Vij: values,
            });
        }

        Ok(Dht { tables })
    }

    fn read_sof(&mut self, sof: u8, reader: &mut Reader<'a>) -> Result<Sof<'a>> {
        let P = reader.read_u8()?;
        let Y = reader.read_u16()?;
        let X = reader.read_u16()?;
        let num_components = reader.read_u8()?;
        let components = reader.read_objs::<FrameComponent>(num_components.into())?;
        Ok(Sof {
            sof: SofMarker(sof),
            P,
            Y,
            X,
            components,
        })
    }

    fn read_sos(&mut self, reader: &mut Reader<'a>) -> Result<Sos<'a>> {
        let num_components = reader.read_u8()?;
        let components = reader.read_objs(num_components.into())?;
        let Ss = reader.read_u8()?;
        let Se = reader.read_u8()?;
        let AhAl = reader.read_u8()?;

        self.reader.position = reader.position;

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
                    // Include all RST markers in the scan data.
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

    fn read_dri(&mut self, reader: &mut Reader<'a>) -> Result<Dri<'a>> {
        let Ri = reader.read_u16()?;
        Ok(Dri {
            Ri,
            _p: PhantomData,
        })
    }

    fn read_com(&mut self, reader: &mut Reader<'a>) -> Result<Com<'a>> {
        Ok(Com {
            com: reader.read_slice(reader.remaining().len())?,
        })
    }

    fn read_app(&mut self, marker: u8, reader: &mut Reader<'a>) -> Result<App<'a>> {
        let n = marker - 0xE0;

        let kind = match n {
            0 => self.read_jfif(reader)?.map(AppKind::Jfif),
            _ => None,
        };

        // Silence the "X bytes remain after decoding" warning for APP segments, since they contain
        // arbitrary data we don't always know about.
        reader.position = reader.buf.len() - 1;

        Ok(App { n, kind })
    }

    fn read_jfif(&mut self, reader: &mut Reader<'a>) -> Result<Option<Jfif<'a>>> {
        const JFIF: &[u8] = b"JFIF\0";

        if reader.read_slice(JFIF.len()).ok() != Some(JFIF) {
            return Ok(None); // Not a JFIF header.
        }

        let major_version = reader.read_u8()?;
        let minor_version = reader.read_u8()?;
        let unit = match reader.read_u8()? {
            0 => DensityUnit::None,
            1 => DensityUnit::DotsPerInch,
            2 => DensityUnit::DotsPerCm,
            e => {
                return Err(Error::from(format!(
                    "JFIF header specifies invalid density unit {e}"
                )))
            }
        };
        let xdensity = reader.read_u16()?;
        let ydensity = reader.read_u16()?;
        let xthumbnail = reader.read_u8()?;
        let ythumbnail = reader.read_u8()?;
        Ok(Some(Jfif {
            major_version,
            minor_version,
            unit,
            xdensity,
            ydensity,
            xthumbnail,
            ythumbnail,
            thumbnail: reader.read_slice(usize::from(xthumbnail) * usize::from(ythumbnail) * 3)?,
        }))
    }
}

#[derive(Debug)]
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

    fn read_objs<T: AnyBitPattern>(&mut self, count: usize) -> Result<&'a [T]> {
        assert_eq!(mem::align_of::<T>(), 1);

        // TODO: bounds check
        let byte_count = count * mem::size_of::<T>();
        let slice = bytemuck::cast_slice(&self.remaining()[..byte_count]);
        self.position += byte_count;
        Ok(slice)
    }

    fn read_length(&mut self) -> Result<u16> {
        // Length parameter is the length of the segment parameters, including the length parameter,
        // but excluding the FF xx marker.

        let len = self.read_u16()?;
        if len < 2 {
            return Err(Error::from(format!("invalid segment length {len}")));
        }
        if self.remaining().len() < (len - 2).into() {
            return Err(Error::from(
                "reached end of data while decoding JPEG stream".to_string(),
            ));
        }
        Ok(len - 2)
    }
}

/// A segment of a JPEG file, introduced by a `0xFF 0xXX` marker.
#[derive(Debug)]
pub struct Segment<'a> {
    marker: u8,
    raw_bytes: &'a [u8],
    offset: usize,
    kind: Option<SegmentKind<'a>>,
}

impl<'a> Segment<'a> {
    /// Returns the offset of the segment's `0xFF 0xXX` marker in the input buffer.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the value of the marker byte indicating the type of the segment.
    ///
    /// All segments begin with a `0xFF 0xXX` marker. This function returns the value of the `0xXX`
    /// byte, which identifies the type of the segment.
    #[inline]
    pub fn marker(&self) -> u8 {
        self.marker
    }

    /// The raw bytes making up this segment, exluding the `0xFF 0xXX` marker and the segment length
    /// indication.
    ///
    /// This only includes as many bytes as specified by the segment header, not any data following
    /// the segment, so for example for an SOS segment it does not include any of the entropy-coded
    /// data following the segment.
    #[inline]
    pub fn raw_bytes(&self) -> &[u8] {
        self.raw_bytes
    }

    #[inline]
    pub fn as_segment_kind(&self) -> Option<&SegmentKind<'a>> {
        self.kind.as_ref()
    }
}

/// An application-specific segment (`APPn`).
#[derive(Debug)]
pub struct App<'a> {
    n: u8,
    kind: Option<AppKind<'a>>,
}

impl<'a> App<'a> {
    /// Returns the type of APP marker (the `n` in `APPn`).
    ///
    /// This is an integer derived from the marker in the beginning of the APP segment. It is always
    /// in range `0..=15`.
    ///
    /// This ID can be used to partially identify the type of data stored in the APP segment. For
    /// example, the JFIF header always uses APP0, so this method would return 0 for them. However,
    /// APP0 can also be used for other purposes, so the JFIF header contains additional identifying
    /// information (a magic number/string).
    #[inline]
    pub fn n(&self) -> u8 {
        self.n
    }

    #[inline]
    pub fn as_app_kind(&self) -> Option<&AppKind<'a>> {
        self.kind.as_ref()
    }
}

/// Enumeration of the known `APPn` segments understood by this parser.
#[derive(Debug)]
#[non_exhaustive]
pub enum AppKind<'a> {
    Jfif(Jfif<'a>),
}

#[derive(Debug)]
pub struct Jfif<'a> {
    major_version: u8,
    minor_version: u8,
    unit: DensityUnit,
    xdensity: u16,
    ydensity: u16,
    xthumbnail: u8,
    ythumbnail: u8,
    thumbnail: &'a [u8],
}

impl<'a> Jfif<'a> {
    #[inline]
    pub fn major_version(&self) -> u8 {
        self.major_version
    }

    #[inline]
    pub fn minor_version(&self) -> u8 {
        self.minor_version
    }

    #[inline]
    pub fn unit(&self) -> DensityUnit {
        self.unit
    }

    #[inline]
    pub fn density_x(&self) -> u16 {
        self.xdensity
    }

    #[inline]
    pub fn density_y(&self) -> u16 {
        self.ydensity
    }

    #[inline]
    pub fn thumbnail_width(&self) -> u8 {
        self.xthumbnail
    }

    #[inline]
    pub fn thumbnail_height(&self) -> u8 {
        self.ythumbnail
    }

    #[inline]
    pub fn thumbnail_data(&self) -> &[u8] {
        self.thumbnail
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DensityUnit {
    None,
    DotsPerInch,
    DotsPerCm,
}

pub struct Com<'a> {
    com: &'a [u8],
}

impl<'a> fmt::Debug for Com<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Com(\"{}\")", self.com.escape_ascii())
    }
}

/// Enumeration of segment kinds understood by this parser.
#[derive(Debug)]
#[non_exhaustive]
pub enum SegmentKind<'a> {
    Dqt(Dqt<'a>),
    Dht(Dht<'a>),
    Dri(Dri<'a>),
    Sof(Sof<'a>),
    Sos(Sos<'a>),
    App(App<'a>),
    Com(Com<'a>),
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

/// **D**efine **Q**uantization **T**ables – sets one or more [`QuantizationTable`]s.
#[derive(Debug)]
pub struct Dqt<'a>(&'a [QuantizationTable]);

impl<'a> Dqt<'a> {
    #[inline]
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

/// **D**efine **R**estart **I**nterval.
///
/// This segment enables the use of *Restart Intervals* and sets the number of MCUs contained in
/// each of them ([`Dri::Ri`]).
///
/// *Restart Intervals* are often used by hardware JPEG encoders to allow parallelizing the encoding
/// process. They enable parallelization of the decoding step as well. Additionally, they make the
/// image more robust against data corruption, by preventing corruption from affecting more than the
/// restart interval it occurs in.
#[derive(Clone, Copy)]
pub struct Dri<'a> {
    Ri: u16,
    _p: PhantomData<&'a ()>,
}

impl<'a> Dri<'a> {
    /// Returns the number of MCUs contained in each restart interval.
    #[inline]
    pub fn Ri(&self) -> u16 {
        self.Ri
    }
}

impl<'a> fmt::Debug for Dri<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dri").field("Ri", &self.Ri).finish()
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
            Self::SOF3 => f.write_str("SOF3"),
            Self::SOF5 => f.write_str("SOF5"),
            Self::SOF6 => f.write_str("SOF6"),
            Self::SOF7 => f.write_str("SOF7"),
            Self::SOF9 => f.write_str("SOF9"),
            Self::SOF10 => f.write_str("SOF10"),
            Self::SOF11 => f.write_str("SOF11"),
            Self::SOF13 => f.write_str("SOF13"),
            Self::SOF14 => f.write_str("SOF14"),
            Self::SOF15 => f.write_str("SOF15"),
            _ => f
                .debug_tuple("SofMarker")
                .field(&format_args!("{:02x}", self.0))
                .finish(),
        }
    }
}

impl SofMarker {
    /// Baseline DCT.
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
