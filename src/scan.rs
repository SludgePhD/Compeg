//! Scan Data preprocessing.
//!
//! The scan data in a JPEG image is uploaded to GPU memory and then decoded by a compute shader.
//! But before that can happen, this module performs some preprocessing on the data. This
//! preprocessing involves:
//!
//! - Finding `RST` markers and saving their locations, so that the compute shader invocations know
//!   where to start.
//! - Finding `0xFF 0x00` byte stuffing sequences and replacing them with `0xFF`.
//! - Aligning the start of each restart interval with a `u32` boundary, so that the shader does
//!   not have to perform byte-wise alignment (WGSL does not have `u8`).

use std::{arch::x86_64::_mm_setzero_si128, cmp, mem::transmute};

use crate::error::Error;

pub struct ScanBuffer {
    /// Data is stored as `u32`s because that's WebGPU's smallest native integer type. Each
    /// restart interval begins on a word boundary, but the contained bytes are stored packed. This
    /// allows each shader invocation to load a `u32` from memory and start decoding it immediately,
    /// rather than having to shift bytes around first.
    words: Vec<u32>,
    /// Start offsets of the restart intervals in `words`.
    start_positions: Vec<u32>,
}

impl ScanBuffer {
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            start_positions: Vec::new(),
        }
    }

    pub fn process(
        &mut self,
        scan_data: &[u8],
        expected_restart_intervals: u32,
    ) -> crate::Result<()> {
        // The RST markers are removed from `scan_data`, but each restart interval is padded to
        // start on a 32-bit boundary. That means at worst a 1-byte restart interval preceded by a
        // 2-byte RST marker will occupy one word and waste 3 bytes in there.
        // This is 1 more byte than the input had, so we have to allocate an extra 1/3rd of the
        // input data length in the output buffer.
        let out_bytes = scan_data.len() + scan_data.len() / 3;
        self.words.resize((out_bytes + 3) / 4, 0);

        let start_pos_buffer_length = (expected_restart_intervals as usize).next_power_of_two();
        let start_pos_index_mask = start_pos_buffer_length - 1;
        self.start_positions.resize(start_pos_buffer_length, 0);
        assert!(self.start_positions.len() > start_pos_index_mask);

        let out: &mut [u8] = bytemuck::cast_slice_mut(&mut self.words);

        let res = preprocess_scalar(out, &mut self.start_positions, scan_data);

        self.words.truncate((res.bytes_out + 3) / 4);
        self.start_positions.truncate(res.ri);

        if res.ri != expected_restart_intervals as usize {
            return Err(Error::from(format!(
                "restart interval count mismatch: counted {}, expected {}",
                res.ri, expected_restart_intervals
            )));
        }

        Ok(())
    }

    /// Returns the preprocessed scan data, ready for upload to GPU memory.
    pub fn processed_scan_data(&self) -> &[u8] {
        bytemuck::cast_slice(&self.words)
    }

    /// Returns the computed start positions, ready for upload to GPU memory.
    pub fn start_positions(&self) -> &[u8] {
        bytemuck::cast_slice(&self.start_positions)
    }
}

struct PreprocessResult {
    ri: usize,
    bytes_out: usize,
}

#[inline(never)]
fn preprocess_scalar(
    out: &mut [u8],
    start_positions: &mut [u32],
    scan_data: &[u8],
) -> PreprocessResult {
    assert!(start_positions.len().is_power_of_two());
    let start_pos_mask = start_positions.len() - 1;

    let mut ri = 1; // One at index 0 is already written.
    let mut write_ptr = 0;
    let mut bytes = scan_data.iter().copied();
    loop {
        match bytes.next() {
            Some(0xff) => match bytes.next() {
                Some(0x00) => {
                    // Byte stuffing sequence, push only `0xFF` to the output.
                    out[write_ptr] = 0xff;
                    write_ptr += 1;
                }
                Some(_) => {
                    // RST marker. We don't check the exact marker type to improve perf. Only
                    // RST is valid here.

                    // Align the next restart interval on a 4-byte boundary.
                    write_ptr = (write_ptr + 0b11) & !0b11;

                    start_positions[ri & start_pos_mask] = (write_ptr / 4) as u32;
                    ri += 1;
                }
                None => break,
            },
            Some(byte) => {
                out[write_ptr] = byte;
                write_ptr += 1;
            }
            None => break,
        }
    }

    PreprocessResult {
        ri,
        bytes_out: write_ptr,
    }
}

#[target_feature(enable = "ssse3")]
#[inline(never)]
unsafe fn preprocess_ssse3(
    out: &mut [u8],
    start_positions: &mut [u32],
    mut scan_data: &[u8],
) -> PreprocessResult {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    fn fmt(v: __m128i) -> String {
        let bytes: [u8; 16] = unsafe { transmute(v) };
        format!("{:02x?}", bytes)
    }

    assert!(start_positions.len().is_power_of_two());
    let start_pos_mask = start_positions.len() - 1;

    let mut ri = 1; // One at index 0 is already written.
    let mut write_ptr = 0;
    let mut bytes_left = scan_data.len();
    let mut last_ff = 0; // 0xff is the last byte of the last iteration was 0xff, 0x00 otherwise
    while bytes_left >= 16 {
        bytes_left -= 16;

        // We could also do scalar processing to align the slice, and then use the aligned load intrinsic
        let mut block = _mm_loadu_si128(scan_data.as_ptr().cast());
        let mut bytecount = 16; // bytes in this block, left-aligned
        scan_data = &scan_data[16..];
        // [ aa 00 66 ff c8 77 ff 00 ... ]

        // Mark the locations of all 0x00 bytes.
        let zero_mask = _mm_cmpeq_epi8(block, _mm_set1_epi8(0));
        // [ 00 ff 00 00 00 00 00 ff ... ]

        // Mark the locations of all 0xFF bytes.
        let ff_mask = _mm_cmpeq_epi8(block, _mm_set1_epi8(-1));
        // [ 00 00 00 ff 00 00 ff 00 ... ]
        let cur_ff = _mm_extract_epi8(ff_mask, 15); // Is the last byte in this block 0xFF?

        // Shift the FF mask to align with the zero/nonzero masks
        let ff_mask = _mm_bslli_si128(ff_mask, 1);
        let ff_mask = _mm_insert_epi8(ff_mask, last_ff, 0);
        // [ 00 00 00 00 ff 00 00 ff ... ]
        last_ff = cur_ff;

        // Compare the shifted mask to compute which bytes are part of byte stuffing sequences (0xFF 0x00).
        let stuff_mask = _mm_and_si128(ff_mask, zero_mask);
        // [ 00 00 00 00 00 00 00 ff ...]
        // The 0xff marks the 0x00 in the orignal data, which has to be discarded.
        if _mm_movemask_epi8(stuff_mask) != 0 {
            // Byte stuffing is present. Use scalar code to remove it, since that's quite hard to do
            // in SIMD before AVX-512.
            let mut bytes: [u8; 16] = transmute(block);
            let stuff: [u8; 16] = transmute(stuff_mask);
            let mut i = 0;
            for (b, stuff) in bytes.into_iter().zip(stuff) {
                if stuff == 0 {
                    bytes[i] = b;
                    i += 1;
                }
            }
            block = transmute(bytes);
            bytecount -= 15 - i;
        }
        dbg!(fmt(block));

        let nonzero_mask = _mm_sub_epi8(_mm_set1_epi8(-1), zero_mask);
        // [ ff 00 ff ff ff ff ff 00 ... ]

        // Do the same with the nonzero mask to find all RST markers.
        let rst_mask = _mm_and_si128(ff_mask, nonzero_mask);
        // [ 00 00 00 00 ff 00 00 00 ... ]
        // 0xff marks the last byte *before* the next restart interval

        // Put the MSb of each byte into a bitmask. We can then use CLZ/CTZ to find out the length
        // of each RST interval.
        let mut bitmask = _mm_movemask_epi8(rst_mask) as u16;
        eprintln!("{:016b}", bitmask);
        let mut remaining = bytecount as u32;
        while remaining > 0 {
            let ctz = bitmask.trailing_zeros();
            dbg!(ctz);
            let mut length = cmp::min(ctz, remaining);
            let last = remaining == length;
            remaining = remaining.saturating_sub(length + 1);

            bitmask >>= length + 2;

            eprintln!("len={length} last={last}");
            if last && cur_ff != 0 {
                // If we're writing the last bytes in this block, but the last byte in the block is
                // `0xFF`, it might be part of another RST marker, so don't emit it.
                // FIXME: is this even needed? the shader won't care if an extra 0xff is at the end
                length -= 1;
            }

            let mask = (1_u128 << length) - 1;
            let mask = transmute(mask);
            _mm_maskmoveu_si128(block, mask, out[write_ptr..].as_mut_ptr().cast());

            // Align the next restart interval on a 4-byte boundary.
            write_ptr = (write_ptr + length as usize + 0b11) & !0b11;

            start_positions[ri & start_pos_mask] = (write_ptr / 4) as u32;
            ri += 1;
        }
    }

    _mm_sfence();

    PreprocessResult {
        ri,
        bytes_out: write_ptr,
    }
}

fn filter_scalar(elems: [u8; 16], drop_mask: u16) -> [u8; 16] {
    let mut out = [0; 16];
    let mut ptr = 0;
    for i in 0..16 {
        if drop_mask & (1 << i) == 0 {
            out[ptr] = elems[i];
            ptr += 1;
        }
    }
    out
}

#[test]
fn test_filter_scalar() {
    assert_eq!(filter_scalar([0xff; 16], !0), [0; 16]);
    assert_eq!(filter_scalar([0xff; 16], 0), [0xff; 16]);

    let elems = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    assert_eq!(
        filter_scalar(elems, 0x0000),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    );
    assert_eq!(
        filter_scalar(elems, 0x8000),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
    );
    assert_eq!(
        filter_scalar(elems, 0x0001),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    );
}

unsafe fn filter_vectorized(elems: [u8; 16], drop_mask: u16) -> [u8; 16] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // LUT for filtering 8 elements
    let mut lut = [[0u8; 8]; 256];
    for i in 0..=255u8 {
        let mut permut = [!0; 8];
        let mut nextidx = 0u8;
        for bit in 0..=7 {
            if i & (1 << bit) == 0 {
                // Keep element.
                permut[bit] = nextidx;
                nextidx += 1;
            }
        }

        lut[usize::from(i)] = permut;
    }

    let elems = transmute::<_, __m128i>(elems);

    todo!()

    /*
    let l = _mm_extract_epi64(elems, 0);
    let h = _mm_extract_epi64(elems, 1);
    let v0 = _mm_setzero_si128();
    let v0 = _mm_insert_epi64(v0, l, 0);
    let v1 = _mm_setzero_si128();
    let v1 = _mm_insert_epi64(v1, h, 0);
    let v0shuf = lut[usize::from(drop_mask >> 8)];
    let v0shuf = _mm_set_epi64x(transmute(v0shuf), 0);
    let v1shuf = lut[usize::from(drop_mask & 0xff)];
    let v1shuf = _mm_set_epi64x(transmute(v1shuf), 0);
    let low = _mm_shuffle_epi8(v0, v0shuf);
    let high = _mm_shuffle_epi8(v1, v1shuf);
    */
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(scan_data: &[u8], output: &[u8], start_positions: &[u32]) {
        let mut buf = ScanBuffer::new();
        buf.process(scan_data, start_positions.len() as u32)
            .unwrap();

        let bytes: &[u8] = bytemuck::cast_slice(&buf.words);
        assert_eq!(output, bytes);

        assert_eq!(buf.start_positions, start_positions);
    }

    fn check_err(scan_data: &[u8], start_positions: &[u32]) -> Error {
        let mut buf = ScanBuffer::new();
        buf.process(scan_data, start_positions.len() as u32)
            .unwrap_err()
    }

    #[test]
    fn process_scan_data() {
        check(&[0x12, 0x34, 0x56, 0x78], &[0x12, 0x34, 0x56, 0x78], &[0]);

        check(&[0xFF, 0xD0, 0xFF, 0xD0], &[], &[0, 0, 0]);

        let scan = &[0xFF, 0x00, 0x44, 0x55, 0xFF, 0xD0, 0x34];
        check(scan, &[0xFF, 0x44, 0x55, 0x00, 0x34, 0, 0, 0], &[0, 1]);
    }

    #[test]
    fn test_expanding_output() {
        // 3 bytes of input data get expanded to 4 bytes of output data. This tests that we allocate
        // enough space in the output buffer.
        check(
            &[0x11, 0xFF, 0xD0, 0x11, 0xFF, 0xD0, 0x11],
            &[0x11, 0, 0, 0, 0x11, 0, 0, 0, 0x11, 0, 0, 0],
            &[0, 1, 2],
        );
    }

    #[test]
    fn test_too_many_rst_markers() {
        // Data corruption can make the actual number of RST markers exceed the expected number.
        let err = check_err(&[0x11, 0xFF, 0xD0, 0x11, 0xFF, 0xD0, 0x11], &[0]);
        assert_eq!(
            err.to_string(),
            "restart interval count mismatch: counted 3, expected 1"
        );
    }

    #[test]
    fn test_ssse3() {
        let mut out = [0; 32];
        let mut start_pos = [0; 16];
        let scan_data = &[
            0xff, 0x00, 0x66, 0xff, 0xc8, 0x77, 0xff, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
            0xde, 0xff,
        ];
        let res = unsafe { preprocess_ssse3(&mut out, &mut start_pos, scan_data) };
        assert_eq!(&out[..res.bytes_out], &[0xff, 0x66, 0x00, 0x00]);
        assert_eq!(&start_pos[..res.ri], &[0,]);
    }
}
