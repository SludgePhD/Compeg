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

    pub fn processed_scan_data(&self) -> &[u8] {
        bytemuck::cast_slice(&self.words)
    }

    pub fn start_positions(&self) -> &[u8] {
        bytemuck::cast_slice(&self.start_positions)
    }

    pub fn process(
        &mut self,
        scan_data: &[u8],
        expected_restart_intervals: u32,
    ) -> crate::Result<()> {
        // The RST markers are removed from `scan_data`, but each restart interval is padded to
        // start on a 32-bit boundary. That means at worst a 1-byte restart interval preceded by a
        // 2-byte RST marker will occupy one word and waste 1 additional byte in there, so we have
        // to allocate an extra 1/3rd of the input data length in the output buffer.
        let out_bytes = scan_data.len() + scan_data.len() / 3;
        self.words.resize((out_bytes + 3) / 4, 0);

        let start_pos_buffer_length = (expected_restart_intervals as usize).next_power_of_two();
        let start_pos_index_mask = start_pos_buffer_length - 1;
        self.start_positions.resize(start_pos_buffer_length, 0);

        let out: &mut [u8] = bytemuck::cast_slice_mut(&mut self.words);
        assert!(out.len() >= scan_data.len());

        let mut ri = 1;
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
                    Some(0xD0..=0xD7) => {
                        // RST marker.

                        // Align the next restart interval on a 4-byte boundary.
                        write_ptr = (write_ptr + 0b11) & !0b11;

                        self.start_positions[ri & start_pos_index_mask] = (write_ptr / 4) as u32;
                        ri += 1;
                    }
                    Some(inv) => {
                        return Err(Error::from(format!(
                            "invalid marker 0x{:02x} found in scan data",
                            inv
                        )));
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
        self.words.truncate((write_ptr + 3) / 4);

        if ri != expected_restart_intervals as usize {
            return Err(Error::from(format!(
                "restart interval count mismatch: counted {}, expected {}",
                ri, expected_restart_intervals
            )));
        }

        Ok(())
    }
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
}
