#![allow(dead_code)]

use crate::huffman::TableData;

// bit stream using only 32-bit integers, as an "exercise" for doing the same in WGSL
struct BitStream<'a> {
    words: &'a [u32],
    cur: u32,
    next: u32,
    left: u32, // 0-64 bits left in `cur` and `next`
}

impl<'a> BitStream<'a> {
    fn new(words: &'a [u32]) -> Self {
        let mut this = Self {
            words,
            cur: 0,
            next: 0,
            left: 0,
        };
        this.refill();
        this
    }

    /// Ensures that there are 32 readable bits in the buffer.
    fn refill(&mut self) {
        if self.left < 32 {
            let w = self.words[0];
            // open-coded `to_be()`
            let w = (w & 0x000000ff) << 24
                | (w & 0x0000ff00) << 8
                | (w & 0x00ff0000) >> 8
                | (w & 0xff000000) >> 24;
            self.words = &self.words[1..];

            self.cur |= w >> self.left;
            self.next = w.checked_shl(32 - self.left).unwrap_or(0);
            self.left += 32;
        }
    }

    fn consume(&mut self, n: u32) {
        assert!(n < 32 && n <= self.left);
        self.cur <<= n;
        self.cur |= (self.next >> 1) >> (31 - n);
        self.next <<= n;
        self.left -= n;
    }

    /// Peeks at the next `n` bits.
    fn peek(&self, n: u32) -> u32 {
        assert!(n < 32 && n <= self.left);
        (self.cur >> 1) >> (31 - n)
    }

    fn huffdecode(&mut self, table: &TableData) -> u8 {
        assert!(self.left >= 16);
        let res = table.lookup((self.cur >> 16) as u16);
        self.consume(res.bits().into());
        res.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitstream() {
        let mut bitstream = BitStream::new(&[
            0b01010101_00000001_11110000_01110011,
            0b00001111,
            0b10000000,
        ]);
        assert_eq!(bitstream.peek(2), 0b01);
        assert_eq!(bitstream.peek(0), 0);
        assert_eq!(bitstream.peek(4), 0b0111);
        bitstream.consume(0);
        assert_eq!(bitstream.peek(8), 0b01110011);
        bitstream.consume(0);
        bitstream.consume(2);
        assert_eq!(bitstream.peek(2), 0b11);
        assert_eq!(bitstream.peek(0), 0);
        bitstream.consume(0);
        assert_eq!(bitstream.peek(6), 0b110011);
        assert_eq!(bitstream.peek(26), 0b110011_11110000_00000001_0101);
        assert_eq!(bitstream.peek(22), 0b110011_11110000_00000001);
        bitstream.consume(22);
        bitstream.refill();
        assert_eq!(bitstream.peek(0), 0);
        assert_eq!(bitstream.peek(8), 0b01010101);
        assert_eq!(bitstream.peek(16), 0b01010101_00001111); // next word
        bitstream.consume(16);
        bitstream.refill();
        assert_eq!(bitstream.peek(24), 0);
        bitstream.consume(24);
        bitstream.refill();
        assert_eq!(bitstream.peek(1), 1);
        assert_eq!(bitstream.peek(0), 0);
        bitstream.consume(0);
        assert_eq!(bitstream.peek(1), 1);
        assert_eq!(bitstream.peek(0), 0);
    }

    #[test]
    fn test_decode() {
        // Taken from the test JPEG.
        let scan_data: &[u8] = &[
            0xEB, 0x77, 0x62, 0x80, 0x01, 0x05, 0x87, 0xAF, 0x22, 0x80, 0x3F, 0xFF,
        ];
        let table = TableData::build(
            &[0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            &[
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
            ],
        );

        let mut bs = BitStream::new(bytemuck::cast_slice(scan_data));
        let dccat = bs.huffdecode(&table);
        let diff = bs.peek(dccat.into());
        bs.consume(dccat.into());

        let diff = huff_extend(diff as i32, dccat as i32);
        assert_eq!(diff, 45);
    }

    fn huff_extend(x: i32, s: i32) -> i32 {
        let vt = 1 << (s - 1);
        if x < vt {
            let vt = (-1 << s) + 1;
            x + vt
        } else {
            x
        }
    }
}
