#![allow(dead_code)]

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

    /// Ensures that there are 31 readable bits in the buffer.
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
        assert!(n > 0 && n < 32 && n <= self.left);
        self.cur <<= n;
        self.cur |= self.next >> (32 - n);
        self.next <<= n;
        self.left -= n;
    }

    /// Peeks at the next `n` bits.
    fn peek(&self, n: u32) -> u32 {
        assert!(n > 0 && n < 32 && n <= self.left);
        self.cur >> (32 - n)
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
        assert_eq!(bitstream.peek(4), 0b0111);
        assert_eq!(bitstream.peek(8), 0b01110011);
        bitstream.consume(2);
        assert_eq!(bitstream.peek(2), 0b11);
        assert_eq!(bitstream.peek(6), 0b110011);
        assert_eq!(bitstream.peek(26), 0b110011_11110000_00000001_0101);
        assert_eq!(bitstream.peek(22), 0b110011_11110000_00000001);
        bitstream.consume(22);
        bitstream.refill();
        assert_eq!(bitstream.peek(8), 0b01010101);
        assert_eq!(bitstream.peek(16), 0b01010101_00001111); // next word
        bitstream.consume(16);
        bitstream.refill();
        assert_eq!(bitstream.peek(24), 0);
        bitstream.consume(24);
        bitstream.refill();
        assert_eq!(bitstream.peek(1), 1);
    }
}
