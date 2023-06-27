use core::fmt;
use std::mem;

use bytemuck::{Pod, Zeroable};

const LUT_LEN: usize = u16::MAX as usize + 1; // u16::MAX must be a valid index (16 1 bits is a valid huffman code)

pub struct TableData {
    // *MASSIVELY* wasteful encoding, but with mediocre PC ports requiring 12 GB of VRAM I think I'll be fine :)
    codes: Box<[LookupResult; LUT_LEN]>,
}

impl TableData {
    #[allow(dead_code)]
    pub fn build(num_codes_per_length: &[u8; 16], codes: &[u8]) -> Self {
        // First, generate all huffman codes from the inputs.
        let mut out = Box::new([LookupResult::new(0, 0); LUT_LEN]);

        // The following is similar in function to the flowcharts in Annex C
        // (`Generate_size_table` and `Generate_code_table`)
        let mut next_code = 0u16;
        let mut code_iter = codes.iter();
        for (code_length, &code_count) in num_codes_per_length.iter().enumerate() {
            let code_length = (code_length + 1) as u8; // 1-based

            next_code <<= 1;

            for _ in 0..code_count {
                let lookup_result = LookupResult::new(code_length, *code_iter.next().unwrap());

                // Because we want to be able to access this `LookupResult` by indexing an array
                // with a `u16`, the most significant bits of which contain the huffman code,
                // we need to set multiple slots in that table to the code.
                let padded_code = next_code << (16 - code_length);

                // For a 16-bit code, there's only 1 slot, for a 2-bit code, there's 2^14 slots to set.
                let duplicates = 1 << (16 - code_length);
                for l in 0..duplicates {
                    out[usize::from(padded_code | l)] = lookup_result;
                }
                next_code += 1;
            }
        }

        Self { codes: out }
    }

    pub fn lookup_msb(&self, msb: u16) -> LookupResult {
        self.codes[msb as usize]
    }

    fn raw(&self) -> &[LookupResult; LUT_LEN] {
        &self.codes
    }

    fn iter(&self) -> impl Iterator<Item = (u16, LookupResult)> {
        let mut out = Vec::new();

        // Starting at all-0, iterate through the table while skipping over long runs of duplicate entries.
        let mut cur = 0u16;
        loop {
            let code = self.codes[usize::from(cur)];
            if code.bits == 0 {
                break;
            }

            out.push((cur >> (16 - code.bits), code));
            cur += 1 << (16 - code.bits);
        }

        out.into_iter()
    }
}

impl fmt::Debug for TableData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (code, lookup)) in self.iter().enumerate() {
            if i != 0 {
                writeln!(f)?;
            }
            let bits = lookup.bits;
            let value = lookup.value;
            write!(f, "{:01$b} -> {2:02x}", code, usize::from(bits), value)?;
        }
        Ok(())
    }
}

/// Storage of all 4 huffman tables that are involved in JPEG decoding.
pub struct HuffmanTables {
    codes: Box<[LookupResult; LUT_LEN * 4]>,
}

impl HuffmanTables {
    pub const TOTAL_SIZE: usize = LUT_LEN * 4 * mem::size_of::<LookupResult>();

    pub fn new() -> Self {
        type TData = (&'static [u8; 16], &'static [u8]);
        const DEFAULT_LUMINANCE_DC: TData = (
            &[0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            &[
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
            ],
        );
        const DEFAULT_LUMINANCE_AC: TData = (
            &[0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125],
            &[
                0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
                0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1,
                0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18,
                0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
                0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57,
                0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75,
                0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92,
                0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
                0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
                0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
                0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2,
                0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
            ],
        );
        const DEFAULT_CHROMINANCE_DC: TData = (
            &[0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            &[
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
            ],
        );
        const DEFAULT_CHROMINANCE_AC: TData = (
            &[0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119],
            &[
                0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07,
                0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09,
                0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25,
                0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
                0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56,
                0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74,
                0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
                0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
                0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
                0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6,
                0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2,
                0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
            ],
        );

        let mut codes: Box<[LookupResult; LUT_LEN * 4]> = vec![LookupResult::ZERO; LUT_LEN * 4]
            .into_boxed_slice()
            .try_into()
            .map_err(drop)
            .unwrap();

        let luma_dc = TableData::build(DEFAULT_LUMINANCE_DC.0, DEFAULT_LUMINANCE_DC.1);
        let luma_ac = TableData::build(DEFAULT_LUMINANCE_AC.0, DEFAULT_LUMINANCE_AC.1);
        let chroma_dc = TableData::build(DEFAULT_CHROMINANCE_DC.0, DEFAULT_CHROMINANCE_DC.1);
        let chroma_ac = TableData::build(DEFAULT_CHROMINANCE_AC.0, DEFAULT_CHROMINANCE_AC.1);
        codes[0..LUT_LEN].copy_from_slice(luma_dc.raw());
        codes[LUT_LEN..LUT_LEN * 2].copy_from_slice(luma_ac.raw());
        codes[LUT_LEN * 2..LUT_LEN * 3].copy_from_slice(chroma_dc.raw());
        codes[LUT_LEN * 3..LUT_LEN * 4].copy_from_slice(chroma_ac.raw());
        Self { codes }
    }

    pub fn set(&mut self, index: u8, data: &TableData) {
        let index = usize::from(index);
        self.codes[LUT_LEN * index..LUT_LEN * (index + 1)].copy_from_slice(data.raw());
    }

    pub fn raw(&self) -> &[LookupResult; LUT_LEN * 4] {
        &self.codes
    }
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LookupResult {
    /// Length of the huffman code in bits (number of bits that need to be consumed from the input).
    bits: u8,
    /// Decoded value. Meaning depends on table class (AC/DC).
    value: u8,
}

impl LookupResult {
    const ZERO: Self = Self { bits: 0, value: 0 };

    fn new(bits: u8, value: u8) -> Self {
        Self { bits, value }
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn value(&self) -> u8 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tablegen() {
        // Default Luminance DC table.
        let num_dc_codes = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let dc_values = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
        ];

        let tbl = TableData::build(&num_dc_codes, &dc_values);
        expect_test::expect![[r#"
            00 -> 00
            010 -> 01
            011 -> 02
            100 -> 03
            101 -> 04
            110 -> 05
            1110 -> 06
            11110 -> 07
            111110 -> 08
            1111110 -> 09
            11111110 -> 0a
            111111110 -> 0b
        "#]]
        .assert_debug_eq(&tbl);
    }

    #[test]
    fn tablegen_large() {
        // Default luminance AC table.
        let num_ac_codes = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];
        let ac_values = [
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
            0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1,
            0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18,
            0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92,
            0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
            0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
            0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
            0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2,
            0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
        ];

        let tbl = TableData::build(&num_ac_codes, &ac_values);
        expect_test::expect![[r#"
            00 -> 01
            01 -> 02
            100 -> 03
            1010 -> 00
            1011 -> 04
            1100 -> 11
            11010 -> 05
            11011 -> 12
            11100 -> 21
            111010 -> 31
            111011 -> 41
            1111000 -> 06
            1111001 -> 13
            1111010 -> 51
            1111011 -> 61
            11111000 -> 07
            11111001 -> 22
            11111010 -> 71
            111110110 -> 14
            111110111 -> 32
            111111000 -> 81
            111111001 -> 91
            111111010 -> a1
            1111110110 -> 08
            1111110111 -> 23
            1111111000 -> 42
            1111111001 -> b1
            1111111010 -> c1
            11111110110 -> 15
            11111110111 -> 52
            11111111000 -> d1
            11111111001 -> f0
            111111110100 -> 24
            111111110101 -> 33
            111111110110 -> 62
            111111110111 -> 72
            111111111000000 -> 82
            1111111110000010 -> 09
            1111111110000011 -> 0a
            1111111110000100 -> 16
            1111111110000101 -> 17
            1111111110000110 -> 18
            1111111110000111 -> 19
            1111111110001000 -> 1a
            1111111110001001 -> 25
            1111111110001010 -> 26
            1111111110001011 -> 27
            1111111110001100 -> 28
            1111111110001101 -> 29
            1111111110001110 -> 2a
            1111111110001111 -> 34
            1111111110010000 -> 35
            1111111110010001 -> 36
            1111111110010010 -> 37
            1111111110010011 -> 38
            1111111110010100 -> 39
            1111111110010101 -> 3a
            1111111110010110 -> 43
            1111111110010111 -> 44
            1111111110011000 -> 45
            1111111110011001 -> 46
            1111111110011010 -> 47
            1111111110011011 -> 48
            1111111110011100 -> 49
            1111111110011101 -> 4a
            1111111110011110 -> 53
            1111111110011111 -> 54
            1111111110100000 -> 55
            1111111110100001 -> 56
            1111111110100010 -> 57
            1111111110100011 -> 58
            1111111110100100 -> 59
            1111111110100101 -> 5a
            1111111110100110 -> 63
            1111111110100111 -> 64
            1111111110101000 -> 65
            1111111110101001 -> 66
            1111111110101010 -> 67
            1111111110101011 -> 68
            1111111110101100 -> 69
            1111111110101101 -> 6a
            1111111110101110 -> 73
            1111111110101111 -> 74
            1111111110110000 -> 75
            1111111110110001 -> 76
            1111111110110010 -> 77
            1111111110110011 -> 78
            1111111110110100 -> 79
            1111111110110101 -> 7a
            1111111110110110 -> 83
            1111111110110111 -> 84
            1111111110111000 -> 85
            1111111110111001 -> 86
            1111111110111010 -> 87
            1111111110111011 -> 88
            1111111110111100 -> 89
            1111111110111101 -> 8a
            1111111110111110 -> 92
            1111111110111111 -> 93
            1111111111000000 -> 94
            1111111111000001 -> 95
            1111111111000010 -> 96
            1111111111000011 -> 97
            1111111111000100 -> 98
            1111111111000101 -> 99
            1111111111000110 -> 9a
            1111111111000111 -> a2
            1111111111001000 -> a3
            1111111111001001 -> a4
            1111111111001010 -> a5
            1111111111001011 -> a6
            1111111111001100 -> a7
            1111111111001101 -> a8
            1111111111001110 -> a9
            1111111111001111 -> aa
            1111111111010000 -> b2
            1111111111010001 -> b3
            1111111111010010 -> b4
            1111111111010011 -> b5
            1111111111010100 -> b6
            1111111111010101 -> b7
            1111111111010110 -> b8
            1111111111010111 -> b9
            1111111111011000 -> ba
            1111111111011001 -> c2
            1111111111011010 -> c3
            1111111111011011 -> c4
            1111111111011100 -> c5
            1111111111011101 -> c6
            1111111111011110 -> c7
            1111111111011111 -> c8
            1111111111100000 -> c9
            1111111111100001 -> ca
            1111111111100010 -> d2
            1111111111100011 -> d3
            1111111111100100 -> d4
            1111111111100101 -> d5
            1111111111100110 -> d6
            1111111111100111 -> d7
            1111111111101000 -> d8
            1111111111101001 -> d9
            1111111111101010 -> da
            1111111111101011 -> e1
            1111111111101100 -> e2
            1111111111101101 -> e3
            1111111111101110 -> e4
            1111111111101111 -> e5
            1111111111110000 -> e6
            1111111111110001 -> e7
            1111111111110010 -> e8
            1111111111110011 -> e9
            1111111111110100 -> ea
            1111111111110101 -> f1
            1111111111110110 -> f2
            1111111111110111 -> f3
            1111111111111000 -> f4
            1111111111111001 -> f5
            1111111111111010 -> f6
            1111111111111011 -> f7
            1111111111111100 -> f8
            1111111111111101 -> f9
            1111111111111110 -> fa
        "#]]
        .assert_debug_eq(&tbl);
    }
}
