use core::fmt;

pub struct TableData {
    codes: Vec<(u16, LookupResult)>,
}

impl TableData {
    #[allow(dead_code)]
    pub fn build(num_codes_per_length: &[u8; 16], codes: &[u8]) -> Self {
        // First, generate all huffman codes from the inputs.
        let mut out = Vec::new();

        // The following is similar in function to the flowcharts in Annex C
        // (`Generate_size_table` and `Generate_code_table`)
        let mut next_code = 0;
        let mut code_iter = codes.iter();
        for (code_length, &code_count) in num_codes_per_length.iter().enumerate() {
            let code_length = (code_length + 1) as u8; // 1-based

            next_code <<= 1;

            for _ in 0..code_count {
                let lookup_result = LookupResult::new(code_length, *code_iter.next().unwrap());
                out.push((next_code, lookup_result));
                next_code += 1;
            }
        }

        // We now have a list of all huffman codes, stored in the least significant bits of the
        // `u16` in `out`. In order to decode them quickly, we need to produce all possible prefixes
        // for each code to fill a 16-bit lookup table with.

        Self { codes: out }
    }
}

impl fmt::Debug for TableData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &(code, lookup) in &self.codes {
            let bits = lookup.bits;
            let value = lookup.value;
            writeln!(
                f,
                "{bits} {:01$b} -> {2:02x}",
                code,
                usize::from(bits),
                value,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct LookupResult {
    /// Length of the huffman code in bits (number of bits that need to be consumed from the input).
    bits: u8,
    /// Decoded value. Meaning depends on table class (AC/DC).
    value: u8,
}

impl LookupResult {
    fn new(bits: u8, value: u8) -> Self {
        Self { bits, value }
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
            2 00 -> 00
            3 010 -> 01
            3 011 -> 02
            3 100 -> 03
            3 101 -> 04
            3 110 -> 05
            4 1110 -> 06
            5 11110 -> 07
            6 111110 -> 08
            7 1111110 -> 09
            8 11111110 -> 0a
            9 111111110 -> 0b

        "#]]
        .assert_debug_eq(&tbl);
    }
}
