use core::fmt;

pub struct TableData {
    /// Stores the encoded bits in the lowest k bits of the word.
    code_table: Vec<u16>,
    codes: Vec<u8>,
}

impl TableData {
    #[allow(dead_code)]
    pub fn build(num_codes_per_length: &[u8; 16], codes: &[u8]) -> Self {
        let mut size_table = generate_size_table(&num_codes_per_length);
        let code_table = generate_code_table(&size_table);
        assert_eq!(size_table.pop(), Some(0)); // remote 0-termination
        assert_eq!(size_table.len(), code_table.len());
        assert_eq!(code_table.len(), codes.len());

        Self {
            code_table,
            codes: codes.to_vec(),
        }
    }
}

impl fmt::Debug for TableData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (&code, &out) in self.code_table.iter().zip(&self.codes) {
            writeln!(f, "{:b} -> {:02x}", code, out)?;
        }
        Ok(())
    }
}

fn generate_size_table(bits: &[u8; 16]) -> Vec<u8> {
    let codes = bits.iter().map(|&v| usize::from(v)).sum::<usize>();

    let mut huffsize = vec![0; codes + 1]; // space for zero-terminator

    let mut k = 0;
    let mut i = 1;
    let mut j = 1;
    loop {
        if i == 16 || j > bits[i] {
            i += 1;
            j = 1;
            if i > 16 {
                huffsize[k] = 0; // zero-terminated list
                break;
            }
        } else {
            huffsize[k] = i as u8;
            k += 1;
            j += 1;
        }
    }

    assert_eq!(k as usize, codes);
    huffsize
}

fn generate_code_table(huffsize: &[u8]) -> Vec<u16> {
    let mut huffcode = vec![0; huffsize.len() - 1];

    let mut k = 0;
    let mut code = 0;
    let mut si = huffsize[0];
    loop {
        huffcode[k] = code;
        code += 1;
        k += 1;

        if huffsize[k] != si {
            if huffsize[k] == 0 {
                // list is zero-terminated
                break;
            }

            while huffsize[k] != si {
                code <<= 1;
                si += 1;
            }
        }
    }

    huffcode
}

#[derive(Clone, Copy)]
struct LookupResult(u8);

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
            0 -> 00
            10 -> 01
            11 -> 02
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
}
