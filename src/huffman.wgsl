// Many of these `u32`s are actually just single bytes, but WGSL doesn't support that data type, so
// we zero-extend them.

struct HuffmanLutL1 {
    // 2 16-bit entries each
    entries: array<u32, 128>,
}

@group(0) @binding(0) var<storage, read> metadata: Metadata;

// Order:
// Index 0 DC
// Index 0 AC
// Index 1 DC
// Index 1 AC
@group(1) @binding(0) var<storage, read> huffman_l1: array<HuffmanLutL1, 4>;

// Level-2 LUT for huffman codes longer than 8 bits. See `huffman.rs` for more detail.
@group(1) @binding(1) var<storage, read> huffman_l2: array<u32>;

// The preprocessed JPEG scan data.
// This is raw byte data, but packed into `u32`s, since WebGPU doesn't have `u8`.
// The preprocessing removes all RST markers, replaces all byte-stuffed 0xFF 0x00 sequences with
// just 0xFF, and aligns every restart interval on a `u32` boundary so that the shader doesn't have
// to do unnecessary bytewise processing.
@group(1) @binding(2) var<storage, read> scan_data: array<u32>;

// List of word indices in `scan_data` where restart intervals begin.
@group(1) @binding(3) var<storage, read> start_positions: array<u32>;

// DCT coefficients for each data unit.
@group(2) @binding(0) var<storage, read_write> coefficients: array<i32>;


struct BitStreamState {
    // Index of the next word in `scan_data` that this bit stream will fetch into the bit buffer.
    next_word: u32,
    // Upper 32 bits of the bit buffer (MSB-aligned).
    cur: u32,
    // Lower 32 bits of the bit buffer (MSB-aligned).
    next: u32,
    // Number of bits left to read from the buffer words `cur` and `next`.
    left: u32,
}

var<private> bitstate: BitStreamState; // 16 bytes per invocation

// Refills the bit stream buffer so that there are at least 32 bits ready to read.
//
// 32 is the "magic number" here, because it allows decoding one huffman code (up to 16 bits) and
// one scalar value (up to 15 bits) without refilling in between.
fn refill() {
    if bitstate.left < 32u {
        var w = scan_data[bitstate.next_word];
        // LSB -> MSB word
        w = (w & 0x000000ffu) << 24u
          | (w & 0x0000ff00u) << 8u
          | (w & 0x00ff0000u) >> 8u
          | (w & 0xff000000u) >> 24u;
        bitstate.next_word += 1u;

        bitstate.cur |= w >> bitstate.left;
        bitstate.next = (w << 1u) << (31u - bitstate.left);
        bitstate.left += 32u;
    }
}

// Advances the bit stream by `n` bits, without refilling it.
fn consume(n: u32) {
    bitstate.cur <<= n;
    bitstate.cur |= (bitstate.next >> 1u) >> (31u - n);
    bitstate.next <<= n;
    bitstate.left -= n;
}

// Peeks at the next `n` bits in the bit stream.
fn peek(n: u32) -> u32 {
    return (bitstate.cur >> 1u) >> (31u - n);
}

// Decodes a huffman code from the bit stream, using huffman table `table`.
//
// Precondition: At least 16 bits left in the reader.
// Postcondition: Consumes up to 16 bits from the bit stream without refilling it.
fn huffdecode(table: u32) -> u32 {
    // The level-1 LUT is indexed by the most significant 8 bits. But we store 2 16-bit entries in
    // the same word, so we have to fetch 2 entries at once.
    let code = bitstate.cur >> 16u;

    let l1idx = code >> 8u;
    var entry = huffman_l1[table].entries[l1idx >> 1u];

    // LSB order, so the low half stores the first entry, the high half the second
    entry = (entry >> ((l1idx & 1u) * 16u)) & 0xffffu;

    // Now, if the MSB is clear, this entry directly stores the lookup value.
    // If the MSB is set, however, we need to access the level-2 LUT.
    if (entry & 0x8000u) != 0u {
        let l2idx = (entry & 0x7fffu) + (code & 0xffu);
        entry = huffman_l2[l2idx >> 1u];

        entry = (entry >> ((l2idx & 1u) * 16u)) & 0xffffu;
    }

    // First byte = The decoded value.
    let value = entry & 0xffu;

    // Second byte = Number of bits to consume.
    let bits = entry >> 8u;

    consume(bits);
    return value;
}


// Huffman decode entry point.
// Each invocation of this shader will decode one restart interval of MCUs.
@compute
@workgroup_size(64)
fn huffman(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    if (id.x >= metadata.start_position_count) {
        return;
    }

    // Initialize bit reader state. The start index is counted in words, so that each invocation
    // starts decoding at a word boundary and no byte shifting is needed.
    bitstate.next_word = start_positions[id.x];
    bitstate.cur = 0u;
    bitstate.next = 0u;
    bitstate.left = 0u;
    refill();

    // DC coefficient prediction is initialized to 0 at the beginning of each restart interval, and
    // updated for each contained MCU.
    var dcpred = vec3(0);

    for (var i = 0u; i < metadata.restart_interval; i++) {
        // Decode 1 MCU.
        // Each MCU contains data units for each component in order, with components that have a
        // sampling factor >1 storing several data units in sequence.

        // Data Unit index in the MCU buffer; starts at 0 and is incremented for each DU we write.
        let mcu_index = id.x * metadata.restart_interval + i;
        var du_index = mcu_index * metadata.dus_per_mcu;

        for (var comp = 0u; comp < 3u; comp++) {
            let qtable = metadata.components[comp].qtable;
            let dchufftable = metadata.components[comp].dchuff;
            let achufftable = metadata.components[comp].achuff;

            for (var v_samp = 0u; v_samp < metadata.components[comp].vsample; v_samp++) {
                for (var h_samp = 0u; h_samp < metadata.components[comp].hsample; h_samp++) {
                    let start_offset = du_index * metadata.retained_coefficients;

                    // Decode 1 data unit.
                    var decoded = array<i32, 64>();

                    // Decode DC coefficient.
                    let dccat = huffdecode(dchufftable); // 16
                    var diff = i32(peek(dccat));         // 11
                    consume(dccat);

                    if dccat == 0u {
                        diff = 0;
                    } else {
                        diff = huff_extend(diff, dccat);
                    }
                    dcpred[comp] += diff;
                    coefficients[start_offset] = dcpred[comp] * dequant(qtable, 0u);

                    // Decode AC coefficients.
                    for (var pos = 1u; pos < 64u; pos++) {
                        refill();

                        let rrrrssss = huffdecode(achufftable); // 16
                        if rrrrssss == 0u {
                            // EOB = Remaining ones are all 0.
                            break;
                        }
                        if rrrrssss == 0xf0u {
                            pos += 16u;
                            continue;
                        }

                        let rrrr = rrrrssss >> 4u;
                        let ssss = rrrrssss & 0x0fu;
                        pos += rrrr;
                        let val = i32(peek(ssss));  // 15
                        consume(ssss);

                        let coeff = huff_extend(val, ssss);
                        if pos < metadata.retained_coefficients {
                            coefficients[start_offset + pos] = coeff * dequant(qtable, pos);
                        }
                    }

                    du_index++;
                }
            }
        }
    }
}

// Returns the quantization table value in `qtable` at `index`.
// Multiplication with a quantized value results in the dequantized value.
fn dequant(qtable: u32, index: u32) -> i32 {
    return metadata.qtables[qtable].values[index];
}

// Performs the `Huff_extend` procedure from the specification.
fn huff_extend(v: i32, t: u32) -> i32 {
    let vt = i32(1) << (t - 1);
    return select(v, v + (i32(-1) << t) + 1, v < vt);
}
