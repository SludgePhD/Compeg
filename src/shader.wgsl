// Most of these `u32`s are actually just single bytes, but WGSL doesn't support that data type.

struct QTable {
    values: array<i32, 64>,   // 64 bytes, zero-padded
}

struct Component {
    vsample: u32,
    hsample: u32,
    qtable: u32,
    dchuff: u32, // TODO change to index into `huffman_luts` directly?
    achuff: u32,
}

struct Metadata {
    qtables: array<QTable, 4>,
    // Ri – number of MCUs per restart interval
    restart_interval: u32,
    width: u32,
    height: u32,
    components: array<Component, 3>,
    start_position_count: atomic<u32>,
}

struct HuffmanTable {
    // 2 16-bit entries each
    lut: array<u32, 32768>,
}

@group(0) @binding(0) var<storage, read_write> metadata: Metadata;

// The raw JPEG scan data, including all RST markers.
// This is raw byte data, but packed into `u32`s, since WebGPU doesn't have `u8`.
@group(0) @binding(1) var<storage, read> scan_data: array<u32>;

// List of byte indexes in `scan_data` where restart intervals begin.
@group(0) @binding(2) var<storage, read_write> start_positions: array<u32>;

@group(0) @binding(3) var out: texture_storage_2d<rgba8uint, write>;

@group(0) @binding(4) var<storage, read_write> debug: array<u32>;

// Order:
// Index 0 DC
// Index 0 AC
// Index 1 DC
// Index 1 AC
@group(0) @binding(5) var<storage, read> huffman_luts: array<HuffmanTable, 4>;

/// Extracts the byte at `index` (0-3) from `word`.
fn extractbyte(word: u32, index: u32) -> u32 {
    return (word >> index * 8u) & 0xffu;
}

fn scan_byte(index: u32) -> u32 {
    let word_index = index >> 2u;
    let byte = index & 3u;

    let word = scan_data[word_index];
    return extractbyte(word, byte);
}

fn push_start_position(pos: u32) {
    let index = atomicAdd(&metadata.start_position_count, 1u);
    if (index < arrayLength(&start_positions)) {
        start_positions[index] = pos;
    }
}

// "Minor" problem: result isn't actually ordered, but we need it to be ordered so that the index
// tells us the part of the output texture to write to.
// That's why the whole sorting module and shader exists.
@compute
@workgroup_size(64)
fn compute_start_positions(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    // Each invocation will process 32 Bytes (up to 33 at the boundary) to find RST markers
    // (0xFF 0xVV, where 0xVV != 0x00).

    let start_pos = id.x * (32u / 4u);
    if (start_pos >= arrayLength(&scan_data)) {
        return;
    }

    var i = 0u;
    for (i = 0u; i < (32u / 4u); i++) {
        let word = scan_data[start_pos + i];
        var ff_mask = 0xffu;
        var nonzero_mask = ff_mask << 8u;

        // Check for 0xFF 0xVV inside the word.
        for (var j = 0u; j < 3u; j++) {
            if (((word & ff_mask) == ff_mask) && ((word & nonzero_mask) != 0u)) {
                let byte_pos = (start_pos + i) * 4u + j + 2u;
                push_start_position(byte_pos);
            }

            ff_mask <<= 8u;
            nonzero_mask <<= 8u;
        }

        // Check for 0xFF 0xVV crossing over to the next word
        // If the last byte in this chunk is 0xFF, we have to check the next chunk's first byte.
        if ((word & 0xff000000u) == 0xff000000u) {
            let next_byte = scan_data[start_pos + i + 1u] & 0xffu;
            if (next_byte != 0x00u) {
                let byte_pos = (start_pos + i + 1u) * 4u + 1u;
                push_start_position(byte_pos);
            }
        }
    }
}

struct BitStreamState {
    next_word: u32,
    cur: u32,
    next: u32,
    left: u32,
}

var<private> bitstate: BitStreamState;

fn refill() {
    if bitstate.left < 32u {
        // FIXME: escape `0xFF 0x00` in bitstream
        var w = scan_data[bitstate.next_word];
        // LSB -> MSB word
        w = (w & 0x000000ffu) << 24u
          | (w & 0x0000ff00u) << 8u
          | (w & 0x00ff0000u) >> 8u
          | (w & 0xff000000u) >> 24u;
        bitstate.next_word += 1u;

        bitstate.cur |= w >> bitstate.left;
        if bitstate.left == 0u {
            bitstate.next = 0u;
        } else {
            bitstate.next = w << (32u - bitstate.left);
        }
        bitstate.left += 32u;
    }
}

fn consume(n: u32) {
    bitstate.cur <<= n;
    bitstate.cur |= (bitstate.next >> 1u) >> (31u - n);
    bitstate.next <<= n;
    bitstate.left -= n;
}

fn peek(n: u32) -> u32 {
    return (bitstate.cur >> 1u) >> (31u - n);
}

// Precondition: At least 16 bits left in the reader.
fn huffdecode(table: u32) -> u32 {
    // Huffman LUTs are generated to be indexed by the top 16 bits in the buffer. But we store 2
    // 16-bit entries in the same word, so we have to fetch 2 entries at once.
    let idx = bitstate.cur >> 16u;

    var entry = huffman_luts[table].lut[idx >> 1u];

    // LSB order, so the low half stores the first entry, the high half the second
    entry = (entry >> ((idx & 1u) * 16u)) & 0xffffu;

    // First byte = Number of bits to consume.
    consume(entry & 0xffu);

    // Second byte = Decode result.
    return entry >> 8u;
}

// Huffman decode entry point.
// Each invocation of this shader will decode one restart interval of MCUs.
@compute
@workgroup_size(64)
fn huffman_decode(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    if (id.x >= metadata.start_position_count) {
        return;
    }

    // Initialize bit reader state.
    let scan_data_byte_index = start_positions[id.x];
    let scan_data_word_index = scan_data_byte_index / 4u;
    let bit_offset = (scan_data_byte_index % 4u) * 8u;
    bitstate.next_word = scan_data_word_index;
    bitstate.cur = 0u;
    bitstate.next = 0u;
    bitstate.left = 0u;
    refill();
    consume(bit_offset);
    refill();

    // DC coefficient prediction is initialized to 0 at the beginning of each restart interval, and
    // updated for each contained MCU.
    var dcpred = vec3(0);

    for (var i = 0u; i < metadata.restart_interval; i++) {
        // Decode 1 MCU.
        // Each MCU contains data units for each component in order, with components that have a
        // sampling factor >1 storing several data units in sequence.

        for (var comp = 0u; comp < 3u; comp++) {
            let qtable = metadata.components[comp].qtable;
            let dchufftable = metadata.components[comp].dchuff;
            let achufftable = metadata.components[comp].achuff;

            for (var v_samp = 0u; v_samp < metadata.components[comp].vsample; v_samp++) {
                for (var h_samp = 0u; h_samp < metadata.components[comp].hsample; h_samp++) {
                    // Decode 1 data unit.
                    var decoded = array<i32, 64>();

                    // Decode DC coefficient.
                    let dccat = huffdecode(dchufftable); // 16
                    var diff = i32(peek(dccat));       // 11
                    consume(dccat);

                    if dccat == 0u {
                        diff = 0;
                    } else {
                        diff = huff_extend(diff, dccat);
                    }
                    dcpred[comp] += diff;
                    decoded[0] = dcpred[comp] * dequantize(qtable, 0u);

                    // Decode AC coefficients.
                    for (var pos = 1u; pos < 64u; pos++) {
                        refill();

                        let rrrrssss = huffdecode(achufftable);
                        if rrrrssss == 0u {
                            // Remaining ones are all 0.
                            break;
                        }
                        if rrrrssss == 0xf0u {
                            pos += 16u;
                            continue;
                        }

                        let rrrr = rrrrssss >> 4u;
                        let ssss = rrrrssss & 0xfu;
                        pos += rrrr;
                        let val = i32(peek(ssss));
                        consume(ssss);

                        let coeff = huff_extend(val, ssss);
                        let i = unzigzag(pos);
                        decoded[i] = coeff * dequantize(qtable, i);
                    }
                }
            }
        }
    }
}

fn dequantize(qtable: u32, value: u32) -> i32 {
    return metadata.qtables[qtable].values[value];
}

fn huff_extend(v: i32, t: u32) -> i32 {
    let vt = 1 << (t - 1u);
    return select(v, v + (-1 << t) + 1, v < vt);
}

fn unzigzag(pos: u32) -> u32 {
    // naga doesn't like this *at all*.
    // move the LUT into `metadata` I guess?
    let lut = array(
         0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    );

    if pos >= 64u {
        return 63;
    } else {
        return lut[pos];
    }
}

/*fn huff_extend(x: i32, s: i32) -> i32
{
    // if x<s return x else return x+offset[s] where offset[s] = ( (-1<<s)+1)
    (x) + ((((x) - (1 << ((s) - 1))) >> 31) & (((-1) << (s)) + 1))
}*/
