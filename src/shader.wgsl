// Many of these `u32`s are actually just single bytes, but WGSL doesn't support that data type, so
// we zero-extend them.

struct QTable {
    values: array<i32, 64>,   // 64 bytes, zero-padded
}

struct Component {
    // Number of vertical data units of this component per MCU.
    vsample: u32,
    // Number of horizontal data units of this component per MCU.
    hsample: u32,
    // Quantization table index to use for coefficients (in `metadata.qtables`).
    qtable: u32,
    // Table index in `huffman_l1` to use for the DC coefficients.
    dchuff: u32,
    // Table index in `huffman_l1` to use for the AC coefficients.
    achuff: u32,
}

struct Metadata {
    qtables: array<QTable, 4>,
    // Ri – number of MCUs per restart interval
    restart_interval: u32,
    components: array<Component, 3>,
    start_position_count: u32,
    width_mcus: u32,
    max_hsample: u32,
    max_vsample: u32,
}

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

@group(2) @binding(0) var out: texture_storage_2d<rgba8uint, write>;


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

struct DataUnitBuf {
    // 8 rows of 8 8-bit pixels, compressed into a `vec2<u32>`
    pixels: array<vec2<u32>, 8>, // 64 bytes
}

// 2x1 data units, enough for 2x1 subsampling
// This is *very* hard-coded for my specific JPEGs, because making it larger immediately impacts
// performance. Once naga supports `override` it can be chosen dynamically.
var<private> mcu_buffer: array<DataUnitBuf, 4>;
// 64 * 4 bytes = 256 bytes per invocation

fn mcu_buffer_store_data_unit(du_index: u32, values_: array<i32, 64>) {
    var values = values_;
    for (var y = 0u; y < 8u; y++) {
        let row = vec2(
            u32(values[y * 8u + 0u]) << 0u |
            u32(values[y * 8u + 1u]) << 8u |
            u32(values[y * 8u + 2u]) << 16u |
            u32(values[y * 8u + 3u]) << 24u,
            u32(values[y * 8u + 4u]) << 0u |
            u32(values[y * 8u + 5u]) << 8u |
            u32(values[y * 8u + 6u]) << 16u |
            u32(values[y * 8u + 7u]) << 24u,
        );

        mcu_buffer[du_index].pixels[y] = row;
    }
}

fn mcu_buffer_flush(mcu_idx: u32) {
    let mcu_coord = vec2(
        mcu_idx % metadata.width_mcus,
        mcu_idx / metadata.width_mcus,
    );

    let mcu_size = vec2(
        metadata.max_hsample * 8u,
        metadata.max_vsample * 8u,
    );

    let top_left = mcu_coord * mcu_size;
    for (var y = 0u; y < mcu_size.y; y++) {
        for (var x = 0u; x < mcu_size.x; x++) {
            let coord = vec2(x, y);

            // FIXME: the computation is hardcoded to my specific JPEGs, it should be made more flexible

            let chroma_x = x / 2u;
            let c_word = u32(x > 3u);
            let cb = (mcu_buffer[2u].pixels[y][c_word] >> ((x & 3u) * 8u)) & 0xffu;
            let cr = (mcu_buffer[3u].pixels[y][c_word] >> ((x & 3u) * 8u)) & 0xffu;

            let du = u32(x > 7u);
            let x = x % 8u;
            let word = u32(x > 3u);
            let luma = (mcu_buffer[du].pixels[y][word] >> ((x & 3u) * 8u)) & 0xffu;

            let rgb = ycbcr2rgb(luma, cb, cr);
            textureStore(out, top_left + coord, vec4(rgb, 0xffu));
        }
    }
}

fn ycbcr2rgb(y_: u32, cb_: u32, cr_: u32) -> vec3<u32> {
    // JFIF specifies a default YCbCr color space according to the BT.601 standard. "Limited range"
    // is not used, the full 256 values are available for luminance information.

    let y = i32(y_);
    let cb = i32(cb_) - 128;
    let cr = i32(cr_) - 128;
    let r = y + ((45 * cr) >> 5u);
    let g = y - ((11 * cb + 23 * cr) >> 5u);
    let b = y + ((113 * cb) >> 6u);
    return vec3<u32>(clamp(vec3(r, g, b), vec3(0), vec3(255)));
}

// JPEG decode entry point.
// Each invocation of this shader will decode one restart interval of MCUs.
@compute
@workgroup_size(64)
fn jpeg_decode(
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
        var du_index = 0u;

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
                    var diff = i32(peek(dccat));         // 11
                    consume(dccat);

                    if dccat == 0u {
                        diff = 0;
                    } else {
                        diff = huff_extend(diff, dccat);
                    }
                    dcpred[comp] += diff;
                    decoded[0] = dcpred[comp] * dequant(qtable, 0u);

                    // Decode AC coefficients.
                    for (var pos = 1u; pos < 64u; pos++) {
                        refill();

                        let rrrrssss = huffdecode(achufftable); // 16
                        if rrrrssss == 0u {
                            // Remaining ones are all 0.
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
                        let i = unzigzag(pos);
                        decoded[i] = coeff * dequant(qtable, i);
                    }

                    // Perform iDCT using the decoded and dequantized coefficients.
                    decoded = idct(decoded);

                    // Write the data unit to the MCU buffer, for later processing.
                    mcu_buffer_store_data_unit(du_index, decoded);
                    du_index += 1u;
                }
            }
        }

        // All components have now been written to the MCU buffer and can be processed together.
        mcu_buffer_flush(id.x * metadata.restart_interval + i);
    }
}

// Returns the quantization table value in `qtable` at `index`.
// Multiplication with a quantized value results in the dequantized value.
fn dequant(qtable: u32, index: u32) -> i32 {
    return metadata.qtables[qtable].values[index];
}

// Performs the `Huff_extend` procedure from the specification.
fn huff_extend(v: i32, t: u32) -> i32 {
    let vt = 1 << (t - 1u);
    return select(v, v + (-1 << t) + 1, v < vt);
}

const UNZIGZAG = array(
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
);

fn unzigzag(pos: u32) -> u32 {
    var unzigzag = UNZIGZAG;
    return u32(unzigzag[pos]);
}

//////////////////////////////////////////////
// IDCT (Inverse Discrete Cosine Transform) //
//////////////////////////////////////////////

// We have several IDCT implementations, primarily just for development and debugging.
// They are chosen statically with the `IDCT_IMPL` parameter.

// IDCT_DC_ONLY
// ------------
// This IDCT ignores the AC values and only decodes the DC coefficient. This is mostly meant for
// debugging. The resulting image will consist entirely of flat 8x8 blocks, but they should be of
// roughly the right color/brightness compared to the correct result.
//
// IDCT_FLOAT_NAIVE
// ----------------
// A naive port of the IDCT routine described by the JPEG specification. This produces correct
// results, but is very computationally expensive.
//
// IDCT_FLOAT_FAST
// ---------------
// A port of the libjpeg-turbo IDCT implementation in `jidctflt.c`. Much faster than the naive IDCT,
// and still produces correct-looking output.
// <https://github.com/libjpeg-turbo/libjpeg-turbo/blob/ec32420f6b5dfa4e86883d42b209e8371e55aeb5/jidctflt.c>
const IDCT_IMPL_CHOICE: u32 = IDCT_FLOAT_FAST;

const IDCT_DC_ONLY: u32 = 0u;
const IDCT_FLOAT_NAIVE: u32 = 1u;
const IDCT_FLOAT_FAST: u32 = 2u;

fn idct(in_vector: array<i32, 64>) -> array<i32, 64> {
    switch IDCT_IMPL_CHOICE {
        case IDCT_DC_ONLY: {
            return idct_dc_only(in_vector);
        }
        case IDCT_FLOAT_NAIVE: {
            return idct_float_naive(in_vector);
        }
        case IDCT_FLOAT_FAST: {
            return idct_float_fast(in_vector);
        }
        default: {
            return array<i32, 64>();
        }
    }
}

fn idct_dc_only(in_vector: array<i32, 64>) -> array<i32, 64> {
    var out_vector = array<i32, 64>();
    let dc = (in_vector[0u] >> 3u) + 128;
    for (var i = 0u; i < 64u; i++) {
        out_vector[i] = clamp(dc, 0, 255);
    }
    return out_vector;
}

const PI = 3.141592;

fn idct_float_naive(in_vector_: array<i32, 64>) -> array<i32, 64> {
    var in_vector = in_vector_;
    var out_vector = array<i32, 64>();

    for (var y = 0u; y < 8u; y++) {
        for (var x = 0u; x < 8u; x++) {
            var val = 0.0;

            for (var u = 0u; u < 8u; u++) {
                for (var v = 0u; v < 8u; v++) {
                    let c_u = select(1.0, 1.0 / sqrt(2.0), u == 0u);
                    let c_v = select(1.0, 1.0 / sqrt(2.0), v == 0u);
                    let s_vu = f32(in_vector[v * 8u + u]);

                    val += c_u * c_v * s_vu
                        * cos((2.0 * f32(x) + 1.0) * f32(u) * PI / 16.0)
                        * cos((2.0 * f32(y) + 1.0) * f32(v) * PI / 16.0);
                }
            }

            val = val / 4.0 + 128.0;
            out_vector[y * 8u + x] = clamp(i32(val), 0, 255);
        }
    }

    return out_vector;
}

const DCT_SIZE = 8u;
fn idct_float_fast(in_vector_: array<i32, 64>) -> array<i32, 64> {
    var in_vector = in_vector_;
    var out_vector = array<i32, 64>();

    var ws = array<f32, 64>();

    var tmp0: f32;
    var tmp1: f32;
    var tmp2: f32;
    var tmp3: f32;
    var tmp4: f32;
    var tmp5: f32;
    var tmp6: f32;
    var tmp7: f32;
    var tmp10: f32;
    var tmp11: f32;
    var tmp12: f32;
    var tmp13: f32;

    for (var icol = 0u; icol < DCT_SIZE; icol++) {
        /* even part */

        tmp0 = f32(in_vector[DCT_SIZE * 0u + icol]) * 0.125;
        tmp1 = f32(in_vector[DCT_SIZE * 2u + icol]) * 0.125;
        tmp2 = f32(in_vector[DCT_SIZE * 4u + icol]) * 0.125;
        tmp3 = f32(in_vector[DCT_SIZE * 6u + icol]) * 0.125;

        tmp10 = tmp0 + tmp2;
        tmp11 = tmp0 - tmp2;

        tmp13 = tmp1 + tmp3;
        tmp12 = (tmp1 - tmp3) * 1.414213562 - tmp13;

        tmp0 = tmp10 + tmp13;
        tmp3 = tmp10 - tmp13;
        tmp1 = tmp11 + tmp12;
        tmp2 = tmp11 - tmp12;

        /* odd part */

        tmp4 = f32(in_vector[DCT_SIZE * 1u + icol]) * 0.125;
        tmp5 = f32(in_vector[DCT_SIZE * 3u + icol]) * 0.125;
        tmp6 = f32(in_vector[DCT_SIZE * 5u + icol]) * 0.125;
        tmp7 = f32(in_vector[DCT_SIZE * 7u + icol]) * 0.125;

        let z13 = tmp6 + tmp5;
        let z10 = tmp6 - tmp5;
        let z11 = tmp4 + tmp7;
        let z12 = tmp4 - tmp7;

        tmp7 = z11 + z13;
        tmp11 = (z11 - z13) * 1.414213562;

        let z5 = (z10 + z12) * 1.847759065;
        tmp10 = z5 - z12 * 1.082392200;
        tmp12 = z5 - z10 * 2.613125930;

        tmp6 = tmp12 - tmp7;
        tmp5 = tmp11 - tmp6;
        tmp4 = tmp10 - tmp5;

        ws[DCT_SIZE * 0u + icol] = tmp0 + tmp7;
        ws[DCT_SIZE * 7u + icol] = tmp0 - tmp7;
        ws[DCT_SIZE * 1u + icol] = tmp1 + tmp6;
        ws[DCT_SIZE * 6u + icol] = tmp1 - tmp6;
        ws[DCT_SIZE * 2u + icol] = tmp2 + tmp5;
        ws[DCT_SIZE * 5u + icol] = tmp2 - tmp5;
        ws[DCT_SIZE * 3u + icol] = tmp3 + tmp4;
        ws[DCT_SIZE * 4u + icol] = tmp3 - tmp4;
    }

    for (var row = 0u; row < DCT_SIZE; row++) {
        /* even part */

        var z5 = ws[row * DCT_SIZE + 0u] + 128.5;
        tmp10 = z5 + ws[row * DCT_SIZE + 4u];
        tmp11 = z5 - ws[row * DCT_SIZE + 4u];

        tmp13 = ws[row * DCT_SIZE + 2u] + ws[row * DCT_SIZE + 6u];
        tmp12 = (ws[row * DCT_SIZE + 2u] - ws[row * DCT_SIZE + 6u]) * 1.414213562 - tmp13;

        tmp0 = tmp10 + tmp13;
        tmp3 = tmp10 - tmp13;
        tmp1 = tmp11 + tmp12;
        tmp2 = tmp11 - tmp12;

        /* odd part */

        let z13 = ws[row * DCT_SIZE + 5u] + ws[row * DCT_SIZE + 3u];
        let z10 = ws[row * DCT_SIZE + 5u] - ws[row * DCT_SIZE + 3u];
        let z11 = ws[row * DCT_SIZE + 1u] + ws[row * DCT_SIZE + 7u];
        let z12 = ws[row * DCT_SIZE + 1u] - ws[row * DCT_SIZE + 7u];

        tmp7 = z11 + z13;
        tmp11 = (z11 - z13) * 1.414213562;

        z5 = (z10 + z12) * 1.847759065;
        tmp10 = z5 - z12 * 1.082392200;
        tmp12 = z5 - z10 * 2.613125930;

        tmp6 = tmp12 - tmp7;
        tmp5 = tmp11 - tmp6;
        tmp4 = tmp10 - tmp5;

        out_vector[row * DCT_SIZE + 0u] = clamp(i32(tmp0 + tmp7), 0, 255);
        out_vector[row * DCT_SIZE + 7u] = clamp(i32(tmp0 - tmp7), 0, 255);
        out_vector[row * DCT_SIZE + 1u] = clamp(i32(tmp1 + tmp6), 0, 255);
        out_vector[row * DCT_SIZE + 6u] = clamp(i32(tmp1 - tmp6), 0, 255);
        out_vector[row * DCT_SIZE + 2u] = clamp(i32(tmp2 + tmp5), 0, 255);
        out_vector[row * DCT_SIZE + 5u] = clamp(i32(tmp2 - tmp5), 0, 255);
        out_vector[row * DCT_SIZE + 3u] = clamp(i32(tmp3 + tmp4), 0, 255);
        out_vector[row * DCT_SIZE + 4u] = clamp(i32(tmp3 - tmp4), 0, 255);
    }

    return out_vector;
}
