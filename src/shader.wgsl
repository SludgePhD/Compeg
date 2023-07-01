// Most of these `u32`s are actually just single bytes, but WGSL doesn't support that data type.

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
    // Table index in `huffman_luts` to use for the DC coefficients.
    dchuff: u32,
    // Table index in `huffman_luts` to use for the AC coefficients.
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
    unzigzag: array<u32, 64>,
}

struct HuffmanTable {
    // 2 16-bit entries each
    lut: array<u32, 32768>,
}

@group(0) @binding(0) var<storage, read> metadata: Metadata;

// Order:
// Index 0 DC
// Index 0 AC
// Index 1 DC
// Index 1 AC
@group(0) @binding(1) var<storage, read> huffman_luts: array<HuffmanTable, 4>;

// The preprocessed JPEG scan data.
// This is raw byte data, but packed into `u32`s, since WebGPU doesn't have `u8`.
// The preprocessing removes all RST markers, replaces all byte-stuffed 0xFF 0x00 sequences with
// just 0xFF, and aligns every restart interval on a `u32` boundary so that the shader doesn't have
// to do unnecessary bytewise processing.
@group(0) @binding(2) var<storage, read> scan_data: array<u32>;

// List of word indices in `scan_data` where restart intervals begin.
@group(0) @binding(3) var<storage, read> start_positions: array<u32>;

@group(0) @binding(4) var out: texture_storage_2d<rgba8uint, write>;


struct BitStreamState {
    next_word: u32,
    cur: u32,
    next: u32,
    left: u32,
}

var<private> bitstate: BitStreamState;

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

// 2x1 data units, enough for 2x1 subsampling
// This is *very* hard-coded for my specific JPEGs, because making it larger immediately impacts
// performance. Once naga supports `override` it can be chosen dynamically.
var<private> mcu_buffer: array<array<vec3<u32>, 16>, 8>;

// 4 * 4 * 16 * 8 bytes = 2048 bytes per invocation, ouch

fn mcu_buffer_clear() {
    for (var y = 0u; y < 16u; y++) {
        for (var x = 0u; x < 16u; x++) {
            mcu_buffer[y][x] = vec3(0u);
        }
    }
}

fn mcu_buffer_store_data_unit(du_coords: vec2<u32>, component: u32, values: array<i32, 64>) {
    var values = values;

    let shift = component * 8u;
    for (var i = 0u; i < 64u; i++) {
        let coords = du_coords * 8u + vec2(i % 8u, i / 8u);
        mcu_buffer[coords.y][coords.x] |= u32(values[i]) << shift;
    }
}

fn mcu_buffer_flush(mcu_idx: u32) {
    // TODO: undo subsampling, convert color models and spaces

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
            let coord = top_left + vec2(x, y);

            let luma = mcu_buffer[y][x] & 0xffu;
            textureStore(out, coord, vec4(luma, 0xffu));
        }
    }

    mcu_buffer_clear();
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
    let start_index = start_positions[id.x];
    bitstate.next_word = start_index;
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

                    // Perform iDCT using the `decoded` coefficients.
                    decoded = idct(decoded);

                    // Write the data unit to the MCU buffer, for later processing.
                    mcu_buffer_store_data_unit(vec2(h_samp, v_samp), comp, decoded);
                }
            }
        }

        // All components have now been written to the MCU buffer and can be processed together.
        mcu_buffer_flush(id.x + i);
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
    if pos >= 64u {
        return 63u;
    } else {
        return metadata.unzigzag[pos];
    }
}

//////////////////////////////////////////////
// IDCT (Inverse Discrete Cosine Transform) //
//////////////////////////////////////////////

// 0 = DC only
// 1 = zune-jpeg integer DCT
// 2 = float DCT
const IDCT_IMPL: u32 = 2u;

// This function is ported from zune-jpeg
fn idct(in_vector: array<i32, 64>) -> array<i32, 64> {
    switch IDCT_IMPL {
        case 0u: {
            return idct_dc_only(in_vector);
        }
        case 1u: {
            return idct_zune(in_vector);
        }
        case 2u: {
            return idct_float(in_vector);
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

fn idct_float(in_vector: array<i32, 64>) -> array<i32, 64> {
    var in_vector = in_vector;
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

fn idct_zune(in_vector: array<i32, 64>) -> array<i32, 64> {
    // FIXME: `const`ify
    let SCALE_BITS = 512 + 65536 + (128 << 17u);

    var in_vector = in_vector;
    var out_vector = array<i32, 64>();

    for (var ptr_ = 0u; ptr_ < 8u; ptr_++) {
        var p1: i32;
        var p2: i32;
        var p3: i32;
        var p4: i32;
        var p5: i32;
        var t0: i32;
        var t1: i32;
        var t2: i32;
        var t3: i32;

        p2 = in_vector[ptr_ + 16u];
        p3 = in_vector[ptr_ + 48u];

        p1 = (p2 + p3) * 2217;

        t2 = p1 + p3 * -7567;
        t3 = p1 + p2 *  3135;

        p2 = in_vector[ptr_];
        p3 = in_vector[ptr_ + 32u];
        t0 = fsh(p2 + p3);
        t1 = fsh(p2 - p3);

        let x0 = t0 + t3 + 512;
        let x3 = t0 - t3 + 512;
        let x1 = t1 + t2 + 512;
        let x2 = t1 - t2 + 512;

        // odd part
        t0 = in_vector[ptr_ + 56u];
        t1 = in_vector[ptr_ + 40u];
        t2 = in_vector[ptr_ + 24u];
        t3 = in_vector[ptr_ + 8u];

        p3 = t0 + t2;
        p4 = t1 + t3;
        p1 = t0 + t3;
        p2 = t1 + t2;
        p5 = (p3 + p4) * 4816;

        t0 *= 1223;
        t1 *= 8410;
        t2 *= 12586;
        t3 *= 6149;

        p1 = p5 + p1 * -3685;
        p2 = p5 + p2 * -10497;
        p3 = p3 * -8034;
        p4 = p4 * -1597;

        t3 += p1 + p4;
        t2 += p2 + p3;
        t1 += p2 + p4;
        t0 += p1 + p3;

        in_vector[ptr_] = (x0 + t3) >> 10u;
        in_vector[ptr_ + 8u] = (x1 + t2) >> 10u;
        in_vector[ptr_ + 16u] = (x2 + t1) >> 10u;
        in_vector[ptr_ + 24u] = (x3 + t0) >> 10u;
        in_vector[ptr_ + 32u] = (x3 - t0) >> 10u;
        in_vector[ptr_ + 40u] = (x2 - t1) >> 10u;
        in_vector[ptr_ + 48u] = (x1 - t2) >> 10u;
        in_vector[ptr_ + 56u] = (x0 - t3) >> 10u;
    }

    var i = 0u;
    for (var ptr_ = 0u; ptr_ < 8u; ptr_++) {
        var p1: i32;
        var p2: i32;
        var p3: i32;
        var p4: i32;
        var p5: i32;
        var t0: i32;
        var t1: i32;
        var t2: i32;
        var t3: i32;

        p2 = in_vector[ptr_ + 2u];
        p3 = in_vector[ptr_ + 6u];

        p1 = (p2 + p3) * 2217;
        t2 = p1 + p3 * -7567;
        t3 = p1 + p3 *  3135;

        p2 = in_vector[ptr_];
        p3 = in_vector[ptr_ + 4u];

        t0 = fsh(p2 + p3);
        t1 = fsh(p2 - p3);

        let x0 = t0 + t3 + SCALE_BITS;
        let x3 = t0 - t3 + SCALE_BITS;
        let x1 = t1 + t2 + SCALE_BITS;
        let x2 = t1 - t2 + SCALE_BITS;
        // odd part
        t0 = in_vector[i + 7u];
        t1 = in_vector[i + 5u];
        t2 = in_vector[i + 3u];
        t3 = in_vector[i + 1u];

        p3 = t0 + t2;
        p4 = t1 + t3;
        p1 = t0 + t3;
        p2 = t1 + t2;
        p5 = (p3 + p4) * f2f(1.175875602);

        t0 *= 1223;
        t1 *= 8410;
        t2 *= 12586;
        t3 *= 6149;

        p1 = p5 + p1 * -3685;
        p2 = p5 + p2 * -10497;
        p3 = p3 * -8034;
        p4 = p4 * -1597;

        t3 += p1 + p4;
        t2 += p2 + p3;
        t1 += p2 + p4;
        t0 += p1 + p3;

        out_vector[i + 0u] = clamp((x0 + t3) >> 17u, 0, 255);
        out_vector[i + 1u] = clamp((x1 + t2) >> 17u, 0, 255);
        out_vector[i + 2u] = clamp((x2 + t1) >> 17u, 0, 255);
        out_vector[i + 3u] = clamp((x3 + t0) >> 17u, 0, 255);
        out_vector[i + 4u] = clamp((x3 - t0) >> 17u, 0, 255);
        out_vector[i + 5u] = clamp((x2 - t1) >> 17u, 0, 255);
        out_vector[i + 6u] = clamp((x1 - t2) >> 17u, 0, 255);
        out_vector[i + 7u] = clamp((x0 - t3) >> 17u, 0, 255);

        i += 8u;
    }

    return out_vector;
}

// Multiply a number by 4096
fn f2f(x: f32) -> i32 {
    return i32(x * 4096.0 + 0.5);
}

// Multiply a number by 4096
fn fsh(x: i32) -> i32 {
    return x << 12u;
}
