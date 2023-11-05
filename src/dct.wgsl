@group(0) @binding(0) var<storage, read> metadata: Metadata;

@group(1) @binding(0) var<storage, read_write> coefficients: array<i32>;

@group(2) @binding(0) var out: texture_storage_2d<rgba8uint, write>;


// DCT entry point. Each invocation will decode 1 DU (not 1 restart interval like the huffman
// shader does).
@compute
@workgroup_size(64)
fn dct(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    if (id.x >= metadata.start_position_count * metadata.restart_interval * metadata.dus_per_mcu) {
        return;
    }

    let start_offset = id.x * 64u;

    var data = array<i32, 64>();
    for (var i = 0u; i < 64u; i++) {
        data[i] = coefficients[start_offset + i];
    }

    data = idct(data);

    // Write the 8x8 8-bit pixels back to the `coefficients` buffer.
    for (var y = 0u; y < 8u; y++) {
        let row = vec2(
            u32(data[y * 8u + 0u]) << 0u |
            u32(data[y * 8u + 1u]) << 8u |
            u32(data[y * 8u + 2u]) << 16u |
            u32(data[y * 8u + 3u]) << 24u,
            u32(data[y * 8u + 4u]) << 0u |
            u32(data[y * 8u + 5u]) << 8u |
            u32(data[y * 8u + 6u]) << 16u |
            u32(data[y * 8u + 7u]) << 24u,
        );

        coefficients[start_offset + y * 2u + 0u] = i32(row[0]);
        coefficients[start_offset + y * 2u + 1u] = i32(row[1]);
    }
}

@compute
@workgroup_size(64)
fn finalize(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let mcu_idx = id.x;
    let first_du = mcu_idx * metadata.dus_per_mcu;

    let mcu_coord = vec2(
        mcu_idx % metadata.width_mcus,
        mcu_idx / metadata.width_mcus,
    );

    let mcu_size = vec2(
        metadata.max_hsample * 8u,
        metadata.max_vsample * 8u,
    );

    let top_left = mcu_coord * mcu_size;

    for (var yblock = 0u; yblock < metadata.max_vsample; yblock++) {
        for (var xblock = 0u; xblock < metadata.max_hsample; xblock++) {
            for (var ypix = 0u; ypix < 8u; ypix++) {
                // Load the 8-pixel rows of each DU for this Y coordinate.
                let luma_row = read_du(first_du + xblock, ypix);
                let cb_row = read_du(first_du + 2u, ypix);
                let cr_row = read_du(first_du + 3u, ypix);

                for (var x = 0u; x < 8u; x++) {
                    let coord = vec2(xblock * 8u + x, yblock * 8u + ypix);

                    // FIXME: the computation is hardcoded to my specific JPEGs, it should be made more flexible

                    let word = u32(x > 3u);
                    let cb = (cb_row[word] >> ((x & 3u) * 8u)) & 0xffu;
                    let cr = (cr_row[word] >> ((x & 3u) * 8u)) & 0xffu;
                    let luma = (luma_row[word] >> ((x & 3u) * 8u)) & 0xffu;

                    let rgb = ycbcr2rgb(luma, cb, cr);
                    textureStore(out, top_left + coord, vec4(rgb, 0xffu));
                }
            }
        }
    }
}

fn read_du(du: u32, y: u32) -> vec2<u32> {
    return vec2(
        u32(coefficients[du * 64u + y * 2u + 0u]),
        u32(coefficients[du * 64u + y * 2u + 1u]),
    );
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

const SCALE = array(
    1.0, 1.387039845, 1.306562965, 1.175875602,
    1.0, 0.785694958, 0.541196100, 0.275899379,
);

const DCT_SIZE = 8u;
fn idct_float_fast(in_vector_: array<i32, 64>) -> array<i32, 64> {
    // Premultiply with the scaling values.
    // See:
    // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/ec32420f6b5dfa4e86883d42b209e8371e55aeb5/jddctmgr.c#L303-L324
    var scale = SCALE;
    var in_vector_int = in_vector_;
    var in_vector = array<f32, 64>();
    for (var col = 0u; col < DCT_SIZE; col++) {
        for (var row = 0u; row < DCT_SIZE; row++) {
            let mul = scale[row] * scale[col];
            in_vector[row * DCT_SIZE + col] = f32(in_vector_int[row * DCT_SIZE + col]) * mul;
        }
    }

    var out_vector = array<i32, 64>();

    var ws = array<f32, 64>();

    for (var icol = 0u; icol < DCT_SIZE; icol++) {
        /* even part */

        let tmp0 = in_vector[DCT_SIZE * 0u + icol] * 0.125;
        let tmp1 = in_vector[DCT_SIZE * 2u + icol] * 0.125;
        let tmp2 = in_vector[DCT_SIZE * 4u + icol] * 0.125;
        let tmp3 = in_vector[DCT_SIZE * 6u + icol] * 0.125;

        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;

        let tmp13 = tmp1 + tmp3;
        let tmp12 = (tmp1 - tmp3) * 1.414213562 - tmp13;

        let tmp0i = tmp10 + tmp13;
        let tmp3i = tmp10 - tmp13;
        let tmp1i = tmp11 + tmp12;
        let tmp2i = tmp11 - tmp12;

        /* odd part */

        let tmp4 = in_vector[DCT_SIZE * 1u + icol] * 0.125;
        let tmp5 = in_vector[DCT_SIZE * 3u + icol] * 0.125;
        let tmp6 = in_vector[DCT_SIZE * 5u + icol] * 0.125;
        let tmp7 = in_vector[DCT_SIZE * 7u + icol] * 0.125;

        let z13 = tmp6 + tmp5;
        let z10 = tmp6 - tmp5;
        let z11 = tmp4 + tmp7;
        let z12 = tmp4 - tmp7;

        let tmp7i = z11 + z13;
        let tmp11i = (z11 - z13) * 1.414213562;

        let z5 = (z10 + z12) * 1.847759065;
        let tmp10i = z5 - z12 * 1.082392200;
        let tmp12i = z5 - z10 * 2.613125930;

        let tmp6i = tmp12i - tmp7i;
        let tmp5i = tmp11i - tmp6i;
        let tmp4i = tmp10i - tmp5i;

        ws[DCT_SIZE * 0u + icol] = tmp0i + tmp7i;
        ws[DCT_SIZE * 7u + icol] = tmp0i - tmp7i;
        ws[DCT_SIZE * 1u + icol] = tmp1i + tmp6i;
        ws[DCT_SIZE * 6u + icol] = tmp1i - tmp6i;
        ws[DCT_SIZE * 2u + icol] = tmp2i + tmp5i;
        ws[DCT_SIZE * 5u + icol] = tmp2i - tmp5i;
        ws[DCT_SIZE * 3u + icol] = tmp3i + tmp4i;
        ws[DCT_SIZE * 4u + icol] = tmp3i - tmp4i;
    }

    for (var row = 0u; row < DCT_SIZE; row++) {
        /* even part */

        let z5 = ws[row * DCT_SIZE + 0u] + 128.5;
        let tmp10 = z5 + ws[row * DCT_SIZE + 4u];
        let tmp11 = z5 - ws[row * DCT_SIZE + 4u];

        let tmp13 = ws[row * DCT_SIZE + 2u] + ws[row * DCT_SIZE + 6u];
        let tmp12 = (ws[row * DCT_SIZE + 2u] - ws[row * DCT_SIZE + 6u]) * 1.414213562 - tmp13;

        let tmp0 = tmp10 + tmp13;
        let tmp3 = tmp10 - tmp13;
        let tmp1 = tmp11 + tmp12;
        let tmp2 = tmp11 - tmp12;

        /* odd part */

        let z13 = ws[row * DCT_SIZE + 5u] + ws[row * DCT_SIZE + 3u];
        let z10 = ws[row * DCT_SIZE + 5u] - ws[row * DCT_SIZE + 3u];
        let z11 = ws[row * DCT_SIZE + 1u] + ws[row * DCT_SIZE + 7u];
        let z12 = ws[row * DCT_SIZE + 1u] - ws[row * DCT_SIZE + 7u];

        let tmp7 = z11 + z13;
        let tmp11i = (z11 - z13) * 1.414213562;

        let z5i = (z10 + z12) * 1.847759065;
        let tmp10i = z5i - z12 * 1.082392200;
        let tmp12i = z5i - z10 * 2.613125930;

        let tmp6 = tmp12i - tmp7;
        let tmp5 = tmp11i - tmp6;
        let tmp4 = tmp10i - tmp5;

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
