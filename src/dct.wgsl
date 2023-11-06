@group(0) @binding(0) var<storage, read> metadata: Metadata;

@group(1) @binding(0) var<storage, read_write> coefficients: array<i32>;

@group(2) @binding(0) var out: texture_storage_2d<rgba8unorm, write>;

// DCT is always performed on 8x8 blocks.
const DCT_SIZE = 8u;

// 8 invocations work together to compute the IDCT of a single DU (8x8 block).
const THREADS_PER_DCT = 8u;

const DCT_WORKGROUP_SIZE = 256u;

const DCTS_PER_WORKGROUP = DCT_WORKGROUP_SIZE / THREADS_PER_DCT;

const WORKGROUP_BUF_LEN = 64u * DCTS_PER_WORKGROUP;

var<workgroup> ws: array<f32, WORKGROUP_BUF_LEN>;
// 4 * 64 * DCTS_PER_WORKGROUP bytes.
// So for 64-wide workgroups this uses 2 KiB.
// For our 256-wide workgroups it uses 8 KiB.

const SCALE = array(
    1.0, 1.387039845, 1.306562965, 1.175875602,
    1.0, 0.785694958, 0.541196100, 0.275899379,
);

const ZIGZAG = array(
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
);

fn zigzag(pos: u32) -> u32 {
    var zigzag = ZIGZAG;
    return u32(zigzag[pos]);
}

// DCT entry point. Each workgroup will decode `DCTS_PER_WORKGROUP` DUs.
@compute
@workgroup_size(DCT_WORKGROUP_SIZE)
fn dct(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let du = id.x / THREADS_PER_DCT;
    let lane = id.x % THREADS_PER_DCT;

    if (du >= metadata.start_position_count * metadata.restart_interval * metadata.dus_per_mcu) {
        return;
    }

    // This DU's start offset into `coefficients`:
    let global_offset = du * 64u;
    // This DU's start offset in `ws`:
    let local_offset = local / THREADS_PER_DCT * 64u;

    // We perform per-column DCT. Each of the 8 threads working on this block does one column.
    // The results are written to workgroup storage (`ws`), then we have a barrier and perform a
    // per-row DCT, with each thread doing one row.

    // We use a port of the libjpeg-turbo IDCT implementation in `jidctflt.c`.
    // <https://github.com/libjpeg-turbo/libjpeg-turbo/blob/ec32420f6b5dfa4e86883d42b209e8371e55aeb5/jidctflt.c>

    // Premultiply the coefficients with the scaling factors.
    // FIXME: should really be precomputed and done as part of dequantization, like libjpeg does it.
    var inputs = array<f32, 8>();
    var scale = SCALE;

    let col = lane;
    for (var row = 0u; row < DCT_SIZE; row++) {
        let mul = scale[row] * scale[col];
        let i = zigzag(row * DCT_SIZE + col);
        inputs[row] = f32(coefficients[global_offset + i]) * mul;
    }

    // Now use the inputs for this column to compute its IDCT, writing the result to `ws`.

    {
        /* even part */
        let tmp0 = inputs[0u] * 0.125;
        let tmp1 = inputs[2u] * 0.125;
        let tmp2 = inputs[4u] * 0.125;
        let tmp3 = inputs[6u] * 0.125;

        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;

        let tmp13 = tmp1 + tmp3;
        let tmp12 = (tmp1 - tmp3) * 1.414213562 - tmp13;

        let tmp0i = tmp10 + tmp13;
        let tmp3i = tmp10 - tmp13;
        let tmp1i = tmp11 + tmp12;
        let tmp2i = tmp11 - tmp12;

        /* odd part */
        let tmp4 = inputs[1u] * 0.125;
        let tmp5 = inputs[3u] * 0.125;
        let tmp6 = inputs[5u] * 0.125;
        let tmp7 = inputs[7u] * 0.125;

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

        ws[local_offset + DCT_SIZE * 0u + col] = tmp0i + tmp7i;
        ws[local_offset + DCT_SIZE * 7u + col] = tmp0i - tmp7i;
        ws[local_offset + DCT_SIZE * 1u + col] = tmp1i + tmp6i;
        ws[local_offset + DCT_SIZE * 6u + col] = tmp1i - tmp6i;
        ws[local_offset + DCT_SIZE * 2u + col] = tmp2i + tmp5i;
        ws[local_offset + DCT_SIZE * 5u + col] = tmp2i - tmp5i;
        ws[local_offset + DCT_SIZE * 3u + col] = tmp3i + tmp4i;
        ws[local_offset + DCT_SIZE * 4u + col] = tmp3i - tmp4i;
    }

    workgroupBarrier();

    // Now perform the per-row IDCT.

    let row = lane;

    {
        let z5 = ws[local_offset + row * DCT_SIZE + 0u] + 128.5;
        let tmp10 = z5 + ws[local_offset + row * DCT_SIZE + 4u];
        let tmp11 = z5 - ws[local_offset + row * DCT_SIZE + 4u];

        let tmp13 = ws[local_offset + row * DCT_SIZE + 2u] + ws[local_offset + row * DCT_SIZE + 6u];
        let tmp12 = (ws[local_offset + row * DCT_SIZE + 2u] - ws[local_offset + row * DCT_SIZE + 6u]) * 1.414213562 - tmp13;

        let tmp0 = tmp10 + tmp13;
        let tmp3 = tmp10 - tmp13;
        let tmp1 = tmp11 + tmp12;
        let tmp2 = tmp11 - tmp12;

        /* odd part */

        let z13 = ws[local_offset + row * DCT_SIZE + 5u] + ws[local_offset + row * DCT_SIZE + 3u];
        let z10 = ws[local_offset + row * DCT_SIZE + 5u] - ws[local_offset + row * DCT_SIZE + 3u];
        let z11 = ws[local_offset + row * DCT_SIZE + 1u] + ws[local_offset + row * DCT_SIZE + 7u];
        let z12 = ws[local_offset + row * DCT_SIZE + 1u] - ws[local_offset + row * DCT_SIZE + 7u];

        let tmp7 = z11 + z13;
        let tmp11i = (z11 - z13) * 1.414213562;

        let z5i = (z10 + z12) * 1.847759065;
        let tmp10i = z5i - z12 * 1.082392200;
        let tmp12i = z5i - z10 * 2.613125930;

        let tmp6 = tmp12i - tmp7;
        let tmp5 = tmp11i - tmp6;
        let tmp4 = tmp10i - tmp5;

        ws[local_offset + row * DCT_SIZE + 0u] = clamp(tmp0 + tmp7, 0.0, 255.0);
        ws[local_offset + row * DCT_SIZE + 7u] = clamp(tmp0 - tmp7, 0.0, 255.0);
        ws[local_offset + row * DCT_SIZE + 1u] = clamp(tmp1 + tmp6, 0.0, 255.0);
        ws[local_offset + row * DCT_SIZE + 6u] = clamp(tmp1 - tmp6, 0.0, 255.0);
        ws[local_offset + row * DCT_SIZE + 2u] = clamp(tmp2 + tmp5, 0.0, 255.0);
        ws[local_offset + row * DCT_SIZE + 5u] = clamp(tmp2 - tmp5, 0.0, 255.0);
        ws[local_offset + row * DCT_SIZE + 3u] = clamp(tmp3 + tmp4, 0.0, 255.0);
        ws[local_offset + row * DCT_SIZE + 4u] = clamp(tmp3 - tmp4, 0.0, 255.0);
    }

    workgroupBarrier();

    // Write the 8x8 8-bit pixels back to the `coefficients` buffer. Each thread writes 1 row.
    let y = lane;

    let rowdata = vec2(
        u32(ws[local_offset + y * 8u + 0u]) << 0u |
        u32(ws[local_offset + y * 8u + 1u]) << 8u |
        u32(ws[local_offset + y * 8u + 2u]) << 16u |
        u32(ws[local_offset + y * 8u + 3u]) << 24u,
        u32(ws[local_offset + y * 8u + 4u]) << 0u |
        u32(ws[local_offset + y * 8u + 5u]) << 8u |
        u32(ws[local_offset + y * 8u + 6u]) << 16u |
        u32(ws[local_offset + y * 8u + 7u]) << 24u,
    );

    coefficients[global_offset + y * 2u + 0u] = i32(rowdata[0]);
    coefficients[global_offset + y * 2u + 1u] = i32(rowdata[1]);
}

/////////////////
// Compositing //
/////////////////

// After each DU has been decoded by the IDCT shader above, they all need to be read from the buffer
// and written to the final texture after conversion to RGB.
// The naive way of doing this has extremely poor performance (presumably because of all the VRAM
// accesses), so we read all the DUs that contribute to an MCU into fast workgroup-local memory and
// operate on that.

// FIXME: make these `override`s once naga supports those.
const DUS_PER_MCU = 4u;
const MCU_HEIGHT = 8u;
const MCU_WIDTH = 16u;

struct DuBuf {
    // data units are stored in the same format as they are in the big buffer: with 8 pixel rows
    // packed into `vec2<u32>`.
    rows: array<vec2<u32>, 8>,
}
// 8 * 8 = 64 Bytes per DU

// Holds the decoded data units belonging to one MCU.
struct McuBuf {
    du: array<DuBuf, DUS_PER_MCU>,
}
// 64 * DUS_PER_MCU bytes per MCU
// (256 Bytes for 4 DUs/MCU)

// We choose the number of MCUs processed per workgroup so that we end up at a "reasonable" amount
// of LDS usage, like 16 KB (GCN has 32K). This ends up at 64 MCUs per workgroup.

// Each thread composites one MCU row.
const THREADS_PER_MCU = MCU_HEIGHT;

// In the lowest-end implementations, WebGPU workgroups can only be 256 threads in size.
const WORKGROUP_SIZE = 256u;

const MCUS_PER_WORKGROUP = WORKGROUP_SIZE / THREADS_PER_MCU;

var<workgroup> databuf: array<McuBuf, MCUS_PER_WORKGROUP>;

@compute
@workgroup_size(WORKGROUP_SIZE)
fn finalize(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let mcu_idx = wgid.x * MCUS_PER_WORKGROUP + local / THREADS_PER_MCU;
    let row = local % THREADS_PER_MCU;

    if (mcu_idx >= metadata.start_position_count * metadata.restart_interval) {
        return;
    }

    let global_du = mcu_idx * metadata.dus_per_mcu;
    let local_mcu = local / THREADS_PER_MCU;  // MCU in this workgroup

    // Each thread composites one row of an MCU, and is responsible for loading one row of each DU.
    for (var i = 0u; i < metadata.dus_per_mcu; i++) {
        let du_offset = (global_du + i) * 64u;
        let data = vec2(
            u32(coefficients[du_offset + row * 2u + 0u]),
            u32(coefficients[du_offset + row * 2u + 1u]),
        );
        databuf[local_mcu].du[i].rows[row] = data;
    }

    workgroupBarrier();

    let mcu_coord = vec2(
        mcu_idx % metadata.width_mcus,
        mcu_idx / metadata.width_mcus,
    );

    let mcu_size = vec2(
        metadata.max_hsample * 8u,
        metadata.max_vsample * 8u,
    );

    let top_left = mcu_coord * mcu_size;
    for (var col = 0u; col < mcu_size.x; col++) {
        let coord = top_left + vec2(col, row);

        var du_offset = 0u;  // DU index where each component starts.
        var components = vec4<u32>();
        for (var comp = 0u; comp < 3u; comp++) {
            let du = du_offset + col * metadata.components[comp].hsample / mcu_size.x;

            // Adjust the rate we sample the sub-sampled components at correctly:
            let xscale = metadata.max_hsample / metadata.components[comp].hsample;
            let yscale = metadata.max_vsample / metadata.components[comp].vsample;
            let x = col / xscale;
            let y = row / yscale;

            let word = u32((x & 7u) > 3u);
            let shift = (x & 7u) * 8u;

            components[comp] = databuf[local_mcu].du[du].rows[y][word] >> shift;

            du_offset += metadata.components[comp].hsample * metadata.components[comp].vsample;
        }

        let rgb = ycbcr2rgb(components[0] & 0xffu, components[1] & 0xffu, components[2] & 0xffu);
        textureStore(out, coord, vec4(f32(rgb.r) / 255.0, f32(rgb.g) / 255.0, f32(rgb.b) / 255.0, 1.0));
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
