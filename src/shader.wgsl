// Most of these `u32`s are actually just single bytes, but WGSL doesn't support that data type.

struct DhtSlot {
    num_dc_codes: array<u32, 16>,  // 16 bytes
    dc: array<u32, 12>,            // 12 bytes
    num_ac_codes: array<u32, 16>,  // 16 bytes
    ac: array<u32, 162>,           // 162 bytes, zero-padded
};

struct QTable {
    values: array<u32, 64>,   // 64 bytes, zero-padded
};

struct Metadata {
    dhts: array<DhtSlot, 2>,
    qtables: array<QTable, 4>,
    // Ri – number of MCUs per restart interval
    restart_interval: u32,
    width: u32,
    height: u32,
    component_qtables: array<u32, 3>,
    component_dchuff: array<u32, 3>,
    component_achuff: array<u32, 3>,
    start_position_count: atomic<u32>,
};

@group(0) @binding(0) var<storage, read_write> metadata: Metadata;

// The raw JPEG scan data, including all RST markers.
// This is raw byte data, but packed into `u32`s, since WebGPU doesn't have `u8`.
@group(0) @binding(1) var<storage, read> scan_data: array<u32>;

// List of byte indexes in `scan_data` where restart intervals begin.
@group(0) @binding(2) var<storage, read_write> start_positions: array<u32>;

@group(0) @binding(3) var out: texture_storage_2d<rgba8uint, write>;

@group(0) @binding(4) var<storage, read_write> debug: array<u32>;


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

// "minor" problem: result isn't actually ordered, but I need it to be ordered
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

// Huffman decode entry point.
@compute
@workgroup_size(64)
fn huffman_decode(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    if (id.x >= metadata.start_position_count) {
        return;
    }

    let x = id.x % metadata.width;
    let y = id.x / metadata.width;
    textureStore(out, vec2(x, y), vec4(vec3(id.x), 255u));
    /* TEST */

    /*var scan_data_byte_index = start_positions[id.x];

    for (var i = 0u; i < metadata.restart_interval; i++) {
        // This loop decodes 1 MCU.
        var decoded = array<i32, 64>();
    }*/
}
