// Most of these `u32`s are actually just single bytes, but WGSL doesn't support that data type.

struct QTable {
    values: array<u32, 64>,   // 64 bytes, zero-padded
}

struct Component {
    vsample: u32,
    hsample: u32,
    qtable: u32,
    dchuff: u32,
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

// 2 slots, 2 classes (DC=0,AC=1)
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

// Huffman decode entry point.
@compute
@workgroup_size(64)
fn huffman_decode(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    if (id.x >= metadata.start_position_count) {
        return;
    }

    var scan_data_byte_index = start_positions[id.x];

    /*TEST*/
    let word = scan_data[scan_data_byte_index / 4u];
    let res = huffman_decode_(word, metadata.components[0].dchuff, 0u);
    if id.x < arrayLength(&debug) {
        debug[id.x * 2u] = word;
        debug[id.x * 2u + 1u] = res;
    }
    /*TEST*/

    for (var i = 0u; i < metadata.restart_interval; i++) {
        // Decode 1 MCU.
        // Each MCU contains data units for each component in order, with components that have a
        // sampling factor >1 storing several data units in sequence.

        for (var comp = 0u; comp < 3u; comp++) {
            for (var v_samp = 0u; v_samp < metadata.components[comp].vsample; v_samp++) {
                for (var h_samp = 0u; h_samp < metadata.components[comp].hsample; h_samp++) {
                    // Decode 1 data unit.
                    var decoded = array<i32, 64>();

                }
            }
        }
    }
}

struct HuffmanDecodeResult {
    bits: u32,  // number of input bits consumed (0-16)
    value: u32, // 8-bit decoded value
}

fn huffman_decode_(
    word: u32, // 16-bit word, with the to-be-decoded huffman code in the most significant 16 bits
    slot: u32, // huffman table index (0 or 1)
    tclass: u32, // huffman table class (DC=0, AC=1)
) -> u32 {
    let index = slot << 1u | tclass;
    var entry = huffman_luts[index].lut[word >> 17u];

    // LSB order, so the low half stores the first entry
    entry = (entry >> ((word & 1u) * 16u)) & 0xffffu;
    return entry;
}
