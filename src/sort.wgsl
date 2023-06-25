// Ported from https://github.com/microsoft/DirectX-Graphics-Samples/blob/5ca41579b6837b3064c8b7333071859425c5c4de/MiniEngine/Core/Shaders/Bitonic32PreSortCS.hlsl
//
// Copyright (c) 2015 Microsoft
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// Invariant: length is a power-of-two.
@group(0) @binding(0) var<storage, read_write> elements: array<u32>;

struct PushParams {
    // This is set to the number of elements to sort. The length of `elements` might be higher than
    // that, since we often have oversized buffers with extra space.
    n_elems: u32,
    k: u32,
    j: u32,
};

var<push_constant> params: PushParams;
// NB: multiple `var<push_constant>` declarations cause a segfault

var<workgroup> buffer: array<u32, 2048>;

// FIXME: the workgroup sizes exceed WebGPU's guarantees (256 max. threads)
// make it a specializable constant once naga/wgpu supports those, and pick automatically.

@compute
@workgroup_size(1024)
fn presort(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) invoc_index: u32,
) {
    let n_elems = params.n_elems;

    // Each workgroup cooperatively sorts 2048 items in the input array, using 1024 invocations.
    let block_start = workgroup_id.x * 2048u;

    // First, copy this workgroup's part of the input array into the workgroup `buffer`.
    load_elem(block_start, invoc_index);
    load_elem(block_start, invoc_index + 1024u);

    // Make sure all stores to the workgroup-scoped `buffer` finish.
    workgroupBarrier();

    // Now perform a cooperative bitonic sort of the 2048 elements.
    for (var k = 2u; k <= 2048u; k <<= 1u) {
        for (var j = k / 2u; j > 0u; j /= 2u) {
            var mask: u32;
            if j == k / 2u {
                mask = k - 1u;
            } else {
                mask = j;
            }

            let i2 = insert_bit(invoc_index, j);
            let i1 = i2 ^ mask;

            if buffer[i1] > buffer[i2] {
                swap_buffer(i1, i2);
            }

            // We write to workgroup memory that the next iteration might immediately read from
            // again (from another invocation), so a barrier is needed.
            workgroupBarrier();
        }
    }

    store_elem(block_start, invoc_index);
    store_elem(block_start, invoc_index + 1024u);
}

@compute
@workgroup_size(1024)
fn bitonic_sort(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    // We don't use `arrayLength` here because the array might have unused trailing capacity.
    // The host sets `n_elems` to the padded power-of-two of elements to sort.
    let N = params.n_elems;
    let k = params.k;
    let j = params.j;

    // We perform `N` invocations per iteration with `k`, assigning the invocation ID to `i`.
    let i = id.x;
    if i >= N {
        return;
    }

    var mask: u32;
    if j == k / 2u {
        mask = k - 1u;
    } else {
        mask = j;
    }

    let i2 = insert_bit(i, j);
    let i1 = i2 ^ mask;

    if elements[i1] > elements[i2] {
        swap_shared(i1, i2);
    }
}

// Once `j` is small enough, element distances are sufficiently small to do blockwise sorting passes
// in workgroup-local memory, improving performance.
@compute
@workgroup_size(1024)
fn sort_small_j(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) invoc_index: u32,
) {
    let n_elems = params.n_elems;
    let k = params.k;

    // Each workgroup cooperatively sorts 2048 items in the input array, using 1024 invocations.
    let block_start = workgroup_id.x * 2048u;

    // First, copy this workgroup's part of the input array into the workgroup `buffer`.
    load_elem(block_start, invoc_index);
    load_elem(block_start, invoc_index + 1024u);

    // Make sure all stores to the workgroup-scoped `buffer` finish.
    workgroupBarrier();

    for (var j = 1024u; j > 0u; j /= 2u) {
        var mask: u32;
        if j == k / 2u {
            mask = k - 1u;
        } else {
            mask = j;
        }

        let i2 = insert_bit(invoc_index, j);
        let i1 = i2 ^ mask;

        if buffer[i1] > buffer[i2] {
            swap_buffer(i1, i2);
        }

        // We write to workgroup memory that the next iteration might immediately read from
        // again (from another invocation), so a barrier is needed.
        workgroupBarrier();
    }

    store_elem(block_start, invoc_index);
    store_elem(block_start, invoc_index + 1024u);
}

// Inserts a single bitmask `bit` into `value`, shifting all its bits to the left to make space.
fn insert_bit(value: u32, bit: u32) -> u32 {
    let mask = bit - 1u;
    return ((value & ~mask) << 1u) | (value & mask) | bit;
}

fn swap_buffer(a: u32, b: u32) {
    let temp = buffer[a];
    buffer[a] = buffer[b];
    buffer[b] = temp;
}

fn swap_shared(a: u32, b: u32) {
    let temp = elements[a];
    elements[a] = elements[b];
    elements[b] = temp;
}

fn load_elem(block_start: u32, i: u32) {
    if block_start + i < params.n_elems {
        buffer[i] = elements[block_start + i];
    } else {
        buffer[i] = 0xffffffffu;
    }
}

fn store_elem(block_start: u32, i: u32) {
    if block_start + i < arrayLength(&elements) {
        elements[block_start + i] = buffer[i];
    }
}
