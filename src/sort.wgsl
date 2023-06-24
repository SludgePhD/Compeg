@group(0) @binding(0) var<storage, read_write> elements: array<u32>;

struct PushParams {
    // This is set to the number of elements to sort. The length of `elements` might be higher than
    // that, since we often have oversized buffers with extra space not needed for *this* specific JPEG.
    n_elems: u32,
    k: u32,
    j: u32,
};

var<push_constant> params: PushParams;
// NB: multiple `var<push_constant>` declarations cause a segfault

// Pads the input array to contain a power of two of elements rather than just `n_elems`.
@compute
@workgroup_size(64)
fn pad(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let index = params.n_elems + id.x;
    if index < arrayLength(&elements) {
        elements[index] = 0xffffffffu;
    }
}

@compute
@workgroup_size(64)
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

    // Generate the index of the paired-up element `ij` by flipping the `j` bit.
    let ij = i ^ j;

    // Only do the comparison+sort if this thread's "base element" index `i` is lower than its pair.
    // Otherwise two threads would race.
    if ij > i {
        if ((i & k) == 0u) && elements[i] > elements[ij] {
            // Swap if i>ij => Sort ascending.
            swap(i, ij);
        }
        if ((i & k) != 0u) && elements[i] < elements[ij] {
            // Swap if i<ij => Sort descending.
            swap(i, ij);
        }
    }
}

fn swap(a: u32, b: u32) {
    let temp = elements[a];
    elements[a] = elements[b];
    elements[b] = temp;
}
