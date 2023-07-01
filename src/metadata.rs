use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct QTable {
    pub values: [u32; 64], // 64 bytes, zero-padded
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Component {
    pub vsample: u32,
    pub hsample: u32,
    pub qtable: u32,
    /// 0-3, indexing into the 4 raw huffman tables.
    pub dchuff: u32,
    /// 0-3, indexing into the 4 raw huffman tables.
    pub achuff: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Metadata {
    pub qtables: [QTable; 4],
    // Ri – number of MCUs per restart interval
    pub restart_interval: u32,
    pub components: [Component; 3],
    /// Written by shader.
    pub total_restart_intervals: u32,
    /// Image width in MCUs.
    pub width_mcus: u32,
    /// Max `Hi` of all the components.
    pub max_hsample: u32,
    /// Max `Vi` of all the components.
    pub max_vsample: u32,
    /// Always set to the constant [`UNZIGZAG`]. This exists only because naga is currently rather
    /// bad at handling in-shader constants and constant arrays, and should be removed when naga
    /// improves.
    pub unzigzag: [u32; 64],
}

#[rustfmt::skip]
pub const UNZIGZAG: [u32; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];
