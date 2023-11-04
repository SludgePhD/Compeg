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
    pub total_restart_intervals: u32,
    /// Image width in MCUs.
    pub width_mcus: u32,
    /// Max `Hi` of all the components.
    pub max_hsample: u32,
    /// Max `Vi` of all the components.
    pub max_vsample: u32,
    pub dus_per_mcu: u32,
}
