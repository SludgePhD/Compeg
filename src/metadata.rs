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
    pub dchuff: u32,
    pub achuff: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Metadata {
    pub qtables: [QTable; 4],
    // Ri – number of MCUs per restart interval
    pub restart_interval: u32,
    pub width: u32,
    pub height: u32,
    pub components: [Component; 3],
    /// Written by shader.
    pub start_position_count: u32,
}
