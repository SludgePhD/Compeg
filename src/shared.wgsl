// File with shared definitions, copied into each of the encoder files.

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
    // Table index in `huffman_l1` to use for the DC coefficients.
    dchuff: u32,
    // Table index in `huffman_l1` to use for the AC coefficients.
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
    dus_per_mcu: u32,
    retained_coefficients: u32,
}
