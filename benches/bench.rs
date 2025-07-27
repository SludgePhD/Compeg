use std::hint::black_box;

use compeg::ScanBuffer;
use divan::counter::BytesCount;

fn main() {
    divan::main();
}

#[divan::bench]
fn scan(benchar: divan::Bencher) {
    let scan = include_bytes!("scan.dat");

    let mut buf = ScanBuffer::new();
    benchar
        .counter(BytesCount::new(scan.len()))
        .bench_local(|| {
            buf.process(black_box(scan), 42876).unwrap();
            buf.processed_scan_data().last().copied()
        });
}
