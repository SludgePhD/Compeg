use compeg::ScanBuffer;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.benchmark_group("scan-preprocessing")
        .throughput(criterion::Throughput::Bytes(
            include_bytes!("scan.dat").len() as u64,
        ))
        .bench_function("scan.dat", |b| {
            let mut buf = ScanBuffer::new();

            b.iter(move || {
                buf.process(black_box(include_bytes!("scan.dat")), 42876)
                    .unwrap();
                buf.processed_scan_data().last().copied()
            })
        });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
