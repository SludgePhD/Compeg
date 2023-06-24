use std::{error::Error, fmt::Write, fs};

use expect_test::{expect_file, ExpectFile};

use super::JpegParser;

fn check(filename: &str, expect: ExpectFile) {
    let bytes = std::fs::read(format!("src/jpeg/test-images/{filename}")).unwrap();
    println!("checking {filename}");
    let mut parser = JpegParser::new(&bytes);

    let mut out = String::new();
    while let Some(segment) = parser.next_segment().unwrap() {
        writeln!(out, "{:04X} {:?}", segment.pos, segment.kind).unwrap();
    }

    expect.assert_eq(&out);
}

#[test]
fn reftests() {
    do_reftests().unwrap();
}

fn do_reftests() -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir("src/jpeg/test-images")? {
        let entry = entry?;
        let path = entry.path();
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let ext = path.extension().unwrap().to_str().unwrap();
        if ext == "log" {
            continue;
        }
        assert_eq!(ext, "jpg", "file '{}' has invalid suffix", path.display());

        if entry.file_type()?.is_file() {
            let log = format!("test-images/{stem}.log");
            check(&format!("{stem}.{ext}"), expect_file![log]);
        }
    }

    Ok(())
}
