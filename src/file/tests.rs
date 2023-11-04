use std::{error::Error, fmt::Write, fs};

use expect_test::{expect_file, ExpectFile};

use super::JpegParser;

fn check(filename: &str, expect: ExpectFile) {
    let bytes = std::fs::read(format!("src/file/test-images/{filename}")).unwrap();
    println!("checking {filename}");
    let mut parser = JpegParser::new(&bytes).unwrap();

    let mut out = String::new();
    while let Some(segment) = parser.next_segment().unwrap() {
        write!(
            out,
            "{:04X} [FF {:02X}] ",
            segment.offset(),
            segment.marker(),
        )
        .unwrap();

        match segment.as_segment_kind() {
            Some(kind) => writeln!(out, "{:?}", kind).unwrap(),
            None => writeln!(out, "{:x?}", segment.raw_bytes()).unwrap(),
        }

        eprintln!("{:04X} [FF {:02X}] ", segment.offset(), segment.marker());
    }

    expect.assert_eq(&out);
}

#[test]
fn reftests() {
    do_reftests().unwrap();
}

fn do_reftests() -> Result<(), Box<dyn Error>> {
    let mut checks = Vec::new();
    for entry in fs::read_dir("src/file/test-images")? {
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
            checks.push((format!("{stem}.{ext}"), expect_file![log]));
        }
    }

    checks.sort_by(|(file1, _), (file2, _)| file1.cmp(file2));

    for (file, expect) in checks {
        check(&file, expect);
    }

    Ok(())
}
