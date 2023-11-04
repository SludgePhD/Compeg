use std::{error::Error, fmt::Write, fs};

use expect_test::{expect, expect_file, Expect, ExpectFile};

use crate::file::SegmentKind;

use super::JpegParser;

fn dump(jpeg: &[u8]) -> String {
    fn dump_impl(jpeg: &[u8], out: &mut String) -> super::Result<()> {
        let mut parser = JpegParser::new(jpeg)?;

        while let Some(segment) = parser.next_segment()? {
            write!(
                out,
                "{:04X} [FF {:02X}] ",
                segment.offset(),
                segment.marker(),
            )
            .unwrap();

            match segment.as_segment_kind() {
                Some(kind) => {
                    write!(out, "{:?}", kind).unwrap();
                    match kind {
                        SegmentKind::APP(app) if app.as_app_kind().is_none() => {
                            // Dump bytes of unknown APP segments too.
                            writeln!(out, " {:x?}", segment.raw_bytes()).unwrap();
                        }
                        _ => writeln!(out).unwrap(),
                    }
                }
                None => writeln!(out, "{:x?}", segment.raw_bytes()).unwrap(),
            }
        }

        if !parser.remaining().is_empty() {
            writeln!(
                out,
                "{} trailing bytes: {:x?}",
                parser.remaining().len(),
                parser.remaining()
            )
            .unwrap();
        }
        Ok(())
    }

    let mut out = String::new();
    if let Err(e) = dump_impl(jpeg, &mut out) {
        writeln!(out, "error: {e}").unwrap();
    }

    out
}

fn check(jpeg: &[u8], expect: Expect) {
    expect.assert_eq(&dump(jpeg));
}

fn check_file(filename: &str, expect: ExpectFile) {
    let bytes = std::fs::read(format!("src/file/test-images/{filename}")).unwrap();
    println!("checking {filename}");
    let out = dump(&bytes);

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
        check_file(&file, expect);
    }

    Ok(())
}

#[test]
fn empty() {
    check(
        &[0xFF],
        expect![[r#"
        error: reached end of data while decoding JPEG stream
    "#]],
    );
    check(
        &[0xFF, 0xD8 /* SOI */],
        expect![[r#"
        error: reached end of data while decoding JPEG stream
    "#]],
    );
    check(
        &[
            0xFF, 0xD8, // SOI
            0xFF, 0xD9, // EOI
        ],
        expect![[""]],
    );
    check(
        &[
            0xFF, 0xD8, // SOI
            0xFF, 0xD9, // EOI
            0xFF, // trailing
        ],
        expect![[r#"
             1 trailing bytes: [ff]
        "#]],
    );
}

#[test]
fn app() {
    check(
        &[
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, // APP0
            0x00, 0x02, // empty
            0xFF, 0xD9, // EOI
        ],
        expect![[r#"
            0002 [FF E0] APP { n: 0, kind: None } []
        "#]],
    );
    check(
        &[
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, // APP0
            0x00, 0x04, // 2 more bytes after this
            0x00, 0x00, // APP0 contents (non-JFIF)
            0xFF, 0xD9, // EOI
        ],
        expect![[r#"
            0002 [FF E0] APP { n: 0, kind: None } [0, 0]
        "#]],
    );
    check(
        &[
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, // APP0
            0x00, 0x04, // 2 more bytes after this
            0x00, 0x00, // APP0 contents (non-JFIF)
            0xFF, 0xDD, // DRI
            0x00, 0x04, // length
            0x00, 0x0F, // Ri
            0xFF, 0xD9, // EOI
        ],
        expect![[r#"
            0002 [FF E0] APP { n: 0, kind: None } [0, 0]
            0008 [FF DD] DRI { Ri: 15 }
        "#]],
    );
}
