//! Dry-run snapshot-load: probeert een snapshot met alle bekende schema's en
//! rapporteert welke matcht. Gebruik ALTIJD vóór je een nieuwe binary deployt
//! om schema-breaks zoals de v2.6.5→v2.6.9 deploy-crisis te detecteren.
//!
//! Gebruik:
//!   cargo run -p snapshot_check --bin verify_snapshot -- <pad>
//!
//! Exit-codes:
//!   0 = snapshot is leesbaar door current binary (direct of via migratie)
//!   1 = alle schema's faalden
//!   2 = usage / file-error
//!
//! Output is machine-leesbaar JSON op stdout zodat tooling het kan parsen.

use std::io::Cursor;
use std::process::ExitCode;

use lucy_state::{Field, schema_v0::FieldV0, schema_v1::FieldV1, schema_v2::FieldV2};
use lucy_state::snapshot_header::{detect_version, SnapshotFormat};

fn try_strict<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    let mut cur = Cursor::new(bytes);
    let val = bincode::deserialize_from(&mut cur).map_err(|e| format!("{}", e))?;
    let consumed = cur.position() as usize;
    if consumed != bytes.len() {
        return Err(format!(
            "trailing {} bytes ({}/{})",
            bytes.len() - consumed,
            consumed,
            bytes.len()
        ));
    }
    Ok(val)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: verify_snapshot <pad>");
        return ExitCode::from(2);
    }
    let path = &args[1];
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            println!("{{\"ok\":false,\"error\":\"read: {}\"}}", e);
            return ExitCode::from(2);
        }
    };

    let (format_name, payload): (&str, &[u8]) = match detect_version(&bytes) {
        SnapshotFormat::Headered(v, off) => {
            let payload = &bytes[off..];
            // Probeer current Field, dan FieldV2 (DIM=256, alle moderne velden), dan FieldV1.
            // Pre-v2.9.0 binaries schreven schema=1 op DIM=256 payload — alleen V2-bridge laadt dat.
            if let Ok(f) = try_strict::<Field>(payload) {
                println!(
                    "{{\"ok\":true,\"schema\":\"headered_v{}_current\",\"tick\":{},\"points\":{},\"bytes\":{}}}",
                    v, f.tick, f.points.len(), bytes.len()
                );
                return ExitCode::SUCCESS;
            }
            if let Ok(f) = try_strict::<FieldV2>(payload) {
                println!(
                    "{{\"ok\":true,\"schema\":\"headered_v{}_v2bridge\",\"tick\":{},\"points\":{},\"bytes\":{}}}",
                    v, f.tick, f.points.len(), bytes.len()
                );
                return ExitCode::SUCCESS;
            }
            if let Ok(f) = try_strict::<FieldV1>(payload) {
                println!(
                    "{{\"ok\":true,\"schema\":\"headered_v{}_v1bridge\",\"tick\":{},\"points\":{},\"bytes\":{}}}",
                    v, f.tick, f.points.len(), bytes.len()
                );
                return ExitCode::SUCCESS;
            }
            let cur_err = try_strict::<Field>(payload).err().unwrap_or_default();
            let v2_err = try_strict::<FieldV2>(payload).err().unwrap_or_default();
            let v1_err = try_strict::<FieldV1>(payload).err().unwrap_or_default();
            println!(
                "{{\"ok\":false,\"schema\":\"headered_v{}\",\"errors\":{{\"current\":\"{}\",\"v2\":\"{}\",\"v1\":\"{}\"}}}}",
                v,
                cur_err.replace('"', "'"),
                v2_err.replace('"', "'"),
                v1_err.replace('"', "'")
            );
            return ExitCode::from(1);
        }
        SnapshotFormat::Legacy => ("legacy", bytes.as_slice()),
    };

    // Legacy ladder: v1 → v0 → current
    let v1_err = match try_strict::<FieldV1>(payload) {
        Ok(f) => {
            println!(
                "{{\"ok\":true,\"schema\":\"legacy_v1\",\"format\":\"{}\",\"tick\":{},\"points\":{},\"bytes\":{}}}",
                format_name, f.tick, f.points.len(), bytes.len()
            );
            return ExitCode::SUCCESS;
        }
        Err(e) => e,
    };
    let v0_err = match try_strict::<FieldV0>(payload) {
        Ok(f) => {
            println!(
                "{{\"ok\":true,\"schema\":\"legacy_v0\",\"format\":\"{}\",\"tick\":{},\"points\":{},\"bytes\":{}}}",
                format_name, f.tick, f.points.len(), bytes.len()
            );
            return ExitCode::SUCCESS;
        }
        Err(e) => e,
    };
    let cur_err = match try_strict::<Field>(payload) {
        Ok(f) => {
            println!(
                "{{\"ok\":true,\"schema\":\"legacy_current\",\"format\":\"{}\",\"tick\":{},\"points\":{},\"bytes\":{}}}",
                format_name, f.tick, f.points.len(), bytes.len()
            );
            return ExitCode::SUCCESS;
        }
        Err(e) => e,
    };

    println!(
        "{{\"ok\":false,\"schema\":\"none\",\"format\":\"{}\",\"errors\":{{\"v1\":\"{}\",\"v0\":\"{}\",\"current\":\"{}\"}}}}",
        format_name,
        v1_err.replace('"', "'"),
        v0_err.replace('"', "'"),
        cur_err.replace('"', "'")
    );
    ExitCode::from(1)
}
