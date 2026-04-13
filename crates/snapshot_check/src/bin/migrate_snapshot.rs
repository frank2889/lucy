//! migrate_snapshot — offline snapshot-conversie tool
//!
//! Usage:
//!   migrate_snapshot <input.snapshot> <output.snapshot>
//!
//! Leest snapshot via `Field::load` (detecteert legacy / headered / versie),
//! migreert naar current schema, schrijft via `Field::save` (= nieuwe header).
//!
//! Exit codes:
//!   0 = succes (conversie voltooid)
//!   1 = gebruiksfout (missende args, file niet gevonden)
//!   2 = load-gefaald (snapshot onleesbaar)
//!   3 = save-gefaald (kan niet schrijven)

use lucy_state::Field;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: migrate_snapshot <input.snapshot> <output.snapshot>");
        eprintln!();
        eprintln!("Leest snapshot (legacy of headered), migreert naar current schema,");
        eprintln!("schrijft output met v2.6.9 SnapshotHeader.");
        return ExitCode::from(1);
    }

    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);

    if !input.exists() {
        eprintln!("❌ Input bestaat niet: {}", input.display());
        return ExitCode::from(1);
    }

    eprintln!("→ Load {} ...", input.display());
    let field = match Field::load(input) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("❌ Load faalde: {}", e);
            return ExitCode::from(2);
        }
    };

    eprintln!("   tick={}, points={}, active={}", field.tick, field.points.len(), field.active_set.len());

    eprintln!("→ Save {} met current header ...", output.display());
    match field.save(output) {
        Ok(()) => {
            let size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
            eprintln!("   ✓ Geschreven: {} bytes", size);
        }
        Err(e) => {
            eprintln!("❌ Save faalde: {}", e);
            return ExitCode::from(3);
        }
    }

    // Round-trip verify
    eprintln!("→ Roundtrip verify: lees output terug ...");
    match Field::load(output) {
        Ok(f2) => {
            if f2.tick != field.tick || f2.points.len() != field.points.len() {
                eprintln!("❌ Roundtrip mismatch: tick {} → {}, points {} → {}",
                    field.tick, f2.tick, field.points.len(), f2.points.len());
                return ExitCode::from(3);
            }
            eprintln!("   ✓ Roundtrip match (tick={}, points={})", f2.tick, f2.points.len());
        }
        Err(e) => {
            eprintln!("❌ Roundtrip load faalde: {}", e);
            return ExitCode::from(3);
        }
    }

    eprintln!();
    eprintln!("🟢 Migratie voltooid.");
    ExitCode::from(0)
}
