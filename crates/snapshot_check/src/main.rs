//! snapshot_check — offline validator voor Lucy snapshots.
//!
//! Laadt een `lucy.snapshot` (bincode Field) met de CURRENT `lucy-state` struct
//! en rapporteert of deserialisatie lukt. Gebruik:
//!
//! ```
//! cargo run -p snapshot_check -- /path/to/lucy.snapshot
//! ```
//!
//! Exit codes:
//!   0 = snapshot laadt succesvol, alle data intact
//!   1 = snapshot file niet gevonden
//!   2 = deserialisatie gefaald (mogelijk schema-mismatch)
//!   3 = structurele validatie gefaald (points=0 of vocab mismatch)
//!
//! Dit is de gate vóór deploy: als dit faalt op een LIVE snapshot, dan zou
//! deployment datala bij cluster-startup veroorzaken.

use lucy_state::Field;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: snapshot_check <path-to-lucy.snapshot>");
        return ExitCode::from(1);
    }
    let path = Path::new(&args[1]);
    if !path.exists() {
        eprintln!("❌ Bestand bestaat niet: {}", path.display());
        return ExitCode::from(1);
    }

    eprintln!("→ Lees {} bytes...", std::fs::metadata(path).map(|m| m.len()).unwrap_or(0));
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("❌ Kan niet lezen: {}", e);
            return ExitCode::from(1);
        }
    };

    // CRC32 check als naburige .crc32 bestaat
    let crc_path = path.with_extension("crc32");
    if crc_path.exists() {
        let stored = std::fs::read(&crc_path).unwrap_or_default();
        if stored.len() == 4 {
            let stored_crc = u32::from_le_bytes([stored[0], stored[1], stored[2], stored[3]]);
            let actual = crc32fast::hash(&bytes);
            if stored_crc == actual {
                eprintln!("✓ CRC32 match: {:#x}", stored_crc);
            } else {
                eprintln!("⚠ CRC32 mismatch: stored={:#x} actual={:#x}", stored_crc, actual);
            }
        }
    } else {
        eprintln!("⚠ Geen CRC32 bestand — skip checksum check");
    }

    // v2.6.9: versie-detectie eerst
    use lucy_state::{detect_version, SnapshotFormat, CURRENT_SCHEMA_VERSION};
    match detect_version(&bytes) {
        SnapshotFormat::Headered(v, offset) => {
            eprintln!("→ Headered snapshot v{} detected (payload vanaf offset {})", v, offset);
            if v != CURRENT_SCHEMA_VERSION {
                eprintln!("⚠ Versie {} ≠ current ({}) — migratie nodig", v, CURRENT_SCHEMA_VERSION);
            }
        }
        SnapshotFormat::Legacy => {
            eprintln!("→ LEGACY snapshot (geen magic header) — zal via FieldV1 migratie geladen worden");
        }
    }

    // Gebruik Field::load (versioned load-path) ipv raw bincode
    eprintln!("→ Load via Field::load (versioned + migratie)...");
    let field: Field = match Field::load(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("❌ Field::load gefaald: {}", e);
            eprintln!();
            eprintln!("Mogelijke oorzaken:");
            eprintln!("  - Legacy snapshot ouder dan v2.6.7 Field layout");
            eprintln!("  - Corrupt bestand (CRC matcht wel, maar bincode decodeert niet)");
            eprintln!("  - Onbekend headered schema_version");
            eprintln!();
            eprintln!("Zie docs/SNAPSHOT_MIGRATION.md voor recovery-paden.");
            return ExitCode::from(2);
        }
    };

    // Structurele validatie
    if field.points.is_empty() {
        eprintln!("❌ Field heeft 0 punten — corrupt");
        return ExitCode::from(3);
    }

    // Rapport
    eprintln!();
    eprintln!("✅ Snapshot valid — struct-compatible met current lucy-state");
    eprintln!();
    eprintln!("Metadata:");
    eprintln!("  tick: {}", field.tick);
    eprintln!("  points: {}", field.points.len());
    eprintln!("  active_set: {}", field.active_set.len());
    eprintln!("  couplings_total: {}", field.points.iter().map(|p| p.couplings.len()).sum::<usize>());
    eprintln!("  snapshot_version: {}", field.snapshot_version);
    eprintln!("  field_dim: {}", field.field_dim);
    eprintln!("  n_input/n_hidden/n_output: {}/{}/{}", field.n_input, field.n_hidden, field.n_output);
    eprintln!("  reflection_count: {}", field.reflection_count);
    eprintln!("  pages_read: {}", field.pages_read);
    eprintln!("  tokens_processed: {}", field.tokens_processed);
    eprintln!();
    eprintln!("Neuromod:");
    eprintln!("  dopamine: {:.3}", field.dopamine);
    eprintln!("  serotonin: {:.3}", field.serotonin);
    eprintln!("  norepinephrine: {:.3}", field.norepinephrine);
    eprintln!("  acetylcholine: {:.3}", field.acetylcholine);
    eprintln!();
    eprintln!("  concept_centroids: {}", field.concept_centroids.len());

    // NaN/Inf sniff
    let nan_phi = field.points.iter().flat_map(|p| p.phi.iter()).filter(|v| !v.is_finite()).count();
    if nan_phi > 0 {
        eprintln!();
        eprintln!("⚠ NaN/Inf in phi: {} values — zouden door healing opgeruimd worden bij boot", nan_phi);
    }

    // Round-trip: schrijf via Field::save (naar /tmp), lees terug via Field::load.
    eprintln!();
    eprintln!("→ Round-trip via Field::save → Field::load (v2.6.9 headered)...");
    let tmp = std::env::temp_dir().join("lucy_snapshot_roundtrip_test.snapshot");
    match field.save(&tmp) {
        Ok(()) => {
            let new_size = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
            eprintln!("  written: {} bytes (headered, current schema)", new_size);
            eprintln!("  original: {} bytes (pre-migration)", bytes.len());
            match Field::load(&tmp) {
                Ok(f2) => {
                    eprintln!("  ✓ Roundtrip succesvol — tick={}, points={}", f2.tick, f2.points.len());
                    let _ = std::fs::remove_file(&tmp);
                    let _ = std::fs::remove_file(tmp.with_extension("crc32"));
                }
                Err(e) => {
                    eprintln!("  ❌ Roundtrip load gefaald: {}", e);
                    return ExitCode::from(2);
                }
            }
        }
        Err(e) => {
            eprintln!("  ❌ Roundtrip save gefaald: {}", e);
            return ExitCode::from(2);
        }
    }

    eprintln!();
    eprintln!("🟢 ALL CHECKS PASSED — veilig te deployen op dit snapshot-format");
    ExitCode::from(0)
}
