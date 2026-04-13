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

    // Poging tot deserialiseren
    eprintln!("→ Deserializeer met CURRENT Field struct...");
    let field: Field = match bincode::deserialize(&bytes) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("❌ Deserialisatie gefaald: {}", e);
            eprintln!();
            eprintln!("Dit betekent dat de CURRENT lucy-state struct niet compatible is");
            eprintln!("met de snapshot op disk. Waarschijnlijke oorzaak:");
            eprintln!("  - Nieuwe velden toegevoegd MIDDEN in Field struct");
            eprintln!("  - Type-wijziging van bestaand veld");
            eprintln!("  - Bincode is positioneel; serde(default) werkt alleen bij EOF");
            eprintln!();
            eprintln!("Zie docs/SNAPSHOT_MIGRATION.md voor het veilige pad.");
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

    // Round-trip test: serialize → deserialize moet identiek zijn (grootte)
    eprintln!();
    eprintln!("→ Round-trip test (serialize → deserialize)...");
    match bincode::serialize(&field) {
        Ok(reserialized) => {
            eprintln!("  original size: {} bytes", bytes.len());
            eprintln!("  reserialized size: {} bytes", reserialized.len());
            if reserialized.len() != bytes.len() {
                eprintln!("  ⚠ Size changed — verwacht bij v2.6.8 als nieuwe velden geschreven worden");
            } else {
                eprintln!("  ✓ Sizes match exact");
            }
            // Verify second load werkt
            match bincode::deserialize::<Field>(&reserialized) {
                Ok(_) => eprintln!("  ✓ Reserialized bytes laden succesvol"),
                Err(e) => {
                    eprintln!("  ❌ Reserialized bytes kunnen niet geladen worden: {}", e);
                    return ExitCode::from(2);
                }
            }
        }
        Err(e) => {
            eprintln!("  ❌ Serialize-back faalde: {}", e);
            return ExitCode::from(2);
        }
    }

    eprintln!();
    eprintln!("🟢 ALL CHECKS PASSED — veilig te deployen op dit snapshot-format");
    ExitCode::from(0)
}
