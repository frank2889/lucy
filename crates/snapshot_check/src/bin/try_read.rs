//! Test direct tegen lucy-state Field; moet exact werken met v2.6.7 snapshots
//! als v2.6.8 velden aan het eind staan.
//!
//! Run: cargo run -p snapshot_check --release --bin try_read -- /path/to/snapshot

use lucy_state::Field;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let bytes = std::fs::read(path).unwrap();
    eprintln!("Read {} bytes", bytes.len());

    // Try met tolerante bincode options
    let opts = bincode::options()
        .with_fixint_encoding()
        .allow_trailing_bytes();

    eprintln!("\n=== Try 1: bincode::deserialize (default, strict) ===");
    match bincode::deserialize::<Field>(&bytes) {
        Ok(f) => eprintln!("✅ OK — tick={}, points={}", f.tick, f.points.len()),
        Err(e) => eprintln!("❌ {}", e),
    }

    eprintln!("\n=== Try 2: DefaultOptions + allow_trailing_bytes ===");
    use bincode::Options;
    match opts.deserialize::<Field>(&bytes) {
        Ok(f) => eprintln!("✅ OK — tick={}, points={}", f.tick, f.points.len()),
        Err(e) => eprintln!("❌ {}", e),
    }
}
