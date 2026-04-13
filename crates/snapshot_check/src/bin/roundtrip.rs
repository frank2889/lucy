//! Roundtrip test: serialize en deserialize vers Field → moet altijd werken
use lucy_state::Field;

fn main() {
    let f = Field::new();
    println!("Fresh Field created, tick={}", f.tick);

    let bytes = bincode::serialize(&f).expect("serialize fresh");
    println!("Serialized {} bytes", bytes.len());

    match bincode::deserialize::<Field>(&bytes) {
        Ok(f2) => println!("✅ Roundtrip OK — tick={}, points={}", f2.tick, f2.points.len()),
        Err(e) => println!("❌ Roundtrip FAILED: {}", e),
    }
}
