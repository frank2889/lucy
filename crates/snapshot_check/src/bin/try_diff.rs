//! Inspect snapshot bytes around waar bincode faalt
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let bytes = std::fs::read(path).unwrap();
    println!("Total bytes: {}", bytes.len());

    // Probeer incrementeel te deserialiseren: lees Field step-by-step
    // via bincode's custom reader, zodat we zien WAAR het faalt.
    // (Bincode biedt geen native instrumentatie — we doen een manual trace.)
    //
    // Voor nu: we printen de LAATSTE 64 bytes + een paar offsets:
    let n = bytes.len();
    if n >= 64 {
        print!("Last 64 bytes (hex): ");
        for i in (n - 64)..n {
            print!("{:02x}", bytes[i]);
        }
        println!();
    }

    // Check: is er een 0x40 byte (64) ergens?
    let occurrences: Vec<usize> = bytes.iter().enumerate()
        .filter(|&(_, &b)| b == 64)
        .map(|(i, _)| i)
        .collect();
    println!("0x40 byte (64) appears at {} positions", occurrences.len());
    if occurrences.len() <= 20 {
        println!("Positions: {:?}", occurrences);
    }

    // Last few positions (last 10%)
    let tail_start = bytes.len() * 9 / 10;
    let tail_64: Vec<usize> = occurrences.iter()
        .filter(|&&i| i >= tail_start)
        .copied().collect();
    println!("0x40 in last 10% (from offset {}): count={}", tail_start, tail_64.len());
}
