//! # perception — Lucy's Universele Zintuigen
//!
//! Elke input (tekst, beeld, audio, sensor, JSON) wordt geprojecteerd naar
//! dezelfde 64-dimensionale ruimte. Het veld maakt niet uit WAAR de input
//! vandaan komt — 64 dims is 64 dims. Lucy leert zelf de patronen.
//!
//! Dit is de basis voor multimodaliteit: Ring deurbel, telefoon, microfoon,
//! thermometer — alles wordt een phi-vector.

use lucy_state::DIM;

/// Wat voor soort input is het?
#[derive(Debug, Clone, PartialEq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Sensor,
    Json,
}

impl Modality {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "image" | "img" | "picture" | "photo" => Modality::Image,
            "audio" | "sound" | "wav" | "mp3" => Modality::Audio,
            "sensor" | "temperature" | "humidity" | "motion" => Modality::Sensor,
            "json" | "api" | "data" => Modality::Json,
            _ => Modality::Text,
        }
    }
}

/// Universele perceptie: converteer ruwe bytes naar een 64-dim phi-vector.
/// Elke modaliteit heeft zijn eigen projectie-strategie.
pub fn perceive(modality: &Modality, data: &[u8]) -> [f32; DIM] {
    match modality {
        Modality::Text => perceive_text(data),
        Modality::Image => perceive_image(data),
        Modality::Audio => perceive_audio(data),
        Modality::Sensor => perceive_sensor(data),
        Modality::Json => perceive_json(data),
    }
}

/// Tekst → 64-dim: character-level hash embedding.
/// Simpel maar effectief voor het veld om patronen in te leren.
fn perceive_text(data: &[u8]) -> [f32; DIM] {
    let mut phi = [0.0f32; DIM];
    let text = String::from_utf8_lossy(data);

    for (i, ch) in text.chars().enumerate() {
        let idx = (ch as usize + i * 7) % DIM;
        phi[idx] += 1.0;
    }

    // Normaliseer
    let norm: f32 = phi.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
    for v in &mut phi {
        *v /= norm;
    }

    phi
}

/// Beeld → 64-dim: resize naar 8×8 grayscale → direct als phi-vector.
/// Werkt met ruwe bytes (grayscale) of neemt gemiddelde van RGB.
fn perceive_image(data: &[u8]) -> [f32; DIM] {
    let mut phi = [0.0f32; DIM];

    if data.is_empty() { return phi; }

    // Strategie: verdeel data in 64 gelijke blokken, neem gemiddelde per blok
    let block_size = (data.len() / DIM).max(1);
    for d in 0..DIM {
        let start = d * block_size;
        let end = ((d + 1) * block_size).min(data.len());
        if start >= data.len() { break; }

        let sum: f32 = data[start..end].iter().map(|&b| b as f32).sum();
        let count = (end - start) as f32;
        phi[d] = sum / count / 255.0; // Normaliseer naar [0, 1]
    }

    // Center rond 0 (het veld werkt beter met symmetrische waarden)
    let mean: f32 = phi.iter().sum::<f32>() / DIM as f32;
    for v in &mut phi {
        *v -= mean;
    }

    phi
}

/// Audio → 64-dim: simplified spectral features.
/// Verdeel in 64 frequentie-achtige banden via lokale energie.
fn perceive_audio(data: &[u8]) -> [f32; DIM] {
    let mut phi = [0.0f32; DIM];

    if data.is_empty() { return phi; }

    // Interpreteer als 16-bit signed PCM (meest voorkomend)
    let samples: Vec<f32> = data.chunks(2)
        .map(|chunk| {
            if chunk.len() == 2 {
                i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0
            } else {
                0.0
            }
        })
        .collect();

    if samples.is_empty() { return phi; }

    // Verdeel samples in 64 banden, bereken energie per band
    let band_size = (samples.len() / DIM).max(1);
    for d in 0..DIM {
        let start = d * band_size;
        let end = ((d + 1) * band_size).min(samples.len());
        if start >= samples.len() { break; }

        let energy: f32 = samples[start..end].iter().map(|s| s * s).sum();
        phi[d] = energy.sqrt();
    }

    // Normaliseer
    let norm: f32 = phi.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
    for v in &mut phi {
        *v /= norm;
    }

    phi
}

/// Sensor → 64-dim: parse numerieke waarden, pad naar 64 dims.
/// Input format: comma-separated floats als UTF-8 string.
fn perceive_sensor(data: &[u8]) -> [f32; DIM] {
    let mut phi = [0.0f32; DIM];
    let text = String::from_utf8_lossy(data);

    for (i, part) in text.split(',').enumerate() {
        if i >= DIM { break; }
        if let Ok(val) = part.trim().parse::<f32>() {
            phi[i] = val;
        }
    }

    phi
}

/// JSON → 64-dim: flatten numerieke waarden uit JSON, hashed key placement.
fn perceive_json(data: &[u8]) -> [f32; DIM] {
    let mut phi = [0.0f32; DIM];
    let text = String::from_utf8_lossy(data);

    // Simpele JSON value extractie (geen full parser nodig)
    // Zoek naar "key": number patronen
    let mut dim_idx = 0;
    for part in text.split(',') {
        let part = part.trim().trim_matches(|c| c == '{' || c == '}' || c == '[' || c == ']');
        if let Some(colon_pos) = part.find(':') {
            let key = &part[..colon_pos].trim().trim_matches('"');
            let val_str = part[colon_pos + 1..].trim().trim_matches('"');

            // Hash de key naar een dimensie-index
            let key_hash: usize = key.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
            let target_dim = key_hash % DIM;

            if let Ok(val) = val_str.parse::<f32>() {
                phi[target_dim] = val;
            } else if val_str == "true" {
                phi[target_dim] = 1.0;
            } else if val_str == "false" {
                phi[target_dim] = 0.0;
            }
        } else if let Ok(val) = part.trim().parse::<f32>() {
            phi[dim_idx % DIM] = val;
            dim_idx += 1;
        }
    }

    phi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceive_text() {
        let phi = perceive(&&Modality::Text, b"Hello Lucy!");
        let norm: f32 = phi.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "text should be normalized");
    }

    #[test]
    fn test_perceive_sensor() {
        let phi = perceive(&Modality::Sensor, b"21.5, 65.0, 1013.25");
        assert!((phi[0] - 21.5).abs() < 0.01);
        assert!((phi[1] - 65.0).abs() < 0.01);
        assert!((phi[2] - 1013.25).abs() < 0.01);
    }

    #[test]
    fn test_perceive_image_empty() {
        let phi = perceive(&Modality::Image, &[]);
        assert!(phi.iter().all(|&v| v == 0.0));
    }
}
