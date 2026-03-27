use lucy_state::*;
use rand::Rng;

/// Initialiseer het veld intelligent vanuit Lucy's eigen source code
pub fn bootstrap_from_source(sources: &[(std::path::PathBuf, String)]) -> Field {
    let mut field = Field::new();
    let mut rng = rand::thread_rng();

    // Tel totale functies/structuren als basis voor topologie
    let mut clusters: Vec<(String, usize)> = Vec::new(); // (naam, startindex)

    for (path, content) in sources {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Elke functie/struct = een cluster van veldpunten
        let mut item_name;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("pub fn ")
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("pub struct ")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("impl ")
            {
                // Nieuwe functie/struct gevonden
                item_name = format!("{}::{}", file_name, trimmed.split('(').next()
                    .unwrap_or(trimmed).split('{').next().unwrap_or(trimmed).trim());
                let start_idx = field.points.len();
                clusters.push((item_name.clone(), start_idx));

                // Creëer ~10 veldpunten per functie
                let cluster_size = 10;
                for i in 0..cluster_size {
                    let mut point = FieldPoint::new();

                    // Seed phi vanuit de source code bytes (niet random!)
                    let bytes = item_name.as_bytes();
                    for d in 0..DIM {
                        let byte_idx = (i * DIM + d) % bytes.len().max(1);
                        point.phi[d] = (bytes[byte_idx] as f32 / 255.0) * 2.0 - 1.0;
                        // Voeg lichte perturbatie toe voor diversiteit
                        point.phi[d] += rng.gen_range(-0.1..0.1);
                    }

                    point.energy = 1.0;
                    field.points.push(point);
                }
            }
        }
    }

    // Creëer koppelingen BINNEN clusters (dicht verbonden)
    for (_, start_idx) in &clusters {
        let cluster_end = (*start_idx + 10).min(field.points.len());
        for i in *start_idx..cluster_end {
            for j in *start_idx..cluster_end {
                if i != j && rng.gen_bool(0.5) {
                    let weight = rng.gen_range(-0.5..0.5); // signed!
                    field.points[i].couplings.push(Coupling {
                        target: j as u32,
                        weight,
                        momentum: 0.0,
                    });
                }
            }
        }
    }

    // Creëer koppelingen TUSSEN clusters (spaarzaam — zoals long-range verbindingen)
    for i in 0..clusters.len() {
        for j in 0..clusters.len() {
            if i != j && rng.gen_bool(0.1) {
                // Verbind een random punt uit cluster i met een random punt uit cluster j
                let src_start = clusters[i].1;
                let dst_start = clusters[j].1;
                let src = src_start + rng.gen_range(0..10.min(field.points.len() - src_start));
                let dst = dst_start + rng.gen_range(0..10.min(field.points.len() - dst_start));

                let weight = rng.gen_range(-0.3..0.3);
                field.points[src].couplings.push(Coupling {
                    target: dst as u32,
                    weight,
                    momentum: 0.0,
                });
            }
        }
    }

    // Markeer eerste ~5% als sensorische punten, laatste ~5% als motor punten
    let n = field.points.len();
    let sensory_end = n / 20;
    let motor_start = n - n / 20;

    for i in 0..sensory_end {
        field.points[i].flags |= FLAG_SENSORY;
    }
    for i in motor_start..n {
        field.points[i].flags |= FLAG_MOTOR;
    }

    // ~20% inhibitory (zoals het echte brein)
    for point in &mut field.points {
        if rng.gen_bool(0.2) {
            point.flags |= FLAG_INHIBITORY;
            // Inhibitory punten hebben negatieve gewichten
            for c in &mut point.couplings {
                c.weight = -c.weight.abs();
            }
        }
    }

    field.total_energy = n as f32;
    field.rebuild_active_set();

    tracing::info!(
        points = n,
        clusters = clusters.len(),
        active = field.active_set.len(),
        "Field bootstrapped from source code"
    );

    field
}
