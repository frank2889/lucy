// Lucy GPU Compute Shader — alle veldpunten parallel updaten
// Elke GPU thread verwerkt 1 veldpunt

const DIM: u32 = 64u;

// Veldpunt data (Structure of Arrays voor GPU-efficiëntie)
@group(0) @binding(0) var<storage, read_write> phi: array<f32>;           // [N * DIM]
@group(0) @binding(1) var<storage, read_write> prediction: array<f32>;    // [N * DIM]
@group(0) @binding(2) var<storage, read_write> memory: array<f32>;        // [N * DIM]
@group(0) @binding(3) var<storage, read_write> tension: array<f32>;       // [N]
@group(0) @binding(4) var<storage, read_write> tension_ema: array<f32>;   // [N]
@group(0) @binding(5) var<storage, read_write> energy: array<f32>;        // [N]
@group(0) @binding(6) var<storage, read_write> eta: array<f32>;           // [N]

// Emoties
@group(0) @binding(7) var<storage, read_write> curiosity: array<f32>;     // [N]
@group(0) @binding(8) var<storage, read_write> satisfaction: array<f32>;  // [N]
@group(0) @binding(9) var<storage, read_write> frustration: array<f32>;   // [N]

// Sparse koppelingen (CSR formaat)
@group(1) @binding(0) var<storage, read> coupling_offsets: array<u32>;    // [N + 1]
@group(1) @binding(1) var<storage, read> coupling_targets: array<u32>;    // [total_couplings]
@group(1) @binding(2) var<storage, read_write> coupling_weights: array<f32>; // [total_couplings]
@group(1) @binding(3) var<storage, read_write> coupling_momentum: array<f32>;

// Clamps
@group(1) @binding(4) var<storage, read> clamp_mask: array<u32>;          // [N] — 1 = geklemd
@group(1) @binding(5) var<storage, read> clamp_values: array<f32>;        // [N * DIM]

// Hyperparameters
struct Params {
    lambda: f32,
    beta: f32,
    alpha: f32,
    rho: f32,
    mu: f32,
    kappa: f32,
    omega: f32,
    ema_decay: f32,
    growth_threshold: f32,
    n_points: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(2) @binding(0) var<uniform> params: Params;

// === UPDATE PHI — elk punt parallel ===
@compute @workgroup_size(64)
fn update_phi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= params.n_points { return; }

    let base = id * DIM;

    // Geklemd punt → forceer phi
    if clamp_mask[id] == 1u {
        for (var d = 0u; d < DIM; d++) {
            prediction[base + d] = phi[base + d];
            phi[base + d] = clamp_values[base + d];
        }
        return;
    }

    // Bereken gradient vanuit buren
    let start = coupling_offsets[id];
    let end = coupling_offsets[id + 1u];
    let n_couplings = end - start;

    var grad: array<f32, 64>;
    for (var d = 0u; d < DIM; d++) {
        grad[d] = 0.0;
    }

    if n_couplings > 0u {
        for (var ci = start; ci < end; ci++) {
            let target = coupling_targets[ci];
            let weight = coupling_weights[ci];
            let target_base = target * DIM;

            for (var d = 0u; d < DIM; d++) {
                grad[d] += weight * (phi[base + d] - phi[target_base + d]);
            }
        }

        let inv_n = 1.0 / f32(n_couplings);
        for (var d = 0u; d < DIM; d++) {
            grad[d] *= inv_n;
        }
    }

    // Update phi
    for (var d = 0u; d < DIM; d++) {
        prediction[base + d] = phi[base + d];
        phi[base + d] -= params.lambda * grad[d];
        phi[base + d] -= params.alpha * sign(phi[base + d]);
        phi[base + d] = clamp(phi[base + d], -10.0, 10.0);
    }
}

// === UPDATE MEMORY — elk punt parallel ===
@compute @workgroup_size(64)
fn update_memory(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= params.n_points { return; }

    let base = id * DIM;

    for (var d = 0u; d < DIM; d++) {
        let eps = phi[base + d] - prediction[base + d];
        memory[base + d] = params.rho * memory[base + d] + (1.0 - params.rho) * eps;
    }
}

// === COMPUTE TENSION + EMOTIES — elk punt parallel ===
@compute @workgroup_size(64)
fn compute_tension_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= params.n_points { return; }

    let base = id * DIM;

    // Prediction error
    var pred_error = 0.0;
    for (var d = 0u; d < DIM; d++) {
        let eps = phi[base + d] - prediction[base + d];
        pred_error += eps * eps;
    }

    // Compute cost + empowerment
    let start = coupling_offsets[id];
    let end = coupling_offsets[id + 1u];
    let compute_cost = f32(end - start) * 0.01;
    var empowerment = 0.0;
    for (var ci = start; ci < end; ci++) {
        empowerment += abs(coupling_weights[ci]);
    }

    let new_tension = max(pred_error + params.kappa * compute_cost - params.omega * empowerment, 0.0);
    let old_ema = tension_ema[id];

    tension[id] = new_tension;
    tension_ema[id] = params.ema_decay * old_ema + (1.0 - params.ema_decay) * new_tension;

    // Emoties
    let emo_decay = 0.95;
    let surprise = max(sqrt(pred_error) - sqrt(old_ema), 0.0);
    curiosity[id] = emo_decay * curiosity[id] + (1.0 - emo_decay) * surprise;

    let improvement = max(old_ema - new_tension, 0.0);
    satisfaction[id] = emo_decay * satisfaction[id] + (1.0 - emo_decay) * improvement;

    let flatness = 1.0 / (1.0 + abs(new_tension - tension_ema[id]) * 10.0);
    let stuck = max(new_tension - params.growth_threshold, 0.0) * flatness;
    frustration[id] = emo_decay * frustration[id] + (1.0 - emo_decay) * stuck;
}

// === UPDATE WEIGHTS — elk punt parallel ===
@compute @workgroup_size(64)
fn update_weights_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= params.n_points { return; }

    let base = id * DIM;
    let start = coupling_offsets[id];
    let end = coupling_offsets[id + 1u];
    let point_eta = eta[id];

    // Prediction error
    var eps: array<f32, 64>;
    for (var d = 0u; d < DIM; d++) {
        eps[d] = phi[base + d] - prediction[base + d];
    }

    for (var ci = start; ci < end; ci++) {
        let target = coupling_targets[ci];
        let target_base = target * DIM;

        var dw = 0.0;
        for (var d = 0u; d < DIM; d++) {
            dw += eps[d] * (phi[base + d] - phi[target_base + d]);
        }
        dw /= f32(DIM);

        coupling_momentum[ci] = params.mu * coupling_momentum[ci] + (1.0 - params.mu) * dw;
        coupling_weights[ci] -= point_eta * dw + params.mu * coupling_momentum[ci];
        coupling_weights[ci] = clamp(coupling_weights[ci], -5.0, 5.0);
    }
}
