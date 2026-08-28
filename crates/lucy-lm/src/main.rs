//! lucy-lm CLI — kleine smoke/demo voor Lucy's eigen taal-decoder (veld-phi -> taal).
//! NB: dit is nog niet gewired in lucy-core; trainen en aanhaken is een volgende fase,
//! gated op compute/host. Zie de crate-docs.

use lucy_lm::{LmStatus, D_HIDDEN, D_PHI, VOCAB_SIZE};

fn main() {
    println!(
        "lucy-lm PhiDecoder — veld-phi ({D_PHI}) -> hidden ({D_HIDDEN}) -> vocab ({VOCAB_SIZE})"
    );
    println!("status: {:?}", LmStatus::default());
    println!(
        "Eigen, zelf-getrainde taal-mond (Visie A). Nog niet gewired in lucy-core; \
         trainen/aanhaken is een volgende fase, gated op compute."
    );
}
