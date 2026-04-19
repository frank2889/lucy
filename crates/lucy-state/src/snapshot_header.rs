//! # snapshot_header — Fixed-size header voor versioned snapshots
//!
//! Elk v2.6.9+ snapshot-bestand begint met een 20-byte header:
//!
//! ```text
//! offset 0:  magic: [u8; 4] = b"LUCY"
//! offset 4:  schema_version: u32    (1 = v2.6.7 Field, 2 = v2.6.8+ Field, 3 = v2.6.9+ Field, ...)
//! offset 8:  payload_size: u64      (bincode body size, post-header)
//! offset 16: reserved: u32 = 0      (future: compression flag, encryption flag, etc)
//! offset 20: -- payload begins --
//! ```
//!
//! **Backward compat:** snapshots geschreven vóór v2.6.9 hebben géén header —
//! hun eerste bytes zijn direct bincode-payload (begint met `points: Vec` length).
//! De `detect_version` functie probeert eerst magic bytes; als niet herkend,
//! valt het terug op legacy (schema_version = 0).
//!
//! **Design keuze:** vaste header ipv bincode-header omdat:
//!   1. Zero-parse detectie — 4 bytes vergelijken is O(1) en cannot fail
//!   2. Geen serde dependencies voor version-check
//!   3. Header-grootte is stabiel voor toekomstige uitbreidingen

use serde::{Deserialize, Serialize};

pub const MAGIC: [u8; 4] = *b"LUCY";
pub const HEADER_SIZE: usize = 20;

/// Current (most recent) schema version. Bump bij elke breaking wijziging aan
/// Field / Explorer / RetrievalStore struct layout.
///
/// Historie:
///   0 = legacy pre-v2.6.9 (geen header, payload = FieldV1 layout = v2.6.7-era)
///   1 = v2.6.9+ Field (bewijs-architectuur, snapshot_header wrapper, DIM=256)
///   2 = v2.9.0+ Field (DIM=512 upgrade, schema_v2 migration support)
///
/// Bump naar 3, 4, ... bij toekomstige breaking Field-wijzigingen.
/// Elke versie MOET een `schema_vN` module + `From<schema_vN::Field> for current::Field`
/// implementatie hebben, anders weigert `load_with_migration` te laden.
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

/// Known schema versions — `load_with_migration` ondersteunt deze.
pub const SUPPORTED_VERSIONS: &[u32] = &[0, 1, 2];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotHeader {
    pub magic: [u8; 4],
    pub schema_version: u32,
    pub payload_size: u64,
    pub reserved: u32,
}

impl SnapshotHeader {
    pub fn new(schema_version: u32, payload_size: u64) -> Self {
        Self { magic: MAGIC, schema_version, payload_size, reserved: 0 }
    }

    pub fn current(payload_size: u64) -> Self {
        Self::new(CURRENT_SCHEMA_VERSION, payload_size)
    }

    /// Serialize naar exact HEADER_SIZE bytes (fixed layout, niet bincode).
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut out = [0u8; HEADER_SIZE];
        out[0..4].copy_from_slice(&self.magic);
        out[4..8].copy_from_slice(&self.schema_version.to_le_bytes());
        out[8..16].copy_from_slice(&self.payload_size.to_le_bytes());
        out[16..20].copy_from_slice(&self.reserved.to_le_bytes());
        out
    }

    /// Parse van raw bytes. Returns None als magic niet klopt.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < HEADER_SIZE { return None; }
        let magic: [u8; 4] = bytes[0..4].try_into().ok()?;
        if magic != MAGIC { return None; }
        let schema_version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let payload_size = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
        let reserved = u32::from_le_bytes(bytes[16..20].try_into().ok()?);
        Some(Self { magic, schema_version, payload_size, reserved })
    }

    pub fn is_supported(&self) -> bool {
        SUPPORTED_VERSIONS.contains(&self.schema_version)
    }
}

/// Detectie-uitkomst voor `detect_version`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotFormat {
    /// Moderne versie met header. (version, payload_offset)
    Headered(u32, usize),
    /// Legacy pre-v2.6.9 snapshot — geen header, payload start op offset 0.
    /// Version wordt afgeleid uit inhoud (poging tot 1, 2 of 3 te deserialiseren).
    Legacy,
}

/// Detecteer of snapshot header heeft. Zero-cost voor moderne snapshots.
pub fn detect_version(bytes: &[u8]) -> SnapshotFormat {
    match SnapshotHeader::from_bytes(bytes) {
        Some(h) if h.is_supported() => SnapshotFormat::Headered(h.schema_version, HEADER_SIZE),
        _ => SnapshotFormat::Legacy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let h = SnapshotHeader::new(3, 80_000_000);
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);
        let parsed = SnapshotHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, h);
    }

    #[test]
    fn detect_legacy_no_magic() {
        // Bytes die NIET met "LUCY" beginnen — legacy snapshot
        let bytes = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
        assert_eq!(detect_version(&bytes), SnapshotFormat::Legacy);
    }

    #[test]
    fn detect_headered() {
        let h = SnapshotHeader::current(12345);
        let bytes = h.to_bytes();
        match detect_version(&bytes) {
            SnapshotFormat::Headered(v, offset) => {
                assert_eq!(v, CURRENT_SCHEMA_VERSION);
                assert_eq!(offset, HEADER_SIZE);
            }
            _ => panic!("Expected headered"),
        }
    }

    #[test]
    fn detect_unsupported_version() {
        let h = SnapshotHeader::new(9999, 100);
        let bytes = h.to_bytes();
        // Unsupported version → valt terug op Legacy (safer to attempt parse)
        assert_eq!(detect_version(&bytes), SnapshotFormat::Legacy);
    }
}
