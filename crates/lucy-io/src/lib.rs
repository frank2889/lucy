//! # lucy-io — Lucy's Zintuigen en Handen
//!
//! I/O laag die Lucy verbindt met de buitenwereld:
//!
//! - **http** — HTTP client (GET/POST) voor het web. Lucy's "ogen" naar het internet.
//! - **fs** — Filesystem interface voor lezen/schrijven. Lucy's "handen" naar disk.
//! - **compiler** — (Toekomstig) Zelf-compilatie vermogen.
//! - **config** — (Toekomstig) Runtime configuratie.
//!
//! Bewust gescheiden van `lucy-core` zodat de I/O laag vervangbaar is
//! (bijv. voor testing met mock HTTP responses).

pub mod http;
pub mod fs;
pub mod compiler;
pub mod config;
