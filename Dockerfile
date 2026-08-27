FROM rust:bookworm AS builder

WORKDIR /lucy
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY docs/ docs/
RUN cargo build --release -p lucy-core

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl3 \
    gcc \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Installeer Rust toolchain — Lucy moet zichzelf kunnen hercompileren
COPY --from=builder /usr/local/rustup /usr/local/rustup
COPY --from=builder /usr/local/cargo /usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"

WORKDIR /lucy

# Binary
COPY --from=builder /lucy/target/release/lucy-core /lucy/lucy-core

# Volledige source — Lucy kan zichzelf aanpassen en hercompileren
COPY --from=builder /lucy/crates/ /lucy/crates/
COPY --from=builder /lucy/Cargo.toml /lucy/Cargo.toml
COPY --from=builder /lucy/Cargo.lock /lucy/Cargo.lock

# v3.1: geen dashboard/frontend meer — Lucy is headless, alleen de HTTP JSON-API.

# v2.5: GEEN mkdir data/ of VOLUME declaratie in Dockerfile
# Data en modules worden ALLEEN via docker-compose named volumes gemount.
# VOLUME declaratie in Dockerfile maakt anonymous volumes die conflicteren
# met named volumes bij container recreatie → DATA VERLIES.
# De directories worden automatisch aangemaakt door Docker bij volume mount.

# Environment
ENV RUST_LOG=info

CMD ["./lucy-core"]
