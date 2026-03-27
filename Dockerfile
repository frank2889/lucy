FROM rust:latest AS builder

WORKDIR /lucy
COPY . .
RUN cargo build --release --workspace

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    rustc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /lucy

# Binary
COPY --from=builder /lucy/target/release/lucy-core /lucy/lucy-core

# Source code — Lucy moet zichzelf kunnen lezen en aanpassen
COPY --from=builder /lucy/crates/ /lucy/crates/
COPY --from=builder /lucy/Cargo.toml /lucy/Cargo.toml

# Persistent directories
RUN mkdir -p /lucy/data /lucy/modules

VOLUME ["/lucy/data", "/lucy/modules"]

# Environment
ENV RUST_LOG=info

CMD ["./lucy-core"]
