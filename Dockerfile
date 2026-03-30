FROM rust:bookworm AS builder

WORKDIR /lucy
COPY . .
RUN cargo build --release -p lucy-core

# Dashboard build
FROM node:20-slim AS dashboard
WORKDIR /dashboard
COPY lucy-dashboard/package.json lucy-dashboard/package-lock.json ./
RUN npm ci
COPY lucy-dashboard/ ./
RUN npm run build

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl3 \
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

# Persistent directories
RUN mkdir -p /lucy/data /lucy/modules /lucy/dashboard

# Dashboard static files
COPY --from=dashboard /dashboard/dist/ /lucy/dashboard/

VOLUME ["/lucy/data", "/lucy/modules"]

# Environment
ENV RUST_LOG=info

CMD ["./lucy-core"]
