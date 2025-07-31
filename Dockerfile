# syntax=docker/dockerfile:1.17
# Multi-stage build for better performance and smaller final image
FROM node:18-alpine AS web-builder

# Install dependencies for native modules
RUN apk add --no-cache python3 make g++

WORKDIR /app/web

# Copy package files first for better layer caching
COPY web/package*.json web/yarn.lock ./

# Configure npm/yarn for better performance and caching
RUN npm config set fetch-retries 5 && \
    npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000 && \
    npm config set cache /tmp/.npm && \
    npm config set prefer-offline true && \
    npm config set maxsockets 1 && \
    npm config set progress false

# Install dependencies with aggressive optimizations and parallel processing
RUN --mount=type=cache,target=/tmp/.yarn \
    yarn config set network-timeout 300000 && \
    yarn config set network-concurrency 8 && \
    yarn config set cache-folder /tmp/.yarn && \
    yarn config set prefer-offline true && \
    yarn install --frozen-lockfile --network-timeout 300000 --prefer-offline --silent --no-progress --ignore-engines --production=false

# Copy source files and build with optimizations
COPY web/ .
RUN --mount=type=cache,target=/tmp/.yarn \
    yarn build --silent --no-progress

# Clean up cache and dependencies to reduce image size
RUN rm -rf /tmp/.npm /tmp/.yarn node_modules package-lock.json yarn.lock

# Final stage
FROM python:3.12-slim

# Install Node.js and Yarn for web interface development
# Combine all apt operations in a single RUN to reduce layers
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g yarn \
    && apt-get clean

# Install uv in a single layer with cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir uv

WORKDIR /app

# Copy Python dependencies
COPY pyproject.toml uv.lock ./

# Install Python dependencies with optimizations and cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy built web interface from builder stage
COPY --from=web-builder /app/web/build ./web/build

# Copy Python application code (excluding web source files)
COPY libs/ ./libs/
COPY main.py ./
COPY Makefile ./
COPY renovate.json ./
COPY setup.cfg ./

ENV USER_AGENT="CLIzilla/3.7 (🤖 still learning; may or may not eat your RAM; report bugs to mom)"

ENTRYPOINT ["uv", "run", "main.py"]

CMD []
