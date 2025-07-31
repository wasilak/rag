# Multi-stage build for better performance and smaller final image
FROM node:18-alpine AS web-builder

# Install dependencies for native modules
RUN apk add --no-cache python3 make g++

WORKDIR /app/web

# Copy package files first for better layer caching
COPY web/package*.json web/yarn.lock ./

# Configure npm/yarn for better performance
RUN npm config set fetch-retries 5 && \
    npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000 && \
    npm config set cache /tmp/.npm && \
    npm config set prefer-offline true

# Install dependencies with optimizations
RUN (yarn config set network-timeout 300000 && \
    yarn config set network-concurrency 1 && \
    yarn config set cache-folder /tmp/.yarn && \
    yarn install --frozen-lockfile --network-timeout 300000 --prefer-offline --silent) || \
    (echo "Yarn failed, trying npm..." && \
    npm ci --no-audit --no-fund --prefer-offline --progress=false --silent)

# Copy source files and build
COPY web/ .
RUN (yarn build --silent) || (npm run build --silent)

# Clean up cache
RUN rm -rf /tmp/.npm /tmp/.yarn node_modules

# Final stage
FROM python:3-slim

# Install Node.js and Yarn for web interface development
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g yarn \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

# Copy Python dependencies
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync

# Copy built web interface from builder stage
COPY --from=web-builder /app/web/build ./web/build

# Copy Python application code
COPY . .

ENV USER_AGENT="CLIzilla/3.7 (🤖 still learning; may or may not eat your RAM; report bugs to mom)"

ENTRYPOINT ["uv", "run", "main.py"]

CMD []
