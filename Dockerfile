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

# Copy application code
COPY . .

# Install web dependencies and build the web interface
WORKDIR /app/web
# Configure npm/yarn for better network reliability
RUN npm config set fetch-retries 5 && \
    npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000

# Try yarn first, fallback to npm if yarn fails
RUN (yarn config set network-timeout 300000 && \
    yarn config set network-concurrency 1 && \
    yarn install --frozen-lockfile --network-timeout 300000 --verbose && \
    yarn build) || \
    (echo "Yarn failed, trying npm..." && \
    npm install --no-audit --no-fund --prefer-offline --progress=false && \
    npm run build)

WORKDIR /app

ENV USER_AGENT="CLIzilla/3.7 (🤖 still learning; may or may not eat your RAM; report bugs to mom)"

ENTRYPOINT ["uv", "run", "main.py"]

CMD []
