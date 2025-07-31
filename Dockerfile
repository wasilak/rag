FROM python:3-slim

# Install Node.js and Yarn for web interface development
RUN apt-get update && apt-get install -y \
    curl \
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

# Install web dependencies (but don't build - will be built at runtime if needed)
WORKDIR /app/web
RUN yarn install --frozen-lockfile --network-timeout 100000

WORKDIR /app

ENV USER_AGENT="CLIzilla/3.7 (🤖 still learning; may or may not eat your RAM; report bugs to mom)"

ENTRYPOINT ["uv", "run", "main.py"]

CMD []
