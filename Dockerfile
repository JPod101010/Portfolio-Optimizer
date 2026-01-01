FROM python:3.12-slim

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install deps (uv is now in /bin/uv, so it's globally accessible)
RUN uv sync --frozen

# Copy app
COPY . .

# Ensure run.sh is executable
RUN chmod +x run.sh

CMD ["./run.sh", "--db-init"]