# Use the official Python image as a base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for Playwright browsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libcups2 \
    fonts-liberation \
    libpango-1.0-0 \
    libatk1.0-0 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install Playwright browsers
RUN python -m playwright install --with-deps

# Copy app source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000  

# Command to run your FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
