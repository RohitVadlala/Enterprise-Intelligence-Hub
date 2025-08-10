# ===== base =====
FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy only dependency files first (better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Cloud Run sets PORT, so use it; Streamlit must bind 0.0.0.0
ENV PORT=8080
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# Streamlit config override (optional but clean)
RUN mkdir -p ~/.streamlit && \
    printf "\
[server]\n\
headless = true\n\
port = \$PORT\n\
address = \"0.0.0.0\"\n\
enableCORS = false\n\
[browser]\n\
gatherUsageStats = false\n" > ~/.streamlit/config.toml

# Entrypoint
CMD ["bash", "-lc", "streamlit run app/streamlit_app.py"]
