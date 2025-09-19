FROM python:3.11-slim

# Minimal system deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy requirements
COPY requirements.txt .

# Normalize CRLF -> LF, drop python-magic-bin (any form), add python-magic
RUN sed -i 's/\r$//' requirements.txt \
 && awk 'tolower($0) !~ /^python-magic-bin([[:space:]]|=|>|<|!|$)/' requirements.txt > requirements.clean \
 && printf '\npython-magic\n' >> requirements.clean \
 && mv requirements.clean requirements.txt

# Install deps with uv
RUN uv pip install --system -r requirements.txt


COPY app.py ./
COPY helpers/ ./helpers/

# Runtime dirs
RUN mkdir -p /app/data /app/faiss_bge_index

EXPOSE 8000

# Non-root
RUN useradd -ms /bin/bash appuser
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
