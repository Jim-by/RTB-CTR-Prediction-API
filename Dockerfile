FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

RUN mkdir -p /app/models

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--timeout-keep-alive", "60"]
