FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend
COPY data/trained ./data/trained

ENV GAMES_PATH=data/trained/games.parquet
ENV NEIGHBORS_PATH=data/trained/hybrid_neighbors.json

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
