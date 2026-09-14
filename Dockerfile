FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend
COPY data/demo ./data/demo

ENV GAMES_PATH=data/demo/games.parquet
ENV NEIGHBORS_PATH=data/demo/hybrid_neighbors.json

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
