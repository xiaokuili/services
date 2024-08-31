FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install TTS flask flask_cors

COPY app.py .

EXPOSE 8721

CMD ["python", "app.py"]
