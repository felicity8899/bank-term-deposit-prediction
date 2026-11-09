# Bank Term Deposit Prediction — prediction service image.
# Builds a small container that serves predict.py with uvicorn.
# model.bin must exist next to this Dockerfile (run `python train.py` first);
# it is committed to the repo so the image builds anywhere.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching: this layer only
# changes when requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + trained model artifact.
COPY predict.py .
COPY model.bin .

EXPOSE 9696

# Shell form so $PORT is expanded at container start: cloud platforms
# (e.g. Render) inject their own PORT; locally it defaults to 9696.
CMD uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}
