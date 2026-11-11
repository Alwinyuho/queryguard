# API image. Uses an OpenAI-compatible LLM endpoint (set LLM_BASE_URL, LLM_MODEL, LLM_API_KEY),
# because a free CPU host cannot run the local 4-bit model. Mount the BIRD databases at /data/bird.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QUERYGUARD_HOME=/app \
    BIRD_DIR=/data/bird

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ui]"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 7860
CMD ["uvicorn", "queryguard.api:app", "--host", "0.0.0.0", "--port", "7860"]
