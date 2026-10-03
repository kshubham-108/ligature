FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The CPU-only wheel is a fraction of the size of the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# Only what serving imports. Installing before the source is copied keeps these layers cached
# when only the code changes.
RUN pip install numpy regex pyyaml huggingface_hub fastapi uvicorn pydantic

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PYTHONPATH=/home/user/app/src
WORKDIR /home/user/app

COPY --chown=user src/ src/
COPY --chown=user app/ app/

EXPOSE 7860
# slim images have no curl, so the check uses Python's standard library.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
