FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the FastEmbed model (avoids cold start)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Copy application
COPY main.py .
COPY context_optimizer ./context_optimizer/

# Run
EXPOSE 8000
CMD ["python", "main.py"]
