FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for SQLite and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Snowflake Arctic Embed XS model
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='Snowflake/snowflake-arctic-embed-xs')"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
