FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .
COPY context_optimizer ./context_optimizer/

# Run
EXPOSE 8000
CMD ["python", "main.py"]
