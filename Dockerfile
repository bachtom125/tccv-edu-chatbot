FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Set the Hugging Face cache directory
ENV TRANSFORMERS_CACHE=/workspace/transformers_cache

# Create the cache directory and ensure it's writable
RUN mkdir -p /workspace/transformers_cache && chmod -R 777 /workspace/transformers_cache

# Copy app
COPY . .

# Expose the port
EXPOSE 7860

# Run the application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 7860 & python tele_bot.py"]