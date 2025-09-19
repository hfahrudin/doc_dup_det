FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port for the web service
EXPOSE 8000

# Run the web server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
