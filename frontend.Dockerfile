FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-frontend.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-frontend.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
