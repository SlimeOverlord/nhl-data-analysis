FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file first for better caching
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Default environment variables (replace with real values or use docker --env)
ENV WANDB_ENTITY=IFT6758-2025-A09
ENV WANDB_PROJECT=ift6758-milestone2
ENV WANDB_API_KEY=your_wandb_api_key_here
ENV FLASK_APP=server/app.py

# Expose port
EXPOSE 5000

# Use gunicorn to serve the Flask app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "server.app:app"]
