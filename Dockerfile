# Use the official Python image as a base
FROM python:3.9-slim

# Set environment variables to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Update and install necessary system packages, including Tesseract and Hindi language data
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-hin \
    libtesseract-dev \
    wget && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Set the TESSDATA_PREFIX environment variable to point to the Tesseract language data
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Create a working directory
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt /app/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . /app/

# Command to run the Python script when the container starts
CMD ["python", "your_script.py"]
