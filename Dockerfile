# 1. Use an official lightweight Python image
FROM python:3.10.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required for XGBoost and general operations
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code
# This includes the dataset/ and model/ folders
COPY . .

# 7. Expose the port Streamlit runs on
EXPOSE 8501

# 8. Healthcheck to ensure the container is running correctly
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 9. Command to run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]