FROM python:3.10

# Set working directory to /app
WORKDIR /app

# Copy the entire project
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Change working directory to /app/frontend
WORKDIR /app/frontend

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "Portfolio_Optimizer.py"]
