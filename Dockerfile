# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Copy the project files
COPY . .

# Sync the project dependencies using uv
RUN uv sync

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["uv", "run", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
