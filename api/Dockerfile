# Use the official Python image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install the dependencies
RUN pip install -r requirements.txt

# Install ImageMagick and dependencies
RUN apt-get update && \
    apt-get install -y imagemagick ffmpeg && \
    apt-get clean

# Copy the rest of the application code
COPY . /app/

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
