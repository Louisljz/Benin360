FROM python:3.11

WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Install ImageMagick
RUN apt-get update && \
    apt-get install -y imagemagick ffmpeg && \
    apt-get clean

COPY . /app/

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]