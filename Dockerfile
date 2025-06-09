# Gunakan image Python yang ringan
FROM python:3.10-slim

# Set working directory di container
WORKDIR /app

# Salin semua file ke container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Port default Railway
ENV PORT=8000

# Jalankan aplikasi
CMD ["python", "api/app.py"]
