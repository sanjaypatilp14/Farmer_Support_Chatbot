# ----------------------------
# Backend Dockerfile (Flask + TensorFlow + eSpeak)
# ----------------------------

# 1️⃣ Base image
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt wordnet omw-1.4

# 4️⃣ Install eSpeak for voice synthesis (supports Kannada, Telugu, etc.)
RUN apt-get update && apt-get install -y espeak-ng && rm -rf /var/lib/apt/lists/*

# 5️⃣ Copy the backend code
COPY . .

# 6️⃣ Expose Flask port
EXPOSE 5000

# 7️⃣ Command to run Flask app
CMD ["python", "chat.py"]
