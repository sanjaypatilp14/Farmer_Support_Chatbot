# ----------------------------
# Frontend Dockerfile (React + Vite)
# ----------------------------

# 1️⃣ Use Node.js base image
FROM node:18-alpine

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# 4️⃣ Copy frontend code
COPY . .

# 5️⃣ Build production bundle
RUN npm run build

# 5️⃣ Expose the Vite development port

EXPOSE 5173

# 7️⃣ Start server
CMD ["npm", "run", "dev", "--", "--host"]
