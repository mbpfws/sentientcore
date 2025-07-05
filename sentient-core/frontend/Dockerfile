FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy application code
COPY . .

# Build the application
RUN npm run build

# Expose port for Next.js
EXPOSE 3000

# Command to run the application in production mode
CMD ["npm", "start"] 