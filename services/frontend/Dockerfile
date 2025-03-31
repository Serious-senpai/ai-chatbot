# Reference: https://docs.docker.com/reference/dockerfile/
FROM node:23.8-alpine AS builder

ARG VITE_APP_BASE_API_URL
COPY services/frontend /app

WORKDIR /app
RUN npm install
RUN npm run build

FROM nginx:1.27-alpine AS runner

COPY services/frontend/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /app/dist /app