# Front end: built with Vite, served by nginx.
#
# Build context is the repository root, not web/, so the nginx config in
# docker/ is reachable. A context of web/ cannot see files above it.

FROM node:22-slim AS build

WORKDIR /app

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./

# Baked in at build time: Vite inlines env vars into the bundle, so this
# cannot be set at container start.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine AS serve

COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
