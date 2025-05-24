# FROM node:20-alpine AS development
# WORKDIR /app_root
# COPY package.json .
# RUN npm install --force
# COPY . .
# CMD ["npm", "run", "dev"]

# Production build
FROM node:20-alpine AS production-dependencies-env
ENV ROOT_PATH=/app_root
WORKDIR ${ROOT_PATH}
COPY package.json package-lock.json .
RUN npm ci --ignore-scripts --force

FROM node:20-alpine AS build-env
ARG COMMIT_HASH
ENV VITE_COMMIT_HASH=$COMMIT_HASH
ENV ROOT_PATH=/app_root
WORKDIR ${ROOT_PATH}
COPY . .
COPY --from=production-dependencies-env ${ROOT_PATH}/node_modules ${ROOT_PATH}/node_modules
RUN npm run build

FROM node:20-alpine AS production
ENV ROOT_PATH=/app_root
WORKDIR ${ROOT_PATH}
COPY package.json package-lock.json ${ROOT_PATH}/
COPY --from=production-dependencies-env ${ROOT_PATH}/node_modules ${ROOT_PATH}/node_modules
COPY --from=build-env ${ROOT_PATH}/build ${ROOT_PATH}/build
CMD ["npm", "run", "start"]