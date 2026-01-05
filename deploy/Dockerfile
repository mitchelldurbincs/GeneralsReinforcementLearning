# Dockerfile for the Generals.io game server
# Builds a lightweight headless game server binary

# -------- build stage --------
FROM golang:1.24-alpine AS builder

# Install build dependencies including protoc
RUN apk add --no-cache git protobuf protobuf-dev

# Install Go protoc plugins
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

WORKDIR /app

# Copy module files first to leverage Docker cache
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of your source tree
COPY . .

# Generate proto files (required for build)
RUN ./scripts/generate-protos.sh

# Build the headless game binary
RUN CGO_ENABLED=0 go build -o generals ./cmd/game_server

# -------- runtime stage --------
FROM alpine:3.21
WORKDIR /app
COPY --from=builder /app/generals ./generals

# Launch the game on container start
ENTRYPOINT ["./generals"]