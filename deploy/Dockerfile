# Dockerfile for the Generals.io game server
# Builds a lightweight headless game server binary

# -------- build stage --------
FROM golang:1.24-alpine AS builder
WORKDIR /app

# Copy module files first to leverage Docker cache
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of your source tree
COPY . .

# Build the headless game binary
RUN CGO_ENABLED=0 go build -o generals ./cmd/game_server

# -------- runtime stage --------
FROM alpine:3.21
WORKDIR /app
COPY --from=builder /app/generals ./generals

# Launch the game on container start
ENTRYPOINT ["./generals"]