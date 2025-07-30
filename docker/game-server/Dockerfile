# Multi-stage build for optimal size
FROM golang:1.24-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make

WORKDIR /build

# Copy go mod files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build with optimizations
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s" \
    -o game-server \
    ./cmd/game_server

# Runtime stage
FROM alpine:3.21

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN adduser -D -g '' appuser

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/game-server .

# Copy any config files if they exist
# COPY --from=builder /build/configs ./configs

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose gRPC port
EXPOSE 50051

# Health check - assumes your server implements a health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/app/game-server", "health"] || exit 1

ENTRYPOINT ["/app/game-server"]