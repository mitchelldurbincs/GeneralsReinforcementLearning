.PHONY: all build test clean generate-protos install-tools

# Default target
all: build

# Build targets
build:
	go build -o bin/game_server ./cmd/game_server
	go build -o bin/ui_client ./cmd/ui_client

# Test targets
test:
	go test ./...

test-coverage:
	go test -cover ./...

test-verbose:
	go test -v ./...

# Proto generation
generate-protos:
	./scripts/generate-protos.sh

# Install required tools
install-tools:
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	@echo "Note: You also need to install protoc compiler manually"
	@echo "Visit: https://grpc.io/docs/protoc-installation/"

# Clean build artifacts
clean:
	rm -rf bin/
	rm -rf pkg/api/

# Development helpers
run-server:
	go run cmd/game_server/main.go

run-ui:
	go run cmd/ui_client/main.go

# Linting and formatting
fmt:
	go fmt ./...

lint:
	golangci-lint run

# Docker targets
docker-build:
	docker build -t generals-game-server .

docker-run:
	docker run -p 50051:50051 generals-game-server