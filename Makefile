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

generate-python-protos:
	./scripts/generate-python-protos.sh

generate-all-protos: generate-protos generate-python-protos

# Install required tools
install-tools:
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	@echo "Note: You also need to install protoc compiler manually"
	@echo "Visit: https://grpc.io/docs/protoc-installation/"

install-python-tools:
	python3 -m venv generalsrl
	./generalsrl/bin/pip install grpcio-tools grpcio protobuf	

# Clean build artifacts
clean:
	rm -rf bin/
	rm -rf pkg/api/
	rm -rf python/generals_pb/

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

# Docker commands
.PHONY: docker-build docker-up docker-down docker-logs docker-test docker-clean

docker-build:
	docker-compose -f docker-compose.dev.yml build

docker-up:
	docker-compose -f docker-compose.dev.yml up -d

docker-down:
	docker-compose -f docker-compose.dev.yml down

docker-logs:
	docker-compose -f docker-compose.dev.yml logs -f

docker-test:
	docker-compose -f docker-compose.dev.yml run --rm game-server go test ./...

docker-clean:
	docker-compose -f docker-compose.dev.yml down -v
	docker rmi $$(docker images -q "generalsreinforcementlearning_*") 2>/dev/null || true

# Run game server in Docker
docker-run-server:
	docker-compose -f docker-compose.dev.yml up game-server

# Build and run in one command
docker-dev: docker-build docker-up
