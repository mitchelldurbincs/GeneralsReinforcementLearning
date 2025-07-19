#!/bin/bash

# Script to generate Go code from proto files

set -e

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Create output directories if they don't exist
mkdir -p pkg/api/game/v1
mkdir -p pkg/api/common/v1

# Generate common proto files
echo "Generating common proto files..."
protoc \
    --go_out=pkg/api \
    --go_opt=paths=source_relative \
    --go-grpc_out=pkg/api \
    --go-grpc_opt=paths=source_relative \
    -I proto \
    proto/common/v1/*.proto 2>/dev/null || echo "No common proto files to generate yet"

# Generate game proto files
echo "Generating game proto files..."
protoc \
    --go_out=pkg/api \
    --go_opt=paths=source_relative \
    --go-grpc_out=pkg/api \
    --go-grpc_opt=paths=source_relative \
    -I proto \
    proto/game/v1/*.proto 2>/dev/null || echo "No game proto files to generate yet"

echo "Proto generation complete!"