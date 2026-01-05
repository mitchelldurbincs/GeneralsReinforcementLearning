#!/bin/bash

# Script to generate Go code from proto files
# Fails loudly on errors instead of silently continuing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure we're in the project root
cd "$(dirname "$0")/.."

echo "=== Proto Generation Script ==="

# Check for protoc
if ! command -v protoc &> /dev/null; then
    echo -e "${RED}ERROR: protoc is not installed${NC}"
    echo ""
    echo "Please install protoc:"
    echo "  - macOS:  brew install protobuf"
    echo "  - Ubuntu: apt install protobuf-compiler"
    echo "  - Or download from: https://github.com/protocolbuffers/protobuf/releases"
    echo ""
    echo "Then run: make install-tools"
    exit 1
fi

# Check for protoc-gen-go
if ! command -v protoc-gen-go &> /dev/null; then
    echo -e "${RED}ERROR: protoc-gen-go is not installed${NC}"
    echo ""
    echo "Run: make install-tools"
    echo "Or:  go install google.golang.org/protobuf/cmd/protoc-gen-go@latest"
    exit 1
fi

# Check for protoc-gen-go-grpc
if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo -e "${RED}ERROR: protoc-gen-go-grpc is not installed${NC}"
    echo ""
    echo "Run: make install-tools"
    echo "Or:  go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"
    exit 1
fi

# Check that proto files exist
if [ ! -f "proto/game/v1/game.proto" ]; then
    echo -e "${RED}ERROR: proto/game/v1/game.proto not found${NC}"
    echo "Are you running this from the project root?"
    exit 1
fi

# Create output directories
echo "Creating output directories..."
mkdir -p pkg/api/game/v1
mkdir -p pkg/api/common/v1
mkdir -p pkg/api/experience/v1

# Generate common proto files
echo -e "${YELLOW}Generating common proto files...${NC}"
protoc \
    --go_out=pkg/api \
    --go_opt=paths=source_relative \
    --go-grpc_out=pkg/api \
    --go-grpc_opt=paths=source_relative \
    -I proto \
    proto/common/v1/*.proto

echo -e "${GREEN}  ✓ common/v1 generated${NC}"

# Generate game proto files
echo -e "${YELLOW}Generating game proto files...${NC}"
protoc \
    --go_out=pkg/api \
    --go_opt=paths=source_relative \
    --go-grpc_out=pkg/api \
    --go-grpc_opt=paths=source_relative \
    -I proto \
    proto/game/v1/*.proto

echo -e "${GREEN}  ✓ game/v1 generated${NC}"

# Generate experience proto files
echo -e "${YELLOW}Generating experience proto files...${NC}"
protoc \
    --go_out=pkg/api \
    --go_opt=paths=source_relative \
    --go-grpc_out=pkg/api \
    --go-grpc_opt=paths=source_relative \
    -I proto \
    proto/experience/v1/*.proto

echo -e "${GREEN}  ✓ experience/v1 generated${NC}"

echo ""
echo -e "${GREEN}=== Proto generation complete! ===${NC}"
echo "Generated files in pkg/api/"
