#!/bin/bash

# Script to download and install protoc compiler

set -e

PROTOC_VERSION="29.5"
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

# Map architecture names
case $ARCH in
    x86_64)
        ARCH="x86_64"
        ;;
    aarch64|arm64)
        ARCH="aarch_64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Construct download URL
PROTOC_ZIP="protoc-${PROTOC_VERSION}-${OS}-${ARCH}.zip"
DOWNLOAD_URL="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${PROTOC_ZIP}"

echo "Downloading protoc ${PROTOC_VERSION} for ${OS}-${ARCH}..."
echo "URL: ${DOWNLOAD_URL}"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "${TEMP_DIR}"

# Download and extract
curl -LO "${DOWNLOAD_URL}"
unzip -q "${PROTOC_ZIP}"

# Install to user's local bin
mkdir -p "$HOME/.local/bin"
cp bin/protoc "$HOME/.local/bin/"
chmod +x "$HOME/.local/bin/protoc"

# Install include files
mkdir -p "$HOME/.local/include"
cp -r include/* "$HOME/.local/include/"

# Clean up
cd -
rm -rf "${TEMP_DIR}"

echo "protoc installed to $HOME/.local/bin/protoc"
echo ""
echo "Add the following to your shell profile if not already present:"
echo 'export PATH="$HOME/.local/bin:$PATH"'
echo ""
echo "Then run: source ~/.bashrc (or ~/.zshrc)"

# Test installation
if command -v protoc &> /dev/null; then
    echo ""
    echo "protoc version:"
    protoc --version
else
    echo ""
    echo "Note: protoc is not in your PATH yet. Please add ~/.local/bin to your PATH."
fi