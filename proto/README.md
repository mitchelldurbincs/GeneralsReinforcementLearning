# Protocol Buffer Definitions

This directory contains the Protocol Buffer (protobuf) definitions for the gRPC services.

## Directory Structure

```
proto/
├── common/
│   └── v1/
│       └── common.proto     # Shared types and enums
└── game/
    └── v1/
        └── game.proto        # GameService definition
```

## Generating Go Code

To generate Go code from the proto files:

```bash
# Install required tools (one-time setup)
make install-tools

# Generate proto files
make generate-protos
```

The generated code will be placed in:
- `pkg/api/common/v1/` - Common types
- `pkg/api/game/v1/` - Game service types and stubs

## Proto Style Guide

1. Use `snake_case` for field names
2. Use `PascalCase` for message and service names
3. Always specify field numbers explicitly
4. Reserve field numbers for deleted fields
5. Use semantic versioning in package paths (v1, v2, etc.)
6. Add comments for all services, methods, and non-obvious fields

## Adding New Services

1. Create a new directory under `proto/<service>/v1/`
2. Define your service in a `.proto` file
3. Update `scripts/generate-protos.sh` to include the new service
4. Run `make generate-protos`