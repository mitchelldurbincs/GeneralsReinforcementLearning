# Configuration Guide

This guide explains how to configure the Generals Reinforcement Learning system using Viper configuration management.

## Configuration Overview

The system uses [Viper](https://github.com/spf13/viper) for configuration management, which provides:
- YAML configuration files
- Environment variable overrides
- Command-line flag overrides
- Hot-reloading of configuration files
- Validation of configuration values

## Configuration Precedence

Configuration values are loaded with the following precedence (highest to lowest):

1. **Command-line flags** - Override all other values
2. **Environment variables** - Override config file values
3. **Environment-specific config files** - e.g., `config.prod.yaml`
4. **Base config file** - `config.yaml`
5. **Default values** - Hardcoded in the application

## File Locations

The system looks for configuration files in the following locations:
- Current directory (`.`)
- `./config` directory
- `/etc/generals-rl` directory

## Environment Variables

All configuration values can be overridden using environment variables with the prefix `GRL_`. Nested values use underscores:

```bash
# Override game.map.city_ratio
export GRL_GAME_MAP_CITY_RATIO=25

# Override server.grpc_server.port
export GRL_SERVER_GRPC_SERVER_PORT=8080
```

## Configuration Files

### Base Configuration (`config.yaml`)

The base configuration file contains all default values for the system. See the provided `config.yaml` for the complete structure.

### Environment-Specific Configurations

- `config.dev.yaml` - Development environment overrides
- `config.prod.yaml` - Production environment overrides

To use an environment-specific config:

```bash
# For development
./ui_client --config config.yaml
# The system will automatically load config.dev.yaml if present

# For production
APP_ENV=production ./grpc_server --config config.yaml
```

## Configuration Categories

### Game Mechanics (`game`)

Controls core game behavior:

```yaml
game:
  map:
    city_ratio: 20              # 1 city per N tiles
    city_start_army: 40         # Initial army count for cities
    min_general_spacing: 5      # Min distance between generals
  production:
    general: 1                  # Armies per turn for generals
    city: 1                     # Armies per turn for cities
    normal: 1                   # Armies per growth interval
    normal_growth_interval: 25  # Turns between normal tile growth
  fog_of_war:
    enabled: true               # Enable fog of war
    visibility_radius: 1        # Visibility radius (3x3 area)
```

### Server Configuration (`server`)

Controls server behavior:

```yaml
server:
  game_server:
    log_level: "info"           # debug, info, warn, error
    log_format: "console"       # console or json
  grpc_server:
    host: "0.0.0.0"
    port: 50051
    turn_timeout: 0             # Milliseconds (0 = no timeout)
    max_games: 100              # Max concurrent games
```

### UI Configuration (`ui`)

Controls the graphical interface:

```yaml
ui:
  window:
    width: 800
    height: 600
    title: "Generals RL UI"
  game:
    tile_size: 32
    turn_interval: 30           # Frames per turn (60 FPS)
  defaults:
    human_player: 0             # Which player is human
    num_players: 2              # Number of players (2-4)
    map_width: 20
    map_height: 15
```

### Colors Configuration (`colors`)

Defines all color values:

```yaml
colors:
  players:
    neutral: [120, 120, 120]    # RGB values
    player_0: [200, 50, 50]     # Red
    player_1: [50, 100, 200]    # Blue
    # etc...
```

## Command-Line Usage

### UI Client

```bash
# Use default config
./ui_client

# Use specific config file
./ui_client --config /path/to/config.yaml

# Override specific values
./ui_client --width 30 --height 20 --players 4

# All AI mode
./ui_client --ai-only
```

### Game Server

```bash
# Use default config
./game_server

# Use specific config file
./game_server --config /path/to/config.yaml
```

### gRPC Server

```bash
# Use default config
./grpc_server

# Override port
./grpc_server --port 8080

# Set log level
./grpc_server --log-level debug

# Production mode with environment config
APP_ENV=production ./grpc_server
```

## Validation

The configuration system validates all values on startup. Common validation rules:

- **Positive integers**: city_ratio, production rates, dimensions
- **Valid ranges**: Colors (0-255), ratios (0-1), ports (1-65535)
- **Logical constraints**: human_player < num_players, num_players (2-4)

## Hot Reloading

The configuration system supports hot reloading of config files. When a config file changes, the system will:

1. Re-read the configuration
2. Validate the new values
3. Apply changes without restart

Note: Not all configuration changes can be applied at runtime. Some require a restart.

## Best Practices

1. **Use environment-specific configs** - Keep production settings separate
2. **Use environment variables for secrets** - Never commit sensitive data
3. **Validate locally** - Test config changes before deployment
4. **Document changes** - Add comments for non-obvious values
5. **Version control configs** - Track all configuration changes

## Troubleshooting

### Config not loading

Check file locations and permissions:
```bash
# Verify config file exists
ls -la config.yaml

# Check current directory
pwd

# Test with explicit path
./ui_client --config $(pwd)/config.yaml
```

### Environment variables not working

Ensure proper prefix and format:
```bash
# Check environment
env | grep GRL_

# Debug with explicit setting
GRL_GAME_MAP_CITY_RATIO=30 ./ui_client
```

### Validation errors

The error message will indicate which value failed:
```
config validation failed: game.map.city_ratio must be positive
```

## Examples

### Running in Development

```bash
# Load dev config with debug logging
./grpc_server --config config.yaml --log-level debug
```

### Running in Production

```bash
# Use production config with environment overrides
export GRL_SERVER_GRPC_SERVER_PORT=8080
export GRL_SERVER_GRPC_SERVER_MAX_GAMES=1000
APP_ENV=production ./grpc_server
```

### Custom Game Settings

```bash
# Large map with 4 players
./ui_client --width 40 --height 30 --players 4 --human 0
```