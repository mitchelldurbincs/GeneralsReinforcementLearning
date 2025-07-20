package config

import (
	"fmt"
	"strings"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/viper"
)

// Config holds all configuration for the application
type Config struct {
	Game        GameConfig        `mapstructure:"game"`
	Server      ServerConfig      `mapstructure:"server"`
	UI          UIConfig          `mapstructure:"ui"`
	Colors      ColorsConfig      `mapstructure:"colors"`
	Performance PerformanceConfig `mapstructure:"performance"`
	Development DevelopmentConfig `mapstructure:"development"`
	Features    FeaturesConfig    `mapstructure:"features"`
}

// GameConfig holds game mechanics configuration
type GameConfig struct {
	Map        MapConfig        `mapstructure:"map"`
	Production ProductionConfig `mapstructure:"production"`
	FogOfWar   FogOfWarConfig   `mapstructure:"fog_of_war"`
}

// MapConfig holds map generation settings
type MapConfig struct {
	CityRatio          int                `mapstructure:"city_ratio"`
	CityStartArmy      int                `mapstructure:"city_start_army"`
	MinGeneralSpacing  int                `mapstructure:"min_general_spacing"`
	MountainVeins      MountainVeinConfig `mapstructure:"mountain_veins"`
}

// MountainVeinConfig holds mountain vein generation settings
type MountainVeinConfig struct {
	Ratio          int     `mapstructure:"ratio"`
	MinLength      int     `mapstructure:"min_length"`
	MaxLengthRatio float64 `mapstructure:"max_length_ratio"`
}

// ProductionConfig holds production rate settings
type ProductionConfig struct {
	General              int `mapstructure:"general"`
	City                 int `mapstructure:"city"`
	Normal               int `mapstructure:"normal"`
	NormalGrowthInterval int `mapstructure:"normal_growth_interval"`
}

// FogOfWarConfig holds fog of war settings
type FogOfWarConfig struct {
	Enabled           bool    `mapstructure:"enabled"`
	VisibilityRadius  int     `mapstructure:"visibility_radius"`
	UpdateThreshold   float64 `mapstructure:"update_threshold"`
}

// ServerConfig holds server configuration
type ServerConfig struct {
	GameServer GameServerConfig `mapstructure:"game_server"`
	GRPCServer GRPCServerConfig `mapstructure:"grpc_server"`
}

// GameServerConfig holds game server specific configuration
type GameServerConfig struct {
	LogLevel  string         `mapstructure:"log_level"`
	LogFormat string         `mapstructure:"log_format"`
	Demo      DemoConfig     `mapstructure:"demo"`
}

// DemoConfig holds demo mode configuration
type DemoConfig struct {
	BoardWidth  int `mapstructure:"board_width"`
	BoardHeight int `mapstructure:"board_height"`
	MaxTurns    int `mapstructure:"max_turns"`
}

// GRPCServerConfig holds gRPC server configuration
type GRPCServerConfig struct {
	Host                   string `mapstructure:"host"`
	Port                   int    `mapstructure:"port"`
	LogLevel               string `mapstructure:"log_level"`
	TurnTimeout            int    `mapstructure:"turn_timeout"`
	MaxGames               int    `mapstructure:"max_games"`
	EnableReflection       bool   `mapstructure:"enable_reflection"`
	GracefulShutdownDelay  int    `mapstructure:"graceful_shutdown_delay"`
}

// UIConfig holds UI/client configuration
type UIConfig struct {
	Window   WindowConfig   `mapstructure:"window"`
	Game     UIGameConfig   `mapstructure:"game"`
	Defaults UIDefaultsConfig `mapstructure:"defaults"`
}

// WindowConfig holds window settings
type WindowConfig struct {
	Width  int    `mapstructure:"width"`
	Height int    `mapstructure:"height"`
	Title  string `mapstructure:"title"`
}

// UIGameConfig holds UI game settings
type UIGameConfig struct {
	TileSize     int `mapstructure:"tile_size"`
	TurnInterval int `mapstructure:"turn_interval"`
}

// UIDefaultsConfig holds default game settings for UI
type UIDefaultsConfig struct {
	HumanPlayer int  `mapstructure:"human_player"`
	NumPlayers  int  `mapstructure:"num_players"`
	MapWidth    int  `mapstructure:"map_width"`
	MapHeight   int  `mapstructure:"map_height"`
	AIOnly      bool `mapstructure:"ai_only"`
}

// ColorsConfig holds all color configurations
type ColorsConfig struct {
	Players   PlayerColorsConfig   `mapstructure:"players"`
	Tiles     TileColorsConfig     `mapstructure:"tiles"`
	UI        UIColorsConfig       `mapstructure:"ui"`
	Rendering RenderingColorsConfig `mapstructure:"rendering"`
}

// PlayerColorsConfig holds player color settings
type PlayerColorsConfig struct {
	Neutral  [3]int `mapstructure:"neutral"`
	Player0  [3]int `mapstructure:"player_0"`
	Player1  [3]int `mapstructure:"player_1"`
	Player2  [3]int `mapstructure:"player_2"`
	Player3  [3]int `mapstructure:"player_3"`
}

// TileColorsConfig holds tile color settings
type TileColorsConfig struct {
	Mountain     [3]int          `mapstructure:"mountain"`
	CityHueShift int             `mapstructure:"city_hue_shift"`
	Text         TextColorsConfig `mapstructure:"text"`
}

// TextColorsConfig holds text color settings
type TextColorsConfig struct {
	General     [3]int `mapstructure:"general"`
	City        [3]int `mapstructure:"city"`
	Army        [3]int `mapstructure:"army"`
	GeneralArmy [3]int `mapstructure:"general_army"`
}

// UIColorsConfig holds UI color settings
type UIColorsConfig struct {
	Background  [3]int `mapstructure:"background"`
	GridLines   [3]int `mapstructure:"grid_lines"`
	FogOfWar    [4]int `mapstructure:"fog_of_war"`
	FogUnknown  [4]int `mapstructure:"fog_unknown"`
}

// RenderingColorsConfig holds rendering color settings
type RenderingColorsConfig struct {
	CityMarkerRatio      float64 `mapstructure:"city_marker_ratio"`
	GeneralMarkerRatio   float64 `mapstructure:"general_marker_ratio"`
	OwnedCityInnerRatio  float64 `mapstructure:"owned_city_inner_ratio"`
}

// PerformanceConfig holds performance/optimization settings
type PerformanceConfig struct {
	PreallocateCapacity int  `mapstructure:"preallocate_capacity"`
	IncrementalUpdates  bool `mapstructure:"incremental_updates"`
}

// DevelopmentConfig holds development/debug settings
type DevelopmentConfig struct {
	VerboseLogging      bool `mapstructure:"verbose_logging"`
	ShowAllTiles        bool `mapstructure:"show_all_tiles"`
	ShowCoordinates     bool `mapstructure:"show_coordinates"`
	ShowPerformanceStats bool `mapstructure:"show_performance_stats"`
}

// FeaturesConfig holds feature flags
type FeaturesConfig struct {
	EnableAI          bool `mapstructure:"enable_ai"`
	EnableMultiplayer bool `mapstructure:"enable_multiplayer"`
	EnableReplay      bool `mapstructure:"enable_replay"`
	EnableSpectator   bool `mapstructure:"enable_spectator"`
	UseOptimizedVisibility bool `mapstructure:"use_optimized_visibility"`
}

var (
	// Global config instance
	cfg *Config
	v   *viper.Viper
)

// setViperDefaults sets all default values using Viper's SetDefault
func setViperDefaults(v *viper.Viper) {
	// Game defaults
	v.SetDefault("game.map.city_ratio", 20)
	v.SetDefault("game.map.city_start_army", 40)
	v.SetDefault("game.map.min_general_spacing", 5)
	v.SetDefault("game.map.mountain_veins.ratio", 50)
	v.SetDefault("game.map.mountain_veins.min_length", 3)
	v.SetDefault("game.map.mountain_veins.max_length_ratio", 0.25)
	
	// Production defaults
	v.SetDefault("game.production.general", 1)
	v.SetDefault("game.production.city", 1)
	v.SetDefault("game.production.normal", 1)
	v.SetDefault("game.production.normal_growth_interval", 25)
	
	// Fog of war defaults
	v.SetDefault("game.fog_of_war.enabled", true)
	v.SetDefault("game.fog_of_war.visibility_radius", 1)
	v.SetDefault("game.fog_of_war.update_threshold", 0.1)
	
	// Server defaults
	v.SetDefault("server.game_server.log_level", "info")
	v.SetDefault("server.game_server.log_format", "console")
	v.SetDefault("server.game_server.demo.board_width", 8)
	v.SetDefault("server.game_server.demo.board_height", 8)
	v.SetDefault("server.game_server.demo.max_turns", 50)
	
	// gRPC server defaults
	v.SetDefault("server.grpc_server.host", "0.0.0.0")
	v.SetDefault("server.grpc_server.port", 50051)
	v.SetDefault("server.grpc_server.log_level", "info")
	v.SetDefault("server.grpc_server.turn_timeout", 0)
	v.SetDefault("server.grpc_server.max_games", 100)
	v.SetDefault("server.grpc_server.enable_reflection", true)
	v.SetDefault("server.grpc_server.graceful_shutdown_delay", 5)
	
	// UI defaults
	v.SetDefault("ui.window.width", 800)
	v.SetDefault("ui.window.height", 600)
	v.SetDefault("ui.window.title", "Generals RL UI")
	v.SetDefault("ui.game.tile_size", 32)
	v.SetDefault("ui.game.turn_interval", 30)
	v.SetDefault("ui.defaults.human_player", 0)
	v.SetDefault("ui.defaults.num_players", 2)
	v.SetDefault("ui.defaults.map_width", 20)
	v.SetDefault("ui.defaults.map_height", 15)
	v.SetDefault("ui.defaults.ai_only", false)
	
	// Color defaults
	v.SetDefault("colors.players.neutral", []int{120, 120, 120})
	v.SetDefault("colors.players.player_0", []int{200, 50, 50})
	v.SetDefault("colors.players.player_1", []int{50, 100, 200})
	v.SetDefault("colors.players.player_2", []int{50, 200, 50})
	v.SetDefault("colors.players.player_3", []int{200, 200, 50})
	
	v.SetDefault("colors.tiles.mountain", []int{80, 80, 80})
	v.SetDefault("colors.tiles.city_hue_shift", 30)
	v.SetDefault("colors.tiles.text.general", []int{255, 255, 255})
	v.SetDefault("colors.tiles.text.city", []int{255, 255, 255})
	v.SetDefault("colors.tiles.text.army", []int{255, 255, 255})
	v.SetDefault("colors.tiles.text.general_army", []int{0, 0, 0})
	
	v.SetDefault("colors.ui.background", []int{0, 0, 0})
	v.SetDefault("colors.ui.grid_lines", []int{50, 50, 50})
	v.SetDefault("colors.ui.fog_of_war", []int{0, 0, 0, 200})
	v.SetDefault("colors.ui.fog_unknown", []int{25, 25, 25, 200})
	
	v.SetDefault("colors.rendering.city_marker_ratio", 0.5)
	v.SetDefault("colors.rendering.general_marker_ratio", 0.5)
	v.SetDefault("colors.rendering.owned_city_inner_ratio", 0.33)
	
	// Performance defaults
	v.SetDefault("performance.preallocate_capacity", 50)
	v.SetDefault("performance.incremental_updates", true)
	
	// Development defaults
	v.SetDefault("development.verbose_logging", false)
	v.SetDefault("development.show_all_tiles", false)
	v.SetDefault("development.show_coordinates", false)
	v.SetDefault("development.show_performance_stats", false)
	
	// Feature flags
	v.SetDefault("features.enable_ai", true)
	v.SetDefault("features.enable_multiplayer", true)
	v.SetDefault("features.enable_replay", false)
	v.SetDefault("features.enable_spectator", false)
	v.SetDefault("features.use_optimized_visibility", false) // Start with false for safe rollout
}

// Init initializes the configuration
func Init(configPath string) error {
	v = viper.New()
	
	// Set defaults before loading any config
	setViperDefaults(v)
	
	// Set config file
	if configPath != "" {
		v.SetConfigFile(configPath)
	} else {
		// Default config locations
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		v.AddConfigPath(".")
		v.AddConfigPath("./config")
		v.AddConfigPath("/etc/generals-rl")
	}
	
	// Set environment variable prefix
	v.SetEnvPrefix("GRL")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()
	
	// Read config file
	if err := v.ReadInConfig(); err != nil {
		// If we have a specific config path and it doesn't exist, that's ok - use defaults
		if configPath != "" {
			// Specific file requested but not found - that's ok, use defaults
		} else if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			// For default locations, only ignore ConfigFileNotFoundError
			return fmt.Errorf("error reading config file: %w", err)
		}
		// Config file not found; use defaults
	}
	
	// Unmarshal into config struct
	cfg = &Config{}
	if err := v.Unmarshal(cfg); err != nil {
		return fmt.Errorf("unable to decode config into struct: %w", err)
	}
	
	// Validate configuration
	if err := Validate(cfg); err != nil {
		return fmt.Errorf("config validation failed: %w", err)
	}
	
	return nil
}

// Get returns the global config instance
func Get() *Config {
	if cfg == nil {
		// Initialize with defaults if not already initialized
		if err := Init(""); err != nil {
			panic("failed to initialize config with defaults: " + err.Error())
		}
	}
	return cfg
}

// GetViper returns the viper instance for advanced usage
func GetViper() *viper.Viper {
	if v == nil {
		panic("config not initialized - call Init() first")
	}
	return v
}

// LoadEnvironmentConfig loads environment-specific config overlay
func LoadEnvironmentConfig(env string) error {
	if env == "" {
		return nil
	}
	
	envFile := fmt.Sprintf("config.%s.yaml", env)
	
	// Try to find environment-specific config
	v.SetConfigFile(envFile)
	if err := v.MergeInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return fmt.Errorf("error merging environment config %s: %w", envFile, err)
		}
	}
	
	// Re-unmarshal with merged config
	if err := v.Unmarshal(cfg); err != nil {
		return fmt.Errorf("unable to decode merged config into struct: %w", err)
	}
	
	return nil
}

// Set allows runtime config updates
func Set(key string, value interface{}) {
	v.Set(key, value)
	// Re-unmarshal to update struct
	v.Unmarshal(cfg)
}

// GetString gets a string value from config
func GetString(key string) string {
	return v.GetString(key)
}

// GetInt gets an int value from config
func GetInt(key string) int {
	return v.GetInt(key)
}

// GetBool gets a bool value from config
func GetBool(key string) bool {
	return v.GetBool(key)
}

// GetFloat64 gets a float64 value from config
func GetFloat64(key string) float64 {
	return v.GetFloat64(key)
}

// ConfigFilePath returns the path of the loaded config file
func ConfigFilePath() string {
	return v.ConfigFileUsed()
}

// WatchConfig enables hot-reloading of config file
func WatchConfig(onChange func()) {
	v.WatchConfig()
	v.OnConfigChange(func(e fsnotify.Event) {
		// Re-unmarshal on change
		v.Unmarshal(cfg)
		if onChange != nil {
			onChange()
		}
	})
}

// Validate validates the configuration values
func Validate(c *Config) error {
	// Validate game mechanics
	if c.Game.Map.CityRatio <= 0 {
		return fmt.Errorf("game.map.city_ratio must be positive")
	}
	if c.Game.Map.CityStartArmy < 0 {
		return fmt.Errorf("game.map.city_start_army must be non-negative")
	}
	if c.Game.Map.MinGeneralSpacing < 1 {
		return fmt.Errorf("game.map.min_general_spacing must be at least 1")
	}
	if c.Game.Production.NormalGrowthInterval <= 0 {
		return fmt.Errorf("game.production.normal_growth_interval must be positive")
	}
	if c.Game.FogOfWar.VisibilityRadius < 0 {
		return fmt.Errorf("game.fog_of_war.visibility_radius must be non-negative")
	}
	if c.Game.FogOfWar.UpdateThreshold < 0 || c.Game.FogOfWar.UpdateThreshold > 1 {
		return fmt.Errorf("game.fog_of_war.update_threshold must be between 0 and 1")
	}
	
	// Validate server configuration
	if c.Server.GRPCServer.Port <= 0 || c.Server.GRPCServer.Port > 65535 {
		return fmt.Errorf("server.grpc_server.port must be between 1 and 65535")
	}
	if c.Server.GRPCServer.MaxGames <= 0 {
		return fmt.Errorf("server.grpc_server.max_games must be positive")
	}
	if c.Server.GRPCServer.TurnTimeout < 0 {
		return fmt.Errorf("server.grpc_server.turn_timeout must be non-negative")
	}
	if c.Server.GRPCServer.GracefulShutdownDelay < 0 {
		return fmt.Errorf("server.grpc_server.graceful_shutdown_delay must be non-negative")
	}
	
	// Validate UI configuration
	if c.UI.Window.Width <= 0 || c.UI.Window.Height <= 0 {
		return fmt.Errorf("ui.window dimensions must be positive")
	}
	if c.UI.Game.TileSize <= 0 {
		return fmt.Errorf("ui.game.tile_size must be positive")
	}
	if c.UI.Game.TurnInterval <= 0 {
		return fmt.Errorf("ui.game.turn_interval must be positive")
	}
	if c.UI.Defaults.NumPlayers < 2 || c.UI.Defaults.NumPlayers > 4 {
		return fmt.Errorf("ui.defaults.num_players must be between 2 and 4")
	}
	if c.UI.Defaults.HumanPlayer < -1 || c.UI.Defaults.HumanPlayer >= c.UI.Defaults.NumPlayers {
		return fmt.Errorf("ui.defaults.human_player must be -1 or valid player index")
	}
	if c.UI.Defaults.MapWidth <= 0 || c.UI.Defaults.MapHeight <= 0 {
		return fmt.Errorf("ui.defaults map dimensions must be positive")
	}
	
	// Validate color values
	validateRGB := func(rgb [3]int, name string) error {
		for i, v := range rgb {
			if v < 0 || v > 255 {
				return fmt.Errorf("%s[%d] must be between 0 and 255", name, i)
			}
		}
		return nil
	}
	
	if err := validateRGB(c.Colors.Players.Neutral, "colors.players.neutral"); err != nil {
		return err
	}
	if err := validateRGB(c.Colors.Players.Player0, "colors.players.player_0"); err != nil {
		return err
	}
	if err := validateRGB(c.Colors.Players.Player1, "colors.players.player_1"); err != nil {
		return err
	}
	if err := validateRGB(c.Colors.Players.Player2, "colors.players.player_2"); err != nil {
		return err
	}
	if err := validateRGB(c.Colors.Players.Player3, "colors.players.player_3"); err != nil {
		return err
	}
	
	// Validate rendering ratios
	if c.Colors.Rendering.CityMarkerRatio <= 0 || c.Colors.Rendering.CityMarkerRatio > 1 {
		return fmt.Errorf("colors.rendering.city_marker_ratio must be between 0 and 1")
	}
	if c.Colors.Rendering.GeneralMarkerRatio <= 0 || c.Colors.Rendering.GeneralMarkerRatio > 1 {
		return fmt.Errorf("colors.rendering.general_marker_ratio must be between 0 and 1")
	}
	if c.Colors.Rendering.OwnedCityInnerRatio <= 0 || c.Colors.Rendering.OwnedCityInnerRatio > 1 {
		return fmt.Errorf("colors.rendering.owned_city_inner_ratio must be between 0 and 1")
	}
	
	// Validate performance settings
	if c.Performance.PreallocateCapacity < 0 {
		return fmt.Errorf("performance.preallocate_capacity must be non-negative")
	}
	
	return nil
}