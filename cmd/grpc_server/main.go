package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/config"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/grpc/gameserver"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

func main() {
	// Command line flags
	configPath := flag.String("config", "", "Path to config file")
	port := flag.Int("port", -1, "The server port (-1 to use config default)")
	host := flag.String("host", "", "The server host (empty to use config default)")
	logLevel := flag.String("log-level", "", "Log level (debug, info, warn, error) (empty to use config default)")
	turnTimeout := flag.Int("turn-timeout", -1, "Default turn timeout in milliseconds (-1 to use config default)")
	maxGames := flag.Int("max-games", -1, "Maximum concurrent games (-1 to use config default)")
	enableReflection := flag.Bool("enable-reflection", false, "Enable gRPC reflection for debugging")
	flag.Parse()
	
	// Initialize configuration
	if err := config.Init(*configPath); err != nil {
		log.Fatal().Err(err).Msg("Failed to initialize config")
	}
	
	cfg := config.Get()
	
	// Use config defaults if not overridden by flags
	if *port == -1 {
		*port = cfg.Server.GRPCServer.Port
	}
	if *host == "" {
		*host = cfg.Server.GRPCServer.Host
	}
	if *logLevel == "" {
		*logLevel = cfg.Server.GRPCServer.LogLevel
	}
	if *turnTimeout == -1 {
		*turnTimeout = cfg.Server.GRPCServer.TurnTimeout
	}
	if *maxGames == -1 {
		*maxGames = cfg.Server.GRPCServer.MaxGames
	}
	// For enableReflection, use config if flag not explicitly set to true
	if !*enableReflection {
		*enableReflection = cfg.Server.GRPCServer.EnableReflection
	}

	// Setup logging
	setupLogging(*logLevel)

	log.Info().
		Int("port", *port).
		Str("host", *host).
		Int("turn_timeout_ms", *turnTimeout).
		Int("max_games", *maxGames).
		Msg("Starting gRPC game server")

	// Create listener
	lis, err := net.Listen("tcp", fmt.Sprintf("%s:%d", *host, *port))
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to listen")
	}

	// Create gRPC server with interceptors
	opts := []grpc.ServerOption{
		grpc.ChainUnaryInterceptor(
			loggingInterceptor,
			recoveryInterceptor,
		),
		grpc.ChainStreamInterceptor(
			streamLoggingInterceptor,
			streamRecoveryInterceptor,
		),
	}
	
	grpcServer := grpc.NewServer(opts...)

	// Register game service
	gameService := gameserver.NewServer()
	gamev1.RegisterGameServiceServer(grpcServer, gameService)

	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	
	// Set health status
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	healthServer.SetServingStatus(gamev1.GameService_ServiceDesc.ServiceName, grpc_health_v1.HealthCheckResponse_SERVING)

	// Register reflection service for debugging
	if *enableReflection {
		reflection.Register(grpcServer)
		log.Info().Msg("gRPC reflection enabled")
	}

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	
	go func() {
		sig := <-sigCh
		log.Info().Str("signal", sig.String()).Msg("Received shutdown signal")
		
		// Set health status to NOT_SERVING
		healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		healthServer.SetServingStatus(gamev1.GameService_ServiceDesc.ServiceName, grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		
		// Give ongoing requests time to complete
		time.Sleep(time.Duration(cfg.Server.GRPCServer.GracefulShutdownDelay) * time.Second)
		
		log.Info().Msg("Gracefully stopping gRPC server")
		grpcServer.GracefulStop()
		cancel()
	}()

	// Start server
	log.Info().Str("address", lis.Addr().String()).Msg("gRPC server listening")
	
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatal().Err(err).Msg("Failed to serve")
		}
	}()

	// TODO: Future enhancements
	// - Prometheus metrics endpoint on separate port
	// - OpenTelemetry tracing integration
	// - Rate limiting per player
	// - Authentication middleware for production
	// - Periodic cleanup of abandoned games
	// - Configuration hot-reload support

	// Wait for shutdown
	<-ctx.Done()
	log.Info().Msg("Server shutdown complete")
}

func setupLogging(level string) {
	// Parse log level
	var logLevel zerolog.Level
	switch level {
	case "debug":
		logLevel = zerolog.DebugLevel
	case "info":
		logLevel = zerolog.InfoLevel
	case "warn":
		logLevel = zerolog.WarnLevel
	case "error":
		logLevel = zerolog.ErrorLevel
	default:
		logLevel = zerolog.InfoLevel
	}
	
	zerolog.SetGlobalLevel(logLevel)

	// Check if we're in production
	if os.Getenv("APP_ENV") == "production" {
		// JSON output for production
		log.Logger = zerolog.New(os.Stdout).With().Timestamp().Logger()
	} else {
		// Pretty console output for development
		log.Logger = log.Output(zerolog.ConsoleWriter{
			Out:        os.Stdout,
			TimeFormat: time.RFC3339,
		})
	}
}

// loggingInterceptor logs all unary RPC calls
func loggingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	start := time.Now()
	
	// Call the handler
	resp, err := handler(ctx, req)
	
	// Log the call
	code := codes.OK
	if err != nil {
		if st, ok := status.FromError(err); ok {
			code = st.Code()
		}
	}
	
	log.Info().
		Str("method", info.FullMethod).
		Str("code", code.String()).
		Dur("duration", time.Since(start)).
		Err(err).
		Msg("gRPC call")
	
	return resp, err
}

// recoveryInterceptor catches panics and returns proper gRPC errors
func recoveryInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
	defer func() {
		if r := recover(); r != nil {
			log.Error().
				Str("method", info.FullMethod).
				Interface("panic", r).
				Msg("Recovered from panic in gRPC handler")
			err = status.Errorf(codes.Internal, "internal server error")
		}
	}()
	
	return handler(ctx, req)
}

// streamLoggingInterceptor logs all streaming RPC calls
func streamLoggingInterceptor(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	start := time.Now()
	
	// Call the handler
	err := handler(srv, ss)
	
	// Log the call
	code := codes.OK
	if err != nil {
		if st, ok := status.FromError(err); ok {
			code = st.Code()
		}
	}
	
	log.Info().
		Str("method", info.FullMethod).
		Str("code", code.String()).
		Dur("duration", time.Since(start)).
		Bool("is_client_stream", info.IsClientStream).
		Bool("is_server_stream", info.IsServerStream).
		Err(err).
		Msg("gRPC stream")
	
	return err
}

// streamRecoveryInterceptor catches panics in streaming handlers
func streamRecoveryInterceptor(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) (err error) {
	defer func() {
		if r := recover(); r != nil {
			log.Error().
				Str("method", info.FullMethod).
				Interface("panic", r).
				Msg("Recovered from panic in gRPC stream handler")
			err = status.Errorf(codes.Internal, "internal server error")
		}
	}()
	
	return handler(srv, ss)
}