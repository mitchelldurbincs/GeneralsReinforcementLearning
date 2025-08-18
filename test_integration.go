package main

import (
	"context"
	"log"
	"net"
	"time"

	"github.com/google/uuid"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/experience"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/grpc/gameserver"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	zlog "github.com/rs/zerolog/log"
	"google.golang.org/grpc"
)

func main() {
	// Set up logging
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	
	// Create buffer manager
	bufferManager := experience.NewBufferManager(10000, zlog.Logger)
	
	// Create experience service
	experienceService := gameserver.NewExperienceService(bufferManager)
	
	// Create gRPC server
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	
	grpcServer := grpc.NewServer()
	experiencepb.RegisterExperienceServiceServer(grpcServer, experienceService)
	
	// Start server in background
	go func() {
		log.Printf("Starting gRPC server on :50051")
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve: %v", err)
		}
	}()
	
	// Give server time to start
	time.Sleep(1 * time.Second)
	
	// Generate some test experiences
	go func() {
		gameID := "test-game-1"
		buffer := bufferManager.GetOrCreateBuffer(gameID)
		
		log.Printf("Starting to generate test experiences for game %s", gameID)
		
		for i := 0; i < 1000; i++ {
			exp := &experiencepb.Experience{
				ExperienceId: uuid.New().String(),
				GameId:       gameID,
				PlayerId:     int32(i % 4),
				Turn:         int32(i),
				State: &experiencepb.TensorState{
					Shape: []int32{9, 20, 20},
					Data:  make([]float32, 9*20*20),
				},
				Action: int32(i % 100),
				Reward: float32(i) * 0.1,
				NextState: &experiencepb.TensorState{
					Shape: []int32{9, 20, 20},
					Data:  make([]float32, 9*20*20),
				},
				Done: i == 999,
			}
			
			if err := buffer.Add(exp); err != nil {
				log.Printf("Error adding experience: %v", err)
			}
			
			if i%100 == 0 {
				log.Printf("Generated %d experiences", i)
			}
			
			time.Sleep(10 * time.Millisecond)
		}
		
		log.Printf("Finished generating experiences")
	}()
	
	// Run for a while
	log.Printf("Server running. Press Ctrl+C to stop...")
	
	// Keep running
	ctx := context.Background()
	<-ctx.Done()
}