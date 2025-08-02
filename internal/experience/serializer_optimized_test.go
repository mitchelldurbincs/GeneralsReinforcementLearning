package experience

import (
	"runtime"
	"sync"
	"testing"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test that optimized serializer produces same results as original
func TestOptimizedSerializerCompatibility(t *testing.T) {
	original := NewSerializer()
	optimized := NewOptimizedSerializer()
	
	// Test different board sizes
	sizes := []int{10, 20, 30}
	
	for _, size := range sizes {
		state := createTestGameState(size, size)
		
		// Test StateToTensor
		for playerID := 0; playerID < 4; playerID++ {
			origTensor := original.StateToTensor(state, playerID)
			optTensor := optimized.StateToTensor(state, playerID)
			
			assert.Equal(t, len(origTensor), len(optTensor), "Tensor sizes should match")
			for i := range origTensor {
				assert.InDelta(t, origTensor[i], optTensor[i], 0.0001, 
					"Tensor values should match at index %d for player %d", i, playerID)
			}
			
			// Return tensor to pool
			optimized.ReturnTensor(optTensor)
		}
		
		// Test GenerateActionMask
		for playerID := 0; playerID < 4; playerID++ {
			origMask := original.GenerateActionMask(state, playerID)
			optMask := optimized.GenerateActionMask(state, playerID)
			
			assert.Equal(t, origMask, optMask, "Action masks should match for player %d", playerID)
			
			// Return mask to pool
			optimized.ReturnActionMask(optMask)
		}
	}
}

// Test tensor pooling functionality
func TestTensorPool(t *testing.T) {
	pool := NewTensorPool()
	
	// Get tensors of different sizes
	tensor1 := pool.Get(100)
	assert.Len(t, tensor1, 100)
	assert.Equal(t, float32(0), tensor1[0]) // Should be zeroed
	
	// Fill with data
	for i := range tensor1 {
		tensor1[i] = float32(i)
	}
	
	// Return to pool
	pool.Put(tensor1)
	
	// Get again - should be cleared
	tensor2 := pool.Get(100)
	assert.Len(t, tensor2, 100)
	assert.Equal(t, float32(0), tensor2[0]) // Should be zeroed
	
	// Verify it's the same slice (reused)
	assert.Equal(t, cap(tensor1), cap(tensor2))
}

// Test visibility caching
func TestVisibilityCache(t *testing.T) {
	optimized := NewOptimizedSerializer()
	state := createTestGameState(10, 10)
	state.FogOfWarEnabled = true
	
	// First call should calculate visibility
	vis1 := optimized.getVisibility(state, 0)
	assert.NotNil(t, vis1)
	
	// Second call should use cache
	vis2 := optimized.getVisibility(state, 0)
	assert.Equal(t, vis1, vis2) // Should be exact same slice
	
	// Different player should calculate new visibility
	vis3 := optimized.getVisibility(state, 1)
	assert.NotNil(t, vis3)
	
	// Clear cache
	optimized.ClearVisibilityCache()
	
	// Should recalculate after clear
	vis4 := optimized.getVisibility(state, 0)
	assert.NotNil(t, vis4)
}

// Test batch processing
func TestBatchStateToTensor(t *testing.T) {
	optimized := NewOptimizedSerializer()
	
	// Create multiple states
	states := make([]*game.GameState, 10)
	for i := range states {
		states[i] = createTestGameState(10, 10)
	}
	
	// Process batch
	tensors := optimized.BatchStateToTensor(states, 0)
	
	assert.Len(t, tensors, 10)
	for _, tensor := range tensors {
		assert.NotNil(t, tensor)
		assert.Len(t, tensor, NumChannels*10*10)
		
		// Clean up
		optimized.ReturnTensor(tensor)
	}
}

// Benchmark comparisons between original and optimized

func BenchmarkStateToTensor_Original_10x10(b *testing.B) {
	serializer := NewSerializer()
	state := createTestGameState(10, 10)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = serializer.StateToTensor(state, 0)
	}
}

func BenchmarkStateToTensor_Optimized_10x10(b *testing.B) {
	serializer := NewOptimizedSerializer()
	state := createTestGameState(10, 10)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor := serializer.StateToTensor(state, 0)
		serializer.ReturnTensor(tensor)
	}
}

func BenchmarkStateToTensor_Original_20x20(b *testing.B) {
	serializer := NewSerializer()
	state := createTestGameState(20, 20)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = serializer.StateToTensor(state, 0)
	}
}

func BenchmarkStateToTensor_Optimized_20x20(b *testing.B) {
	serializer := NewOptimizedSerializer()
	state := createTestGameState(20, 20)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor := serializer.StateToTensor(state, 0)
		serializer.ReturnTensor(tensor)
	}
}

func BenchmarkStateToTensor_Original_50x50(b *testing.B) {
	serializer := NewSerializer()
	state := createTestGameState(50, 50)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = serializer.StateToTensor(state, 0)
	}
}

func BenchmarkStateToTensor_Optimized_50x50(b *testing.B) {
	serializer := NewOptimizedSerializer()
	state := createTestGameState(50, 50)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor := serializer.StateToTensor(state, 0)
		serializer.ReturnTensor(tensor)
	}
}

// Benchmark action mask generation

func BenchmarkGenerateActionMask_Original(b *testing.B) {
	serializer := NewSerializer()
	state := createTestGameState(20, 20)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = serializer.GenerateActionMask(state, 0)
	}
}

func BenchmarkGenerateActionMask_Optimized(b *testing.B) {
	serializer := NewOptimizedSerializer()
	state := createTestGameState(20, 20)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mask := serializer.GenerateActionMask(state, 0)
		serializer.ReturnActionMask(mask)
	}
}

// Benchmark concurrent access

func BenchmarkStateToTensor_Optimized_Concurrent(b *testing.B) {
	serializer := NewOptimizedSerializer()
	state := createTestGameState(20, 20)
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			tensor := serializer.StateToTensor(state, 0)
			serializer.ReturnTensor(tensor)
		}
	})
}

// Benchmark batch processing

func BenchmarkBatchStateToTensor_Small(b *testing.B) {
	serializer := NewOptimizedSerializer()
	states := make([]*game.GameState, 4)
	for i := range states {
		states[i] = createTestGameState(20, 20)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensors := serializer.BatchStateToTensor(states, 0)
		for _, tensor := range tensors {
			serializer.ReturnTensor(tensor)
		}
	}
}

func BenchmarkBatchStateToTensor_Large(b *testing.B) {
	serializer := NewOptimizedSerializer()
	states := make([]*game.GameState, 16)
	for i := range states {
		states[i] = createTestGameState(20, 20)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensors := serializer.BatchStateToTensor(states, 0)
		for _, tensor := range tensors {
			serializer.ReturnTensor(tensor)
		}
	}
}

// Memory allocation benchmark

func BenchmarkMemoryAllocations_Original(b *testing.B) {
	serializer := NewSerializer()
	state := createTestGameState(20, 20)
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		_ = serializer.StateToTensor(state, 0)
		_ = serializer.GenerateActionMask(state, 0)
	}
}

func BenchmarkMemoryAllocations_Optimized(b *testing.B) {
	serializer := NewOptimizedSerializer()
	state := createTestGameState(20, 20)
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		tensor := serializer.StateToTensor(state, 0)
		mask := serializer.GenerateActionMask(state, 0)
		serializer.ReturnTensor(tensor)
		serializer.ReturnActionMask(mask)
	}
}

// Test concurrent tensor pool access
func TestTensorPoolConcurrency(t *testing.T) {
	pool := NewTensorPool()
	const goroutines = 100
	const iterations = 1000
	
	var wg sync.WaitGroup
	wg.Add(goroutines)
	
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				size := 100 + (j % 10) * 100 // Vary sizes
				tensor := pool.Get(size)
				require.Len(t, tensor, size)
				
				// Use the tensor
				for k := range tensor {
					tensor[k] = float32(k)
				}
				
				// Return it
				pool.Put(tensor)
			}
		}()
	}
	
	wg.Wait()
}

// Profile memory usage
func TestMemoryUsage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping memory usage test in short mode")
	}
	
	// Force GC before starting
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	
	// Original serializer
	original := NewSerializer()
	state := createTestGameState(50, 50)
	
	for i := 0; i < 1000; i++ {
		_ = original.StateToTensor(state, i%4)
		_ = original.GenerateActionMask(state, i%4)
	}
	
	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	originalMem := m2.Alloc - m1.Alloc
	
	// Optimized serializer
	runtime.GC()
	runtime.ReadMemStats(&m1)
	
	optimized := NewOptimizedSerializer()
	for i := 0; i < 1000; i++ {
		tensor := optimized.StateToTensor(state, i%4)
		mask := optimized.GenerateActionMask(state, i%4)
		optimized.ReturnTensor(tensor)
		optimized.ReturnActionMask(mask)
	}
	
	runtime.GC()
	runtime.ReadMemStats(&m2)
	optimizedMem := m2.Alloc - m1.Alloc
	
	t.Logf("Original memory usage: %d bytes", originalMem)
	t.Logf("Optimized memory usage: %d bytes", optimizedMem)
	t.Logf("Memory reduction: %.2f%%", float64(originalMem-optimizedMem)/float64(originalMem)*100)
	
	// Optimized should use significantly less memory
	assert.Less(t, optimizedMem, originalMem/2, "Optimized serializer should use less than half the memory")
}