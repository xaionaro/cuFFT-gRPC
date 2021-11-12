package main

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/cpmech/gosl/fun/fftw"
	pb "github.com/xaionaro/cuFFT-gRPC/client_example/go/protobufgen/github.com/xaionaro/cuFFT-gRPC/protobuf"
	"google.golang.org/grpc"
)

type fttHandlerInterface interface {
	Batch(in [][]complex128, isInverse bool)
}

type cuFFTThroughGRPCHandler struct {
	pb.FTServiceClient
}

func newCuFFTThroughGRPCHandler() *cuFFTThroughGRPCHandler {
	conn, err := grpc.Dial("localhost:11216", grpc.WithInsecure())
	assertNoError(err)
	return &cuFFTThroughGRPCHandler{FTServiceClient: pb.NewFTServiceClient(conn)}
}

func (h *cuFFTThroughGRPCHandler) Batch(inputs [][]complex128, isInverse bool) {
	batch := make([]complex128, len(inputs[0])*len(inputs))
	for idx, in := range inputs {
		copy(batch[idx*len(in):], in)
	}
	taskType := pb.FTType_Z2Z_FORWARD
	if isInverse {
		taskType = pb.FTType_Z2Z_INVERSE
	}
	_, err := h.FTServiceClient.Exec(context.Background(), &pb.FTRequest{
		Values: castComplex128SliceToBytes(batch),
		Type:   taskType,
		Size:   []uint32{uint32(len(inputs[0]))},
		Tasks:  uint32(len(inputs)),
	})
	assertNoError(err)
}

type fftwDirectlyHandler struct {
}

func newFFTWDirectlyHandler() *fftwDirectlyHandler {
	return &fftwDirectlyHandler{}
}

func (fftwDirectlyHandler) Batch(inputs [][]complex128, isInverse bool) {
	input := make([]complex128, len(inputs[0]))
	fftwPlan := fftw.NewPlan1d(input, isInverse, true)
	defer fftwPlan.Free()
	for _, in := range inputs {
		copy(input, in)
		fftwPlan.Execute()
	}
}

func BenchmarkCuFFTThroughGRPC(b *testing.B) {
	benchmark(b, newCuFFTThroughGRPCHandler())
}

func BenchmarkFFTWDirectly(b *testing.B) {
	benchmark(b, newFFTWDirectlyHandler())
}

func prepareBatch(batchSize, taskSize uint) [][]complex128 {
	result := make([][]complex128, int(batchSize))
	for idx := range result {
		result[idx] = make([]complex128, int(taskSize))
		for subIdx := range result[idx] {
			result[idx][subIdx] = complex(rand.Float64(), rand.Float64())
		}
	}
	return result
}

func benchmark(b *testing.B, fftHandler fttHandlerInterface) {
	for _, batchSize := range []uint{1, 10, 100} {
		b.Run(fmt.Sprintf("batchSize%d", batchSize), func(b *testing.B) {
			for _, taskSize := range []uint{2, 10, 100, 1000, 10000, 100000} {
				b.Run(fmt.Sprintf("taskSize%d", taskSize), func(b *testing.B) {
					for _, isInverse := range []bool{false, true} {
						b.Run(fmt.Sprintf("isInverse%v", isInverse), func(b *testing.B) {
							if batchSize*taskSize > 100000 {
								b.Skip()
								return
							}
							batch := prepareBatch(batchSize, taskSize)
							b.ReportAllocs()
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								fftHandler.Batch(batch, isInverse)
							}
						})
					}
				})
			}
		})
	}
}
