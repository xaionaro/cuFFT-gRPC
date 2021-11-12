//go:generate echo -e "IF FAILS then please execute:\n\tgo install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \\\n\tgo install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest\n"
//go:generate protoc --plugin protoc-gen-go=$GOPATH/bin/protoc-gen-go --plugin protoc-gen-go-grpc=$GOPATH/bin/protoc-gen-go-grpc -I=../../protobuf --go_out=protobufgen --go-grpc_out=protobufgen ../../protobuf/service.proto

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"reflect"
	"unsafe"

	pb "github.com/xaionaro/cuFFT-gRPC/client_example/go/protobufgen/github.com/xaionaro/cuFFT-gRPC/protobuf"
	"google.golang.org/grpc"
)

func castFloat64SliceToBytes(s []float64) []byte {
	sliceHdr := *(*reflect.SliceHeader)((unsafe.Pointer)(&s))
	sliceHdr.Len *= 8
	sliceHdr.Cap *= 8
	return *(*[]byte)((unsafe.Pointer)(&sliceHdr))
}

func castBytesToFloat64Slice(s []byte) []float64 {
	sliceHdr := *(*reflect.SliceHeader)((unsafe.Pointer)(&s))
	sliceHdr.Len /= 8
	sliceHdr.Cap /= 8
	return *(*[]float64)((unsafe.Pointer)(&sliceHdr))
}

func castComplex128SliceToBytes(s []complex128) []byte {
	sliceHdr := *(*reflect.SliceHeader)((unsafe.Pointer)(&s))
	sliceHdr.Len *= 16
	sliceHdr.Cap *= 16
	return *(*[]byte)((unsafe.Pointer)(&sliceHdr))
}

func castBytesToComplex128Slice(s []byte) []complex128 {
	sliceHdr := *(*reflect.SliceHeader)((unsafe.Pointer)(&s))
	sliceHdr.Len /= 16
	sliceHdr.Cap /= 16
	return *(*[]complex128)((unsafe.Pointer)(&sliceHdr))
}

func assertNoError(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	addr := flag.String("addr", "localhost:11216", "server address to connect to")
	flag.Parse()

	conn, err := grpc.Dial(*addr, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()

	ctx, cancelFn := context.WithCancel(context.Background())
	defer cancelFn()

	client := pb.NewFTServiceClient(conn)
	response, err := client.Exec(ctx, &pb.FTRequest{
		Values: castFloat64SliceToBytes([]float64{1, 2, 1, 3, 1, 1, 0, 1, 0.5, 0}),
		Type:   pb.FTType_D2Z,
		Size:   []uint32{10},
		Tasks:  1,
	})
	assertNoError(err)

	result := castBytesToComplex128Slice(response.Values)
	fmt.Println(result)

	response, err = client.Exec(ctx, &pb.FTRequest{
		Values: castComplex128SliceToBytes(result),
		Type:   pb.FTType_Z2D,
		Size:   []uint32{10},
		Tasks:  1,
	})
	assertNoError(err)

	reversed := castBytesToFloat64Slice(response.Values)
	for idx := range reversed {
		reversed[idx] /= float64(len(reversed))
	}
	fmt.Println(reversed)
}
