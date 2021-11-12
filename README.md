# Server
```
xaionaro@alien:~/go/src/github.com/xaionaro/cuFFT-gRPC$ rm -rf build && mkdir build && (cd build && cmake .. && make && ./cuFFT-gRPC)
-- The CXX compiler identification is GNU 10.3.0
-- The CUDA compiler identification is NVIDIA 11.2.152
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/local/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Found CUDA: /usr (found version "11.2")
-- Found CUDAToolkit: /usr/include (found version "11.2.152")
-- Found Protobuf: /usr/lib/x86_64-linux-gnu/libprotobuf.so;-lpthread (found version "3.12.4")
-- Configuring done
-- Generating done
-- Build files have been written to: /home/xaionaro/go/src/github.com/xaionaro/cuFFT-gRPC/build
[ 16%] Generating service.pb.cc, service.pb.h, service.grpc.pb.cc, service.grpc.pb.h
Scanning dependencies of target svc_grpc_proto
[ 33%] Building CXX object CMakeFiles/svc_grpc_proto.dir/service.grpc.pb.cc.o
[ 50%] Building CXX object CMakeFiles/svc_grpc_proto.dir/service.pb.cc.o
[ 66%] Linking CXX static library libsvc_grpc_proto.a
[ 66%] Built target svc_grpc_proto
Scanning dependencies of target cuFFT-gRPC
[ 83%] Building CXX object CMakeFiles/cuFFT-gRPC.dir/src/main.cpp.o
[100%] Linking CXX executable cuFFT-gRPC
[100%] Built target cuFFT-gRPC
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1112 21:27:23.749905 2125847 main.cpp:284] Server listening on 0.0.0.0:11216
```

# Client
```go
...
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
...
```

```
xaionaro@alien:~/go/src/github.com/xaionaro/cuFFT-gRPC$ cd client_example/go && go generate . && go run .
IF FAILS then please execute:
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

[(10.5+0i) (0.036474508437579045-4.140997047615304i) (-1.5225424859373684-0.06937863785644394i) (1.713525491562421-1.383706418154278i) (1.2725424859373684-2.0143700267352034i) (-3.5+0i)]
[1 2.0000000000000004 1.0000000000000002 3.0000000000000004 1 1 -8.881784197001253e-17 0.9999999999999997 0.49999999999999983 0]
```
