
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <glog/logging.h>
#include "service.grpc.pb.h"

using namespace std;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

typedef std::complex<double> Complex;

#define _TO_STR(a) # a
#define X_TO_STR(a) _TO_STR(a)

#define CUFFT_CHECK(condition) { \
    cufftResult result = condition; \
    CHECK(result == CUFFT_SUCCESS) << X_TO_STR(result); \
}

#define CUDA_CHECK(condition) { \
    cudaError_t result = condition; \
    CHECK(result == cudaSuccess) << cudaGetErrorString(result); \
}

static_assert(sizeof(double) == sizeof(cufftDoubleReal), "sizeof(double) != sizeof(cufftDoubleReal)");
static_assert(sizeof(Complex) == sizeof(cufftDoubleComplex), "sizeof(Complex) != sizeof(cufftDoubleComplex)");

cufftType task_type_from_protobuf(const FTType t) {
    switch(t) {
    case FTType::R2C:
        return cufftType::CUFFT_R2C;
    case FTType::C2C_FORWARD:
    case FTType::C2C_INVERSE:
        return cufftType::CUFFT_C2C;
    case FTType::C2R:
        return cufftType::CUFFT_C2R;
    case FTType::D2Z:
        return cufftType::CUFFT_D2Z;
    case FTType::Z2D:
        return cufftType::CUFFT_Z2D;
    case FTType::Z2Z_FORWARD:
    case FTType::Z2Z_INVERSE:
        return cufftType::CUFFT_Z2Z;
    }
    assert(false); // should never happen
}

int ft_direction(const FTType t) {
    switch(t) {
    case FTType::R2C:
    case FTType::C2C_FORWARD:
    case FTType::D2Z:
    case FTType::Z2Z_FORWARD:
        return CUFFT_FORWARD;
    case FTType::C2C_INVERSE:
    case FTType::C2R:
    case FTType::Z2D:
    case FTType::Z2Z_INVERSE:
        return CUFFT_INVERSE;
    }
    assert(false); // should never happen
}

inline unsigned int item_size_output(
    const cufftType task_type
) {
    switch(task_type) {
    case cufftType::CUFFT_C2R:
        return 4;
    case cufftType::CUFFT_Z2D:
    case cufftType::CUFFT_C2C:
    case cufftType::CUFFT_R2C:
        return 8;
    case cufftType::CUFFT_Z2Z:
    case cufftType::CUFFT_D2Z:
        return 16;
    default:
        assert(false); // should never happen
    }
}

inline unsigned int item_size_input(
    const cufftType task_type
) {
    switch(task_type) {
    case cufftType::CUFFT_R2C:
        return 4;
    case cufftType::CUFFT_D2Z:
    case cufftType::CUFFT_C2R:
    case cufftType::CUFFT_C2C:
        return 8;
    case cufftType::CUFFT_Z2D:
    case cufftType::CUFFT_Z2Z:
        return 16;
    default:
        assert(false); // should never happen
    }
}

inline unsigned int calc_memsize(
    const unsigned int batch_size,
    const unsigned int dims,
    const int* const sizes,
    const unsigned int item_size
) {
    unsigned int result = batch_size * item_size;
    for (int dim = 0; dim < dims; dim++) {
        result *= sizes[dim];
    }
    return result;
}

inline unsigned int calc_output_memsize(
    const unsigned int batch_size,
    const unsigned int dims,
    const int* const sizes,
    const cufftType task_type
) {
    switch(task_type) {
    case cufftType::CUFFT_R2C:
    case cufftType::CUFFT_D2Z: {
        if (dims > 1) {
            LOG(WARNING) << "not supported, yet";
            return false;
        }
        const int size = sizes[0]/2+1;
        return calc_memsize(batch_size, dims, &size, item_size_output(task_type));
    }
    case cufftType::CUFFT_C2R:
    case cufftType::CUFFT_C2C:
    case cufftType::CUFFT_Z2D:
    case cufftType::CUFFT_Z2Z:
        return calc_memsize(batch_size, dims, sizes, item_size_output(task_type));
    default:
        assert(false); // should never happen
    }
}


inline unsigned int calc_input_memsize(
    const unsigned int batch_size,
    const unsigned int dims,
    const int* const _sizes,
    const cufftType task_type
) {
    int *sizes;
    switch (task_type) {
    case cufftType::CUFFT_C2R:
    case cufftType::CUFFT_Z2D: {
        if (dims < 1 || dims > 3) {
            LOG(WARNING) << "invalid amount of dimensions";
            return -1;
        }
        if (dims > 1) {
            LOG(WARNING) << "not supported, yet";
            return -1;
        }
        int size = _sizes[0]/2+1;
        sizes = &size;
        break;
    }
    case cufftType::CUFFT_R2C:
    case cufftType::CUFFT_D2Z:
    case cufftType::CUFFT_C2C:
    case cufftType::CUFFT_Z2Z:
        sizes = (int *)_sizes;
        break;
    default:
        assert(false); // should never happen
    }

    const unsigned int item_size = item_size_input(task_type);
    return calc_memsize(batch_size, dims, sizes, item_size);
}

class ServiceImpl final : public FTService::Service {
public:
    ::grpc::Status Exec(
        ::grpc::ServerContext* context,
        const ::FTRequest* request,
        ::FTResponse* response
    ) override {
        cufftType task_type = task_type_from_protobuf(request->type());
        int direction = ft_direction(request->type());
        const unsigned int batch_size = request->tasks();
        int exchange_fd = open(request->datafilepath().c_str(), O_RDWR);
        if (exchange_fd < 0) {
            LOG(WARNING) << "unable to open file " << request->datafilepath();
            close(exchange_fd);
            return ::grpc::Status::CANCELLED;
        }
        struct stat exchange_fstats;
        int stat_rc = fstat(exchange_fd, &exchange_fstats);
        if (stat_rc != 0) {
            LOG(WARNING) << "unable to fstat() file " << request->datafilepath();
            close(exchange_fd);
            return ::grpc::Status::CANCELLED;
        }
        const unsigned int exchange_memsize = exchange_fstats.st_size;
        void *exchange_buffer = mmap(0, exchange_memsize, PROT_WRITE, MAP_SHARED, exchange_fd, 0);
        if (exchange_buffer == MAP_FAILED) {
            LOG(WARNING) << "unable to mmap() file " << request->datafilepath() << ": " << strerror(errno);
            close(exchange_fd);
            return ::grpc::Status::CANCELLED;
        }
        int dims = request->sizes_size();
        int *sizes = (int *)malloc(dims * sizeof(int));
        {
            auto _sizes = request->sizes().data();
            for (int dim = 0; dim < dims; dim++) {
                sizes[dim] = _sizes[dim];
            }
        }
        const unsigned int input_memsize = calc_input_memsize(batch_size, dims, sizes, task_type);
        if (input_memsize < 0) {
            munmap(exchange_buffer, exchange_memsize);
            close(exchange_fd);
            return ::grpc::Status::CANCELLED;
        }

        void *d_input;
        CUDA_CHECK(cudaMalloc(&d_input, input_memsize));
        CUDA_CHECK(cudaMemcpy(d_input, exchange_buffer, input_memsize, cudaMemcpyHostToDevice));

        void *d_output;
        const unsigned int output_memsize = calc_output_memsize(batch_size, dims, sizes, task_type);
        CUDA_CHECK(cudaMalloc(&d_output, output_memsize));

        cufftHandle plan;
        CUFFT_CHECK(cufftPlanMany(
                        &plan, dims, sizes,
                        NULL, 0, 0,
                        NULL, 0, 0,
                        task_type,
                        batch_size
                    ));

        switch(task_type) {
        case cufftType::CUFFT_C2C:
            cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output, direction);
            break;
        case cufftType::CUFFT_Z2Z:
            cufftExecZ2Z(plan, (cufftDoubleComplex *)d_input, (cufftDoubleComplex *)d_output, direction);
            break;
        case cufftType::CUFFT_R2C:
            cufftExecR2C(plan, (cufftReal *)d_input, (cufftComplex *)d_output);
            break;
        case cufftType::CUFFT_D2Z:
            cufftExecD2Z(plan, (cufftDoubleReal *)d_input, (cufftDoubleComplex *)d_output);
            break;
        case cufftType::CUFFT_C2R:
            cufftExecC2R(plan, (cufftComplex *)d_input, (cufftReal *)d_output);
            break;
        case cufftType::CUFFT_Z2D:
            cufftExecZ2D(plan, (cufftDoubleComplex *)d_input, (cufftDoubleReal *)d_output);
            break;
        default:
            assert(false); // should never happen
        }
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(exchange_buffer, d_output, output_memsize, cudaMemcpyDeviceToHost));
        cufftDestroy(plan);
        CUDA_CHECK(cudaFree(d_input));
        if (d_output != d_input) {
            CUDA_CHECK(cudaFree(d_output));
        }
        free(sizes);

        munmap(exchange_buffer, exchange_memsize);
        close(exchange_fd);
        return ::grpc::Status::OK;
    };
};


int main(int argc, char **argv) {
    std::string server_address("0.0.0.0:11216");
    ServiceImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    LOG(INFO) << "Server listening on " << server_address;
    server->Wait();
    return 0;
}
