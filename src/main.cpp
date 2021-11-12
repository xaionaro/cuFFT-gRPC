
#include <assert.h>
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

class ServiceImpl final : public FTService::Service {
public:
    ::grpc::Status Exec(
        ::grpc::ServerContext* context,
        const ::FTRequest* request,
        ::FTResponse* response
    ) override {
        cufftType taskType = task_type_from_protobuf(request->type());
        const unsigned int batch_size = request->tasks();
        const void *input = request->values().data();
        const unsigned int input_memsize = request->values().length();

        void *d_input;
        CUDA_CHECK(cudaMalloc(&d_input, input_memsize));

        void *d_output;
        unsigned int output_memsize = -1;
        switch(taskType) {
        case cufftType::CUFFT_C2C:
        case cufftType::CUFFT_Z2Z:
            output_memsize = input_memsize;
            break;
        case cufftType::CUFFT_R2C:
        case cufftType::CUFFT_D2Z:
            output_memsize = input_memsize*2;
            break;
        case cufftType::CUFFT_C2R:
        case cufftType::CUFFT_Z2D:
            output_memsize = input_memsize/2;
            break;
        default:
            assert(false); // should never happen
        }
        CUDA_CHECK(cudaMalloc(&d_output, output_memsize));

        int dims = request->size_size();
        int *sizes = (int *)malloc(dims * sizeof(int));
        {
            auto _sizes = request->size().data();
            for (int dim = 0; dim < dims; dim++) {
                sizes[dim] = _sizes[dim];
            }
        }

        cufftHandle plan;
        CUFFT_CHECK(cufftPlanMany(
                        &plan, dims, sizes,
                        NULL, 0, 0,
                        NULL, 0, 0,
                        taskType,
                        batch_size
                    ));

        switch(taskType) {
        case cufftType::CUFFT_C2C:
            cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output, ft_direction(request->type()));
            break;
        case cufftType::CUFFT_Z2Z:
            cufftExecZ2Z(plan, (cufftDoubleComplex *)d_input, (cufftDoubleComplex *)d_output, ft_direction(request->type()));
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

        void *output = (void *)malloc(output_memsize);
        CUDA_CHECK(cudaMemcpy(output, d_output, output_memsize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_input));
        if (d_output != d_input) {
            CUDA_CHECK(cudaFree(d_output));
        }

        std::string *output_string = new std::string((char *)output, output_memsize);
        response->set_allocated_values(output_string);
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
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
    return 0;
}
