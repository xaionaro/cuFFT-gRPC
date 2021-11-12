# DO NOT COPY&PASTE FROM THIS CmakeLists.txt.
# THE LICENSE OF THIS CODE IS UNCLEAR:
#
# SOME PARTS ARE COPIED FROM THE OFFICIAL GRPC DOCUMENTATION,
# OTHER PARTS ARE COPIED FROM https://github.com/mnicely/cufft_examples/blob/master/CMakeLists.txt
# AND https://github.com/iclementine/cufft_examples/blob/master/CMakeLists.txt, 
# AND THE REST IS WRITTEN BY ME (this you can use under WTFPLv2 or CC-0).

cmake_minimum_required(VERSION 3.0)
project(cuFFT-gRPC VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 14)
include(CPack)

include(CTest)
enable_testing()

# == main ==

add_executable(${PROJECT_NAME} src/main.cpp)

# == glog ==

find_library(GLOG_LIBRARY glog)
target_link_libraries(${PROJECT_NAME} ${GLOG_LIBRARY})

# == CUDA ==

find_package(CUDA REQUIRED)
find_package(CUDAToolkit)
# Check if we can use --threads for faster compile
if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11)
  if(${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 2)
    set(CMAKE_CUDA_FLAGS "--threads=0")
  endif()
endif()
include_directories(include ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} cufft)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 72)

# == protobuf ==

find_package(Protobuf REQUIRED)
set(PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
target_link_libraries(${PROJECT_NAME} protobuf::libprotobuf)

# == gRPC ==

set(GRPC_CPP grpc++)
set(GRPC_CPP_REFLECTION grpc++_reflection)
find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
# Proto file
get_filename_component(svc_proto "protobuf/service.proto" ABSOLUTE)
get_filename_component(svc_proto_path "${svc_proto}" PATH)
# Generated sources
set(svc_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/service.pb.cc")
set(svc_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/service.pb.h")
set(svc_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/service.grpc.pb.cc")
set(svc_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/service.grpc.pb.h")
add_custom_command(
      OUTPUT "${svc_proto_srcs}" "${svc_proto_hdrs}" "${svc_grpc_srcs}" "${svc_grpc_hdrs}"
      COMMAND ${PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${svc_proto_path}"
        --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${svc_proto}"
      DEPENDS "${svc_proto}")
# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
# svc_grpc_proto
add_library(svc_grpc_proto
  ${svc_grpc_srcs}
  ${svc_grpc_hdrs}
  ${svc_proto_srcs}
  ${svc_proto_hdrs})
target_link_libraries(svc_grpc_proto
  ${GRPC_CPP_REFLECTION}
  ${GRPC_CPP})
# Final result
target_link_libraries(${PROJECT_NAME} svc_grpc_proto)
