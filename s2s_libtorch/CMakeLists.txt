cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(s2s) 
 
SET(CMAKE_C_COMPILER g++)
# add_definitions(--std=c++14)


find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# the path of pre-libTorch
set(Torch_DIR /home/wz/Downloads/libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
message(STATUS "Torch library status:")
message(STATUS "    version: ${TORCH_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBS}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")
message(STATUS "    torch lib : ${TORCH_LIBRARIES} ")
 
# include_directories(${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
add_executable(s2s_feature_extract s2s_feature_extract.cpp)
target_link_libraries(s2s_feature_extract ${TORCH_LIBRARIES} ${PCL_LIBRARIES})


