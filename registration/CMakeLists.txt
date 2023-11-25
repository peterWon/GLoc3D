project (LoPcGR)

cmake_minimum_required (VERSION 3.8)

SET(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/cmake")
set(OpenCV_DIR "/home/wz/study_hub/opencv-3.3.1/install/share/OpenCV")
# set(Torch_DIR "/home/wz/Downloads/libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu/libtorch/share/cmake/Torch")
set(Torch_DIR "/home/wz/Downloads/libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117/libtorch/share/cmake/Torch")


find_package(PCL REQUIRED)
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)
find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)
message(STATUS "Torch library status:")
message(STATUS "    version: ${TORCH_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBS}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")
message(STATUS "    torch lib : ${TORCH_LIBRARIES} ")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} 
                    ${TORCH_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


file(GLOB_RECURSE 3D_LIBRARY_SRCS "3d/*.cpp")
file(GLOB_RECURSE 2D_LIBRARY_SRCS "2d/*.cpp")


add_executable(save_probability_img "save_probability_img.cpp" 
  ${3D_LIBRARY_SRCS} ${2D_LIBRARY_SRCS} ${SC_LIBRARY_SRCS}) 
target_link_libraries(save_probability_img ${PCL_LIBRARIES} ${OpenCV_LIBS} glog ${CERES_LIBRARIES})


add_executable(global_localization "global_localization.cpp" 
  ground_estimator.cpp loop_detector.cpp ${3D_LIBRARY_SRCS} ${2D_LIBRARY_SRCS} ${SC_LIBRARY_SRCS})  
target_link_libraries(global_localization ${PCL_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS} glog faiss ${CERES_LIBRARIES} )