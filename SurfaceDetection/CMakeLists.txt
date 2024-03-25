cmake_minimum_required(VERSION 3.10.2)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
project(SurfaceDetection)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(SurfaceDetection SurfaceDetection.cxx)

target_link_libraries(SurfaceDetection ${ITK_LIBRARIES})