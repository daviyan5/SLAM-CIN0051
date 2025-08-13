cmake_minimum_required(VERSION 3.22)
project(SLAM-CIN0051 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(
    $<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF>
)

find_package(OpenCV 4.12 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog 1.15.3 REQUIRED)
find_package(yaml-cpp 0.8.0 REQUIRED)
find_package(fbow REQUIRED)
find_package(OpenMP REQUIRED) 

message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Found Eigen version: ${Eigen3_VERSION}")
message(STATUS "Found spdlog version: ${spdlog_VERSION}")
message(STATUS "Found fmt version: ${fmt_VERSION}")
message(STATUS "Found yaml-cpp version: ${yaml-cpp_VERSION}")
message(STATUS "Found fbow version: ${fbow_VERSION}")
message(STATUS "Found OpenMP version: ${OpenMP_CXX_VERSION}")

set(SLAM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(tools/cli)

install(
    DIRECTORY ${SLAM_INCLUDE_DIRS}
    DESTINATION include
)
