cmake_minimum_required(VERSION 3.22)
project(SLAM-CIN0051 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(
    $<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF>
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Found Eigen version: ${EIGEN3_VERSION_STRING}")
message(STATUS "Found spdlog version: ${spdlog_VERSION}")

set(SLAM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(tools/cli)

install(
    DIRECTORY ${SLAM_INCLUDE_DIRS}
    DESTINATION include
)