add_library(slam_loop_closure loop_closure.cpp)

target_include_directories(
    slam_loop_closure
    PUBLIC
        ${SLAM_INCLUDE_DIRS}
)

target_link_libraries(
    slam_loop_closure
    PRIVATE
        fbow::fbow
        OpenMP::OpenMP_CXX
        spdlog::spdlog
        fmt::fmt
        opencv::opencv
        Eigen3::Eigen
        yaml-cpp::yaml-cpp
)

install(TARGETS slam_loop_closure DESTINATION lib)