add_library(slam_backend loop_closure.cpp)

target_include_directories(
    slam_backend
    PUBLIC
        ${SLAM_INCLUDE_DIRS}
        ${spdlog_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
        ${fbow_INCLUDE_DIRS}
)

target_link_libraries(
    slam_backend
    PUBLIC
        slam_frontend
        fbow::fbow
        OpenMP::OpenMP_CXX
    PRIVATE
        spdlog::spdlog
        fmt::fmt
        opencv::opencv
        Eigen3::Eigen
)

install(TARGETS slam_backend DESTINATION lib)
