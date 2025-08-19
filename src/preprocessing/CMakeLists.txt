add_library(slam_preprocessing preprocessor.cpp)

target_include_directories(
    slam_preprocessing
    PUBLIC
        ${SLAM_INCLUDE_DIRS}
    PRIVATE
        ${spdlog_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
        ${yaml-cpp_INCLUDE_DIRS}
)

target_link_libraries(
    slam_preprocessing
    PRIVATE
        spdlog::spdlog
        fmt::fmt
        opencv::opencv
        Eigen3::Eigen
        yaml-cpp::yaml-cpp
)

install(TARGETS slam_preprocessing DESTINATION lib)
