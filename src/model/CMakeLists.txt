add_library(slam_model model.cpp)

target_include_directories(
    slam_model 
    PUBLIC
        ${SLAM_INCLUDE_DIRS}
    PRIVATE
        ${spdlog_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
)

target_link_libraries(
    slam_model
    PUBLIC
        slam_preprocessing
    PRIVATE
        spdlog::spdlog
        fmt::fmt
        opencv::opencv
        Eigen3::Eigen
)

install(TARGETS slam_model DESTINATION lib)
