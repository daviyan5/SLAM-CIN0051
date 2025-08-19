add_executable(cli cli.cpp)

target_include_directories(
    cli
    PUBLIC
        ${SLAM_INCLUDE_DIRS}
    PRIVATE
        ${spdlog_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
)

target_link_libraries(
    cli
    PUBLIC
        slam_model
    PRIVATE
        spdlog::spdlog
        fmt::fmt
        opencv::opencv
        Eigen3::Eigen
)

install(TARGETS cli DESTINATION bin)
