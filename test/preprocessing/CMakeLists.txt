add_executable(test_preprocessing test_preprocessing.cpp)

target_include_directories(
    test_preprocessing
    PUBLIC
        ${SLAM_INCLUDE_DIRS}
    PRIVATE
        ${spdlog_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
)

target_link_libraries(
    test_preprocessing
    PUBLIC
        slam_preprocessing
    PRIVATE
        spdlog::spdlog
        fmt::fmt
        opencv::opencv
        Eigen3::Eigen
)

install(TARGETS test_preprocessing DESTINATION test)
