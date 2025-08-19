add_executable(test_feature_detector test_feature_detector.cpp)
add_executable(test_feature_matcher test_feature_matcher.cpp)

target_include_directories(
    test_feature_detector
    PUBLIC
        ${SLAM_INCLUDE_DIRS}
    PRIVATE
        ${spdlog_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
        ${yaml-cpp_INCLUDE_DIRS}
)
target_include_directories(
    test_feature_matcher
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
    test_feature_detector
    PUBLIC
        slam_frontend
    PRIVATE
        spdlog::spdlog
        fmt::fmt
        opencv::opencv
        Eigen3::Eigen
        yaml-cpp::yaml-cpp
)
target_link_libraries(
    test_feature_matcher
    PUBLIC
        slam_frontend
    PRIVATE
        spdlog::spdlog
        fmt::fmt
        opencv::opencv
        Eigen3::Eigen
        yaml-cpp::yaml-cpp
)
install(TARGETS test_feature_detector test_feature_matcher
        DESTINATION test)

