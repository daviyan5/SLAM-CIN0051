add_executable(test_loop_closure test_loop_closure.cpp)

target_include_directories(
    test_loop_closure
    PUBLIC
        ${SLAM_INCLUDE_DIRS}
    PRIVATE
        ${spdlog_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
        ${fbow_INCLUDE_DIRS}
)

target_link_libraries(
    test_loop_closure
    PUBLIC
        slam_backend
        slam_frontend
    PRIVATE
        fbow::fbow
        OpenMP::OpenMP_CXX
        spdlog::spdlog
        fmt::fmt
        opencv::opencv
        Eigen3::Eigen
)

install(TARGETS test_loop_closure DESTINATION test)
