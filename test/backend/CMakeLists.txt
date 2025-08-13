add_executable(test_loop_closure test_loop_closure.cpp)

target_include_directories(
    test_loop_closure
    PUBLIC
        ${SLAM_INCLUDE_DIRS}
)

target_link_libraries(
    test_loop_closure
    PUBLIC
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

install(TARGETS test_loop_closure DESTINATION test)