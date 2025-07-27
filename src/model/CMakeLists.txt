add_library(slam_model model.cpp)

target_include_directories(
    slam_model 
    PUBLIC
    ${SLAM_INCLUDE_DIRS}
    PRIVATE 
    ${quill_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS} 
)

target_link_libraries(
    slam_model PRIVATE
    OpenCV::OpenCV
    Eigen3::Eigen
)

install(TARGETS slam_model DESTINATION lib)