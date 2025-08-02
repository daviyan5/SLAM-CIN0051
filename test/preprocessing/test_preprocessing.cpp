#include <Eigen/Eigen>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#include <slam/common/common.hpp>

int main() {
    spdlog::set_level(spdlog::level::debug);
    slam::Camera camera("./data/camera.yml");
    cv::Mat image = cv::imread("./data/images/0000000000.png", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        spdlog::error("Failed to load test image.");
        return -1;
    }

    Eigen::MatrixXd undistortedImage{camera.undistortImage(std::move(image))};

    cv::Mat undistortedImageCv;

    // Undistorted is between 0 to 1. Multiply by 255 to convert to 8-bit image.
    slam::EigenGrayMatrix undistortedImageU8 =
        (undistortedImage.array() * slam::COLOR_RANGE).cast<unsigned char>();

    // Copy data from Eigen matrix to OpenCV
    undistortedImageCv = cv::Mat(undistortedImageU8.rows(), undistortedImageU8.cols(), CV_8UC1);
    cv::eigen2cv(undistortedImageU8, undistortedImageCv);

    bool success = cv::imwrite("./results/undistorted_output.png", undistortedImageCv);
    if (success) {
        spdlog::info("Successfully saved undistorted image to undistorted_output.png");
    } else {
        spdlog::error("Failed to save the image.");
    }

    return 0;
}
