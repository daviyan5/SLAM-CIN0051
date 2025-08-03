#include <Eigen/Eigen>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/fmt/chrono.h>
#include <spdlog/spdlog.h>

#include <slam/common/common.hpp>
#include <slam/preprocessing/preprocessor.hpp>

int main() {
    try {
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
        undistortedImageCv = cv::Mat(static_cast<int>(undistortedImageU8.rows()),
                                     static_cast<int>(undistortedImageU8.cols()), CV_8UC1);
        cv::eigen2cv(undistortedImageU8, undistortedImageCv);

        bool success = cv::imwrite("./results/undistorted_output.png", undistortedImageCv);
        if (success) {
            spdlog::info("Successfully saved undistorted image to undistorted_output.png");
        } else {
            spdlog::error("Failed to save the image.");
        }

        // Test preprocessor
        slam::Preprocessor preprocessor("./data/images", camera);

        constexpr int N_FRAMES = 10;
        for (int i = 0; i < N_FRAMES; ++i) {
            try {
                slam::MatrixTimePair pair{preprocessor.yield()};
                if (pair.first.rows() == 0 and pair.first.cols() == 0) {
                    spdlog::error("No frames yielded from preprocessor.");
                    return -1;
                }
                spdlog::info("Timestamp received: {}", pair.second);
            } catch (const std::exception& e) {
                spdlog::error("Failed to prepare preprocessor: {}", e.what());
                return -1;
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("An error occurred: {}", e.what());
        return -1;
    }

    return 0;
}
