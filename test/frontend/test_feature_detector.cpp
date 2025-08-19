#include <chrono>

#include <Eigen/Eigen>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#include <slam/frontend/feature_detector.hpp>

int main() {
    try {
        spdlog::set_level(spdlog::level::debug);
        slam::FeatureDetector featureDetector("./data/feature_detector.yml");
        cv::Mat cvImage = cv::imread("./data/images/0000000000.png", cv::IMREAD_GRAYSCALE);

        if (cvImage.empty()) {
            spdlog::error("Failed to load test image.");
            return -1;
        }

        // Convert OpenCV image to Eigen matrix
        slam::EigenGrayMatrix eigenImage(cvImage.rows, cvImage.cols);
        for (int i = 0; i < cvImage.rows; ++i) {
            for (int j = 0; j < cvImage.cols; ++j) {
                eigenImage(i, j) = cvImage.at<uchar>(i, j);
            }
        }

        std::vector<slam::Keypoint> keypoints;
        slam::DescriptorMatrix descriptors;

        auto detectStart = std::chrono::high_resolution_clock::now();
        featureDetector.detect(eigenImage, keypoints);
        auto detectEnd = std::chrono::high_resolution_clock::now();

        auto computeStart = std::chrono::high_resolution_clock::now();
        featureDetector.compute(eigenImage, keypoints, descriptors);
        auto computeEnd = std::chrono::high_resolution_clock::now();

        auto detectDuration =
            std::chrono::duration_cast<std::chrono::milliseconds>(detectEnd - detectStart);
        auto computeDuration =
            std::chrono::duration_cast<std::chrono::milliseconds>(computeEnd - computeStart);

        spdlog::info("Detected {} keypoints", keypoints.size());
        spdlog::info("Computed {} descriptors", descriptors.rows());
        spdlog::info("Keypoint detection: {} ms", detectDuration.count());
        spdlog::info("Descriptor computation: {} ms", computeDuration.count());

        // Convert Eigen keypoints back to OpenCV for visualization
        std::vector<cv::KeyPoint> cvKeypoints;
        cvKeypoints.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            cv::KeyPoint cvKp;
            cvKp.pt.x = kp.x;
            cvKp.pt.y = kp.y;
            cvKp.size = kp.size;
            cvKp.angle = kp.angle;
            cvKp.response = kp.response;
            cvKeypoints.push_back(cvKp);
        }

        cv::Mat keypointsImage;
        cv::drawKeypoints(cvImage, cvKeypoints, keypointsImage);

        bool success = cv::imwrite("./results/keypoints_output.png", keypointsImage);
        if (success) {
            spdlog::info("Successfully saved image with keypoints to keypoints_output.png");
        } else {
            spdlog::error("Failed to save the image.");
        }

        spdlog::info("Testing detectAndCompute...");
        std::vector<slam::Keypoint> keypoints2;
        slam::DescriptorMatrix descriptors2;

        auto detectAndComputeStart = std::chrono::high_resolution_clock::now();
        featureDetector.detectAndCompute(eigenImage, keypoints2, descriptors2);
        auto detectAndComputeEnd = std::chrono::high_resolution_clock::now();

        auto detectAndComputeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            detectAndComputeEnd - detectAndComputeStart);

        spdlog::info("DetectAndCompute: {} keypoints, {} descriptors in {} ms", keypoints2.size(),
                     descriptors2.rows(), detectAndComputeDuration.count());

    } catch (const std::exception& e) {
        spdlog::error("An error occurred: {}", e.what());
        return -1;
    }

    return 0;
}
