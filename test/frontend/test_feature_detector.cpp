#include <chrono>

#include <Eigen/Eigen>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#include <slam/frontend/feature_detector.hpp>

int main() {
    spdlog::set_level(spdlog::level::debug);
    slam::FeatureDetector featureDetector("./data/feature_detector.yml");
    cv::Mat image = cv::imread("./data/images/0000000000.png", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        spdlog::error("Failed to load test image.");
        return -1;
    }

    std::vector<slam::KeyDescriptorPair> keyDescriptorPairs;

    auto detectStart = std::chrono::high_resolution_clock::now();
    featureDetector.detect(image, keyDescriptorPairs);
    auto detectEnd = std::chrono::high_resolution_clock::now();

    auto computeStart = std::chrono::high_resolution_clock::now();
    featureDetector.compute(image, keyDescriptorPairs);
    auto computeEnd = std::chrono::high_resolution_clock::now();

    auto detectDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(detectEnd - detectStart);
    auto computeDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(computeEnd - computeStart);

    spdlog::info("Keypoint detection: {} ms", detectDuration.count());
    spdlog::info("Descriptor computation: {} ms", computeDuration.count());

    std::vector<cv::KeyPoint> keypoints;
    for (const auto& keyDescriptorPair : keyDescriptorPairs) {
        keypoints.push_back(keyDescriptorPair.first);
    }

    cv::Mat keypointsImage;
    cv::drawKeypoints(image, keypoints, keypointsImage, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    bool success = cv::imwrite("./results/keypoints_output.png", keypointsImage);
    if (success) {
        spdlog::info("Successfully saved image with keypoints to keypoints_output.png");
    } else {
        spdlog::error("Failed to save the image.");
    }

    return 0;
}
