#include <chrono>

#include <Eigen/Eigen>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#include <slam/frontend/feature_detector.hpp>

int main() {
    spdlog::set_level(spdlog::level::debug);
    slam::FeatureDetector feature_detector("./data/feature_detector.yml");
    cv::Mat image = cv::imread("./data/images/0000000000.png", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        spdlog::error("Failed to load test image.");
        return -1;
    }

    std::vector<slam::KeyDescriptorPair> keyDescriptorPairs;

    auto detect_start = std::chrono::high_resolution_clock::now();
    feature_detector.detect(image, keyDescriptorPairs);
    auto detect_end = std::chrono::high_resolution_clock::now();

    auto compute_start = std::chrono::high_resolution_clock::now();
    feature_detector.compute(image, keyDescriptorPairs);
    auto compute_end = std::chrono::high_resolution_clock::now();

    auto detect_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start);
    auto compute_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start);

    spdlog::info("Keypoint detection: {} ms", detect_duration.count());
    spdlog::info("Descriptor computation: {} ms", compute_duration.count());

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
