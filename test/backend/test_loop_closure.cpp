#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <spdlog/spdlog.h>

#include <slam/backend/loop_closure.hpp>
#include <slam/common/common.hpp>
#include <slam/frontend/feature_detector.hpp>
#include <slam/frontend/feature_matcher.hpp>

int main() {
    spdlog::set_level(spdlog::level::info);

    const std::string vocabFile = "./data/vocabulary/orb_mur.fbow";
    const std::string loopClosureConfigFile = "./data/loop_closure.yml";
    const std::string detectorConfigFile = "./data/feature_detector.yml";
    const std::string matcherConfigFile = "./data/feature_matcher.yml";
    const std::string cameraConfigFile = "./data/camera.yml";
    const std::string imageDirectory = "./data/images_test_loop2/";

    for (const auto& path : {vocabFile, loopClosureConfigFile, detectorConfigFile,
                             matcherConfigFile, cameraConfigFile}) {
        if (!std::filesystem::exists(path)) {
            spdlog::error("Required file does not exist: {}", path);
            return -1;
        }
    }
    try {
        slam::Camera camera(cameraConfigFile);
        slam::FeatureDetector featureDetector(detectorConfigFile);
        slam::FeatureMatcher matcher(matcherConfigFile);
        slam::LoopClosure detector(vocabFile, loopClosureConfigFile, matcher);

        std::vector<slam::Keypoint> lastKeypoints;
        slam::DescriptorMatrix lastDescriptors;

        spdlog::info("Processing image sequence to build database...");
        constexpr int N_IMAGES = 10;  // Number of images to process
        for (int i = 0; i < N_IMAGES; ++i) {
            std::string imagePath = imageDirectory + std::to_string(i) + ".png";
            cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                spdlog::error("Failed to load image: {}", imagePath);
                continue;
            }

            slam::EigenGrayMatrix eigenImage;
            cv::cv2eigen(img, eigenImage);

            std::vector<slam::Keypoint> kpsSlam;
            slam::DescriptorMatrix descSlam;

            featureDetector.detectAndCompute(eigenImage, kpsSlam, descSlam);

            std::vector<Eigen::Vector3d> mapPoints;
            mapPoints.reserve(kpsSlam.size());

            for (const auto& kp : kpsSlam) {
                mapPoints.emplace_back(kp.x, kp.y, 1.0);
            }

            detector.addKeyframe(i, descSlam, kpsSlam, mapPoints);

            if (i == N_IMAGES - 1) {
                lastKeypoints = kpsSlam;
                lastDescriptors = descSlam;
            }
        }

        spdlog::info("Attempting to detect loop with the last keyframe...");
        auto result = detector.detect(lastDescriptors, lastKeypoints, camera);

        if (result && result->matchedKeyframeId == 0) {
            spdlog::info("SUCCESS: Loop detected with the correct keyframe (ID {}).",
                         result->matchedKeyframeId);
        } else if (result) {
            spdlog::error("FAILURE: Loop detected with incorrect keyframe (ID {}). Expected 0.",
                          result->matchedKeyframeId);
            return -1;
        } else {
            spdlog::error("FAILURE: No loop was detected or verified.");
            return -1;
        }

    } catch (const std::exception& e) {
        spdlog::error("An unhandled exception occurred: {}", e.what());
        return -1;
    }

    return 0;
}
