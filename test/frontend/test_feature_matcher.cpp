#include <chrono>

#include <Eigen/Eigen>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#include <slam/frontend/feature_detector.hpp>
#include <slam/frontend/feature_matcher.hpp>

int main() {
    try {
        spdlog::set_level(spdlog::level::debug);

        slam::FeatureDetector featureDetector("./data/feature_detector.yml");
        slam::FeatureMatcher featureMatcher("./data/feature_matcher.yml");

        cv::Mat cvImage1 = cv::imread("./data/images/0000000000.png", cv::IMREAD_GRAYSCALE);
        cv::Mat cvImage2 = cv::imread("./data/images/0000000001.png", cv::IMREAD_GRAYSCALE);

        if (cvImage1.empty() || cvImage2.empty()) {
            spdlog::error("Failed to load test images.");
            return -1;
        }

        slam::EigenGrayMatrix eigenImage1(cvImage1.rows, cvImage1.cols);
        slam::EigenGrayMatrix eigenImage2(cvImage2.rows, cvImage2.cols);

        for (int i = 0; i < cvImage1.rows; ++i) {
            for (int j = 0; j < cvImage1.cols; ++j) {
                eigenImage1(i, j) = cvImage1.at<uchar>(i, j);
                eigenImage2(i, j) = cvImage2.at<uchar>(i, j);
            }
        }

        std::vector<slam::Keypoint> keypoints1, keypoints2;
        slam::DescriptorMatrix descriptors1, descriptors2;

        auto detect1Start = std::chrono::high_resolution_clock::now();
        featureDetector.detectAndCompute(eigenImage1, keypoints1, descriptors1);
        auto detect1End = std::chrono::high_resolution_clock::now();

        auto detect2Start = std::chrono::high_resolution_clock::now();
        featureDetector.detectAndCompute(eigenImage2, keypoints2, descriptors2);
        auto detect2End = std::chrono::high_resolution_clock::now();

        auto detect1Duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(detect1End - detect1Start);
        auto detect2Duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(detect2End - detect2Start);

        spdlog::info("Image 1: Detected {} keypoints with {} descriptors in {} ms",
                     keypoints1.size(), descriptors1.rows(), detect1Duration.count());
        spdlog::info("Image 2: Detected {} keypoints with {} descriptors in {} ms",
                     keypoints2.size(), descriptors2.rows(), detect2Duration.count());

        std::vector<slam::Match> matches;

        auto matchStart = std::chrono::high_resolution_clock::now();
        featureMatcher.match(descriptors1, descriptors2, matches, keypoints1, keypoints2);
        auto matchEnd = std::chrono::high_resolution_clock::now();

        auto matchDuration =
            std::chrono::duration_cast<std::chrono::milliseconds>(matchEnd - matchStart);

        spdlog::info("Found {} matches in {} ms", matches.size(), matchDuration.count());

        std::vector<cv::KeyPoint> cvKeypoints1, cvKeypoints2;
        cvKeypoints1.reserve(keypoints1.size());
        cvKeypoints2.reserve(keypoints2.size());

        for (const auto& kp : keypoints1) {
            cv::KeyPoint cvKp;
            cvKp.pt.x = kp.x;
            cvKp.pt.y = kp.y;
            cvKp.size = kp.size;
            cvKp.angle = kp.angle;
            cvKp.response = kp.response;
            cvKeypoints1.push_back(cvKp);
        }

        for (const auto& kp : keypoints2) {
            cv::KeyPoint cvKp;
            cvKp.pt.x = kp.x;
            cvKp.pt.y = kp.y;
            cvKp.size = kp.size;
            cvKp.angle = kp.angle;
            cvKp.response = kp.response;
            cvKeypoints2.push_back(cvKp);
        }

        std::vector<cv::DMatch> cvMatches;
        cvMatches.reserve(matches.size());
        for (const auto& match : matches) {
            cvMatches.emplace_back(match.queryIdx, match.trainIdx, match.distance);
        }

        cv::Mat matchesImage;
        cv::drawMatches(cvImage1, cvKeypoints1, cvImage2, cvKeypoints2, cvMatches, matchesImage);

        bool success = cv::imwrite("./results/matches_output.png", matchesImage);
        if (success) {
            spdlog::info("Successfully saved matches visualization to matches_output.png");
        } else {
            spdlog::error("Failed to save the matches image.");
        }

        if (!matches.empty()) {
            float minDist = std::numeric_limits<float>::max();
            float maxDist = 0.0F;
            float avgDist = 0.0F;

            for (const auto& match : matches) {
                minDist = std::min(minDist, match.distance);
                maxDist = std::max(maxDist, match.distance);
                avgDist += match.distance;
            }
            avgDist /= static_cast<float>(matches.size());

            spdlog::info("Match distances: min={:.2f}, max={:.2f}, avg={:.2f}", minDist, maxDist,
                         avgDist);
        }

    } catch (const std::exception& e) {
        spdlog::error("An error occurred: {}", e.what());
        return -1;
    }

    return 0;
}
