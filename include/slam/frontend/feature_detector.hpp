#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <spdlog/spdlog.h>

#include <slam/common/common.hpp>

namespace slam {

namespace constants {
constexpr int CIRCLE_PERIMETER = 16;
constexpr int BRIEF_PAIRS = 8;
constexpr int BLUR_KERNEL_SIZE = 5;
constexpr float DEFAULT_KEYPOINT_SIZE = 6.0F;
constexpr int CARDINAL_DIRECTION_STEP = 8;
constexpr int FULL_CIRCLE_TEST_COUNT = CIRCLE_PERIMETER * 2;
constexpr float RADIANS_TO_DEGREES = 180.0F / CV_PI;
constexpr float DEGREES_TO_RADIANS = CV_PI / 180.0F;
}  // namespace constants

class FeatureDetector {
public:
    /**
     * @brief Constructor for the feature detector.
     * @param configPath Path to the configuration file.
     */
    explicit FeatureDetector(const std::filesystem::path& configPath) {
        cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Could not open feature detector file: " +
                                     configPath.string());
        }

        fs["IntensityThreshold"] >> m_intensityThreshold;
        if (m_intensityThreshold < 0 || m_intensityThreshold > constants::COLOR_RANGE) {
            throw std::runtime_error("Intensity threshold must be in the range [0, 255].");
        }

        fs["ContiguousPixelsThreshold"] >> m_contiguousPixelsThreshold;
        if (m_contiguousPixelsThreshold < 0 ||
            m_contiguousPixelsThreshold > constants::CIRCLE_PERIMETER) {
            throw std::runtime_error("Contiguous pixels threshold must be in the range [0, 16].");
        }

        int tmpMaxSuppressionValue{};
        fs["NonMaxSuppression"] >> tmpMaxSuppressionValue;

        if (tmpMaxSuppressionValue != 0 && tmpMaxSuppressionValue != 1) {
            throw std::runtime_error("Non-max suppression must be either 0 (false) or 1 (true).");
        }

        m_nonMaxSuppression = static_cast<bool>(tmpMaxSuppressionValue);

        fs["SuppressionWindowSize"] >> m_suppressionWindowSize;
        if (m_suppressionWindowSize <= 0) {
            throw std::runtime_error("Suppression window size must be a positive integer.");
        }

        fs["PatchSize"] >> m_patchSize;
        if (m_patchSize <= 0 || m_patchSize % 2 == 0) {
            throw std::runtime_error("Patch size must be a positive odd integer.");
        }

        fs["NumBRIEFPairs"] >> m_numBRIEFPairs;
        if (m_numBRIEFPairs <= 0 || m_numBRIEFPairs % constants::BRIEF_PAIRS != 0) {
            throw std::runtime_error("Number of BRIEF pairs must be a positive multiple of 8.");
        }
        fs.release();

        // BRIEF uses a fixed pattern for sampling, so we generate it once here.
        // This pattern is used to compute the BRIEF descriptors.
        m_briefPattern = generateBRIEFPattern();

        SPDLOG_DEBUG("FAST intensity threshold: {}", m_intensityThreshold);
        SPDLOG_DEBUG("FAST contiguous pixels threshold: {}", m_contiguousPixelsThreshold);
        SPDLOG_DEBUG("FAST non-max suppression: {}", m_nonMaxSuppression);
        SPDLOG_DEBUG("FAST suppression window size: {}", m_suppressionWindowSize);

        SPDLOG_DEBUG("BRIEF patch size: {}", m_patchSize);
        SPDLOG_DEBUG("BRIEF number of pairs: {}", m_numBRIEFPairs);
    }

    /**
     * @brief Detects keypoints in the given image.
     * @param image The input image in which to detect features.
     * @param keypoints Output vector of detected keypoints.
     */
    void detect(const cv::Mat& image,                   // in
                std::vector<cv::KeyPoint>& keypoints);  // out

    /**
     * @brief Computes descriptors for the detected keypoints.
     * @param image The input image in which to compute descriptors.
     * @param keypoints The input vector of keypoints for which to compute descriptors.
     * @param descriptors Output matrix of computed descriptors.
     */
    void compute(const cv::Mat& image,                  // in
                 std::vector<cv::KeyPoint>& keypoints,  // in
                 cv::Mat& descriptors);                 // out

    /**
     * @brief Detects keypoints and computes descriptors in the given image.
     * @param image The input image in which to detect features and compute descriptors.
     * @param keypoints Output vector of detected keypoints.
     * @param descriptors Output matrix of computed descriptors.
     */
    void detectAndCompute(const cv::Mat& image,                  // in
                          std::vector<cv::KeyPoint>& keypoints,  // out
                          cv::Mat& descriptors);                 // out

private:
    static constexpr std::array<std::array<int, 2>, 16> M_PIXEL_OFFSETS = {{{0, -3},
                                                                            {1, -3},
                                                                            {2, -2},
                                                                            {3, -1},
                                                                            {3, 0},
                                                                            {3, 1},
                                                                            {2, 2},
                                                                            {1, 3},
                                                                            {0, 3},
                                                                            {-1, 3},
                                                                            {-2, 2},
                                                                            {-3, 1},
                                                                            {-3, 0},
                                                                            {-3, -1},
                                                                            {-2, -2},
                                                                            {-1, -3}}};

    int m_intensityThreshold{};
    int m_contiguousPixelsThreshold{};

    bool m_nonMaxSuppression{};
    int m_suppressionWindowSize{};

    int m_patchSize{};
    int m_numBRIEFPairs{};
    std::vector<std::pair<cv::Point2i, cv::Point2i>> m_briefPattern;

    [[nodiscard]] bool isFASTCorner(const cv::Mat& image, int x, int y) const;

    void applyNonMaxSuppression(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) const;

    [[nodiscard]] static float computeFASTScore(const cv::Mat& image, int x, int y);

    [[nodiscard]] float computeOrientation(const cv::Mat& image,
                                           const cv::KeyPoint& keypoint) const;

    [[nodiscard]] cv::Mat computeBRIEFDescriptor(const cv::Mat& image,
                                                 const cv::KeyPoint& keypoint) const;
    void detectFASTKeypoints(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    [[nodiscard]] std::vector<std::pair<cv::Point2i, cv::Point2i>> generateBRIEFPattern() const;
};

}  // namespace slam
