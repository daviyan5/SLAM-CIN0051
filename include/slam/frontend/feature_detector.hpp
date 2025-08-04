#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <spdlog/spdlog.h>

#include <slam/common/common.hpp>

namespace slam {

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
        if (m_intensityThreshold < 0 || m_intensityThreshold > 255) {
            throw std::runtime_error("Intensity threshold must be in the range [0, 255].");
        }

        fs["ContiguousPixelsThreshold"] >> m_contiguousPixelsThreshold;
        if (m_contiguousPixelsThreshold < 0 || m_contiguousPixelsThreshold > 16) {
            throw std::runtime_error("Contiguous pixels threshold must be in the range [0, 16].");
        }

        fs["NonMaxSuppression"] >> m_nonMaxSuppression;
        if (m_nonMaxSuppression != 0 && m_nonMaxSuppression != 1) {
            throw std::runtime_error("Non-max suppression must be either 0 (false) or 1 (true).");
        }

        fs["SuppressionWindowSize"] >> m_suppressionWindowSize;
        if (m_suppressionWindowSize <= 0) {
            throw std::runtime_error("Suppression window size must be a positive integer.");
        }
        fs.release();

        SPDLOG_DEBUG("FAST intensity threshold: {}", m_intensityThreshold);
        SPDLOG_DEBUG("FAST contiguous pixels threshold: {}", m_contiguousPixelsThreshold);
        SPDLOG_DEBUG("FAST non-max suppression: {}", m_nonMaxSuppression);
        SPDLOG_DEBUG("FAST suppression window size: {}", m_suppressionWindowSize);
    }

    /**
     * @brief Detects keypoints in the given image.
     * @param image The input image in which to detect features.
     * @param keyDescriptorPairs Output vector of keypoints.
     */
    void detect(const cv::Mat& image,                                 // in
                std::vector<KeyDescriptorPair>& keyDescriptorPairs);  // out

    /**
     * @brief Computes descriptors for the detected keypoints.
     * @param image The input image in which to compute descriptors.
     * @param keyDescriptorPairs Output vector of keypoints and their descriptors.
     */
    void compute(const cv::Mat& image,                                 // in
                 std::vector<KeyDescriptorPair>& keyDescriptorPairs);  // out

    /**
     * @brief Detects keypoints and computes descriptors in the given image.
     * @param image The input image in which to detect features.
     * @param keyDescriptorPairs Output vector of keypoints and their descriptors.
     */
    void detectAndCompute(const cv::Mat& image,                                 // in
                          std::vector<KeyDescriptorPair>& keyDescriptorPairs);  // out

private:
    const int m_pixelOffsets[16][2] = {{0, -3}, {1, -3},  {2, -2},  {3, -1}, {3, 0},  {3, 1},
                                       {2, 2},  {1, 3},   {0, 3},   {-1, 3}, {-2, 2}, {-3, 1},
                                       {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

    int m_intensityThreshold;
    int m_contiguousPixelsThreshold;

    bool m_nonMaxSuppression;
    int m_suppressionWindowSize;

    void detectFASTKeypoints(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    bool isFASTCorner(const cv::Mat& image, int x, int y);
    void applyNonMaxSuppression(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    float computeFASTScore(const cv::Mat& image, int x, int y);
};

}  // namespace slam
