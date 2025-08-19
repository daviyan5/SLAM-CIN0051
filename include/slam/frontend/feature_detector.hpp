#pragma once

#include <filesystem>
#include <vector>

#include <Eigen/Dense>

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
constexpr float RADIANS_TO_DEGREES = 180.0F / M_PI;
constexpr float DEGREES_TO_RADIANS = M_PI / 180.0F;
}  // namespace constants

/**
 * @brief Structure representing a keypoint in an image.
 */
struct Keypoint {
    float x{};         // x-coordinate
    float y{};         // y-coordinate
    float size{};      // size/scale of the keypoint
    float angle{};     // orientation angle in degrees
    float response{};  // response/score of the keypoint

    Keypoint(float X, float Y, float size = constants::DEFAULT_KEYPOINT_SIZE)
        : x(X), y(Y), size(size) {
    }
};

/**
 * @brief Typedef for descriptor matrix.
 * Each row represents a descriptor for a keypoint.
 * For BRIEF descriptors, each element is a byte (8 bits).
 */
using DescriptorMatrix = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
     * @param image The input image as an Eigen matrix (grayscale, values 0-255).
     * @param keypoints Output vector of detected keypoints.
     */
    void detect(const EigenGrayMatrix& image,       // in
                std::vector<Keypoint>& keypoints);  // out

    /**
     * @brief Computes descriptors for the detected keypoints.
     * @param image The input image as an Eigen matrix (grayscale, values 0-255).
     * @param keypoints The input vector of keypoints for which to compute descriptors.
     * @param descriptors Output matrix of computed descriptors.
     */
    void compute(const EigenGrayMatrix& image,      // in
                 std::vector<Keypoint>& keypoints,  // in/out (angles are updated)
                 DescriptorMatrix& descriptors);    // out

    /**
     * @brief Detects keypoints and computes descriptors in the given image.
     * @param image The input image as an Eigen matrix (grayscale, values 0-255).
     * @param keypoints Output vector of detected keypoints.
     * @param descriptors Output matrix of computed descriptors.
     */
    void detectAndCompute(const EigenGrayMatrix& image,      // in
                          std::vector<Keypoint>& keypoints,  // out
                          DescriptorMatrix& descriptors);    // out

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
    std::vector<std::pair<Eigen::Vector2i, Eigen::Vector2i>> m_briefPattern;

    [[nodiscard]] bool isFASTCorner(const EigenGrayMatrix& image, int x, int y) const;

    void applyNonMaxSuppression(const EigenGrayMatrix& image,
                                std::vector<Keypoint>& keypoints) const;

    [[nodiscard]] static float computeFASTScore(const EigenGrayMatrix& image, int x, int y);

    [[nodiscard]] float computeOrientation(const EigenGrayMatrix& image,
                                           const Keypoint& keypoint) const;

    [[nodiscard]] Eigen::Matrix<uint8_t, 1, Eigen::Dynamic> computeBRIEFDescriptor(
        const EigenGrayMatrix& image, const Keypoint& keypoint) const;

    void detectFASTKeypoints(const EigenGrayMatrix& image, std::vector<Keypoint>& keypoints);

    [[nodiscard]] std::vector<std::pair<Eigen::Vector2i, Eigen::Vector2i>> generateBRIEFPattern()
        const;

    /**
     * @brief Apply Gaussian blur to an image using Eigen.
     * @param image Input image
     * @param kernelSize Size of the Gaussian kernel (must be odd)
     * @param sigma Standard deviation of the Gaussian
     * @return Blurred image
     */
    [[nodiscard]] static EigenGrayMatrix gaussianBlur(const EigenGrayMatrix& image, int kernelSize,
                                                      double sigma);
};

}  // namespace slam
