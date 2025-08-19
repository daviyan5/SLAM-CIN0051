#pragma once

#include <filesystem>
#include <vector>

#include <Eigen/Dense>

#include <slam/frontend/feature_detector.hpp>

namespace slam {
namespace constants {
constexpr uint32_t MAX_JUMP_RADIUS = 500;
}  // namespace constants

/**
 * @brief Structure representing a feature match between two images.
 */
struct Match {
    int queryIdx;    // Index of the descriptor in the query set
    int trainIdx;    // Index of the descriptor in the train set
    float distance;  // Distance between the two descriptors

    Match(int qIdx, int tIdx, float dist) : queryIdx(qIdx), trainIdx(tIdx), distance(dist) {
    }
};

/**
 * @class FeatureMatcher
 * @brief A simple brute-force feature matcher using Eigen data types.
 *
 * This class implements a basic brute-force matching strategy. For each feature
 * in the first set, it finds the single best corresponding feature in the
 * second set by minimizing a distance metric. It can optionally filter these
 * matches to return only the top N best matches overall.
 */
class FeatureMatcher {
public:
    /**
     * @brief Enum to define the distance metric for matching descriptors.
     * HAMMING is for binary descriptors (e.g., BRIEF, ORB).
     * L2 (Euclidean) is for floating-point descriptors (e.g., SIFT, SURF).
     */
    enum class DistanceType : uint8_t { HAMMING, L2 };

    /**
     * @brief Constructs a new Feature Matcher from configuration file.
     *
     * @param configPath Path to the configuration file.
     */
    explicit FeatureMatcher(const std::filesystem::path& configPath);

    /**
     * @brief Finds the best match for each descriptor from the first set in the second set.
     *
     * @param descriptors1 The set of query descriptors (for HAMMING: uint8_t matrix, for L2: float
     * matrix).
     * @param descriptors2 The set of train descriptors to search within.
     * @param matches The output vector of matches. If filtering is enabled, this will
     * contain the top 'goodMatchesCount' matches. Otherwise, it will
     * contain the single best match for each query descriptor.
     * @param keypoints1 Optional vector of keypoints corresponding to descriptors1.
     * @param keypoints2 Optional vector of keypoints corresponding to descriptors2.
     */
    void match(const DescriptorMatrix& descriptors1, const DescriptorMatrix& descriptors2,
               std::vector<Match>& matches, const std::vector<Keypoint>& keypoints1 = {},
               const std::vector<Keypoint>& keypoints2 = {}) const;

private:
    void validateInputs(const DescriptorMatrix& d1, const DescriptorMatrix& d2) const;

    static void findBestMatchesL2(const Eigen::MatrixXf& descriptors1,
                                  const Eigen::MatrixXf& descriptors2,
                                  std::vector<Match>& bestMatches);

    void findBestMatchesHamming(const DescriptorMatrix& descriptors1,
                                const DescriptorMatrix& descriptors2,
                                const std::vector<Keypoint>& keypoints1,
                                const std::vector<Keypoint>& keypoints2,
                                std::vector<Match>& bestMatches) const;

    void filterAndSortMatches(std::vector<Match>& matchesToFilter) const;

    DistanceType m_distanceType{DistanceType::HAMMING};
    bool m_filterMatches{false};
    int m_goodMatchesCount{};
    float m_ratioTestThreshold{};
    bool m_useRatioTest{false};
};

}  // namespace slam
