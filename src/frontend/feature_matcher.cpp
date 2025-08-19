#include <algorithm>
#include <array>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/persistence.hpp>  // For cv::FileStorage only

#include <spdlog/spdlog.h>

#include <slam/frontend/feature_matcher.hpp>

using slam::FeatureMatcher;

constexpr uint8_t popcount(int n) {
    uint8_t count = 0;
    while (n > 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}

constexpr std::array<uint8_t, slam::constants::POSSIBLE_VALUES> generatePopcountTable() {
    std::array<uint8_t, slam::constants::POSSIBLE_VALUES> table{};
    for (int i = 0; i < slam::constants::POSSIBLE_VALUES; ++i) {
        table.at(i) = popcount(i);
    }
    return table;
}

static const auto POPCOUNT_TABLE = generatePopcountTable();

FeatureMatcher::FeatureMatcher(const std::filesystem::path& configPath) {
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Could not open feature matcher config file: " +
                                 configPath.string());
    }

    std::string distanceTypeStr;
    fs["DistanceType"] >> distanceTypeStr;
    if (distanceTypeStr == "HAMMING") {
        m_distanceType = DistanceType::HAMMING;
    } else if (distanceTypeStr == "L2") {
        m_distanceType = DistanceType::L2;
    } else {
        throw std::runtime_error("Invalid distance type. Must be 'HAMMING' or 'L2'.");
    }

    int tmpFilterMatches{};
    fs["FilterMatches"] >> tmpFilterMatches;
    if (tmpFilterMatches != 0 && tmpFilterMatches != 1) {
        throw std::runtime_error("FilterMatches must be either 0 (false) or 1 (true).");
    }
    m_filterMatches = static_cast<bool>(tmpFilterMatches);

    fs["GoodMatchesCount"] >> m_goodMatchesCount;
    if (m_filterMatches && m_goodMatchesCount <= 0) {
        throw std::runtime_error("GoodMatchesCount must be positive when filtering is enabled.");
    }

    int tmpUseRatioTest{};
    fs["UseRatioTest"] >> tmpUseRatioTest;
    if (tmpUseRatioTest != 0 && tmpUseRatioTest != 1) {
        throw std::runtime_error("UseRatioTest must be either 0 (false) or 1 (true).");
    }
    m_useRatioTest = static_cast<bool>(tmpUseRatioTest);

    fs["RatioTestThreshold"] >> m_ratioTestThreshold;
    if (m_ratioTestThreshold < 0.0F || m_ratioTestThreshold > 1.0F) {
        throw std::runtime_error("RatioTestThreshold must be in the range [0, 1].");
    }

    fs.release();

    SPDLOG_DEBUG("Feature Matcher Configuration:");
    SPDLOG_DEBUG("  Distance Type: {}", distanceTypeStr);
    SPDLOG_DEBUG("  Filter Matches: {}", m_filterMatches);
    SPDLOG_DEBUG("  Good Matches Count: {}", m_goodMatchesCount);
    SPDLOG_DEBUG("  Use Ratio Test: {}", m_useRatioTest);
    SPDLOG_DEBUG("  Ratio Test Threshold: {:.2f}", m_ratioTestThreshold);

    SPDLOG_INFO("FeatureMatcher initialized with {} distance metric", distanceTypeStr);
}

void FeatureMatcher::match(const slam::DescriptorMatrix& descriptors1,
                           const slam::DescriptorMatrix& descriptors2,
                           std::vector<slam::Match>& matches,
                           const std::vector<Keypoint>& keypoints1,
                           const std::vector<Keypoint>& keypoints2) const {
    matches.clear();
    validateInputs(descriptors1, descriptors2);

    SPDLOG_DEBUG("Matching {} descriptors against {} descriptors", descriptors1.rows(),
                 descriptors2.rows());

    std::vector<slam::Match> allBestMatches;
    if (m_distanceType == DistanceType::HAMMING) {
        findBestMatchesHamming(descriptors1, descriptors2, keypoints1, keypoints2, allBestMatches);
    } else {
        throw std::runtime_error("L2 distance requires float descriptors. Use the float overload.");
    }

    if (m_filterMatches) {
        filterAndSortMatches(allBestMatches);
    }

    matches = std::move(allBestMatches);
    SPDLOG_INFO("Matched {} features successfully", matches.size());
}

void FeatureMatcher::validateInputs(const slam::DescriptorMatrix& d1,
                                    const slam::DescriptorMatrix& d2) const {
    if (d1.rows() == 0 || d2.rows() == 0) {
        SPDLOG_DEBUG("Empty descriptors provided to matcher");
        throw std::invalid_argument("Empty descriptors provided.");
    }

    if (m_distanceType != DistanceType::HAMMING) {
        throw std::runtime_error("DescriptorMatrix (uint8_t) requires HAMMING distance.");
    }

    if (d1.cols() != d2.cols()) {
        throw std::runtime_error("Descriptor dimensions must match.");
    }
}

void FeatureMatcher::findBestMatchesL2(const Eigen::MatrixXf& descriptors1,
                                       const Eigen::MatrixXf& descriptors2,
                                       std::vector<slam::Match>& bestMatches) {
    const Eigen::VectorXf d1NormsSq = descriptors1.rowwise().squaredNorm();
    const Eigen::VectorXf d2NormsSq = descriptors2.rowwise().squaredNorm();

    const Eigen::MatrixXf distsSq = d1NormsSq.replicate(1, descriptors2.rows()) +
                                    d2NormsSq.transpose().replicate(descriptors1.rows(), 1) -
                                    2.0F * (descriptors1 * descriptors2.transpose());

    bestMatches.reserve(distsSq.rows());
    for (Eigen::Index i = 0; i < distsSq.rows(); ++i) {
        Eigen::MatrixXf::Index minIndex{};
        float minSqDist = distsSq.row(i).minCoeff(&minIndex);
        bestMatches.emplace_back(static_cast<int>(i), static_cast<int>(minIndex),
                                 std::sqrt(std::max(0.0F, minSqDist)));
    }
}

template <typename Derived1, typename Derived2>
static int calculateHammingDistance(const Eigen::MatrixBase<Derived1>& d1,
                                    const Eigen::MatrixBase<Derived2>& d2) {
    int dist = 0;
    for (Eigen::Index k = 0; k < d1.size(); ++k) {
        const uint32_t index = d1(k) ^ d2(k);
        dist += POPCOUNT_TABLE.at(index);
    }
    return dist;
}

static void updateBestMatches(int dist, Eigen::Index j, int& bestDist, int& secondBestDist,
                              Eigen::Index& bestIndex) {
    if (dist < bestDist) {
        secondBestDist = bestDist;
        bestDist = dist;
        bestIndex = j;
    } else if (dist < secondBestDist) {
        secondBestDist = dist;
    }
}

void FeatureMatcher::findBestMatchesHamming(const slam::DescriptorMatrix& descriptors1,
                                            const slam::DescriptorMatrix& descriptors2,
                                            const std::vector<Keypoint>& keypoints1,
                                            const std::vector<Keypoint>& keypoints2,
                                            std::vector<slam::Match>& bestMatches) const {
    const Eigen::Index desc1Count = descriptors1.rows();
    const Eigen::Index desc2Count = descriptors2.rows();
    const bool nonEmptyKeypoints = !keypoints1.empty() && !keypoints2.empty();
    bestMatches.reserve(desc1Count);

    for (Eigen::Index i = 0; i < desc1Count; ++i) {
        int bestDist = std::numeric_limits<int>::max();
        int secondBestDist = std::numeric_limits<int>::max();
        Eigen::Index bestIndex = -1;

        for (Eigen::Index j = 0; j < desc2Count; ++j) {
            int dist = calculateHammingDistance(descriptors1.row(i), descriptors2.row(j));

            if (nonEmptyKeypoints) {
                const float dx = keypoints1[i].x - keypoints2[j].x;
                const float dy = keypoints1[i].y - keypoints2[j].y;
                const float imageDistance = std::sqrt(dx * dx + dy * dy);
                if (imageDistance > slam::constants::MAX_JUMP_RADIUS) {
                    dist *= slam::constants::JUMP_PENALTY;
                }
            }

            updateBestMatches(dist, j, bestDist, secondBestDist, bestIndex);
        }

        bool matchIsGood = true;
        if (m_useRatioTest) {
            // Apply Lowe's ratio test
            if (static_cast<float>(bestDist) >=
                m_ratioTestThreshold * static_cast<float>(secondBestDist)) {
                matchIsGood = false;
            }
        }

        if (matchIsGood && bestIndex != -1) {
            bestMatches.emplace_back(static_cast<int>(i), static_cast<int>(bestIndex),
                                     static_cast<float>(bestDist));
        }
    }
}

void FeatureMatcher::filterAndSortMatches(std::vector<slam::Match>& matchesToFilter) const {
    const auto sortPredicate = [](const slam::Match& a, const slam::Match& b) {
        return a.distance < b.distance;
    };

    if (matchesToFilter.size() > static_cast<size_t>(m_goodMatchesCount)) {
        std::partial_sort(matchesToFilter.begin(), matchesToFilter.begin() + m_goodMatchesCount,
                          matchesToFilter.end(), sortPredicate);
        matchesToFilter.erase(matchesToFilter.begin() + m_goodMatchesCount, matchesToFilter.end());
    } else {
        std::sort(matchesToFilter.begin(), matchesToFilter.end(), sortPredicate);
    }
    SPDLOG_DEBUG("Filtered to {} best matches", matchesToFilter.size());
}
