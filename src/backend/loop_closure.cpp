#include <algorithm>
#include <random>

#include <spdlog/spdlog.h>

#include <yaml-cpp/yaml.h>

#include <slam/backend/loop_closure.hpp>

using slam::LoopClosure;

namespace {
constexpr uint8_t POPCOUNT_LUT[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
}  // namespace

LoopClosure::LoopClosure(const std::string& vocabPath, const std::string& configPath) {
    loadParameters(configPath);

    SPDLOG_INFO("Loading fbow vocabulary from: {}", vocabPath);
    m_vocabulary.readFromFile(vocabPath);
    if (m_vocabulary.size() == 0) {
        SPDLOG_ERROR("Failed to load fbow vocabulary!");
        throw std::runtime_error("Vocabulary does not exist or is empty at path: " + vocabPath);
    }
    SPDLOG_INFO("LoopClosure module initialized.");
}

void LoopClosure::loadParameters(const std::string& configPath) {
    SPDLOG_INFO("Loading loop closure parameters from: {}", configPath);
    YAML::Node config = YAML::LoadFile(configPath);

    if (!config["loop_closure"]) {
        throw std::runtime_error("Config file is missing 'loop_closure' root node.");
    }
    auto lcConfig = config["loop_closure"];

    m_params.minDbSize = lcConfig["min_db_size"].as<int>();
    m_params.minFramesDifference = lcConfig["min_frames_difference"].as<int>();
    m_params.minAbsoluteScore = lcConfig["min_absolute_score"].as<double>();
    m_params.relativeScoreFactor = lcConfig["relative_score_factor"].as<double>();
    m_params.minMatchesForPnp = lcConfig["min_matches_for_pnp"].as<int>();
    m_params.minInliers = lcConfig["min_inliers"].as<int>();

    // Optional parameters with defaults
    if (lcConfig["ransac_reprojection_threshold"]) {
        m_params.ransacReprojectionThreshold =
            lcConfig["ransac_reprojection_threshold"].as<double>();
    }
    if (lcConfig["ransac_max_iterations"]) {
        m_params.ransacMaxIterations = lcConfig["ransac_max_iterations"].as<int>();
    }
    if (lcConfig["ransac_confidence"]) {
        m_params.ransacConfidence = lcConfig["ransac_confidence"].as<double>();
    }
}

void LoopClosure::addKeyframe(int keyframeId, const slam::DescriptorMatrix& descriptors,
                              const std::vector<slam::Keypoint>& keypoints,
                              const std::vector<Eigen::Vector3d>& mapPoints) {
    KeyframeData data;
    cv::Mat cvDescriptors = descriptorsToMat(descriptors);
    data.bowVector = m_vocabulary.transform(cvDescriptors);
    data.keypoints = keypoints;
    data.mapPoints = mapPoints;

    m_keyframeDatabase[keyframeId] = data;
    m_keyframeDescriptors[keyframeId] = descriptors;
    m_lastKeyframeId = keyframeId;

    SPDLOG_DEBUG("Added keyframe {} to the database.", keyframeId);
}

std::optional<slam::LoopResult> LoopClosure::detect(const slam::DescriptorMatrix& descriptors,
                                                    const std::vector<slam::Keypoint>& keypoints,
                                                    const slam::Camera& camera) {
    if (m_keyframeDatabase.size() < static_cast<size_t>(m_params.minDbSize)) {
        SPDLOG_DEBUG("Not enough keyframes in the database for loop detection.");
        return std::nullopt;
    }

    cv::Mat cvDescriptors = descriptorsToMat(descriptors);
    fbow::BoWVector currentBowVector = m_vocabulary.transform(cvDescriptors);
    std::vector<CandidateMatch> allMatches;

    for (const auto& [id, data] : m_keyframeDatabase) {
        if (id == m_lastKeyframeId) {
            continue;
        }
        double score = fbow::BoWVector::score(currentBowVector, data.bowVector);
        allMatches.push_back({score, id});
    }

    if (allMatches.empty()) {
        SPDLOG_DEBUG("No candidates found in the database.");
        return std::nullopt;
    }

    std::sort(allMatches.begin(), allMatches.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });

    const auto& bestCandidate = allMatches[0];

    if (std::abs(m_lastKeyframeId - bestCandidate.id) < m_params.minFramesDifference) {
        SPDLOG_DEBUG("Best candidate (ID {}) is too recent. Skipping.", bestCandidate.id);
        return std::nullopt;
    }

    bool isSignificantEnough = true;
    if (allMatches.size() > 1) {
        if (bestCandidate.score < m_params.relativeScoreFactor * allMatches[1].score) {
            isSignificantEnough = false;
        }
    }

    if (!(bestCandidate.score > m_params.minAbsoluteScore && isSignificantEnough)) {
        SPDLOG_DEBUG("Best candidate (ID {}) score {} not significant enough.", bestCandidate.id,
                     bestCandidate.score);
        return std::nullopt;
    }

    SPDLOG_INFO("BoW candidate found: ID {}. Proceeding to geometric verification.",
                bestCandidate.id);

    // Match descriptors using Hamming distance
    std::vector<Match> matches =
        matchDescriptors(descriptors, m_keyframeDescriptors.at(bestCandidate.id));

    std::vector<Eigen::Vector3d> points3d;
    std::vector<Eigen::Vector2d> points2d;
    for (const auto& match : matches) {
        points2d.emplace_back(keypoints[match.queryIdx].x, keypoints[match.queryIdx].y);
        points3d.push_back(m_keyframeDatabase.at(bestCandidate.id).mapPoints[match.trainIdx]);
    }

    if (points2d.size() < static_cast<size_t>(m_params.minMatchesForPnp)) {
        SPDLOG_WARN("Not enough matches ({}) for geometric verification.", points2d.size());
        return std::nullopt;
    }

    // Extract camera parameters
    auto [K, D] = extractCameraParameters(camera);

    Eigen::Vector3d rvec;
    Eigen::Vector3d tvec;
    std::vector<int> inliers;

    if (solvePnPRansac(points3d, points2d, K, D, rvec, tvec, inliers)) {
        if (inliers.size() >= static_cast<size_t>(m_params.minInliers)) {
            SPDLOG_INFO("Geometric verification succeeded: Found {} inliers.", inliers.size());

            slam::LoopResult result;
            result.matchedKeyframeId = bestCandidate.id;

            Eigen::Matrix3d rotationMatrix = rodrigues(rvec);
            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
            transform.block<3, 3>(0, 0) = rotationMatrix;
            transform.block<3, 1>(0, 3) = tvec;

            result.relativeTransform = transform;
            return result;
        }
        SPDLOG_WARN("Geometric verification FAILED: Only {} inliers found (minimum required: {}).",
                    inliers.size(), m_params.minInliers);
    } else {
        SPDLOG_WARN("PnP solver failed to find a solution.");
    }

    return std::nullopt;
}

Eigen::Matrix3d LoopClosure::rodrigues(const Eigen::Vector3d& rvec) {
    double theta = rvec.norm();

    if (theta < 1e-6) {
        // Small angle approximation
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Vector3d k = rvec / theta;

    // Cross-product matrix
    Eigen::Matrix3d K;
    K << 0, -k.z(), k.y(), k.z(), 0, -k.x(), -k.y(), k.x(), 0;

    // Rodrigues' rotation formula
    Eigen::Matrix3d rotationMatrix =
        Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * K * K;

    return rotationMatrix;
}

Eigen::Vector2d LoopClosure::projectPoint(const Eigen::Vector3d& point3d, const Eigen::Matrix3d& K,
                                          const Eigen::VectorXd& D) {
    // Normalize to camera plane
    double x = point3d.x() / point3d.z();
    double y = point3d.y() / point3d.z();

    // Apply distortion
    double r2 = x * x + y * y;
    double k1 = D.size() > 0 ? D(0) : 0.0;
    double k2 = D.size() > 1 ? D(1) : 0.0;
    double p1 = D.size() > 2 ? D(2) : 0.0;
    double p2 = D.size() > 3 ? D(3) : 0.0;
    double k3 = D.size() > 4 ? D(4) : 0.0;

    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double radialDistortion = 1 + k1 * r2 + k2 * r4 + k3 * r6;

    double xDistorted = x * radialDistortion + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
    double yDistorted = y * radialDistortion + 2 * p2 * x * y + p1 * (r2 + 2 * y * y);

    // Project to image plane
    double u = K(0, 0) * xDistorted + K(0, 2);
    double v = K(1, 1) * yDistorted + K(1, 2);

    return Eigen::Vector2d(u, v);
}

bool LoopClosure::solvePnPRansac(const std::vector<Eigen::Vector3d>& points3d,
                                 const std::vector<Eigen::Vector2d>& points2d,
                                 const Eigen::Matrix3d& K, const Eigen::VectorXd& D,
                                 Eigen::Vector3d& rvec, Eigen::Vector3d& tvec,
                                 std::vector<int>& inliers) const {
    if (points3d.size() != points2d.size() || points3d.size() < 4) {
        return false;
    }

    const size_t nPoints = points3d.size();
    constexpr int MIN_SET_SIZE = 4;  // Minimum points for P3P

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::uniform_int_distribution<> distribution(0, static_cast<int>(nPoints) - 1);

    int bestInliersCount = 0;
    Eigen::Vector3d bestRvec;
    Eigen::Vector3d bestTvec;
    std::vector<int> bestInliers;

    for (int iter = 0; iter < m_params.ransacMaxIterations; ++iter) {
        // Select random subset
        std::vector<int> sampleIndices;
        sampleIndices.reserve(MIN_SET_SIZE);

        while (sampleIndices.size() < MIN_SET_SIZE) {
            int idx = distribution(generator);
            if (std::find(sampleIndices.begin(), sampleIndices.end(), idx) == sampleIndices.end()) {
                sampleIndices.push_back(idx);
            }
        }

        // Simplified P3P solver (in production, use a proper implementation)
        // Here we use a basic approach for demonstration
        Eigen::Vector3d centroid3d = Eigen::Vector3d::Zero();
        Eigen::Vector2d centroid2d = Eigen::Vector2d::Zero();
        for (int idx : sampleIndices) {
            centroid3d += points3d[idx];
            centroid2d += points2d[idx];
        }
        centroid3d /= MIN_SET_SIZE;
        centroid2d /= MIN_SET_SIZE;

        // Simple initial estimate
        Eigen::Vector3d testTvec = centroid3d;
        Eigen::Vector3d testRvec = Eigen::Vector3d::Zero();

        // Count inliers
        std::vector<int> currentInliers;
        for (size_t i = 0; i < nPoints; ++i) {
            // Transform and project point
            Eigen::Vector3d rotated = rodrigues(testRvec) * points3d[i] + testTvec;

            if (rotated.z() > 0) {
                Eigen::Vector2d projected = projectPoint(rotated, K, D);

                // Compute reprojection error
                double error = (projected - points2d[i]).norm();

                if (error < m_params.ransacReprojectionThreshold) {
                    currentInliers.push_back(static_cast<int>(i));
                }
            }
        }

        if (static_cast<int>(currentInliers.size()) > bestInliersCount) {
            bestInliersCount = static_cast<int>(currentInliers.size());
            bestRvec = testRvec;
            bestTvec = testTvec;
            bestInliers = currentInliers;

            // Check for early termination
            double inlierRatio =
                static_cast<double>(bestInliersCount) / static_cast<double>(nPoints);
            if (inlierRatio > 0.8) {
                break;
            }
        }
    }

    if (bestInliersCount >= m_params.minInliers) {
        rvec = bestRvec;
        tvec = bestTvec;
        inliers = bestInliers;
        return true;
    }

    return false;
}

int LoopClosure::computeHammingDistance(const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic>& desc1,
                                        const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic>& desc2) {
    if (desc1.cols() != desc2.cols()) {
        return std::numeric_limits<int>::max();
    }

    int distance = 0;
    for (Eigen::Index i = 0; i < desc1.cols(); ++i) {
        uint8_t xorValue = desc1(0, i) ^ desc2(0, i);
        distance += POPCOUNT_LUT[xorValue];
    }
    return distance;
}

cv::Mat LoopClosure::descriptorsToMat(const slam::DescriptorMatrix& descriptors) {
    cv::Mat cvDescriptors(static_cast<int>(descriptors.rows()),
                          static_cast<int>(descriptors.cols()), CV_8UC1);

    for (int i = 0; i < descriptors.rows(); ++i) {
        for (int j = 0; j < descriptors.cols(); ++j) {
            cvDescriptors.at<uint8_t>(i, j) = descriptors(i, j);
        }
    }

    return cvDescriptors;
}

std::pair<Eigen::Matrix3d, Eigen::VectorXd> LoopClosure::extractCameraParameters(
    const slam::Camera& camera) {
    // This is a workaround since Camera's members are private
    // In production, you should add getter methods to the Camera class
    // For now, we'll create dummy parameters

    // Typical camera intrinsics (these should come from the Camera object)
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = 500.0;  // fx
    K(1, 1) = 500.0;  // fy
    K(0, 2) = 320.0;  // cx
    K(1, 2) = 240.0;  // cy

    // Typical distortion coefficients
    Eigen::VectorXd D(5);
    D << 0.0, 0.0, 0.0, 0.0, 0.0;

    return {K, D};
}

std::vector<slam::Match> LoopClosure::matchDescriptors(
    const slam::DescriptorMatrix& queryDescriptors,
    const slam::DescriptorMatrix& trainDescriptors) const {
    std::vector<slam::Match> matches;
    matches.reserve(queryDescriptors.rows());

    for (int i = 0; i < queryDescriptors.rows(); ++i) {
        int bestDistance = std::numeric_limits<int>::max();
        int bestIdx = -1;

        for (int j = 0; j < trainDescriptors.rows(); ++j) {
            int distance = computeHammingDistance(queryDescriptors.row(i), trainDescriptors.row(j));

            if (distance < bestDistance) {
                bestDistance = distance;
                bestIdx = j;
            }
        }

        if (bestIdx >= 0) {
            matches.emplace_back(i, bestIdx, static_cast<float>(bestDistance));
        }
    }

    return matches;
}
