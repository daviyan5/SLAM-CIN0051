#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>

#include <Eigen/SVD>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <spdlog/spdlog.h>

#include <slam/backend/loop_closure.hpp>

using slam::LoopClosure;

LoopClosure::LoopClosure(const std::string& vocabPath, const std::string& configPath,
                         const FeatureMatcher& matcher)
    : m_matcher(matcher) {
    loadParameters(configPath);

    spdlog::info("Loading fbow vocabulary from: {}", vocabPath);
    m_vocabulary.readFromFile(vocabPath);
    if (m_vocabulary.size() == 0) {
        throw std::runtime_error("Vocabulary is empty at path: " + vocabPath);
    }
    spdlog::info("LoopClosure module initialized.");
}

void LoopClosure::loadParameters(const std::filesystem::path& configPath) {
    SPDLOG_INFO("Loading Loop Closure parameters from: {}", configPath.string());

    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Could not open Loop Closure config file: " + configPath.string());
    }

    fs["MinDbSize"] >> m_params.minDbSize;
    if (m_params.minDbSize < 0) {
        throw std::runtime_error("'MinDbSize' must be a non-negative integer.");
    }

    fs["MinFramesDifference"] >> m_params.minFramesDifference;
    if (m_params.minFramesDifference <= 0) {
        throw std::runtime_error("'MinFramesDifference' must be a positive integer.");
    }

    fs["MinAbsoluteScore"] >> m_params.minAbsoluteScore;
    if (m_params.minAbsoluteScore < 0.0) {
        throw std::runtime_error("'MinAbsoluteScore' must be non-negative.");
    }

    fs["RelativeScoreFactor"] >> m_params.relativeScoreFactor;
    if (m_params.relativeScoreFactor < 0.0) {
        throw std::runtime_error("'RelativeScoreFactor' must be non-negative.");
    }

    fs["MinMatchesForPnP"] >> m_params.minMatchesForPnP;
    if (m_params.minMatchesForPnP <= 3) {
        throw std::runtime_error("'MinMatchesForPnP' must be greater than 3 for PnP.");
    }

    fs["MinInliersForPnP"] >> m_params.minInliersForPnP;
    if (m_params.minInliersForPnP <= 3) {
        throw std::runtime_error("'MinInliersForPnP' must be greater than 3 for PnP.");
    }
    if (m_params.minInliersForPnP > m_params.minMatchesForPnP) {
        throw std::runtime_error("'MinInliersForPnP' cannot be greater than 'MinMatchesForPnP'.");
    }

    fs["RansacMaxIterations"] >> m_params.ransacMaxIterations;
    if (m_params.ransacMaxIterations <= 0) {
        throw std::runtime_error("'RansacMaxIterations' must be a positive integer.");
    }

    fs["RansacReprojectionThreshold"] >> m_params.ransacReprojectionThreshold;
    if (m_params.ransacReprojectionThreshold <= 0.0) {
        throw std::runtime_error("'RansacReprojectionThreshold' must be a positive value.");
    }

    fs.release();

    SPDLOG_DEBUG("Loop Closure Configuration Loaded:");
    SPDLOG_DEBUG("  MinDbSize: {}", m_params.minDbSize);
    SPDLOG_DEBUG("  MinFramesDifference: {}", m_params.minFramesDifference);
    SPDLOG_DEBUG("  MinAbsoluteScore: {:.3f}", m_params.minAbsoluteScore);
    SPDLOG_DEBUG("  RelativeScoreFactor: {:.3f}", m_params.relativeScoreFactor);
    SPDLOG_DEBUG("  MinMatchesForPnP: {}", m_params.minMatchesForPnP);
    SPDLOG_DEBUG("  MinInliersForPnP: {}", m_params.minInliersForPnP);
    SPDLOG_DEBUG("  RansacMaxIterations: {}", m_params.ransacMaxIterations);
    SPDLOG_DEBUG("  RansacReprojectionThreshold: {:.3f}", m_params.ransacReprojectionThreshold);

    SPDLOG_INFO("Loop Closure parameters loaded successfully.");
}

void LoopClosure::addKeyframe(int keyframeId, const DescriptorMatrix& descriptors,
                              const std::vector<Keypoint>& keypoints,
                              const std::vector<Eigen::Vector3d>& mapPoints) {
    KeyframeData data;
    cv::Mat cvDescriptors;
    cv::eigen2cv(descriptors, cvDescriptors);
    data.bowVector = m_vocabulary.transform(cvDescriptors);
    data.keypoints = keypoints;
    data.mapPoints = mapPoints;

    m_keyframeDatabase[keyframeId] = data;
    m_keyframeDescriptors[keyframeId] = descriptors;
    m_lastKeyframeId = keyframeId;
}

std::optional<slam::LoopResult> LoopClosure::detect(const DescriptorMatrix& descriptors,
                                                    const std::vector<Keypoint>& keypoints,
                                                    const Camera& camera) {
    if (m_keyframeDatabase.size() < static_cast<size_t>(m_params.minDbSize)) {
        return std::nullopt;
    }

    cv::Mat cvDescriptors;
    cv::eigen2cv(descriptors, cvDescriptors);
    fbow::BoWVector currentBowVector = m_vocabulary.transform(cvDescriptors);

    if (currentBowVector.empty()) {
        return std::nullopt;
    }

    int bestCandidateId = -1;
    double maxScore = 0.0;
    double secondMaxScore = 0.0;

    for (const auto& [id, data] : m_keyframeDatabase) {
        if (std::abs(m_lastKeyframeId - id) < m_params.minFramesDifference) {
            continue;
        }
        double score = fbow::BoWVector::score(currentBowVector, data.bowVector);
        if (score > maxScore) {
            secondMaxScore = maxScore;
            maxScore = score;
            bestCandidateId = id;
        } else if (score > secondMaxScore) {
            secondMaxScore = score;
        }
    }

    if (bestCandidateId == -1 || maxScore < m_params.minAbsoluteScore ||
        maxScore < m_params.relativeScoreFactor * secondMaxScore) {
        return std::nullopt;
    }

    spdlog::info("BoW candidate found: ID {}. Verifying geometry...", bestCandidateId);
    return verifyGeometricConsistency(descriptors, keypoints, bestCandidateId, camera);
}

std::optional<slam::LoopResult> LoopClosure::verifyGeometricConsistency(
    const DescriptorMatrix& queryDescriptors, const std::vector<Keypoint>& queryKeypoints,
    int candidateId, const Camera& camera) {
    std::vector<Match> matches;
    m_matcher.get().match(queryDescriptors, m_keyframeDescriptors.at(candidateId), matches,
                          queryKeypoints, m_keyframeDatabase.at(candidateId).keypoints);

    if (matches.size() < static_cast<size_t>(m_params.minMatchesForPnP)) {
        return std::nullopt;
    }

    std::vector<Eigen::Vector3d> points3d;
    std::vector<Eigen::Vector2d> points2d;
    points3d.reserve(matches.size());
    points2d.reserve(matches.size());
    for (const auto& match : matches) {
        points2d.emplace_back(queryKeypoints[match.queryIdx].x, queryKeypoints[match.queryIdx].y);
        points3d.push_back(m_keyframeDatabase.at(candidateId).mapPoints[match.trainIdx]);
    }

    Eigen::Matrix3d bestRotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d bestTranslation = Eigen::Vector3d::Zero();
    int maxInliers = 0;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, static_cast<int>(matches.size()) - 1);

    for (int i = 0; i < m_params.ransacMaxIterations; ++i) {
        std::vector<Eigen::Vector3d> sample3d;
        std::vector<Eigen::Vector2d> sample2d;
        std::vector<int> sampleIndices;

        constexpr int sampleSize = 6;
        while (sampleIndices.size() < sampleSize) {
            int idx = dist(rng);
            if (std::find(sampleIndices.begin(), sampleIndices.end(), idx) == sampleIndices.end()) {
                sampleIndices.push_back(idx);
                sample3d.push_back(points3d[idx]);
                sample2d.push_back(points2d[idx]);
            }
        }

        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        if (!solvePnP(sample3d, sample2d, rotation, translation)) {
            continue;
        }

        int currentInliers = 0;
        for (size_t j = 0; j < points3d.size(); ++j) {
            Eigen::Vector3d transformedPoint = rotation * points3d[j] + translation;
            if (transformedPoint.z() <= 0) {
                continue;
            }

            Eigen::Vector3d projected =
                camera.getIntrinsicMatrix() * (transformedPoint / transformedPoint.z());
            double error = (points2d[j] - projected.head<2>()).norm();

            if (error < m_params.ransacReprojectionThreshold) {
                currentInliers++;
            }
        }

        if (currentInliers > maxInliers) {
            maxInliers = currentInliers;
            bestRotation = rotation;
            bestTranslation = translation;
        }
    }

    if (maxInliers >= m_params.minInliersForPnP) {
        spdlog::info("Geometric verification SUCCEEDED: Found {} inliers.", maxInliers);
        LoopResult result;
        result.matchedKeyframeId = candidateId;
        result.relativeTransform = Eigen::Matrix4d::Identity();
        result.relativeTransform.block<3, 3>(0, 0) = bestRotation;
        result.relativeTransform.block<3, 1>(0, 3) = bestTranslation;
        return result;
    }

    spdlog::warn("Geometric verification FAILED: Only {} inliers found.", maxInliers);
    return std::nullopt;
}

bool LoopClosure::solvePnP(const std::vector<Eigen::Vector3d>& points3d,
                           const std::vector<Eigen::Vector2d>& points2d, Eigen::Matrix3d& rotation,
                           Eigen::Vector3d& translation) {
    static constexpr int minPointsForPnP = 6;
    static constexpr int projectionMatrixCols = 12;

    size_t n = points3d.size();
    if (n < minPointsForPnP) {
        return false;
    }

    Eigen::MatrixXd A(2 * n, projectionMatrixCols);
    for (int64_t = 0; i < n; ++i) {
        const double X = points3d[i].x(), Y = points3d[i].y(), Z = points3d[i].z();
        const double u = points2d[i].x(), v = points2d[i].y();
        A.row(2 * i) << X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u;
        A.row(2 * i + 1) << 0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd p = svd.matrixV().col(projectionMatrixCols - 1);

    Eigen::Matrix<double, 3, 4> P = Eigen::Map<Eigen::Matrix<double, 3, 4>>(p.data());

    Eigen::Matrix3d R = P.block<3, 3>(0, 0);
    Eigen::Vector3d t = P.col(3);

    Eigen::JacobiSVD<Eigen::Matrix3d> svdR(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double det = (svdR.matrixU() * svdR.matrixV().transpose()).determinant();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    I(2, 2) = det;

    rotation = svdR.matrixU() * I * svdR.matrixV().transpose();
    translation = t / R.norm();

    return true;
}
