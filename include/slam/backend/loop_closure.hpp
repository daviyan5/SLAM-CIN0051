#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <slam/common/common.hpp>
#include <slam/frontend/feature_matcher.hpp>

#include <fbow/fbow.h>

namespace slam {

struct LoopResult {
    int matchedKeyframeId{};
    Eigen::Matrix4d relativeTransform = Eigen::Matrix4d::Identity();
};
class LoopClosure {
public:
    explicit LoopClosure(const std::string& vocabPath, const std::string& configPath,
                         const FeatureMatcher& matcher);

    void addKeyframe(int keyframeId, const DescriptorMatrix& descriptors,
                     const std::vector<Keypoint>& keypoints,
                     const std::vector<Eigen::Vector3d>& mapPoints);

    [[nodiscard]] std::optional<LoopResult> detect(const DescriptorMatrix& descriptors,
                                                   const std::vector<Keypoint>& keypoints,
                                                   const Camera& camera);

private:
    struct LoopClosureParameters {
        int minDbSize{};
        int minFramesDifference{};
        double minAbsoluteScore{};
        double relativeScoreFactor{};
        int minMatchesForPnP{};
        int minInliersForPnP{};
        int ransacMaxIterations{};
        double ransacReprojectionThreshold{};
    };

    struct KeyframeData {
        fbow::BoWVector bowVector;
        std::vector<Keypoint> keypoints;
        std::vector<Eigen::Vector3d> mapPoints;
    };

    void loadParameters(const std::filesystem::path& configPath);

    std::optional<LoopResult> verifyGeometricConsistency(
        const DescriptorMatrix& queryDescriptors, const std::vector<Keypoint>& queryKeypoints,
        int candidateId, const Camera& camera);

    static bool solvePnP(const std::vector<Eigen::Vector3d>& points3d,
                         const std::vector<Eigen::Vector2d>& points2d, Eigen::Matrix3d& rotation,
                         Eigen::Vector3d& translation);

    LoopClosureParameters m_params;
    fbow::Vocabulary m_vocabulary;
    std::reference_wrapper<const FeatureMatcher> m_matcher;

    std::map<int, KeyframeData> m_keyframeDatabase;
    std::map<int, DescriptorMatrix> m_keyframeDescriptors;
    int m_lastKeyframeId{-1};
};

}  // namespace slam
