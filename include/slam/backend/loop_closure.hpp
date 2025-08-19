#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <slam/common/common.hpp>
#include <slam/frontend/feature_detector.hpp>
#include <slam/frontend/feature_matcher.hpp>

#include <fbow/fbow.h>

namespace slam {

struct KeyframeData {
    fbow::BoWVector bowVector;
    std::vector<Keypoint> keypoints;
    std::vector<Eigen::Vector3d> mapPoints;  // 3D points corresponding to each keypoint/descriptor
};

struct LoopResult {
    int matchedKeyframeId;
    Eigen::Matrix4d relativeTransform;  // The transform FROM the current frame TO the matched frame
};

class LoopClosure {
public:
    // struct to hold all configurable parameters with their defaults
    struct Params {
        int minDbSize = 2;
        int minFramesDifference = 2;
        double minAbsoluteScore = 0.005;
        double relativeScoreFactor = 1.5;
        int minMatchesForPnp = 20;
        int minInliers = 5;
        double ransacReprojectionThreshold = 4.0;
        int ransacMaxIterations = 100;
        double ransacConfidence = 0.99;
    };

    /**
     * @brief Constructor for LoopClosure.
     * @param vocabPath Path to the pre-trained fbow vocabulary file.
     * @param configPath Path to the YAML configuration file.
     */
    explicit LoopClosure(const std::string& vocabPath, const std::string& configPath);

    /**
     * @brief Add a keyframe to the loop closure database
     * @param keyframeId Unique identifier for the keyframe
     * @param descriptors Descriptors matrix for the keyframe
     * @param keypoints Keypoints detected in the keyframe
     * @param mapPoints 3D world points corresponding to each keypoint
     */
    void addKeyframe(int keyframeId, const DescriptorMatrix& descriptors,
                     const std::vector<Keypoint>& keypoints,
                     const std::vector<Eigen::Vector3d>& mapPoints);

    /**
     * @brief Detect potential loop closure
     * @param descriptors Current frame descriptors
     * @param keypoints Current frame keypoints
     * @param camera Camera parameters
     * @return Optional LoopResult if a loop is detected
     */
    [[nodiscard]] std::optional<LoopResult> detect(const DescriptorMatrix& descriptors,
                                                   const std::vector<Keypoint>& keypoints,
                                                   const Camera& camera);

private:
    struct CandidateMatch {
        double score;
        int id;
    };

    void loadParameters(const std::string& configPath);

    /**
     * @brief Solve PnP problem using RANSAC with Eigen matrices
     * @param points3d 3D world points
     * @param points2d 2D image points
     * @param K Camera intrinsic matrix
     * @param D Camera distortion coefficients
     * @param rvec Output rotation vector (angle-axis representation)
     * @param tvec Output translation vector
     * @param inliers Output indices of inlier points
     * @return true if successful, false otherwise
     */
    [[nodiscard]] bool solvePnPRansac(const std::vector<Eigen::Vector3d>& points3d,
                                      const std::vector<Eigen::Vector2d>& points2d,
                                      const Eigen::Matrix3d& K, const Eigen::VectorXd& D,
                                      Eigen::Vector3d& rvec, Eigen::Vector3d& tvec,
                                      std::vector<int>& inliers) const;

    /**
     * @brief Convert rotation vector (angle-axis) to rotation matrix
     * @param rvec Rotation vector (angle-axis representation)
     * @return 3x3 rotation matrix
     */
    [[nodiscard]] static Eigen::Matrix3d rodrigues(const Eigen::Vector3d& rvec);

    /**
     * @brief Project 3D point to 2D image plane
     * @param point3d 3D point in camera coordinates
     * @param K Camera intrinsic matrix
     * @param D Camera distortion coefficients
     * @return 2D point in image plane
     */
    [[nodiscard]] static Eigen::Vector2d projectPoint(const Eigen::Vector3d& point3d,
                                                      const Eigen::Matrix3d& K,
                                                      const Eigen::VectorXd& D);

    /**
     * @brief Compute Hamming distance between two binary descriptors
     * @param desc1 First descriptor
     * @param desc2 Second descriptor
     * @return Hamming distance
     */
    [[nodiscard]] static int computeHammingDistance(
        const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic>& desc1,
        const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic>& desc2);

    /**
     * @brief Convert DescriptorMatrix to OpenCV Mat for fbow
     * @param descriptors Input descriptor matrix
     * @return OpenCV Mat containing the descriptors
     */
    [[nodiscard]] static cv::Mat descriptorsToMat(const DescriptorMatrix& descriptors);

    /**
     * @brief Extract camera intrinsics from Camera object
     * @param camera Camera object
     * @return Pair of intrinsic matrix K and distortion coefficients D
     */
    [[nodiscard]] static std::pair<Eigen::Matrix3d, Eigen::VectorXd> extractCameraParameters(
        const Camera& camera);

    /**
     * @brief Match descriptors between current frame and keyframe
     * @param queryDescriptors Current frame descriptors
     * @param trainDescriptors Keyframe descriptors
     * @return Vector of matches
     */
    [[nodiscard]] std::vector<Match> matchDescriptors(
        const DescriptorMatrix& queryDescriptors, const DescriptorMatrix& trainDescriptors) const;

    Params m_params;
    fbow::Vocabulary m_vocabulary;
    std::map<int, KeyframeData> m_keyframeDatabase;
    std::map<int, DescriptorMatrix> m_keyframeDescriptors;
    int m_lastKeyframeId{-1};
};

}  // namespace slam
