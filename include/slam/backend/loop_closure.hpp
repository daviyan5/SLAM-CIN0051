#pragma once

#include <fbow/fbow.h>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>

namespace slam {

/**
 * @brief A struct to hold all necessary data for a keyframe in the database.
 */
struct KeyframeData {
    fbow::BoWVector bow_vector;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point3f> map_points; // 3D points corresponding to each keypoint/descriptor
};

/**
 * @brief struct to hold the confirmed result of a loop detection.
 */
struct LoopResult {
    int matched_keyframe_id;
    Eigen::Matrix4d relative_transform; // The transform FROM the current frame TO the matched frame
};

/**
 * @brief Camera struct to hold intrinsic parameters.
 */
struct Camera {
    cv::Mat K; // Intrinsic matrix
    cv::Mat D; // Distortion coefficients
};

/**
 * @brief Class to handle loop detection 
 */
class LoopClosure {
public:
    /**
     * @brief Constructor for LoopClosure.
     * @param vocab_path Path to the pre-trained fbow vocabulary file.
     */
    explicit LoopClosure(const std::string& vocab_path);

    /**
     * @brief Adds a new keyframe's data to the database.
     * @param keyframe_id The unique ID of the keyframe.
     * @param descriptors The feature descriptors.
     * @param keypoints The 2D feature keypoints.
     * @param map_points The corresponding 3D map points.
     */
    void addKeyframe(int keyframe_id, const cv::Mat& descriptors, const std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::Point3f>& map_points);

    /**
     * @brief Detects and verifies a loop.
     * @param descriptors The descriptors of the current keyframe.
     * @param keypoints The 2D keypoints of the current keyframe.
     * @param camera The camera intrinsics.
     * @return An optional containing the confirmed LoopResult (matched ID and transform).
     */
    std::optional<LoopResult> detect(const cv::Mat& descriptors, const std::vector<cv::KeyPoint>& keypoints, const Camera& camera);

private:
    fbow::Vocabulary m_vocabulary;
    std::map<int, KeyframeData> m_keyframe_database;
    std::map<int, cv::Mat> m_keyframe_descriptors; 

    int m_last_keyframe_id{-1};
};

} // namespace slam