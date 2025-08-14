#pragma once

#include <fbow/fbow.h>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>

namespace slam {

struct KeyframeData {
    fbow::BoWVector bow_vector;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point3f> map_points; // 3D points corresponding to each keypoint/descriptor
};

struct LoopResult {
    int matched_keyframe_id;
    Eigen::Matrix4d relative_transform; // The transform FROM the current frame TO the matched frame
};

struct Camera {
    cv::Mat K; // Intrinsic matrix
    cv::Mat D; // Distortion coefficients
};

class LoopClosure {
public:
    // struct to hold all configurable parameters with their defaults
    struct Params {
        int min_db_size = 2;
        int min_frames_difference = 2;
        double min_absolute_score = 0.005;
        double relative_score_factor = 1.5;
        int min_matches_for_pnp = 20;
        int min_inliers = 5;
    };

    /**
     * @brief Constructor for LoopClosure.
     * @param vocab_path Path to the pre-trained fbow vocabulary file.
     * @param config_path Path to the YAML configuration file.
     */
    explicit LoopClosure(const std::string& vocab_path, const std::string& config_path);

    void addKeyframe(int keyframe_id, const cv::Mat& descriptors, const std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::Point3f>& map_points);
    std::optional<LoopResult> detect(const cv::Mat& descriptors, const std::vector<cv::KeyPoint>& keypoints, const Camera& camera);

private:
    void loadParameters(const std::string& config_path);

    Params m_params; 
    fbow::Vocabulary m_vocabulary;
    std::map<int, KeyframeData> m_keyframe_database;
    std::map<int, cv::Mat> m_keyframe_descriptors;
    int m_last_keyframe_id{-1};
};

} // namespace slam