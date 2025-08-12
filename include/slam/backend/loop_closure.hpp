#pragma once

#include <map>
#include <memory>
#include <optional>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace slam {

struct LoopClosureResult {
    int matched_keyframe_id;
    Eigen::Matrix4d relative_transform; // transform from current frame to matched frame
};

/**
 * @brief Class to handle loop detection and verification.
 * It maintains a database of keyframe descriptors and checks for potential loops.
 */
class LoopClosure {
public:
    /**
     * @brief Constructor for LoopClosure.
     */
    LoopClosure();

    /**
     * @brief Adds a new keyframe's descriptors to the internal database.
     * @param keyframe_id - The unique ID of the keyframe.
     * @param descriptors - The feature descriptors for this keyframe.
     */
    void addKeyframe(int keyframe_id, const cv::Mat& descriptors);

    /**
     * @brief Detects if a loop has occurred with the given keyframe.
     * @param current_keyframe_id - The ID of the current keyframe.
     * @param current_descriptors - The descriptors of the current keyframe.
     * @return An optional LoopClosureResult. If a loop is found, the optional will contain
     * the result; otherwise, it will be empty (std::nullopt).
     */
    std::optional<LoopClosureResult> detect(int current_keyframe_id, const cv::Mat& current_descriptors);

private:
    std::map<int, cv::Mat> m_keyframe_database; // TODO: Change to DBoW3

    // TODO: Add members for feature matcher, thresholds, etc.
    // std::unique_ptr<cv::DescriptorMatcher> m_matcher;
};

} // namespace slam