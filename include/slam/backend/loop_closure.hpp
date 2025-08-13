#pragma once

#include <fbow/fbow.h>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

namespace slam {

/**
 * @brief Class to handle loop detection using the FBoW model.
 */
class LoopClosure {
public:
    /**
     * @brief Constructor for LoopClosure.
     * @param vocab_path Path to the pre-trained fbow vocabulary file.
     */
    explicit LoopClosure(const std::string& vocab_path);

    /**
     * @brief Adds a new keyframe's descriptors to the database.
     * @param keyframe_id The unique ID of the keyframe.
     * @param descriptors The feature descriptors for this keyframe.
     */
    void addKeyframe(int keyframe_id, const cv::Mat& descriptors);

    /**
     * @brief Detects if a loop has occurred with the given keyframe's descriptors.
     * @param descriptors The descriptors of the current keyframe.
     * @return An optional containing the ID of the matched keyframe candidate.
     */
    std::optional<int> detect(const cv::Mat& descriptors);

private:
    fbow::Vocabulary m_vocabulary;
    std::map<int, fbow::BoWVector> m_keyframe_bow_vectors;

    // To avoid detecting loops with very recent frames
    int m_last_keyframe_id{-1};
};

} 