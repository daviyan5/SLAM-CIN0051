#include <spdlog/spdlog.h>

#include <slam/backend/loop_closure.hpp>

using slam::LoopClosure;

LoopClosure::LoopClosure() {
    SPDLOG_INFO("LoopClosure module initialized.");
    // TODO: Initialize the feature matcher 
    // m_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
}

void LoopClosure::addKeyframe(int keyframe_id, const cv::Mat& descriptors) {
    SPDLOG_DEBUG("Adding keyframe {} to loop closure database.", keyframe_id);
    m_keyframe_database[keyframe_id] = descriptors.clone();
}

std::optional<slam::LoopClosureResult> LoopClosure::detect(int current_keyframe_id, const cv::Mat& current_descriptors) {
    SPDLOG_DEBUG("Attempting to detect loop for keyframe {}.", current_keyframe_id);

    if (m_keyframe_database.size() < 20) { 
        return std::nullopt;
    }

    SPDLOG_TRACE("No loop detected for keyframe {}.", current_keyframe_id);
    return std::nullopt;
    
}