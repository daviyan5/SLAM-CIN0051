#include <spdlog/spdlog.h>
#include <algorithm> 
#include <slam/backend/loop_closure.hpp>

using slam::LoopClosure;

LoopClosure::LoopClosure(const std::string& vocab_path) {
    SPDLOG_INFO("Loading fbow vocabulary from: {}", vocab_path);
    m_vocabulary.readFromFile(vocab_path);
    
    if (m_vocabulary.size() == 0) {
        SPDLOG_ERROR("Failed to load fbow vocabulary!");
        throw std::runtime_error("Vocabulary does not exist or is empty at path: " + vocab_path);
    }
    SPDLOG_INFO("LoopClosure module initialized with fbow vocabulary.");
}

void LoopClosure::addKeyframe(int keyframe_id, const cv::Mat& descriptors) {
    fbow::BoWVector bow_vector = m_vocabulary.transform(descriptors);

    m_keyframe_bow_vectors[keyframe_id] = bow_vector;
    m_last_keyframe_id = keyframe_id;
    SPDLOG_DEBUG("Added keyframe {} to the fbow database.", keyframe_id);
}

// hold match results for sorting.
struct Match {
    double score;
    int id;
};

std::optional<int> LoopClosure::detect(const cv::Mat& descriptors) {
    constexpr int MIN_DB_SIZE = 2;
    if (m_keyframe_bow_vectors.size() < MIN_DB_SIZE) {
        SPDLOG_DEBUG("Not enough keyframes in the database for loop detection. Current size: {}",
                     m_keyframe_bow_vectors.size());
        return std::nullopt;
    }     

    fbow::BoWVector current_bow_vector = m_vocabulary.transform(descriptors);

    std::vector<Match> all_matches;
    for (const auto& [keyframe_id, past_bow_vector] : m_keyframe_bow_vectors) {
        if (keyframe_id == m_last_keyframe_id) continue;
        
        double score = fbow::BoWVector::score(current_bow_vector, past_bow_vector);
        all_matches.push_back({score, keyframe_id});
    }

    if (all_matches.empty()) {
        SPDLOG_DEBUG("No matches found in the database for the current keyframe.");
        return std::nullopt;
    }

    std::sort(all_matches.begin(), all_matches.end(), [](const Match& a, const Match& b) {
        return a.score > b.score;
    });

    SPDLOG_DEBUG("Top 5 scores for current frame (ID {}):", m_last_keyframe_id);
    for (size_t i = 0; i < std::min<size_t>(5, all_matches.size()); ++i) {
        SPDLOG_DEBUG("  - Match ID: {}, Score: {}", all_matches[i].id, all_matches[i].score);
    }

    const auto& best_candidate = all_matches[0];

    constexpr int MIN_FRAMES_DIFFERENCE = 2;
    if (std::abs(m_last_keyframe_id - best_candidate.id) < MIN_FRAMES_DIFFERENCE) {
        SPDLOG_DEBUG("Best candidate keyframe {} is too recent (last keyframe: {}). Skipping loop detection.",
                     best_candidate.id, m_last_keyframe_id);
        return std::nullopt;
    }

    bool is_significant_enough = true;
    if (all_matches.size() > 1) {
        const auto& second_best_candidate = all_matches[1];
        if (best_candidate.score < 1.5 * second_best_candidate.score) {
            is_significant_enough = false;
        }
    }

    SPDLOG_DEBUG("Loop detection scores:");
    for (const auto& match : all_matches) {
        SPDLOG_DEBUG("Keyframe {}: Score = {}", match.id, match.score);
    }

    constexpr double MIN_ABSOLUTE_SCORE = 0.005;
    if (best_candidate.score > MIN_ABSOLUTE_SCORE && is_significant_enough) {
        SPDLOG_INFO("Loop candidate found: Matched with keyframe {} with a score of {}",
                    best_candidate.id, best_candidate.score);
        return best_candidate.id;
    }

    return std::nullopt;
}