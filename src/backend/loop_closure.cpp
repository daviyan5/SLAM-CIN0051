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

void LoopClosure::addKeyframe(int keyframe_id, const cv::Mat& descriptors, const std::vector<cv::KeyPoint>& keypoints, const std::vector<cv::Point3f>& map_points) {
    KeyframeData data;
    data.bow_vector = m_vocabulary.transform(descriptors);
    data.keypoints = keypoints;
    data.map_points = map_points;
    
    m_keyframe_database[keyframe_id] = data;
    m_keyframe_descriptors[keyframe_id] = descriptors.clone();
    m_last_keyframe_id = keyframe_id;

    SPDLOG_DEBUG("Added keyframe {} to the database.", keyframe_id);
}

struct Match { double score; int id; };

std::optional<slam::LoopResult> LoopClosure::detect(const cv::Mat& descriptors, const std::vector<cv::KeyPoint>& keypoints, const slam::Camera& camera) {
    if (m_keyframe_database.size() < 2) {
        SPDLOG_DEBUG("Not enough keyframes in the database for loop detection.");
        return std::nullopt;
    } 

    // BoW vector computation and candidate retrieval
    fbow::BoWVector current_bow_vector = m_vocabulary.transform(descriptors);
    std::vector<Match> all_matches;
    for (const auto& [id, data] : m_keyframe_database) {
        if (id == m_last_keyframe_id) continue;
        double score = fbow::BoWVector::score(current_bow_vector, data.bow_vector);
        all_matches.push_back({score, id});
    }

    if (all_matches.empty()) {
        SPDLOG_DEBUG("No candidates found in the database.");
        return std::nullopt;
    } 

    std::sort(all_matches.begin(), all_matches.end(), [](const auto& a, const auto& b) { return a.score > b.score; });

    const auto& best_candidate = all_matches[0];
    
    // Temporal consistency check
    constexpr int MIN_FRAMES_DIFFERENCE = 2;
    if (std::abs(m_last_keyframe_id - best_candidate.id) < MIN_FRAMES_DIFFERENCE) {
        SPDLOG_DEBUG("Best candidate (ID {}) is too recent. Skipping.", best_candidate.id);
        return std::nullopt;
    }
    
    bool is_significant_enough = true;
    if (all_matches.size() > 1) {
        if (best_candidate.score < 1.5 * all_matches[1].score) is_significant_enough = false;
    }

    constexpr double MIN_ABSOLUTE_SCORE = 0.005;
    if (!(best_candidate.score > MIN_ABSOLUTE_SCORE && is_significant_enough)) {
        SPDLOG_DEBUG("Best candidate (ID {}) score {} not significant enough.", best_candidate.id, best_candidate.score);
        return std::nullopt;
    }
    
    SPDLOG_INFO("BoW candidate found: ID {}. Proceeding to geometric verification.", best_candidate.id);

    // Geometric verification 
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors, m_keyframe_descriptors.at(best_candidate.id), matches);

    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    for (const auto& match : matches) {
        points_2d.push_back(keypoints[match.queryIdx].pt);
        points_3d.push_back(m_keyframe_database.at(best_candidate.id).map_points[match.trainIdx]);
    }

    constexpr int MIN_MATCHES_FOR_PNP = 20;
    if (points_2d.size() < MIN_MATCHES_FOR_PNP) {
        SPDLOG_WARN("Not enough matches ({}) for geometric verification.", points_2d.size());
        return std::nullopt;
    }

    cv::Mat rvec, tvec;
    cv::Mat inliers;
    cv::solvePnPRansac(points_3d, points_2d, camera.K, camera.D, rvec, tvec, false, 100, 4.0, 0.99, inliers);

    constexpr int MIN_INLIERS = 5; 
    if (inliers.rows >= MIN_INLIERS) {
        SPDLOG_INFO("Geometric verification succeeded: Found {} inliers.", inliers.rows);
        
        slam::LoopResult result;
        result.matched_keyframe_id = best_candidate.id;

        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) transform(i,j) = R.at<double>(i,j);
            transform(i,3) = tvec.at<double>(i);
        }
        
        result.relative_transform = transform;
        return result;
    } else {
        SPDLOG_WARN("Geometric verification FAILED: Only {} inliers found.", inliers.rows);
    }
    
    return std::nullopt;
}