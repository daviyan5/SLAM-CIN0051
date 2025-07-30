#pragma once

#include <filesystem>

#include <slam/backend/backend.hpp>
#include <slam/common/common.hpp>
#include <slam/frontend/feature_detector.hpp>
#include <slam/frontend/feature_matcher.hpp>
#include <slam/frontend/pose_estimator.hpp>
#include <slam/postprocessing/visualizer.hpp>
#include <slam/preprocessing/preprocessor.hpp>

namespace slam {

class SLAMModel {
public:
    SLAMModel(const std::filesystem::path& configPath, const std::filesystem::path& videoPath);
    void run();  // Inicia o pipeline principal do SLAM.
private:
    Camera m_camera;
    Preprocessor m_preprocessor;
    FeatureDetector m_featureDetector;
    FeatureMatcher m_featureMatcher;
    PoseEstimator m_poseEstimator;
    Map m_map;
    Backend m_backend;
    Visualizer m_visualizer;
};

};  // namespace slam
