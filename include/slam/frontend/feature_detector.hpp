#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <slam/common/common.hpp>

namespace slam {

class FeatureDetector {
public:
    FeatureDetector(const std::filesystem::path& configPath);
    void detectAndCompute(const cv::Mat& image,                                 // in
                          std::vector<KeyDescriptorPair>& keyDescriptorPairs);  // out
};

}  // namespace slam
