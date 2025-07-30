#pragma once

#include <filesystem>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace slam {

using KeyDescriptorPair = std::pair<cv::KeyPoint, cv::Mat>;

class Camera {
public:
    Camera(const std::filesystem::path& config_path) {
    }

private:
    cv::Mat m_K;  // Camera intrinsic matrix
};

}  // namespace slam