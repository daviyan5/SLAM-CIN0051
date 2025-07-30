#pragma once

#include <filesystem>
#include <functional>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <slam/common/common.hpp>

namespace slam {

class Preprocessor {
public:
    Preprocessor(std::filesystem::path &videoPath, const slam::Camera &camera);
    cv::Mat yield();

private:
    int m_frame{0};
    std::reference_wrapper<const slam::Camera> m_camera;  // Referência para evitar cópia
};

}  // namespace slam
