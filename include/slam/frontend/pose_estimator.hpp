#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <slam/common/common.hpp>

namespace slam {

class PoseEstimator {
public:
    PoseEstimator(const slam::Camera& camera);
    void estimate(const std::vector<KeyDescriptorPair>& pairs1,
                  const std::vector<KeyDescriptorPair>& pairs2,
                  const std::vector<std::pair<int, int>>& matches, cv::Mat& R, cv::Mat& t);

private:
    const slam::Camera& m_camera;
};

}  // namespace slam
