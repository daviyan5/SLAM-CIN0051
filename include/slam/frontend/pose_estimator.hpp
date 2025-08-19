#pragma once

#include <functional>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <slam/common/common.hpp>

namespace slam {

class PoseEstimator {
public:
    explicit PoseEstimator(const slam::Camera& camera);
    void estimate(const std::vector<KeyDescriptorPair>& pairs1,
                  const std::vector<KeyDescriptorPair>& pairs2,
                  const std::vector<std::pair<int, int>>& matches, cv::Mat& R, cv::Mat& t);

    /**
     * @brief Triangulates 3D points from 2D matches and a known camera motion.
     * @param keypoints1 Keypoints from the first image.
     * @param keypoints2 Keypoints from the second image.
     * @param matches The indices of matching keypoints.
     * @param R The rotation matrix from frame 1 to 2.
     * @param t The translation vector from frame 1 to 2.
     * @return A vector of the triangulated 3D points.
     */
    std::vector<cv::Point3d> triangulatePoints(const std::vector<cv::KeyPoint>& keypoints1,
                                               const std::vector<cv::KeyPoint>& keypoints2,
                                               const std::vector<cv::DMatch>& matches,
                                               const cv::Mat& R, const cv::Mat& t);

private:
    std::reference_wrapper<const slam::Camera> m_camera;
};

}  // namespace slam
