#pragma once

#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief Recovers the relative pose (R, t) from an essential matrix and point correspondences.
 *        This is a simplified version, not using OpenCV's recoverPose.
 * @param E Essential matrix (3x3)
 * @param points1 Vector of normalized points from image 1
 * @param points2 Vector of normalized points from image 2
 * @param K Camera intrinsic matrix (3x3)
 * @param R Output rotation matrix (3x3)
 * @param t Output translation vector (3x1)
 */
void simpleRecoverPose(const cv::Mat& E,
                      const std::vector<cv::Point2f>& points1,
                      const std::vector<cv::Point2f>& points2,
                      const cv::Mat& K,
                      cv::Mat& R,
                      cv::Mat& t);