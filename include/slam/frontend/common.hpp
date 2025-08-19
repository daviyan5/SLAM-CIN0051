#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

namespace slam {

void triangulatePoints(const cv::Mat& K, const cv::Mat& T1, const cv::Mat& T2,
                       const std::vector<cv::Point2f>& points1,
                       const std::vector<cv::Point2f>& points2,
                       std::vector<cv::Point3d>& points_3d);

void triangulation(const std::vector<cv::KeyPoint>& keypoints1,
                   const std::vector<cv::KeyPoint>& keypoints2,
                   const std::vector<cv::DMatch>& matches, const cv::Mat& R, const cv::Mat& t)

}  // namespace slam
