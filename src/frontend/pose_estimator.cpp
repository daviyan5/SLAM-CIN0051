#include "slam/frontend/pose_estimator.hpp"

#include <vector>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

using slam::PoseEstimator;

PoseEstimator::PoseEstimator(const slam::Camera& camera) : m_camera(camera) {
}

void PoseEstimator::estimate(const std::vector<KeyDescriptorPair>& pairs1,
                             const std::vector<KeyDescriptorPair>& pairs2,
                             const std::vector<std::pair<int, int>>& matches, cv::Mat& R,
                             cv::Mat& t) {
    if (matches.size() < 8) {
        SPDLOG_WARN("Cannot estimate pose, not enough matches ({}). Required at least 8.",
                    matches.size());
        return;
    }

    // 1. Convert the matches into vector<Point2f> as required by OpenCV's epipolar geometry
    // functions.
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (const auto& match : matches) {
        points1.push_back(pairs1[match.first].first.pt);
        points2.push_back(pairs2[match.second].first.pt);
    }

    // 2. Calculate the Essential Matrix using the 2D point correspondences.
    //    found via the epipolar constraint: x2^T * E * x1 = 0.
    //    use RANSAC to be robust against outlier matches.
    cv::Mat K_cv;
    cv::eigen2cv(m_camera.get().K(), K_cv);  // Convert Eigen K matrix to cv::Mat
    cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, K_cv, cv::RANSAC);

    if (essential_matrix.empty()) {
        SPDLOG_WARN("Essential Matrix could not be computed.");
        return;
    }

    // 3. Recover the Rotation (R) and Translation (t) from the Essential Matrix.
    //    decomposing E gives four possible solutions for R and t.
    //    `recoverPose` internally uses triangulation to check the depth of points
    //    and select the one correct solution where the points are in front of both cameras.
    cv::recoverPose(essential_matrix, points1, points2, K_cv, R, t);
}

std::vector<cv::Point3d> PoseEstimator::triangulatePoints(
    const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches, const cv::Mat& R, const cv::Mat& t) {
    cv::Mat K_cv;
    cv::eigen2cv(m_camera.get().K(), K_cv);

    // Create the projection matrices for both camera poses
    // Pose 1 is the origin [I | 0]
    cv::Mat T1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    // Pose 2 is [R | t]
    cv::Mat T2 = (cv::Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
                  R.at<double>(0, 2), t.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(1, 1),
                  R.at<double>(1, 2), t.at<double>(1, 0), R.at<double>(2, 0), R.at<double>(2, 1),
                  R.at<double>(2, 2), t.at<double>(2, 0));

    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Triangulate points to get them in 4D homogeneous coordinates
    cv::Mat pts_4d;
    cv::triangulatePoints(K_cv * T1, K_cv * T2, points1, points2, pts_4d);

    // homogeneous to 3D coordinates
    std::vector<cv::Point3d> points_3d;
    for (int i = 0; i < pts_4d.cols; i++) {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points_3d.push_back(p);
    }

    return points_3d;
}