#include <slam/frontend/common.hpp>

namespace slam {

void triangulatePoints(const cv::Mat& K, const cv::Mat& T1, const cv::Mat& T2,
                       const std::vector<cv::Point2f>& points1,
                       const std::vector<cv::Point2f>& points2,
                       std::vector<cv::Point3d>& points_3d) {
    cv::Mat P1 = K * T1;
    cv::Mat P2 = K * T2;

    points_3d.clear();
    points_3d.reserve(points1.size());

    for (size_t i = 0; i < points1.size(); i++) {
        cv::Point2f p1 = points1[i];
        cv::Point2f p2 = points2[i];

        cv::Matx44d design;

        cv::Mat row0 = p1.x * P1.row(2) - P1.row(0);
        cv::Mat row1 = p1.y * P1.row(2) - P1.row(1);
        cv::Mat row2 = p2.x * P2.row(2) - P2.row(0);
        cv::Mat row3 = p2.y * P2.row(2) - P2.row(1);

        for (int j = 0; j < 4; j++) {
            design(0, j) = row0.at<double>(j);
            design(1, j) = row1.at<double>(j);
            design(2, j) = row2.at<double>(j);
            design(3, j) = row3.at<double>(j);
        }

        cv::Vec4d Xhomogeneous;
        cv::SVD::solveZ(design, Xhomogeneous);

        if (std::abs(Xhomogeneous[3]) > 1e-6) {
            cv::Point3d p3d(Xhomogeneous[0] / Xhomogeneous[3], Xhomogeneous[1] / Xhomogeneous[3],
                            Xhomogeneous[2] / Xhomogeneous[3]);

            points_3d.push_back(p3d);
        }
    }
}

std::vector<cv::Point3d> PoseEstimator::triangulation(
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
    std::vector<cv::Point3d> points_3d;
    slam::triangulation(K_cv, T1, T2, points1, points2, points_3d);

    return points_3d;
}

};  // namespace slam

// namespace slam