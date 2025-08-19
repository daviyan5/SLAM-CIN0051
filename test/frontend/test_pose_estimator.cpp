#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#include <slam/common/common.hpp>
#include <slam/frontend/feature_detector.hpp>
#include <slam/frontend/feature_matcher.hpp>
#include <slam/frontend/pose_estimator.hpp>

void convertToCvKeypoints(const std::vector<slam::Keypoint>& slamKeypoints,
                          std::vector<cv::KeyPoint>& cvKeypoints) {
    cvKeypoints.clear();
    cvKeypoints.reserve(slamKeypoints.size());
    for (const auto& skp : slamKeypoints) {
        cvKeypoints.emplace_back(skp.x, skp.y, skp.size, skp.angle, skp.response);
    }
}

void convertToCvMatches(const std::vector<slam::Match>& slamMatches,
                        std::vector<cv::DMatch>& cvMatches) {
    cvMatches.clear();
    cvMatches.reserve(slamMatches.size());
    for (const auto& sm : slamMatches) {
        cvMatches.emplace_back(sm.queryIdx, sm.trainIdx, 0.0f);
    }
}

bool isRotationMatrix(const cv::Mat& R) {
    if (R.empty()) {
        return false;
    }
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
    return cv::norm(I, shouldBeIdentity, cv::NORM_L2) < 1e-6;
}

void visualizeMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& keypoints1,
                      const cv::Mat& img2, const std::vector<cv::KeyPoint>& keypoints2,
                      const std::vector<cv::DMatch>& matches,
                      const std::vector<cv::Point3d>& points_3d, const std::string& output_path) {
    cv::Mat out_img;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, out_img);

    if (points_3d.empty()) {
        cv::imwrite(output_path, out_img);
        SPDLOG_INFO("Saved visualization to {}", output_path);
        return;
    }

    std::vector<double> depths;
    for (const auto& p : points_3d) {
        if (p.z > 0) {
            depths.push_back(p.z);
        }
    }

    if (depths.size() < 2) {
        cv::imwrite(output_path, out_img);
        SPDLOG_INFO("Saved visualization to {}", output_path);
        return;
    }

    std::sort(depths.begin(), depths.end());
    double min_z = depths[static_cast<size_t>(depths.size() * 0.05)];
    double max_z = depths[static_cast<size_t>(depths.size() * 0.95)];

    cv::Mat lut(1, 256, CV_8UC3);
    for (int i = 0; i < 256; i++) {
        cv::Mat c(1, 1, CV_8UC1, cv::Scalar(i));
        cv::applyColorMap(c, c, cv::COLORMAP_JET);
        lut.at<cv::Vec3b>(0, i) = c.at<cv::Vec3b>(0, 0);
    }

    for (size_t i = 0; i < matches.size(); ++i) {
        if (i >= points_3d.size() || points_3d[i].z <= 0) {
            continue;
        }

        const auto& p3d = points_3d[i];
        double depth_scale = (p3d.z - min_z) / (max_z - min_z);
        int color_index = static_cast<int>(depth_scale * 255.0);
        color_index = std::max(0, std::min(255, color_index));

        cv::Vec3b color_bgr = lut.at<cv::Vec3b>(0, color_index);
        cv::Scalar color(color_bgr[0], color_bgr[1], color_bgr[2]);

        const auto& match = matches[i];
        cv::Point2f p1 = keypoints1[match.queryIdx].pt;
        cv::Point2f p2 =
            keypoints2[match.trainIdx].pt + cv::Point2f(static_cast<float>(img1.cols), 0);

        cv::circle(out_img, p1, 5, color, -1);
        cv::circle(out_img, p2, 5, color, -1);
    }

    cv::imwrite(output_path, out_img);
    SPDLOG_INFO("Saved visualization to {}", output_path);
}

int main() {
    spdlog::set_level(spdlog::level::debug);

    try {
        SPDLOG_INFO("Starting pose estimation test...");
        slam::Camera camera("./data/camera.yml");
        slam::PoseEstimator pose_estimator(camera);
        slam::FeatureDetector detector("./data/feature_detector.yml");
        slam::FeatureMatcher matcher("./data/feature_matcher.yml");
        SPDLOG_INFO("Loaded camera parameters and feature detector/matcher.");
        cv::Mat img1 = cv::imread("./data/test_images/0.png", cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread("./data/test_images/1.png", cv::IMREAD_GRAYSCALE);
        if (img1.empty() || img2.empty()) {
            SPDLOG_ERROR("Could not load test images.");
            return -1;
        }
        SPDLOG_INFO("Loaded test images: {}x{} and {}x{}", img1.cols, img1.rows, img2.cols, img2.rows);
        slam::EigenGrayMatrix eigenImg1;
        cv::cv2eigen(img1, eigenImg1);
        std::vector<slam::Keypoint> slamKps1;
        slam::DescriptorMatrix slamDesc1;
        detector.detectAndCompute(eigenImg1, slamKps1, slamDesc1);

        slam::EigenGrayMatrix eigenImg2;
        cv::cv2eigen(img2, eigenImg2);
        std::vector<slam::Keypoint> slamKps2;
        slam::DescriptorMatrix slamDesc2;
        detector.detectAndCompute(eigenImg2, slamKps2, slamDesc2);

        std::vector<slam::Match> slamMatches;
        matcher.match(slamDesc1, slamDesc2, slamMatches);

        std::vector<cv::KeyPoint> kps1, kps2;
        convertToCvKeypoints(slamKps1, kps1);
        convertToCvKeypoints(slamKps2, kps2);

        cv::Mat desc1, desc2;
        cv::eigen2cv(slamDesc1, desc1);
        cv::eigen2cv(slamDesc2, desc2);

        std::vector<cv::DMatch> all_matches;
        convertToCvMatches(slamMatches, all_matches);

        std::vector<slam::KeyDescriptorPair> pairs1, pairs2;
        for (size_t i = 0; i < kps1.size(); ++i) {
            pairs1.push_back({kps1[i], desc1.row(static_cast<int>(i))});
        }
        for (size_t i = 0; i < kps2.size(); ++i) {
            pairs2.push_back({kps2[i], desc2.row(static_cast<int>(i))});
        }

        std::vector<std::pair<int, int>> matches_for_estimation;
        for (const auto& match : all_matches) {
            matches_for_estimation.push_back({match.queryIdx, match.trainIdx});
        }
        SPDLOG_INFO("Found {} total matches.", matches_for_estimation.size());

        cv::Mat R, t;
        pose_estimator.estimate(pairs1, pairs2, matches_for_estimation, R, t);

        if (isRotationMatrix(R)) {
            SPDLOG_INFO("Pose estimation successful.");
            std::stringstream ss_R, ss_t;
            ss_R << R;
            ss_t << t;
            SPDLOG_INFO("Rotation Matrix:\n{}", ss_R.str());
            SPDLOG_INFO("Translation Vector:\n{}", ss_t.str());

            std::vector<cv::Point3d> points_3d =
                pose_estimator.triangulatePoints(kps1, kps2, all_matches, R, t);

            int points_in_front_of_camera = 0;
            for (const auto& p : points_3d) {
                if (p.z > 0) {
                    points_in_front_of_camera++;
                }
            }

            SPDLOG_INFO("Triangulated {} points.", points_3d.size());
            SPDLOG_INFO("{} points are in front of the camera.", points_in_front_of_camera);

            visualizeMatches(img1, kps1, img2, kps2, all_matches, points_3d,
                             "results/pose_estimation_result.png");

            double inlier_ratio =
                points_3d.empty()
                    ? 0
                    : static_cast<double>(points_in_front_of_camera) / points_3d.size();

            if (inlier_ratio > 0.75) {
                SPDLOG_INFO("A high percentage of triangulated points are valid.");
            } else {
                SPDLOG_WARN("Most triangulated points are behind the camera, likely a pose error.");
            }
        } else {
            SPDLOG_ERROR("Pose estimation failed or produced an invalid rotation matrix.");
            return -1;
        }

    } catch (const std::exception& e) {
        SPDLOG_ERROR("An error occurred: {}", e.what());
        return -1;
    }
    return 0;
}
