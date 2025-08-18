#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <slam/common/common.hpp>
#include <slam/frontend/pose_estimator.hpp>
#include <sstream>

// check if a matrix is a valid rotation matrix (R' * R = I)
bool isRotationMatrix(const cv::Mat& R) {
    if (R.empty()) return false;
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
    return cv::norm(I, shouldBeIdentity, cv::NORM_L2) < 1e-6;
}

void visualizeMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& keypoints1,
                      const cv::Mat& img2, const std::vector<cv::KeyPoint>& keypoints2,
                      const std::vector<cv::DMatch>& matches,
                      const std::vector<cv::Point3d>& points_3d,
                      const std::string& output_path) {

    cv::Mat out_img;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, out_img);

    std::vector<double> depths;
    for (const auto& p : points_3d) {
        if (p.z > 0) { 
            depths.push_back(p.z);
        }
    }
    std::sort(depths.begin(), depths.end());
    
    double min_z = depths[depths.size() * 0.05];
    double max_z = depths[depths.size() * 0.95];

    cv::Mat lut(1, 256, CV_8UC3);
    for (int i = 0; i < 256; i++) {
        cv::Mat c(1, 1, CV_8UC1, cv::Scalar(i));
        cv::applyColorMap(c, c, cv::COLORMAP_JET);
        lut.at<cv::Vec3b>(0, i) = c.at<cv::Vec3b>(0, 0);
    }

    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& match = matches[i];
        if (i < points_3d.size() && points_3d[i].z > 0) {
            const auto& p3d = points_3d[i];
            
            double depth_scale = (p3d.z - min_z) / (max_z - min_z);
            int color_index = static_cast<int>(depth_scale * 255.0);
            color_index = std::max(0, std::min(255, color_index)); 
            
            cv::Vec3b color_bgr = lut.at<cv::Vec3b>(0, color_index);
            cv::Scalar color(color_bgr[0], color_bgr[1], color_bgr[2]);

            cv::Point2f p1 = keypoints1[match.queryIdx].pt;
            cv::Point2f p2 = keypoints2[match.trainIdx].pt + cv::Point2f(img1.cols, 0);

            cv::circle(out_img, p1, 5, color, -1);
            cv::circle(out_img, p2, 5, color, -1);
        }
    }
    
    cv::imwrite(output_path, out_img);
    SPDLOG_INFO("Saved visualization to {}", output_path);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_project_root>" << std::endl;
        return -1;
    }
    spdlog::set_level(spdlog::level::info);

    std::string project_root = argv[1];

    try {
        slam::Camera camera(project_root + "/data/camera.yml");
        slam::PoseEstimator pose_estimator(camera);

        cv::Mat img1 = cv::imread(project_root + "/data/test_images/0.png", cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(project_root + "/data/test_images/1.png", cv::IMREAD_GRAYSCALE);
        if (img1.empty() || img2.empty()) {
            SPDLOG_ERROR("Could not load test images.");
            return -1;
        }

        auto orb = cv::ORB::create();
        std::vector<slam::KeyDescriptorPair> pairs1, pairs2;
        cv::Mat desc1, desc2;
        std::vector<cv::KeyPoint> kps1, kps2;
        orb->detectAndCompute(img1, cv::Mat(), kps1, desc1);
        orb->detectAndCompute(img2, cv::Mat(), kps2, desc2);

        for(size_t i = 0; i < kps1.size(); ++i) pairs1.push_back({kps1[i], desc1.row(i)});
        for(size_t i = 0; i < kps2.size(); ++i) pairs2.push_back({kps2[i], desc2.row(i)});

        auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
        std::vector<cv::DMatch> all_matches;
        matcher->match(desc1, desc2, all_matches);

        std::vector<std::pair<int, int>> matches_for_estimation;
        for(const auto& match : all_matches) {
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
            
            std::vector<cv::Point3d> points_3d = pose_estimator.triangulatePoints(kps1, kps2, all_matches, R, t);
            
            int points_in_front_of_camera = 0;
            for (const auto& p : points_3d) {
                if (p.z > 0) { 
                    points_in_front_of_camera++;
                }
            }

            SPDLOG_INFO("Triangulated {} points.", points_3d.size());
            SPDLOG_INFO("{} points are in front of the camera.", points_in_front_of_camera);

            visualizeMatches(img1, kps1, img2, kps2, all_matches, points_3d, "pose_estimation_result.png");

            double inlier_ratio = points_3d.empty() ? 0 : static_cast<double>(points_in_front_of_camera) / points_3d.size();

            if (inlier_ratio > 0.75) { 
                 SPDLOG_INFO("A high percentage of triangulated points are valid.");
            } else {
                SPDLOG_ERROR("Most triangulated points are behind the camera, likely a pose error.");
                return -1;
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