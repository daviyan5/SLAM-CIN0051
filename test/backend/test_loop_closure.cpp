#include <filesystem>
#include <spdlog/spdlog.h>
#include <vector>
#include <string>

#include <slam/backend/loop_closure.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_project_root>" << std::endl;
        return -1;
    }
    spdlog::set_level(spdlog::level::info);

    const std::string project_root = argv[1];
    const std::string vocab_file = project_root + "/data/vocabulary/orb_mur.fbow";
    const std::string config_file = project_root + "/config/loop_closure.yaml";

    if (!std::filesystem::exists(vocab_file)) {
        SPDLOG_ERROR("Vocabulary file not found at: {}", vocab_file);
        return -1;
    }
    if (!std::filesystem::exists(config_file)) {
        SPDLOG_ERROR("Config file not found at: {}", config_file);
        return -1;
    }
    
    slam::Camera camera;
    camera.K = (cv::Mat_<double>(3, 3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);
    camera.D = cv::Mat::zeros(5, 1, CV_64F);

    try {
        slam::LoopClosure detector(vocab_file, config_file);
        auto orb = cv::ORB::create();
        
        std::vector<cv::KeyPoint> last_keypoints;
        cv::Mat last_descriptors;

        for (int i = 0; i < 10; ++i) {
            std::string path = project_root + "/data/images_test_loop2/" + std::to_string(i) + ".png";
            cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                SPDLOG_ERROR("Failed to load image: {}", path);
                return -1;
            }

            std::vector<cv::KeyPoint> kps;
            cv::Mat desc;
            orb->detectAndCompute(img, cv::Mat(), kps, desc);

            std::vector<cv::Point3f> map_points;
            for(const auto& kp : kps) {
                map_points.emplace_back(kp.pt.x, kp.pt.y, 1.0);
            }
            
            detector.addKeyframe(i, desc, kps, map_points);
            
            if (i == 9) {
                last_keypoints = kps;
                last_descriptors = desc;
            }
        }
        
        auto result = detector.detect(last_descriptors, last_keypoints, camera);

        if (result && result->matched_keyframe_id == 0) {
            SPDLOG_INFO("Loop with initial keyframe {}.", result->matched_keyframe_id);
        } else if (result) {
            SPDLOG_INFO("Loop detected with keyframe {}.", result->matched_keyframe_id);
            return -1;
        } else {
            SPDLOG_ERROR("No loop was detected or verified.");
            return -1;
        }

    } catch (const std::exception& e) {
        SPDLOG_ERROR("An error occurred: {}", e.what());
        return -1;
    }

    return 0;
}