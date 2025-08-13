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
    spdlog::set_level(spdlog::level::debug); 

    const std::string vocab_file = std::string(argv[1]) + "data/vocabulary/orb_mur.fbow";
    if (!std::filesystem::exists(vocab_file)) {
        SPDLOG_ERROR("Vocabulary file not found at: {}", vocab_file);
        return -1;
    }

    // fake camera for testing PnP
    slam::Camera camera;
    camera.K = (cv::Mat_<double>(3, 3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);
    camera.D = cv::Mat::zeros(5, 1, CV_64F); 

    try {
        slam::LoopClosure detector(vocab_file);
        auto orb = cv::ORB::create();
        
        std::vector<cv::KeyPoint> last_keypoints;
        cv::Mat last_descriptors;

        for (int i = 0; i < 4; ++i) {
            std::string path = std::string(argv[1]) + "data/images_test_loop/" + std::to_string(i) + ".png";
            cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                SPDLOG_ERROR("Failed to load image: {}", path);
                return -1;
            }

            std::vector<cv::KeyPoint> kps;
            cv::Mat desc;
            orb->detectAndCompute(img, cv::Mat(), kps, desc);

            // fake 3D points by projecting 2D points onto a plane at Z=1
            std::vector<cv::Point3f> map_points;
            for(const auto& kp : kps) {
                map_points.emplace_back(kp.pt.x, kp.pt.y, 1.0);
            }
            
            detector.addKeyframe(i, desc, kps, map_points);
            
            if (i == 3) { // last frame's data for the detect call
                last_keypoints = kps;
                last_descriptors = desc;
            }
        }
        
        // last frame to detect a loop
        auto result = detector.detect(last_descriptors, last_keypoints, camera);

        // expected result is a match with keyframe 0
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