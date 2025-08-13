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

    const std::string vocab_file = std::string(argv[1]) + "/data/vocabulary/orb_mur.fbow";

    if (!std::filesystem::exists(vocab_file)) {
        SPDLOG_ERROR("Vocabulary file not found at: {}", vocab_file);
        return -1;
    }

    try {
        slam::LoopClosure detector(vocab_file);
        auto orb = cv::ORB::create();
        std::vector<cv::Mat> all_descriptors;
        
        for (int i = 0; i < 4; ++i) {
            std::string path = std::string(argv[1]) + "/data/images_test_loop/" + std::to_string(i) + ".png";
            cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {  
                SPDLOG_ERROR("Failed to load image: {}", path);
                return -1;
            }
            std::vector<cv::KeyPoint> kps;
            cv::Mat desc;
            orb->detectAndCompute(img, cv::Mat(), kps, desc);
            detector.addKeyframe(i, desc);
            all_descriptors.push_back(desc);
        }

        auto result = detector.detect(all_descriptors.back());

        if (result && *result == 0) {
            SPDLOG_INFO("Loop detected with first keyframe {}.", *result);
        } else if (result) {
            SPDLOG_INFO("Loop detected with keyframe {}.", *result);
            return -1;
        } else {
            SPDLOG_INFO("No loop was detected.");
            return -1;
        }

    } catch (const std::exception& e) {
        SPDLOG_ERROR("An error occurred: {}", e.what());
        return -1;
    }

    return 0;
}