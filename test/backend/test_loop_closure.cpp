#include <spdlog/spdlog.h>

#include <slam/backend/loop_closure.hpp>

int main() {
    spdlog::set_level(spdlog::level::debug);

    try {
        slam::LoopClosure detector;

        cv::Mat dummy_descriptors_1(100, 32, CV_8U);
        cv::randu(dummy_descriptors_1, cv::Scalar::all(0), cv::Scalar::all(255));

        cv::Mat dummy_descriptors_2(120, 32, CV_8U);
        cv::randu(dummy_descriptors_2, cv::Scalar::all(0), cv::Scalar::all(255));

        detector.addKeyframe(0, dummy_descriptors_1);
        SPDLOG_INFO("Added keyframe 0 to database.");

        auto result = detector.detect(25, dummy_descriptors_2);

        if (result) {
            SPDLOG_INFO("Loop detected with keyframe {}!", result->matched_keyframe_id);
        } else {
            SPDLOG_INFO("No loop was detected. (expected behavior)");
        }

    } catch (const std::exception& e) {
        spdlog::error("An error occurred during test: {}", e.what());
        return -1;
    }

    return 0;
}