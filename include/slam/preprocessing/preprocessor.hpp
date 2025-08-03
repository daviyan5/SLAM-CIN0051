#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>

#include <Eigen/Eigen>

#include <opencv2/opencv.hpp>

#include <slam/common/common.hpp>

namespace slam {

using MatrixTimePair = std::pair<Eigen::MatrixXd, std::chrono::system_clock::time_point>;

/**
 * @brief Class to abstract the whole of preprocessing. For now, it only reads the image from a
 * stream and undistorts it.
 */
class Preprocessor {
public:
    /**
     * @brief Constructor for Preprocessor.
     * @param streamPath - Path to the stream, can be a directory or a video file.
     * @param camera - Camera object containing intrinsic parameters.
     * @param frameSkip - Number of frames to skip between each processed frame.
     */
    Preprocessor(const std::filesystem::path &streamPath, const slam::Camera &camera,
                 int frameSkip = 0);

    /**
     * @brief User-defined destructor to release resources.
     * @details Releases the video capture if it was opened.
     */
    ~Preprocessor();

    /**
     * @brief Returns the next undistorted image and its timestamp.
     * @return A pair containing the undistorted image as an Eigen matrix and the timestamp
     */
    MatrixTimePair yield();

private:
    void prepareDirectory();
    void prepareVideo();

    int m_frameNumber{0};
    int m_totalFrames{0};
    int m_frameSkip{0};
    bool m_isDirectory{false};
    bool m_isVideo{false};

    std::vector<std::chrono::system_clock::time_point> m_timestamps;
    std::vector<std::filesystem::path> m_files;

    std::shared_ptr<cv::VideoCapture> m_vc{nullptr};
    std::filesystem::path m_streamPath;
    std::reference_wrapper<const slam::Camera> m_camera;
};

}  // namespace slam
