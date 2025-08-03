#include <fstream>
#include <iostream>

#include <spdlog/spdlog.h>

#include <slam/preprocessing/preprocessor.hpp>

using slam::Preprocessor;

Preprocessor::Preprocessor(std::filesystem::path streamPath, const slam::Camera &camera,
                           const int frameSkip)
    : m_camera(camera), m_streamPath(std::move(streamPath)), m_frameSkip(frameSkip) {
    if (std::filesystem::is_directory(m_streamPath)) {
        m_isDirectory = true;
        prepareDirectory();
    } else if (std::filesystem::is_regular_file(m_streamPath)) {
        m_isVideo = true;
        prepareVideo();
    } else {
        throw std::runtime_error("Unsupported stream type: " + m_streamPath.string());
    }
}

void Preprocessor::prepareDirectory() {
    // Load m_files with the name of the files in the directory in lexical order
    SPDLOG_INFO("Preparing directory: {}", m_streamPath.string());
    if (!std::filesystem::exists(m_streamPath)) {
        throw std::runtime_error("Directory does not exist: " + m_streamPath.string());
    }
    if (!std::filesystem::is_directory(m_streamPath)) {
        throw std::runtime_error("Path is not a directory: " + m_streamPath.string());
    }
    for (const auto &entry : std::filesystem::directory_iterator(m_streamPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".png") {
            m_totalFrames++;
            m_files.push_back(entry.path());
        }
    }

    std::sort(m_files.begin(), m_files.end());
    SPDLOG_INFO("Sucessfully parsed directory {} with {} frames.", m_streamPath.string(),
                m_totalFrames);

    std::ifstream file(m_streamPath / "timestamps.txt");
    if (!file) {
        throw std::runtime_error("Could not open timestamps.txt in directory: " +
                                 m_streamPath.string());
    }

    std::string line;
    while (std::getline(file, line)) {
        auto decimalPos = line.find('.');
        if (decimalPos == std::string::npos) {
            SPDLOG_ERROR("Invalid Format: {}", line);
            continue;
        }

        std::string mainPart = line.substr(0, decimalPos);
        std::string nanoPart = line.substr(decimalPos + 1);

        std::tm timeStruct = {};
        std::stringstream ss(mainPart);
        ss >> std::get_time(&timeStruct, "%Y-%m-%d %H:%M:%S");

        if (ss.fail()) {
            SPDLOG_ERROR("Failed to parse time from line: {}", line);
            continue;
        }

        auto timePoint = std::chrono::system_clock::from_time_t(std::mktime(&timeStruct));

        int64_t nanoseconds = std::stoll(nanoPart);
        timePoint += std::chrono::nanoseconds(nanoseconds);

        m_timestamps.push_back(timePoint);
    }
    SPDLOG_INFO("Sucessfully read timestamps.");
    if (m_timestamps.size() != m_totalFrames) {
        throw std::runtime_error("Number of timestamps does not match number of frames.");
    }
}

void Preprocessor::prepareVideo() {
    SPDLOG_INFO("Preparing video: {}", m_streamPath.string());
    m_vc = std::make_unique<cv::VideoCapture>(m_streamPath.string());
    if (!m_vc->isOpened()) {
        throw std::runtime_error("Could not open video file: " + m_streamPath.string());
    }
    m_totalFrames = static_cast<int>(m_vc->get(cv::CAP_PROP_FRAME_COUNT));
    SPDLOG_INFO("Sucessfully opened video {} with {} frames.", m_streamPath.string(),
                m_totalFrames);
}

slam::MatrixTimePair Preprocessor::yield() {
    slam::MatrixTimePair pair;
    cv::Mat frame;
    auto timestamp = std::chrono::system_clock::now();

    if (m_frameNumber >= m_totalFrames) {
        SPDLOG_DEBUG("Reached end of stream: {} frames processed.", m_frameNumber);
        return pair;
    }

    SPDLOG_INFO("Yielding frame {} of {}.", m_frameNumber, m_totalFrames);

    if (m_isDirectory) {
        if (m_frameNumber >= m_files.size()) {
            throw std::runtime_error("Frame index out of bounds for directory stream.");
        }
        frame = cv::imread(m_files[m_frameNumber].string(), cv::IMREAD_COLOR);
        if (frame.empty()) {
            throw std::runtime_error("Failed to read image from file: " +
                                     m_files[m_frameNumber].string());
        }
        constexpr double NANOSECONDS_PER_MILLISECOND = 1.0e6;
        double timestampMs =
            static_cast<double>(m_timestamps[m_frameNumber].time_since_epoch().count()) /
            NANOSECONDS_PER_MILLISECOND;
        timestamp = std::chrono::system_clock::time_point(
            std::chrono::milliseconds(static_cast<int64_t>(timestampMs)));

    } else if (m_isVideo) {
        m_vc->set(cv::CAP_PROP_POS_FRAMES, m_frameNumber);
        if (!m_vc->read(frame)) {
            throw std::runtime_error("Failed to read frame from video.");
        }
        m_frameNumber += 1 + m_frameSkip;
        double timestampMs = m_vc->get(cv::CAP_PROP_POS_MSEC);
        timestamp = std::chrono::system_clock::time_point(
            std::chrono::milliseconds(static_cast<int64_t>(timestampMs)));

    } else {
        throw std::runtime_error("Unsupported stream type for yielding frames.");
    }
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    pair.first = m_camera.get().undistortImage(std::move(grayFrame));
    pair.second = timestamp;
    m_frameNumber += 1 + m_frameSkip;
    return pair;
}
