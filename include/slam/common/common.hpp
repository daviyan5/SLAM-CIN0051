#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>

#include <spdlog/spdlog.h>

namespace slam {
// This helper function is self-contained and can be defined first.
constexpr uint8_t popcount(int n) {
    uint8_t count = 0;
    while (n > 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}

namespace constants {
constexpr double COLOR_RANGE = 255.0;
constexpr size_t POSSIBLE_VALUES = 256;
static const auto POPCOUNT_TABLE = []() {
    std::array<uint8_t, POSSIBLE_VALUES> table{};
    for (int i = 0; i < POSSIBLE_VALUES; ++i) {
        table.at(i) = popcount(i);
    }
    return table;
}();

}  // namespace constants

template <typename Derived1, typename Derived2>
static int calculateHammingDistance(const Eigen::MatrixBase<Derived1>& d1,
                                    const Eigen::MatrixBase<Derived2>& d2) {
    int dist = 0;
    for (Eigen::Index k = 0; k < d1.size(); ++k) {
        // Assuming the descriptor elements are 8-bit, the XOR result will be in [0, 255]
        const uint32_t index = d1(k) ^ d2(k);
        dist += constants::POPCOUNT_TABLE.at(index);
    }
    return dist;
}

/**
 * @brief A pair of keypoint and its descriptor.
 * This should be used to store keypoints and their corresponding descriptors in a single structure.
 */
using KeyDescriptorPair = std::pair<cv::KeyPoint, cv::Mat>;

/**
 * @brief A Eigen matrix for grayscale images.
 */
using EigenGrayMatrix =
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Class representing the Camera.
 */
class Camera {
public:
    Camera() = default;

    /**
     * @brief Constructor for a single camera.
     * @param configPath - Path to the OpenCV-style YAML file.
     * @param cameraIndex - The index of the camera to load (e.g., 1 for K1/D1, 2 for K2/D2).
     */
    explicit Camera(const std::filesystem::path& configPath, int cameraIndex = 0)
        : m_cameraIndex(cameraIndex) {
        cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Could not open calibration file: " + configPath.string());
        }

        std::string kKey = "K" + std::to_string(m_cameraIndex);
        std::string dKey = "D" + std::to_string(m_cameraIndex);

        if (fs[kKey].empty() || fs[dKey].empty()) {
            throw std::runtime_error("Could not find keys " + kKey + " or " + dKey + " in file.");
        }

        cv::Mat cvK, cvD;
        cv::Size cvImageSize;
        fs[kKey] >> cvK;
        fs[dKey] >> cvD;
        fs["ImageSize"] >> cvImageSize;
        fs.release();

        cv::cv2eigen(cvK, m_K);
        cv::cv2eigen(cvD, m_D);
        m_imageSize = Eigen::Vector2i(cvImageSize.width, cvImageSize.height);

        int rows = m_imageSize(1);
        int cols = m_imageSize(0);

        m_fx = m_K(0, 0);
        m_fy = m_K(1, 1);
        m_cx = m_K(0, 2);
        m_cy = m_K(1, 2);

        m_k1 = m_D.size() > 0 ? m_D(0) : 0.0;
        m_k2 = m_D.size() > 1 ? m_D(1) : 0.0;
        m_p1 = m_D.size() > 2 ? m_D(2) : 0.0;
        m_p2 = m_D.size() > 3 ? m_D(3) : 0.0;
        m_k3 = m_D.size() > 4 ? m_D(4) : 0.0;

        SPDLOG_DEBUG("Camera initialized with image size [{}, {}]", rows, cols);
        SPDLOG_DEBUG("Camera intrinsics: fx={}, fy={}, cx={}, cy={}", m_fx, m_fy, m_cx, m_cy);
        SPDLOG_DEBUG("Camera distortion coefficients: k1={}, k2={}, p1={}, p2={}, k3={}", m_k1,
                     m_k2, m_p1, m_p2, m_k3);
    }

    /**
     * @brief Undistorts the CV Image into a Eigen matrix
     * @param rawImage - An rvalue reference to the image to be undistorted. The function takes
     * ownership of the data. The image should be in grayscale format.
     * @return The undistorted image as Eigen matrix
     */
    Eigen::MatrixXd undistortImage(cv::Mat&& rawImage) const {
        cv::Mat image{std::move(rawImage)};

        if (image.empty()) {
            throw std::runtime_error("Input image is empty.");
        }
        if (image.rows != m_imageSize(1) || image.cols != m_imageSize(0)) {
            throw std::runtime_error("Input image size does not match camera image size.");
        }

        Eigen::Map<EigenGrayMatrix> mappedImage(image.data, image.rows, image.cols);
        Eigen::MatrixXd eigenImage = mappedImage.cast<double>() / constants::COLOR_RANGE;

        int rows = m_imageSize(1);
        int cols = m_imageSize(0);

        auto uGrid = Eigen::RowVectorXd::LinSpaced(cols, 0, cols - 1).replicate(rows, 1);
        auto vGrid = Eigen::VectorXd::LinSpaced(rows, 0, rows - 1).replicate(1, cols);

        Eigen::ArrayXXd x = (uGrid.array() - m_cx) / m_fx;
        Eigen::ArrayXXd y = (vGrid.array() - m_cy) / m_fy;

        Eigen::ArrayXXd r = (x.square() + y.square()).sqrt();

        Eigen::ArrayXXd xDist = x * (1 + m_k1 * r.square() + m_k2 * r.pow(4)) + 2 * m_p1 * x * y +
                                m_p2 * (r.square() + 2 * x.square());
        Eigen::ArrayXXd yDist = y * (1 + m_k1 * r.square() + m_k2 * r.pow(4)) + 2 * m_p2 * x * y +
                                m_p1 * (r.square() + 2 * y.square());

        Eigen::ArrayXXd uDist = m_fx * xDist + m_cx;
        Eigen::ArrayXXd vDist = m_fy * yDist + m_cy;

        Eigen::MatrixXd undistortedImage =
            Eigen::MatrixXd::NullaryExpr(rows, cols, [&](Eigen::Index i, Eigen::Index j) -> double {
                int uDistVal = static_cast<int>(std::round(uDist(i, j)));
                int vDistVal = static_cast<int>(std::round(vDist(i, j)));

                double retVal = 0.0;
                if (uDistVal >= 0 && vDistVal >= 0 && uDistVal < eigenImage.cols() &&
                    vDistVal < eigenImage.rows()) {
                    retVal = eigenImage(vDistVal, uDistVal);
                }
                return retVal;
            });

        return undistortedImage;
    }

    [[nodiscard]] const Eigen::Matrix3d& getIntrinsicMatrix() const {
        return m_K;
    }
    [[nodiscard]] const Eigen::VectorXd& getDistortionCoefficients() const {
        return m_D;
    }

private:
    int m_cameraIndex{};
    double m_fx{}, m_fy{}, m_cx{}, m_cy{};
    double m_k1{}, m_k2{}, m_p1{}, m_p2{}, m_k3{};

    Eigen::Matrix3d m_K;
    Eigen::VectorXd m_D;
    Eigen::Vector2i m_imageSize;
};

/**
 * @brief Triangulates 3D points from two camera projection matrices and corresponding 2D points.
 *
 * @param P1 The projection matrix of the first camera.
 * @param P2 The projection matrix of the second camera.
 * @param points1 The 2D points in the first image.
 * @param points2 The 2D points in the second image.
 * @param points_4d Output 4D homogeneous coordinates of the triangulated points.
 */
inline void triangulate(const cv::Mat& P1, const cv::Mat& P2,
                        const std::vector<cv::Point2f>& points1,
                        const std::vector<cv::Point2f>& points2, cv::Mat& points_4d) {
    points_4d = cv::Mat(4, points1.size(), CV_32F);

    for (size_t i = 0; i < points1.size(); i++) {
        cv::Mat A(4, 4, CV_64F);

        A.row(0) = points1[i].x * P1.row(2) - P1.row(0);
        A.row(1) = points1[i].y * P1.row(2) - P1.row(1);
        A.row(2) = points2[i].x * P2.row(2) - P2.row(0);
        A.row(3) = points2[i].y * P2.row(2) - P2.row(1);

        cv::Mat u, w, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        cv::Mat x_homogeneous = vt.row(3).t();

        x_homogeneous.copyTo(points_4d.col(static_cast<int>(i)));
    }
}

}  // namespace slam
