#include <cmath>
#include <random>

#include <slam/frontend/feature_detector.hpp>

namespace slam {
void FeatureDetector::detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) {
    keypoints.clear();

    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    detectFASTKeypoints(grayImage, keypoints);

    if (m_nonMaxSuppression) {
        applyNonMaxSuppression(grayImage, keypoints);
    }

    SPDLOG_DEBUG("Detected {} keypoints", keypoints.size());
}

void FeatureDetector::compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                              cv::Mat& descriptors) {
    if (keypoints.empty()) {
        return;
    }

    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    descriptors.create(static_cast<int>(keypoints.size()), m_numBRIEFPairs / constants::BRIEF_PAIRS,
                       CV_8UC1);

    // BRIEF deals with images at pixel level, so it is very noise-sensitive. This sensitivity can
    // be reduced by applying a blur to the image, thus increasing the stability and repeatability
    // of the descriptors.
    cv::Mat blurredImage;
    cv::Size kernelSize(constants::BLUR_KERNEL_SIZE, constants::BLUR_KERNEL_SIZE);

    cv::GaussianBlur(grayImage, blurredImage, kernelSize, 1.0);

    for (auto& keypoint : keypoints) {
        float angle = computeOrientation(blurredImage, keypoint);
        keypoint.angle = angle;

        cv::Mat descriptor = computeBRIEFDescriptor(blurredImage, keypoint);

        int index = static_cast<int>(&keypoint - keypoints.data());
        descriptor.copyTo(descriptors.row(index));
    }

    SPDLOG_DEBUG("Computed descriptors for {} keypoints", keypoints.size());
}

void FeatureDetector::detectAndCompute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                                       cv::Mat& descriptors) {
    detect(image, keypoints);
    compute(image, keypoints, descriptors);
}

void FeatureDetector::detectFASTKeypoints(const cv::Mat& image,
                                          std::vector<cv::KeyPoint>& keypoints) {
    keypoints.clear();
    const int border = 3;

    for (int row = border; row < image.rows - border; row++) {
        for (int col = border; col < image.cols - border; col++) {
            if (isFASTCorner(image, col, row)) {
                keypoints.emplace_back(col, row, constants::DEFAULT_KEYPOINT_SIZE);
            }
        }
    }
}

[[nodiscard]] bool FeatureDetector::isFASTCorner(const cv::Mat& image, int x, int y) const {
    const uchar pixelIntensity = image.at<uchar>(y, x);

    // To exclude a large number of non-corners pixels, we first examine the four neighbor pixels at
    // positions 0, 8, 4 and 12 (First, we check if pixels 0 and 8 are brighter or darker than the
    // center pixel. If so, we proceed to check pixels 5 and 13). If the center pixel is a corner,
    // then at least three of the tested neighbors must all be brighter than or darker than it.
    int brighterPixels = 0;
    int darkerPixels = 0;

    for (int i = 0; i < 2; i++) {
        const int32_t OFFSET_INDEX{i * constants::CARDINAL_DIRECTION_STEP};
        int neighborX = x + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(0);
        int neighborY = y + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(1);
        const uchar neighborIntensity = image.at<uchar>(neighborY, neighborX);

        if (neighborIntensity > pixelIntensity + m_intensityThreshold) {
            brighterPixels++;
        } else if (neighborIntensity < pixelIntensity - m_intensityThreshold) {
            darkerPixels++;
        }
    }

    if (brighterPixels == 0 && darkerPixels == 0) {
        return false;
    }

    for (int i = 0; i < 2; i++) {
        const int32_t OFFSET_INDEX{(i * constants::CARDINAL_DIRECTION_STEP) + 4};
        int neighborX = x + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(0);
        int neighborY = y + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(1);
        const uchar neighborIntensity = image.at<uchar>(neighborY, neighborX);

        if (neighborIntensity > pixelIntensity + m_intensityThreshold) {
            brighterPixels++;
        } else if (neighborIntensity < pixelIntensity - m_intensityThreshold) {
            darkerPixels++;
        }
    }

    if (brighterPixels < 3 && darkerPixels < 3) {
        return false;
    }

    // If the pixel passes the test above, we perform a full segment test by examining all pixels in
    // the circle. We iterate 32 times to check all 16 pixels in the circle twice, this helps handle
    // wraparounds.
    brighterPixels = 0;
    darkerPixels = 0;

    for (int i = 0; i < constants::FULL_CIRCLE_TEST_COUNT; i++) {
        const int32_t OFFSET_INDEX{i % constants::CIRCLE_PERIMETER};
        int neighborX = x + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(0);
        int neighborY = y + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(1);
        const uchar neighborIntensity = image.at<uchar>(neighborY, neighborX);

        if (neighborIntensity > pixelIntensity + m_intensityThreshold) {
            brighterPixels++;
            darkerPixels = 0;
        } else if (neighborIntensity < pixelIntensity - m_intensityThreshold) {
            darkerPixels++;
            brighterPixels = 0;
        } else {
            brighterPixels = 0;
            darkerPixels = 0;
        }

        if (brighterPixels >= m_contiguousPixelsThreshold ||
            darkerPixels >= m_contiguousPixelsThreshold) {
            return true;
        }
    }

    return false;
}

void FeatureDetector::applyNonMaxSuppression(const cv::Mat& image,
                                             std::vector<cv::KeyPoint>& keypoints) const {
    if (keypoints.empty()) {
        return;
    }

    std::vector<cv::KeyPoint> suppressedKeypoints;

    for (auto& keypoint : keypoints) {
        keypoint.response = computeFASTScore(image, static_cast<int>(keypoint.pt.x),
                                             static_cast<int>(keypoint.pt.y));
    }

    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });

    std::vector<bool> suppressed(keypoints.size(), false);

    for (size_t i = 0; i < keypoints.size(); i++) {
        if (suppressed[i]) {
            continue;
        }

        suppressedKeypoints.push_back(keypoints[i]);

        for (size_t j = i + 1; j < keypoints.size(); j++) {
            if (suppressed[j]) {
                continue;
            }

            float dx = keypoints[i].pt.x - keypoints[j].pt.x;
            float dy = keypoints[i].pt.y - keypoints[j].pt.y;
            float distance = std::sqrt((dx * dx) + (dy * dy));

            if (distance < static_cast<float>(m_suppressionWindowSize)) {
                suppressed[j] = true;
            }
        }
    }

    keypoints = suppressedKeypoints;
}

[[nodiscard]] float FeatureDetector::computeFASTScore(const cv::Mat& image, int x, int y) {
    const uchar pixelIntensity = image.at<uchar>(y, x);
    float score = 0.0F;

    for (int i = 0; i < constants::CIRCLE_PERIMETER; i++) {
        int neighborX = x + M_PIXEL_OFFSETS.at(i).at(0);
        int neighborY = y + M_PIXEL_OFFSETS.at(i).at(1);
        const uchar neighborIntensity = image.at<uchar>(neighborY, neighborX);
        score += static_cast<float>(std::abs(neighborIntensity - pixelIntensity));
    }

    return score;
}

[[nodiscard]] float FeatureDetector::computeOrientation(const cv::Mat& image,
                                                        const cv::KeyPoint& keypoint) const {
    int x = static_cast<int>(keypoint.pt.x);
    int y = static_cast<int>(keypoint.pt.y);

    const int radius = m_patchSize / 2;
    if (x - radius < 0 || x + radius >= image.cols || y - radius < 0 || y + radius >= image.rows) {
        return 0.0F;
    }

    float m01 = 0.0F;
    float m10 = 0.0F;

    for (int v = -radius; v <= radius; v++) {
        for (int u = -radius; u <= radius; u++) {
            if (u * u + v * v <= radius * radius) {
                uchar pixelIntensity = image.at<uchar>(y + v, x + u);
                m01 += static_cast<float>(v) * static_cast<float>(pixelIntensity);
                m10 += static_cast<float>(u) * static_cast<float>(pixelIntensity);
            }
        }
    }

    auto angle = static_cast<float>(std::atan2(m01, m10) * constants::RADIANS_TO_DEGREES);
    return angle;
}

[[nodiscard]] cv::Mat FeatureDetector::computeBRIEFDescriptor(const cv::Mat& image,
                                                              const cv::KeyPoint& keypoint) const {
    const int descriptorSize = m_numBRIEFPairs / constants::BRIEF_PAIRS;
    cv::Mat descriptor = cv::Mat::zeros(1, descriptorSize, CV_8UC1);

    int x = static_cast<int>(keypoint.pt.x);
    int y = static_cast<int>(keypoint.pt.y);

    if (x - m_patchSize / 2 < 0 || x + m_patchSize / 2 >= image.cols || y - m_patchSize / 2 < 0 ||
        y + m_patchSize / 2 >= image.rows) {
        return descriptor;
    }

    float angle = keypoint.angle * constants::DEGREES_TO_RADIANS;
    float cosAngle = std::cos(angle);
    float sinAngle = std::sin(angle);

    int bitIndex = 0;
    for (size_t i = 0;
         i < m_briefPattern.size() && bitIndex < descriptorSize * constants::BRIEF_PAIRS; i++) {
        cv::Point2i p1 = m_briefPattern[i].first;
        cv::Point2i p2 = m_briefPattern[i].second;

        const float p1X{static_cast<float>(p1.x)};
        const float p1Y{static_cast<float>(p1.y)};
        const float p2X{static_cast<float>(p2.x)};
        const float p2Y{static_cast<float>(p2.y)};

        int x1 = static_cast<int>(p1X * cosAngle - p1Y * sinAngle) + x;
        int y1 = static_cast<int>(p1X * sinAngle + p1Y * cosAngle) + y;
        int x2 = static_cast<int>(p2X * cosAngle - p2Y * sinAngle) + x;
        int y2 = static_cast<int>(p2X * sinAngle + p2Y * cosAngle) + y;

        if (x1 >= 0 && x1 < image.cols && y1 >= 0 && y1 < image.rows && x2 >= 0 &&
            x2 < image.cols && y2 >= 0 && y2 < image.rows) {
            uchar pixelIntensity1 = image.at<uchar>(y1, x1);
            uchar pixelIntensity2 = image.at<uchar>(y2, x2);

            if (pixelIntensity1 < pixelIntensity2) {
                int byteIndex = bitIndex / constants::BRIEF_PAIRS;
                int bitPosition = bitIndex % constants::BRIEF_PAIRS;
                descriptor.at<uchar>(0, byteIndex) |= (1 << bitPosition);
            }

            bitIndex++;
        }
    }

    return descriptor;
}

[[nodiscard]] std::vector<std::pair<cv::Point2i, cv::Point2i>>
FeatureDetector::generateBRIEFPattern() const {
    std::vector<std::pair<cv::Point2i, cv::Point2i>> pattern;

    // Here we utilize a random distribution to generate 256 pairs of points around the center of
    // the patch. These pairs of points will be used to compute the BRIEF descriptor. The points are
    // sampled from a Gaussian distribution centered at (0, 0) with a standard deviation of 1.0. The
    // points are then scaled to fit within the patch size.
    const float scale = static_cast<float>(m_patchSize) / 2.0F;

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0F, 1.0F);

    for (int i = 0; i < m_numBRIEFPairs; i++) {
        float x1 = distribution(generator) * scale;
        float y1 = distribution(generator) * scale;
        float x2 = distribution(generator) * scale;
        float y2 = distribution(generator) * scale;

        if (std::abs(x1) < scale && std::abs(y1) < scale && std::abs(x2) < scale &&
            std::abs(y2) < scale) {
            pattern.emplace_back(cv::Point2i(static_cast<int>(x1), static_cast<int>(y1)),
                                 cv::Point2i(static_cast<int>(x2), static_cast<int>(y2)));
        }
    }

    return pattern;
}

}  // namespace slam
