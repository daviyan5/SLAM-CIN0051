#include <slam/frontend/feature_detector.hpp>

namespace slam {
void FeatureDetector::detect(const cv::Mat& image,
                             std::vector<KeyDescriptorPair>& keyDescriptorPairs) {
    keyDescriptorPairs.clear();

    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    std::vector<cv::KeyPoint> keypoints;
    detectFASTKeypoints(grayImage, keypoints);

    if (m_nonMaxSuppression) {
        applyNonMaxSuppression(grayImage, keypoints);
    }

    for (const auto& keypoint : keypoints) {
        cv::Mat descriptor;
        keyDescriptorPairs.emplace_back(keypoint, descriptor);
    }

    SPDLOG_DEBUG("Detected {} keypoints", keypoints.size());
}

void FeatureDetector::compute(const cv::Mat& image,
                              std::vector<KeyDescriptorPair>& keyDescriptorPairs) {
}

void FeatureDetector::detectAndCompute(const cv::Mat& image,
                                       std::vector<KeyDescriptorPair>& keyDescriptorPairs) {
    detect(image, keyDescriptorPairs);
    compute(image, keyDescriptorPairs);
}

void FeatureDetector::detectFASTKeypoints(const cv::Mat& image,
                                          std::vector<cv::KeyPoint>& keypoints) {
    keypoints.clear();
    const int border = 3;

    for (int y = border; y < image.rows - border; y++) {
        for (int x = border; x < image.cols - border; x++) {
            if (isFASTCorner(image, x, y)) {
                keypoints.emplace_back(x, y, 6.0f);
            }
        }
    }
}

bool FeatureDetector::isFASTCorner(const cv::Mat& image, int x, int y) {
    const uchar pixelIntensity = image.at<uchar>(y, x);

    // To exclude a large number of non-corners pixels, we first examine the four neighbor pixels at
    // positions 0, 8, 4 and 12 (First, we check if pixels 0 and 8 are brighter or darker than the
    // center pixel. If so, we proceed to check pixels 5 and 13). If the center pixel is a corner,
    // then at least three of the tested neighbors must all be brighter than or darker than it.
    int brighterPixels = 0;
    int darkerPixels = 0;

    for (int i = 0; i < 2; i++) {
        int neighborX = x + m_pixelOffsets[i * 8][0];
        int neighborY = y + m_pixelOffsets[i * 8][1];
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
        int neighborX = x + m_pixelOffsets[i * 8 + 4][0];
        int neighborY = y + m_pixelOffsets[i * 8 + 4][1];
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

    for (int i = 0; i < 32; i++) {
        int neighborX = x + m_pixelOffsets[i % 16][0];
        int neighborY = y + m_pixelOffsets[i % 16][1];
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
                                             std::vector<cv::KeyPoint>& keypoints) {
    if (keypoints.empty()) {
        return;
    }

    std::vector<cv::KeyPoint> suppressedKeypoints;

    for (auto& keypoint : keypoints) {
        keypoint.response = computeFASTScore(image, keypoint.pt.x, keypoint.pt.y);
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
            float distance = std::sqrt(dx * dx + dy * dy);

            if (distance < m_suppressionWindowSize) {
                suppressed[j] = true;
            }
        }
    }

    keypoints = suppressedKeypoints;
}

float FeatureDetector::computeFASTScore(const cv::Mat& image, int x, int y) {
    const uchar pixelIntensity = image.at<uchar>(y, x);
    float score = 0.0f;

    for (int i = 0; i < 16; i++) {
        int neighborX = x + m_pixelOffsets[i][0];
        int neighborY = y + m_pixelOffsets[i][1];
        const uchar neighborIntensity = image.at<uchar>(neighborY, neighborX);
        score += std::abs(neighborIntensity - pixelIntensity);
    }

    return score;
}

float FeatureDetector::computeOrientation(const cv::Mat& image, const cv::KeyPoint& keypoint) {
    int x = static_cast<int>(keypoint.pt.x);
    int y = static_cast<int>(keypoint.pt.y);

    const int radius = m_patchSize / 2;
    if (x - radius < 0 || x + radius >= image.cols || y - radius < 0 || y + radius >= image.rows) {
        return 0.0f;
    }

    float m01 = 0.0f;
    float m10 = 0.0f;

    for (int v = -radius; v <= radius; v++) {
        for (int u = -radius; u <= radius; u++) {
            if (u * u + v * v <= radius * radius) {
                uchar pixelIntensity = image.at<uchar>(y + v, x + u);
                m01 += v * pixelIntensity;
                m10 += u * pixelIntensity;
            }
        }
    }

    float angle = std::atan2(m01, m10) * 180.0f / CV_PI;
    return angle;
}
}  // namespace slam
