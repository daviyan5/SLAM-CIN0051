#include <cmath>
#include <random>

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
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // BRIEF deals with images at pixel level, so it is very noise-sensitive. This sensitivity can
    // be reduced by applying a blur to the image, thus increasing the stability and repeatability
    // of the descriptors.
    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 1.0);

    for (auto& keyDescriptorPair : keyDescriptorPairs) {
        cv::KeyPoint& keypoint = keyDescriptorPair.first;
        keypoint.angle = computeOrientation(blurredImage, keypoint);
        cv::Mat& descriptor = keyDescriptorPair.second;
        descriptor = computeBRIEFDescriptor(blurredImage, keypoint);
    }

    SPDLOG_DEBUG("Computed descriptors for {} keypoints", keyDescriptorPairs.size());
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

cv::Mat FeatureDetector::computeBRIEFDescriptor(const cv::Mat& image,
                                                const cv::KeyPoint& keypoint) {
    const int descriptorSize = m_numBRIEFPairs / 8;
    cv::Mat descriptor = cv::Mat::zeros(1, descriptorSize, CV_8UC1);

    int x = static_cast<int>(keypoint.pt.x);
    int y = static_cast<int>(keypoint.pt.y);

    if (x - m_patchSize / 2 < 0 || x + m_patchSize / 2 >= image.cols || y - m_patchSize / 2 < 0 ||
        y + m_patchSize / 2 >= image.rows) {
        return descriptor;
    }

    float angle = keypoint.angle * CV_PI / 180.0f;
    float cosAngle = std::cos(angle);
    float sinAngle = std::sin(angle);

    int bitIndex = 0;
    for (size_t i = 0; i < m_briefPattern.size() && bitIndex < descriptorSize * 8; i++) {
        cv::Point2i p1 = m_briefPattern[i].first;
        cv::Point2i p2 = m_briefPattern[i].second;

        int x1 = static_cast<int>(p1.x * cosAngle - p1.y * sinAngle) + x;
        int y1 = static_cast<int>(p1.x * sinAngle + p1.y * cosAngle) + y;
        int x2 = static_cast<int>(p2.x * cosAngle - p2.y * sinAngle) + x;
        int y2 = static_cast<int>(p2.x * sinAngle + p2.y * cosAngle) + y;

        if (x1 >= 0 && x1 < image.cols && y1 >= 0 && y1 < image.rows && x2 >= 0 &&
            x2 < image.cols && y2 >= 0 && y2 < image.rows) {
            uchar pixelIntensity1 = image.at<uchar>(y1, x1);
            uchar pixelIntensity2 = image.at<uchar>(y2, x2);

            if (pixelIntensity1 < pixelIntensity2) {
                int byteIndex = bitIndex / 8;
                int bitPosition = bitIndex % 8;
                descriptor.at<uchar>(0, byteIndex) |= (1 << bitPosition);
            }

            bitIndex++;
        }
    }

    return descriptor;
}

std::vector<std::pair<cv::Point2i, cv::Point2i>> FeatureDetector::generateBRIEFPattern() {
    std::vector<std::pair<cv::Point2i, cv::Point2i>> pattern;

    // Here we utilize a random distribution to generate 256 pairs of points around the center of
    // the patch. These pairs of points will be used to compute the BRIEF descriptor. The points are
    // sampled from a Gaussian distribution centered at (0, 0) with a standard deviation of 1.0. The
    // points are then scaled to fit within the patch size.
    const float scale = m_patchSize / 2.0f;

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);

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
