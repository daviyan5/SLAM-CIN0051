#include <cmath>
#include <random>

#include <slam/frontend/feature_detector.hpp>

using slam::FeatureDetector;

void FeatureDetector::detect(const slam::EigenGrayMatrix& image, std::vector<Keypoint>& keypoints) {
    keypoints.clear();

    detectFASTKeypoints(image, keypoints);

    if (m_nonMaxSuppression) {
        applyNonMaxSuppression(image, keypoints);
    }

    SPDLOG_DEBUG("Detected {} keypoints", keypoints.size());
}

void FeatureDetector::compute(const slam::EigenGrayMatrix& image, std::vector<Keypoint>& keypoints,
                              DescriptorMatrix& descriptors) {
    if (keypoints.empty()) {
        descriptors = DescriptorMatrix(0, 0);
        return;
    }

    descriptors.resize(static_cast<int64_t>(keypoints.size()),
                       m_numBRIEFPairs / slam::constants::BRIEF_PAIRS);

    // BRIEF deals with images at pixel level, so it is very noise-sensitive. This sensitivity can
    // be reduced by applying a blur to the image, thus increasing the stability and repeatability
    // of the descriptors.
    slam::EigenGrayMatrix blurredImage =
        gaussianBlur(image, slam::constants::BLUR_KERNEL_SIZE, 1.0);

    for (int64_t i = 0; i < keypoints.size(); ++i) {
        float angle = computeOrientation(blurredImage, keypoints[i]);
        keypoints[i].angle = angle;

        Eigen::Matrix<uint8_t, 1, Eigen::Dynamic> descriptor =
            computeBRIEFDescriptor(blurredImage, keypoints[i]);

        descriptors.row(i) = descriptor;
    }

    SPDLOG_DEBUG("Computed descriptors for {} keypoints", keypoints.size());
}

void FeatureDetector::detectAndCompute(const slam::EigenGrayMatrix& image,
                                       std::vector<Keypoint>& keypoints,
                                       DescriptorMatrix& descriptors) {
    detect(image, keypoints);
    compute(image, keypoints, descriptors);
}

void FeatureDetector::detectFASTKeypoints(const slam::EigenGrayMatrix& image,
                                          std::vector<Keypoint>& keypoints) {
    keypoints.clear();
    const int border = 3;

    for (int row = border; row < image.rows() - border; row++) {
        for (int col = border; col < image.cols() - border; col++) {
            if (isFASTCorner(image, col, row)) {
                keypoints.emplace_back(static_cast<float>(col), static_cast<float>(row));
            }
        }
    }
}

[[nodiscard]] bool FeatureDetector::isFASTCorner(const slam::EigenGrayMatrix& image, int x,
                                                 int y) const {
    const uint8_t pixelIntensity = image(y, x);

    // To exclude a large number of non-corners pixels, we first examine the four neighbor pixels at
    // positions 0, 8, 4 and 12 (First, we check if pixels 0 and 8 are brighter or darker than the
    // center pixel. If so, we proceed to check pixels 5 and 13). If the center pixel is a corner,
    // then at least three of the tested neighbors must all be brighter than or darker than it.
    int brighterPixels = 0;
    int darkerPixels = 0;

    for (int i = 0; i < 2; i++) {
        const int32_t OFFSET_INDEX{i * slam::constants::CARDINAL_DIRECTION_STEP};
        int neighborX = x + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(0);
        int neighborY = y + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(1);
        const uint8_t neighborIntensity = image(neighborY, neighborX);

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
        const int32_t OFFSET_INDEX{(i * slam::constants::CARDINAL_DIRECTION_STEP) + 4};
        int neighborX = x + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(0);
        int neighborY = y + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(1);
        const uint8_t neighborIntensity = image(neighborY, neighborX);

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

    for (int i = 0; i < slam::constants::FULL_CIRCLE_TEST_COUNT; i++) {
        const int32_t OFFSET_INDEX{i % slam::constants::CIRCLE_PERIMETER};
        int neighborX = x + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(0);
        int neighborY = y + M_PIXEL_OFFSETS.at(OFFSET_INDEX).at(1);
        const uint8_t neighborIntensity = image(neighborY, neighborX);

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

void FeatureDetector::applyNonMaxSuppression(const slam::EigenGrayMatrix& image,
                                             std::vector<Keypoint>& keypoints) const {
    if (keypoints.empty()) {
        return;
    }

    std::vector<Keypoint> suppressedKeypoints;

    for (auto& keypoint : keypoints) {
        keypoint.response =
            computeFASTScore(image, static_cast<int>(keypoint.x), static_cast<int>(keypoint.y));
    }

    std::sort(keypoints.begin(), keypoints.end(),
              [](const Keypoint& a, const Keypoint& b) { return a.response > b.response; });

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

            float dx = keypoints[i].x - keypoints[j].x;
            float dy = keypoints[i].y - keypoints[j].y;
            float distance = std::sqrt((dx * dx) + (dy * dy));

            if (distance < static_cast<float>(m_suppressionWindowSize)) {
                suppressed[j] = true;
            }
        }
    }

    keypoints = suppressedKeypoints;
}

[[nodiscard]] float FeatureDetector::computeFASTScore(const slam::EigenGrayMatrix& image, int x,
                                                      int y) {
    const uint8_t pixelIntensity = image(y, x);
    float score = 0.0F;

    for (int i = 0; i < slam::constants::CIRCLE_PERIMETER; i++) {
        int neighborX = x + M_PIXEL_OFFSETS.at(i).at(0);
        int neighborY = y + M_PIXEL_OFFSETS.at(i).at(1);
        const uint8_t neighborIntensity = image(neighborY, neighborX);
        score += static_cast<float>(std::abs(neighborIntensity - pixelIntensity));
    }

    return score;
}

[[nodiscard]] float FeatureDetector::computeOrientation(const slam::EigenGrayMatrix& image,
                                                        const Keypoint& keypoint) const {
    int x = static_cast<int>(keypoint.x);
    int y = static_cast<int>(keypoint.y);

    const int radius = m_patchSize / 2;
    if (x - radius < 0 || x + radius >= image.cols() || y - radius < 0 ||
        y + radius >= image.rows()) {
        return 0.0F;
    }

    float m01 = 0.0F;
    float m10 = 0.0F;

    for (int v = -radius; v <= radius; v++) {
        for (int u = -radius; u <= radius; u++) {
            if (u * u + v * v <= radius * radius) {
                uint8_t pixelIntensity = image(y + v, x + u);
                m01 += static_cast<float>(v) * static_cast<float>(pixelIntensity);
                m10 += static_cast<float>(u) * static_cast<float>(pixelIntensity);
            }
        }
    }

    auto angle = static_cast<float>(std::atan2(m01, m10) * slam::constants::RADIANS_TO_DEGREES);
    return angle;
}

[[nodiscard]] Eigen::Matrix<uint8_t, 1, Eigen::Dynamic> FeatureDetector::computeBRIEFDescriptor(
    const slam::EigenGrayMatrix& image, const Keypoint& keypoint) const {
    const int descriptorSize = m_numBRIEFPairs / slam::constants::BRIEF_PAIRS;
    Eigen::Matrix<uint8_t, 1, Eigen::Dynamic> descriptor =
        Eigen::Matrix<uint8_t, 1, Eigen::Dynamic>::Zero(1, descriptorSize);

    int x = static_cast<int>(keypoint.x);
    int y = static_cast<int>(keypoint.y);

    if (x - m_patchSize / 2 < 0 || x + m_patchSize / 2 >= image.cols() || y - m_patchSize / 2 < 0 ||
        y + m_patchSize / 2 >= image.rows()) {
        return descriptor;
    }

    float angle = keypoint.angle * slam::constants::DEGREES_TO_RADIANS;
    float cosAngle = std::cos(angle);
    float sinAngle = std::sin(angle);

    int bitIndex = 0;
    for (size_t i = 0;
         i < m_briefPattern.size() && bitIndex < descriptorSize * slam::constants::BRIEF_PAIRS;
         i++) {
        Eigen::Vector2i p1 = m_briefPattern[i].first;
        Eigen::Vector2i p2 = m_briefPattern[i].second;

        const auto p1X = static_cast<float>(p1.x());
        const auto p1Y = static_cast<float>(p1.y());
        const auto p2X = static_cast<float>(p2.x());
        const auto p2Y = static_cast<float>(p2.y());

        auto x1 = static_cast<int>((p1X * cosAngle) - (p1Y * sinAngle)) + x;
        auto y1 = static_cast<int>((p1X * sinAngle) + (p1Y * cosAngle)) + y;
        auto x2 = static_cast<int>((p2X * cosAngle) - (p2Y * sinAngle)) + x;
        auto y2 = static_cast<int>((p2X * sinAngle) + (p2Y * cosAngle)) + y;

        if (x1 >= 0 && x1 < image.cols() && y1 >= 0 && y1 < image.rows() && x2 >= 0 &&
            x2 < image.cols() && y2 >= 0 && y2 < image.rows()) {
            uint8_t pixelIntensity1 = image(y1, x1);
            uint8_t pixelIntensity2 = image(y2, x2);

            if (pixelIntensity1 < pixelIntensity2) {
                int byteIndex = bitIndex / slam::constants::BRIEF_PAIRS;
                int bitPosition = bitIndex % slam::constants::BRIEF_PAIRS;
                descriptor(0, byteIndex) |= (1 << bitPosition);
            }

            bitIndex++;
        }
    }

    return descriptor;
}

[[nodiscard]] std::vector<std::pair<Eigen::Vector2i, Eigen::Vector2i>>
FeatureDetector::generateBRIEFPattern() const {
    std::vector<std::pair<Eigen::Vector2i, Eigen::Vector2i>> pattern;

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
            pattern.emplace_back(Eigen::Vector2i(static_cast<int>(x1), static_cast<int>(y1)),
                                 Eigen::Vector2i(static_cast<int>(x2), static_cast<int>(y2)));
        }
    }

    return pattern;
}

[[nodiscard]] slam::EigenGrayMatrix FeatureDetector::gaussianBlur(
    const slam::EigenGrayMatrix& image, int kernelSize, double sigma) {
    if (kernelSize % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd");
    }

    // Generate Gaussian kernel
    int halfSize = kernelSize / 2;
    Eigen::MatrixXd kernel(kernelSize, kernelSize);
    double sum = 0.0;

    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            double value = std::exp(-((i * i) + (j * j)) / (2 * sigma * sigma));
            kernel(i + halfSize, j + halfSize) = value;
            sum += value;
        }
    }

    // Normalize kernel
    kernel /= sum;

    // Apply convolution
    slam::EigenGrayMatrix blurred = slam::EigenGrayMatrix::Zero(image.rows(), image.cols());

    for (int y = halfSize; y < image.rows() - halfSize; y++) {
        for (int x = halfSize; x < image.cols() - halfSize; x++) {
            double pixelValue = 0.0;

            for (int ky = -halfSize; ky <= halfSize; ky++) {
                for (int kx = -halfSize; kx <= halfSize; kx++) {
                    pixelValue += static_cast<double>(image(y + ky, x + kx)) *
                                  kernel(ky + halfSize, kx + halfSize);
                }
            }

            blurred(y, x) = static_cast<uint8_t>(std::round(pixelValue));
        }
    }

    // Copy border pixels from original image
    for (int i = 0; i < halfSize; i++) {
        blurred.row(i) = image.row(i);
        blurred.row(image.rows() - 1 - i) = image.row(image.rows() - 1 - i);
        blurred.col(i) = image.col(i);
        blurred.col(image.cols() - 1 - i) = image.col(image.cols() - 1 - i);
    }

    return blurred;
}
