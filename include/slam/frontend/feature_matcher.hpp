#pragma once

#include <filesystem>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <slam/common/common.hpp>

namespace slam {

class FeatureMatcher {
public:
    explicit FeatureMatcher(const std::filesystem::path& configPath);
    void match(const std::vector<KeyDescriptorPair>& pairs1,  // in
               const std::vector<KeyDescriptorPair>& pairs2,  // in
               std::vector<std::pair<int, int>>& matches);    // out
};

}  // namespace slam
