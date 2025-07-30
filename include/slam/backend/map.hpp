#pragma once

#include <mutex>

#include <slam/common/common.hpp>

namespace slam {

class Map {
public:
    Map();
    void insertKeyframe(/* ... */);
    void insertMapPoint(/* ... */);

private:
    // Membros para armazenar keyframes e map points.
    std::mutex m_mapMutex;
};

}  // namespace slam
