#pragma once

#include <thread>

#include <slam/backend/map.hpp>
#include <slam/common/common.hpp>

namespace slam {

class Visualizer {
public:
    explicit Visualizer(const slam::Map& map);
    void run();  // Inicia a thread de visualização.
private:
    std::reference_wrapper<const slam::Map> m_map;
    std::thread m_visualizerThread;
};

}  // namespace slam
