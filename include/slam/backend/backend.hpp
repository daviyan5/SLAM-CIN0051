#pragma once

#include <thread>

#include <slam/backend/map.hpp>
#include <slam/common/common.hpp>

namespace slam {

class Backend {
public:
    Backend(slam::Map& map);
    void run();  // Inicia a thread do backend. Realiza as otimizações
private:
    slam::Map& m_map;
    std::thread m_backendThread;
};

}  // namespace slam
