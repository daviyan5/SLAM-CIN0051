#pragma once

#include <thread>

#include <slam/backend/map.hpp>
#include <slam/common/common.hpp>

namespace slam {

class Backend {
public:
    explicit Backend(slam::Map& map);
    void run();  // Inicia a thread do backend. Realiza as otimizações
private:
    // Não é const pois o Backend chamará funções não-const de Map
    std::reference_wrapper<slam::Map> m_map;
    std::thread m_backendThread;
};

}  // namespace slam
