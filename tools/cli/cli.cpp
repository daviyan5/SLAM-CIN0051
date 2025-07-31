#include <getopt.h>

#include <spdlog/spdlog.h>

#include <slam/model/model.hpp>

void printHelp() {
}

int main(int argc, char** argv) {
    /*
    General form of the command:
    cli -c <config_file> -v <stream_path>
    */
    spdlog::set_level(spdlog::level::debug);

    int c{};
    std::string streamPath{};
    std::string configPath{};
    while ((c = getopt(argc, argv, "hv:c:")) != -1) {
        switch (c) {
            case 'h':
                printHelp();
                return 0;
            case 'v':
                streamPath = optarg;
                break;
            case 'c':
                configPath = optarg;
                break;
            default:
                printHelp();
                return 0;
        }
    }
    SPDLOG_DEBUG("Running model with config [{}] and stream [{}]", configPath, streamPath);
    slam::SLAMModel model(configPath, streamPath);
    return 0;
}
