.DEFAULT_GOAL := help

.PHONY: all release debug deploy clean format help

all: release

release:
	@conan install . --build=missing -s build_type=Release -d 'full_deploy' --deployer-folder build/Debug
	@cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake
	@cmake --build build/Release
	@cmake --install build/Release --prefix ./stage/release

debug:
	@conan install . --build=missing -s build_type=Debug -d 'full_deploy' --deployer-folder build/Debug
	@cmake -S . -B build/Debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=build/Debug/generators/conan_toolchain.cmake
	@cmake --build build/Debug
	@cmake --install build/Debug --prefix ./stage/debug

format:
	@find src tools include -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

clean:
	@rm -rf build stage

help:
	@echo "Available targets:"
	@echo "  make release         - Build the project in Release mode and create a distributable package."
	@echo "  make debug           - Build the project in Debug mode and create a distributable package."
	@echo "  make format          - Run clang-format on all source and header files."
	@echo "  make clean           - Remove all generated build and install files."
	@echo "  make help            - (Default) Shows this help message."