.DEFAULT_GOAL := help

.PHONY: all release debug install-release install-debug deploy clean format help

all: release

release:
	@conan install . --build=missing -s build_type=Release
	@cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=build/release/conan_toolchain.cmake
	@cmake --build build/release
	@cmake --install build/release --prefix ./stage/release
	@conan deploy --deployer=full_deploy -of stage/release --build-folder=build/release	

debug:
	@conan install . --build=missing -s build_type=Debug
	@cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=build/debug/conan_toolchain.cmake
	@cmake --build build/debug
	@cmake --install build/debug --prefix ./stage/debug
	@conan deploy --deployer=full_deploy -of stage/debug --build-folder=build/debug

format:
	@find src tools -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

clean:
	@rm -rf build stage

help:
	@echo "Available targets:"
	@echo "  make release         - (Default) Build the project in Release mode."
	@echo "  make debug           - Build the project in Debug mode."
	@echo "  make install-release - Install your application to the 'stage/release' directory."
	@echo "  make deploy          - Create a full, self-contained package in 'stage/release'."
	@echo "  make format          - Run clang-format on all source and header files."
	@echo "  make clean           - Remove all generated build and install files."
