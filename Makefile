.DEFAULT_GOAL := help

.PHONY: all release debug deploy clean format tidy test help

all: release

release:
	@conan install . --build=missing -s build_type=Release -d 'full_deploy' --deployer-folder build/Release
	@cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake
	@cmake --build build/Release -j 16
	@cmake --install build/Release --prefix ./stage/release

debug:
	@conan install . --build=missing -s build_type=Debug -d 'full_deploy' --deployer-folder build/Debug
	@cmake -S . -B build/Debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=build/Debug/generators/conan_toolchain.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	@cmake --build build/Debug -j 16
	@cmake --install build/Debug --prefix ./stage/debug

format:
	@find src tools include -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

tidy:
	@if [ ! -f build/Debug/compile_commands.json ]; then \
		echo "Compilation database not found. Run 'make debug' first to generate it."; \
		exit 1; \
	fi
	@run-clang-tidy -p build/Debug -j 16 \
		-header-filter='(src|tools|include)/.*'
		
test:
	@cd stage/debug/test && for test_name in $$(find . -maxdepth 1 -type f -executable -name "test_*"); do \
		echo "--- Running $$test_name ---"; \
		$$test_name || exit 1; \
	done

clean:
	@rm -rf build stage

help:
	@echo "Available targets:"
	@echo "  make release         - Build the project in Release mode and create a distributable package."
	@echo "  make debug           - Build the project in Debug mode and create a distributable package."
	@echo "  make format          - Run clang-format on all source and header files."
	@echo "  make tidy            - Run clang-tidy on all source files (requires 'make debug' to be run first)."
	@echo "  make clean           - Remove all generated build and install files."
	@echo "  make help            - (Default) Shows this help message."
