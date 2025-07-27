# Visual SLAM Project (SLAM-CIN0051)

This repository contains the source code for a monocular Visual SLAM system developed in C++ using OpenCV and Eigen.

## Prerequisites

### Core Tools

* A C++17 compliant compiler (e.g., GCC, Clang)
* CMake (version 3.22 or higher)
* Python 3
* `clang-format` (for code formatting)

### System Libraries

To properly run the build scripts and manage dependencies, you may need to install the following system libraries:

```bash
sudo apt update
sudo apt install -y libva-dev libvdpau-dev libx11-xcb-dev libfontenc-dev libxaw7-dev libxkbfile-dev libxmu-dev libxmuu-dev libxpm-dev libxres-dev libxtst-dev libxcb-glx0-dev libxcb-render-util0-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-dri3-dev libxcb-cursor-dev libxcb-dri2-0-dev libxcb-dri3-dev libxcb-present-dev libxcb-composite0-dev libxcb-ewmh-dev libxcb-res0-dev libxcb-util-dev libxcb-util0-dev
````

## 1\. Setup

This project uses a Python virtual environment to manage Conan, the C++ package manager.

To set up the environment, run the provided setup script. This will create a local virtual environment in `.SLAM-venv/`, activate it, and install Conan.

```bash
chmod +x venv
source venv
```

**Note:** After the initial setup, you will need to activate the environment in your shell session whenever you want to build the project:

```bash
source venv
```

## 2\. Building the Project

The project is managed via a `Makefile` that simplifies the build process. The primary commands are:

  * **`make release`**: Compiles the project in `Release` mode. This is the recommended build for performance. The final package will be placed in `stage/release`.
  * **`make debug`**: Compiles the project in `Debug` mode. This enables detailed logging with Quill (`QUILL_ACTIVE=1`) and includes debugging symbols. The final package will be placed in `stage/debug`.

## 3\. Available Commands

The following `make` commands are available:

| Command        | Description                                                                                              |
| :------------- | :------------------------------------------------------------------------------------------------------- |
| `make release` | Builds the project in Release mode and creates a full, self-contained package in `stage/release`.          |
| `make debug`   | Builds the project in Debug mode and creates a full, self-contained package in `stage/debug`.            |
| `make format`  | Automatically formats all `.cpp` and `.hpp` files in the `src/` and `tools/` directories using `.clang-format`. |
| `make clean`   | Removes the `build/` and `stage/` directories, cleaning all build artifacts and installed packages.      |
| `make help`    | Displays a list of all available commands.                                                               |

## 4\. Code Formatting

To maintain a consistent code style across the project, we use `clang-format`. Before committing any changes, please run the formatting command:

```bash
make format
```