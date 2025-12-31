cmake_minimum_required(VERSION 3.5.1)
project(MixformerV2)

add_compile_options(-std=c++11)
add_definitions(-DENABLE_DVPP_INTERFACE)

# 输出目录配置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
message(STATUS "可执行文件输出目录: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# ====================== 依赖库配置 ======================
# ACL 核心库
set(ACL_INCLUDE_PATH "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include")
set(ACL_LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64")

# OpenCV 库（系统 3.2.0，）
set(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv/")
find_package(OpenCV REQUIRED)
message(STATUS "使用系统 OpenCV 版本: ${OpenCV_VERSION}")

# ====================== 项目自身路径配置 ======================
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_INCLUDE ${PROJECT_ROOT}/include)
set(PROJECT_SRC ${PROJECT_ROOT}/src)

# ====================== 头文件包含目录 ======================
include_directories(
    ${PROJECT_INCLUDE}
    ${ACL_INCLUDE_PATH}
    ${OpenCV_INCLUDE_DIRS}  # 系统 OpenCV 头文件
)

# ====================== 库文件链接目录 ======================
link_directories(
    ${ACL_LIB_PATH}
    /usr/local/Ascend/thirdpart/aarch64/lib/
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib/
    ${OpenCV_LIB_DIRS}  # OpenCV 
    /usr/lib/aarch64-linux-gnu/
)

# 源文件收集
aux_source_directory(${PROJECT_SRC} PROJECT_SOURCES)

# 可执行文件编译
add_executable(main ${PROJECT_SOURCES})

# ====================== 链接依赖库 ======================
target_link_libraries(main
    # 系统基础库
    pthread stdc++ m z lzma dl rt

    # ACL 库
    ascendcl acl_dvpp

    # OpenCV 
    ${OpenCV_LIBS}
)