cmake_minimum_required(VERSION 3.22.1)
project(ml2recorder LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable SSE4.1 for non-temporal streaming copy from camera DMA buffers.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options(-msse4.1)
endif()

# ── Dependency downloads ──────────────────────────────────────────────────────
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# CMake 4.x removed compatibility with cmake_minimum_required < 3.5.
# Several fetched deps (lz4 v1.9.4, zstd, xxhash) set an older minimum.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    set(CMAKE_POLICY_VERSION_MINIMUM "3.5")
endif()

# Suppress CMake 3.30+ deprecation warning for FetchContent_Populate (single-arg form).
# We use it for header-only deps and VRS (which we build manually).
if(POLICY CMP0168)
    cmake_policy(SET CMP0168 OLD)
endif()

FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
)

FetchContent_Declare(lz4
    GIT_REPOSITORY https://github.com/lz4/lz4.git
    GIT_TAG        v1.9.4
    SOURCE_SUBDIR  build/cmake
)

FetchContent_Declare(zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG        v1.5.5
    SOURCE_SUBDIR  build/cmake
)

FetchContent_Declare(xxhash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG        v0.8.2
    SOURCE_SUBDIR  cmake_unofficial
)

# RapidJSON is header-only; download but don't add subdirectory.
# v1.1.0 has a const-assignment bug in document.h that breaks C++17 on Clang 21.
# Use master which has the fix.
FetchContent_Declare(rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        master
)

# VRS: download only. We build it manually below to avoid its system-package
# requirements (Boost, libpng, libjpeg, TurboJPEG) which are unavailable on
# the Android NDK toolchain.
FetchContent_Declare(vrs
    GIT_REPOSITORY https://github.com/facebookresearch/vrs.git
    GIT_TAG        main
)

# ── fmt ──────────────────────────────────────────────────────────────────────
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_TEST   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(fmt)

# ── lz4 ──────────────────────────────────────────────────────────────────────
set(LZ4_BUILD_CLI          OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_LEGACY_LZ4C  OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(lz4)

# ── zstd ─────────────────────────────────────────────────────────────────────
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED   OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC   ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(zstd)

# ── xxhash ───────────────────────────────────────────────────────────────────
# Disable xxhsum CLI tool (it requires a main() we don't need)
set(XXHASH_BUILD_XXHSUM OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(xxhash)
# xxhash cmake_unofficial creates target xxHash::xxhash

# ── RapidJSON (header-only) ──────────────────────────────────────────────────
FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
    FetchContent_Populate(rapidjson)
endif()
add_library(rapidjson_iface INTERFACE)
target_include_directories(rapidjson_iface INTERFACE
    ${rapidjson_SOURCE_DIR}/include
)
target_compile_definitions(rapidjson_iface INTERFACE
    RAPIDJSON_HAS_STDSTRING=1
)
add_library(rapidjson::rapidjson ALIAS rapidjson_iface)

# ── VRS (manual build — write path only) ─────────────────────────────────────
#
# VRS's CMakeLists.txt requires system packages (Boost, libpng, libjpeg,
# TurboJPEG, RapidJSON) that are not available via the Android NDK toolchain.
# We download VRS source only and build the minimal write-path components
# ourselves, providing a Boost shim for the two Boost usages in vrs/os.
#
# Components compiled:
#   vrs/*.cpp             — core RecordFileWriter / Recordable / DataLayout
#   vrs/os/*.cpp          — OS abstraction (Semaphore patched via boost_shim/)
#   vrs/helpers/*.cpp     — string / serialisation helpers (needs RapidJSON)
#   vrs/oss/logging/*.cpp — minimal logging (replaces Meta's internal logger)
#   vrs/utils/xxhash/*.cpp — VRS's xxhash checksum wrapper

FetchContent_GetProperties(vrs)
if(NOT vrs_POPULATED)
    FetchContent_Populate(vrs)
endif()

file(GLOB vrs_core_srcs         "${vrs_SOURCE_DIR}/vrs/*.cpp")
file(GLOB vrs_os_srcs           "${vrs_SOURCE_DIR}/vrs/os/*.cpp")
file(GLOB vrs_helpers_srcs      "${vrs_SOURCE_DIR}/vrs/helpers/*.cpp")
file(GLOB vrs_logging_srcs      "${vrs_SOURCE_DIR}/vrs/oss/logging/*.cpp")
file(GLOB vrs_xxhash_srcs       "${vrs_SOURCE_DIR}/vrs/utils/xxhash/*.cpp")

add_library(vrslib STATIC
    ${vrs_core_srcs}
    ${vrs_os_srcs}
    ${vrs_helpers_srcs}
    ${vrs_logging_srcs}
    ${vrs_xxhash_srcs}
)

target_include_directories(vrslib PUBLIC
    # VRS headers live at vrs_SOURCE_DIR/vrs/*.h — consumers include <vrs/Foo.h>
    ${vrs_SOURCE_DIR}
    # Logging headers are at vrs/oss/logging/Log.h — included as <logging/Log.h>
    ${vrs_SOURCE_DIR}/vrs/oss
    # Boost shim replaces boost/interprocess and boost/uuid for Android NDK
    ${CMAKE_CURRENT_SOURCE_DIR}/boost_shim
    # zstd headers (libzstd_static doesn't propagate INTERFACE_INCLUDE_DIRECTORIES)
    ${zstd_SOURCE_DIR}/lib
)

target_compile_definitions(vrslib PUBLIC
    OSS_BUILD_MODE              # Required: selects the open-source code paths in VRS
    RAPIDJSON_HAS_STDSTRING=1
)

target_link_libraries(vrslib PUBLIC
    fmt::fmt
    lz4_static
    libzstd_static
    xxHash::xxhash
    rapidjson::rapidjson
    log                         # Android log library (used by vrs_logging)
)

# ── Application sources ───────────────────────────────────────────────────────
set(SOURCES
    main.cpp
    timestamp.cpp
    session.cpp
    write_queue.cpp
    rgb_capture.cpp
    h264_encoder.cpp
    perception_capture.cpp
    depth_capture.cpp
    world_camera_capture.cpp
    imu_capture.cpp
    audio_capture.cpp
    meshing_capture.cpp
    metadata_writer.cpp
    storage_monitor.cpp
    thermal_monitor.cpp
    csv_writer.cpp
    vrs_writer.cpp
)

add_library(ml2recorder SHARED ${SOURCES})

target_include_directories(ml2recorder PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue
)

target_sources(ml2recorder PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c
)

target_link_libraries(ml2recorder
    android
    log
    mediandk
    OpenSLES
    vrslib
)

# ── Magic Leap SDK ────────────────────────────────────────────────────────────
# MLSDK is proprietary — obtain it from https://ml2-developer.magicleap.com/
# and point MLSDK at the unpacked directory, e.g.
#   export MLSDK=$HOME/MagicLeap/mlsdk/v1.12.0
if(NOT DEFINED ENV{MLSDK})
    message(FATAL_ERROR
        "MLSDK environment variable is not set. "
        "Download MLSDK from https://ml2-developer.magicleap.com/ and set "
        "MLSDK to the unpacked path before building.")
endif()
set(MLSDK $ENV{MLSDK})

if(NOT EXISTS ${MLSDK}/include)
    message(FATAL_ERROR "MLSDK=${MLSDK} does not contain an include/ directory.")
endif()

message(STATUS "Using MLSDK at: ${MLSDK}")
target_include_directories(ml2recorder PRIVATE ${MLSDK}/include)
target_link_directories(ml2recorder PRIVATE ${MLSDK}/lib/ml2)
target_link_libraries(ml2recorder
    camera.magicleap
    perception.magicleap
)
