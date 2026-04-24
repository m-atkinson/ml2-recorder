cmake_minimum_required(VERSION 3.14)
project(ml2recorder_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Source files under test
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp)

add_executable(ml2recorder_tests
    test_csv_writer.cpp
    test_timestamp.cpp
    test_ring_buffer.cpp
    test_session.cpp
    ${SRC_DIR}/csv_writer.cpp
    ${SRC_DIR}/timestamp.cpp
    ${SRC_DIR}/ring_buffer.cpp
    ${SRC_DIR}/session.cpp
)

target_include_directories(ml2recorder_tests PRIVATE ${SRC_DIR})
target_link_libraries(ml2recorder_tests gtest_main)

include(GoogleTest)
gtest_discover_tests(ml2recorder_tests)
