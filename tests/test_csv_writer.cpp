#include "csv_writer.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

namespace fs = std::filesystem;

class CsvWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = fs::temp_directory_path() / "ml2_csv_test";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }

    std::string read_file(const std::string& path) {
        std::ifstream f(path);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    fs::path test_dir_;
};

TEST_F(CsvWriterTest, WritesHeaderOnOpen) {
    ml2::CsvWriter writer;
    auto path = (test_dir_ / "test.csv").string();
    ASSERT_TRUE(writer.open(path, {"timestamp_ns", "filename"}));
    writer.close();

    EXPECT_EQ(read_file(path), "timestamp_ns,filename\n");
}

TEST_F(CsvWriterTest, WritesDataRows) {
    ml2::CsvWriter writer;
    auto path = (test_dir_ / "test.csv").string();
    ASSERT_TRUE(writer.open(path, {"timestamp_ns", "value"}));
    writer.write_row({"1000000000", "hello"});
    writer.write_row({"2000000000", "world"});
    writer.close();

    EXPECT_EQ(read_file(path),
        "timestamp_ns,value\n"
        "1000000000,hello\n"
        "2000000000,world\n");
}

TEST_F(CsvWriterTest, TimestampConvenienceMethod) {
    ml2::CsvWriter writer;
    auto path = (test_dir_ / "test.csv").string();
    ASSERT_TRUE(writer.open(path, {"timestamp_ns", "x", "y"}));
    writer.write_row(int64_t(1000000000), {"1.5", "2.5"});
    writer.close();

    EXPECT_EQ(read_file(path),
        "timestamp_ns,x,y\n"
        "1000000000,1.5,2.5\n");
}

TEST_F(CsvWriterTest, RowCountTracking) {
    ml2::CsvWriter writer;
    auto path = (test_dir_ / "test.csv").string();
    ASSERT_TRUE(writer.open(path, {"a", "b"}));
    EXPECT_EQ(writer.row_count(), 0u);
    writer.write_row({"1", "2"});
    EXPECT_EQ(writer.row_count(), 1u);
    writer.write_row({"3", "4"});
    EXPECT_EQ(writer.row_count(), 2u);
}

TEST_F(CsvWriterTest, ThreadSafety) {
    ml2::CsvWriter writer;
    auto path = (test_dir_ / "test.csv").string();
    ASSERT_TRUE(writer.open(path, {"timestamp_ns", "thread_id"}));

    constexpr int kThreads = 4;
    constexpr int kRowsPerThread = 250;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&writer, t]() {
            for (int i = 0; i < kRowsPerThread; ++i) {
                writer.write_row({std::to_string(i), std::to_string(t)});
            }
        });
    }
    for (auto& th : threads) th.join();
    writer.close();

    EXPECT_EQ(writer.row_count(), size_t(kThreads * kRowsPerThread));

    // Verify file line count = header + data rows
    std::ifstream f(path);
    int line_count = 0;
    std::string line;
    while (std::getline(f, line)) ++line_count;
    EXPECT_EQ(line_count, 1 + kThreads * kRowsPerThread);
}

TEST_F(CsvWriterTest, OpenFailsForBadPath) {
    ml2::CsvWriter writer;
    EXPECT_FALSE(writer.open("/nonexistent/dir/test.csv", {"a"}));
}

TEST_F(CsvWriterTest, EmptyValuesHandled) {
    ml2::CsvWriter writer;
    auto path = (test_dir_ / "test.csv").string();
    ASSERT_TRUE(writer.open(path, {"a", "b", "c"}));
    writer.write_row({"1", "", "3"});
    writer.close();

    EXPECT_EQ(read_file(path), "a,b,c\n1,,3\n");
}
