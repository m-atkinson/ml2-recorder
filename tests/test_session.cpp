#include "session.h"

#include <filesystem>
#include <gtest/gtest.h>

namespace fs = std::filesystem;

class SessionTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_base_ = fs::temp_directory_path() / "ml2_session_test";
        fs::create_directories(test_base_);
    }

    void TearDown() override {
        fs::remove_all(test_base_);
    }

    fs::path test_base_;
    ml2::Session session_;
};

TEST_F(SessionTest, CreateMakesDirectory) {
    ASSERT_TRUE(session_.create(test_base_.string()));
    EXPECT_TRUE(fs::is_directory(session_.path()));
}

TEST_F(SessionTest, NameFollowsFormat) {
    ASSERT_TRUE(session_.create(test_base_.string()));
    EXPECT_EQ(session_.name().substr(0, 8), "session_");
    // session_YYYYMMDD_HHMMSS = 23 chars
    EXPECT_EQ(session_.name().size(), 23u);
}

TEST_F(SessionTest, CreateSubdir) {
    ASSERT_TRUE(session_.create(test_base_.string()));
    auto rgb_dir = session_.create_subdir("rgb");
    EXPECT_TRUE(fs::is_directory(rgb_dir));
    EXPECT_EQ(rgb_dir, session_.path() + "/rgb");
}

TEST_F(SessionTest, Filepath) {
    ASSERT_TRUE(session_.create(test_base_.string()));
    auto path = session_.filepath("rgb.csv");
    EXPECT_EQ(path, session_.path() + "/rgb.csv");
}

TEST_F(SessionTest, MultipleSubdirs) {
    ASSERT_TRUE(session_.create(test_base_.string()));
    std::vector<std::string> dirs = {
        "rgb", "depth", "depth_confidence",
        "world_cam_0", "world_cam_1", "world_cam_2", "audio"
    };
    for (const auto& d : dirs) {
        EXPECT_TRUE(fs::is_directory(session_.create_subdir(d)));
    }
}
