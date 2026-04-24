#include "timestamp.h"

#include <chrono>
#include <thread>
#include <gtest/gtest.h>

TEST(TimestampTest, ReturnsPositiveValue) {
    EXPECT_GT(ml2::now_ns(), 0);
}

TEST(TimestampTest, IsMonotonicallyIncreasing) {
    int64_t ts1 = ml2::now_ns();
    volatile int x = 0;
    for (int i = 0; i < 10000; ++i) x += i;
    int64_t ts2 = ml2::now_ns();
    EXPECT_GT(ts2, ts1);
}

TEST(TimestampTest, ReasonableResolution) {
    int64_t ts1 = ml2::now_ns();
    int64_t ts2 = ml2::now_ns();
    EXPECT_LT(ts2 - ts1, 1'000'000'000LL);
}

TEST(TimestampTest, SensorTsConversionIsIdentity) {
    EXPECT_EQ(ml2::sensor_ts_to_ns(1234567890), 1234567890);
}

TEST(TimestampTest, SleepProducesExpectedDelta) {
    int64_t ts1 = ml2::now_ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    int64_t ts2 = ml2::now_ns();
    int64_t delta_ms = (ts2 - ts1) / 1'000'000;
    EXPECT_GE(delta_ms, 20);
    EXPECT_LE(delta_ms, 200);
}
