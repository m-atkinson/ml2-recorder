#include "ring_buffer.h"

#include <atomic>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

TEST(RingBufferTest, EmptyOnConstruction) {
    ml2::RingBuffer buf(4);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.drops(), 0u);
}

TEST(RingBufferTest, PushAndPop) {
    ml2::RingBuffer buf(4);
    uint8_t data[] = {1, 2, 3, 4};
    buf.push(data, sizeof(data), 1000);

    EXPECT_EQ(buf.size(), 1u);

    bool popped = buf.pop([](const uint8_t* d, size_t s, int64_t ts) {
        EXPECT_EQ(s, 4u);
        EXPECT_EQ(d[0], 1);
        EXPECT_EQ(d[3], 4);
        EXPECT_EQ(ts, 1000);
    });
    EXPECT_TRUE(popped);
    EXPECT_TRUE(buf.empty());
}

TEST(RingBufferTest, PopFromEmptyReturnsFalse) {
    ml2::RingBuffer buf(4);
    bool popped = buf.pop([](const uint8_t*, size_t, int64_t) {
        FAIL() << "Should not be called on empty buffer";
    });
    EXPECT_FALSE(popped);
}

TEST(RingBufferTest, FIFOOrder) {
    ml2::RingBuffer buf(4);
    for (int i = 0; i < 3; ++i) {
        uint8_t d = static_cast<uint8_t>(i);
        buf.push(&d, 1, i * 100);
    }

    for (int i = 0; i < 3; ++i) {
        buf.pop([i](const uint8_t* d, size_t s, int64_t ts) {
            EXPECT_EQ(s, 1u);
            EXPECT_EQ(d[0], static_cast<uint8_t>(i));
            EXPECT_EQ(ts, i * 100);
        });
    }
}

TEST(RingBufferTest, DropOldestWhenFull) {
    ml2::RingBuffer buf(3);
    for (int i = 0; i < 5; ++i) {
        uint8_t d = static_cast<uint8_t>(i);
        buf.push(&d, 1, i * 100);
    }

    EXPECT_EQ(buf.size(), 3u);
    EXPECT_EQ(buf.drops(), 2u);

    buf.pop([](const uint8_t* d, size_t, int64_t ts) {
        EXPECT_EQ(d[0], 2);
        EXPECT_EQ(ts, 200);
    });
}

TEST(RingBufferTest, Clear) {
    ml2::RingBuffer buf(4);
    uint8_t d = 42;
    buf.push(&d, 1, 100);
    buf.push(&d, 1, 200);
    buf.clear();
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0u);
}

TEST(RingBufferTest, ConcurrentPushPop) {
    ml2::RingBuffer buf(32);
    constexpr int kItems = 1000;
    std::atomic<bool> done{false};

    std::thread producer([&]() {
        for (int i = 0; i < kItems; ++i) {
            uint8_t d = static_cast<uint8_t>(i & 0xFF);
            buf.push(&d, 1, i);
        }
        done.store(true, std::memory_order_release);
    });

    int consumed = 0;
    std::thread consumer([&]() {
        while (!done.load(std::memory_order_acquire) || !buf.empty()) {
            if (buf.pop([](const uint8_t*, size_t, int64_t) {})) {
                ++consumed;
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(consumed + buf.drops(), static_cast<size_t>(kItems));
}

TEST(RingBufferTest, LargeFrameData) {
    ml2::RingBuffer buf(2);
    std::vector<uint8_t> frame(1024 * 1024, 0xAB);
    buf.push(frame.data(), frame.size(), 999);

    buf.pop([&](const uint8_t* d, size_t s, int64_t ts) {
        EXPECT_EQ(s, frame.size());
        EXPECT_EQ(d[0], 0xAB);
        EXPECT_EQ(d[s - 1], 0xAB);
        EXPECT_EQ(ts, 999);
    });
}
