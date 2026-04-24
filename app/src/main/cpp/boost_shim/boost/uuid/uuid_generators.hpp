// Minimal shim for boost::uuids::random_generator used by vrs/os/System.cpp.
// Generates a version-4 UUID using std::random_device.
#pragma once

#include <boost/uuid/uuid.hpp>
#include <cstdint>
#include <random>

namespace boost {
namespace uuids {

struct random_generator {
  uuid operator()() {
    uuid u;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    const uint64_t a = dist(gen);
    const uint64_t b = dist(gen);
    for (int i = 0; i < 8; ++i) {
      u.data[i]     = static_cast<uint8_t>((a >> (i * 8)) & 0xff);
      u.data[i + 8] = static_cast<uint8_t>((b >> (i * 8)) & 0xff);
    }
    // Version 4: random UUID
    u.data[6] = (u.data[6] & 0x0f) | 0x40;
    // Variant bits: 10xxxxxx
    u.data[8] = (u.data[8] & 0x3f) | 0x80;
    return u;
  }
};

} // namespace uuids
} // namespace boost
