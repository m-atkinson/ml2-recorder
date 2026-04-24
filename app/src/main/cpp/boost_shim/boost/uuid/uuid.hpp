// Minimal shim for boost::uuids::uuid used by vrs/os/System.cpp.
#pragma once

#include <array>
#include <cstdint>

namespace boost {
namespace uuids {

struct uuid {
  std::array<uint8_t, 16> data{};
};

} // namespace uuids
} // namespace boost
