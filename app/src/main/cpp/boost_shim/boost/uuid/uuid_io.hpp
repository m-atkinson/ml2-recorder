// Minimal shim for boost::uuids ostream operator used by vrs/os/System.cpp.
// Formats uuid as: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#pragma once

#include <boost/uuid/uuid.hpp>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace boost {
namespace uuids {

inline std::ostream& operator<<(std::ostream& os, const uuid& u) {
  std::ios_base::fmtflags flags = os.flags();
  os << std::hex << std::setfill('0');
  // 4 bytes: xxxxxxxx
  for (int i = 0; i < 4; ++i) os << std::setw(2) << static_cast<int>(u.data[i]);
  os << '-';
  // 2 bytes: xxxx
  for (int i = 4; i < 6; ++i) os << std::setw(2) << static_cast<int>(u.data[i]);
  os << '-';
  // 2 bytes: xxxx
  for (int i = 6; i < 8; ++i) os << std::setw(2) << static_cast<int>(u.data[i]);
  os << '-';
  // 2 bytes: xxxx
  for (int i = 8; i < 10; ++i) os << std::setw(2) << static_cast<int>(u.data[i]);
  os << '-';
  // 6 bytes: xxxxxxxxxxxx
  for (int i = 10; i < 16; ++i) os << std::setw(2) << static_cast<int>(u.data[i]);
  os.flags(flags);
  return os;
}

} // namespace uuids
} // namespace boost
