// ps2x USB (OHCI) port: force-included into every file of the port, standing in for PCSX2's precompiled header
// (the standard headers the verbatim qemu-usb files use without including: std::min, std::size, malloc, abort).
#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
