#pragma once // ps2xRuntime shim for the program logger's file open
#include <cstdio>
#include <string>
namespace FileSystem { inline std::FILE* OpenCFile(const char* path, const char* mode) { return std::fopen(path, mode); } }
