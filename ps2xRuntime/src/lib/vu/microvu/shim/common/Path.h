#pragma once // ps2xRuntime shim for the program logger's path helper
#include <string>
namespace Path { inline std::string Combine(const std::string& a, const std::string& b) { return a + "/" + b; } }
namespace EmuFolders { inline std::string Logs = "tmp"; }
