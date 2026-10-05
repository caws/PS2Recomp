#pragma once // ps2xRuntime shim: microVU's one StringUtil use is in the html program logger
#include <string>
namespace StringUtil { inline std::string StdStringFromFormat(const char* fmt, ...) { return std::string(fmt); } }
