// ps2x SIO2 port shim: the parts of PCSX2 pcsx2/Config.h the copied pad/SIO2 files use -- the binding/setting table
// types (PCSX2 UI data, kept so the tables compile verbatim) and EmuConfig.Pad's multitap query (no multitap here).
#pragma once
#include "common/Pcsx2Types.h"
#include <string>
#include <utility>
#include <vector>

enum class GenericInputBinding : u8
{
    Unknown, DPadUp, DPadRight, DPadLeft, DPadDown, LeftStickUp, LeftStickRight, LeftStickDown, LeftStickLeft, L3,
    RightStickUp, RightStickRight, RightStickDown, RightStickLeft, R3, Triangle, Circle, Cross, Square, Select, Start,
    System, L1, L2, R1, R2, SmallMotor, LargeMotor, Count,
};

struct InputBindingInfo
{
    enum class Type : u8 { Unknown, Button, Axis, HalfAxis, Motor, Pointer, Keyboard, Device, Macro };
    const char *name;
    const char *display_name;
    const char *icon_name;
    Type bind_type;
    u16 bind_index;
    GenericInputBinding generic_mapping;
};

struct SettingInfo
{
    using GetOptionsCallback = std::vector<std::pair<std::string, std::string>> (*)();
    enum class Type { Boolean, Integer, IntegerList, Float, String, StringList, Path };
    Type type;
    const char *name;
    const char *display_name;
    const char *description;
    const char *default_value;
    const char *min_value;
    const char *max_value;
    const char *step_value;
    const char *format;
    const char *const *options;
    GetOptionsCallback get_options;
    float multiplier;
};

// PCSX2 Pcsx2Config::PadOptions::NUM_PORTS: pad ports across both multitaps.
struct Pcsx2Config
{
    struct PadOptions
    {
        static constexpr u32 NUM_PORTS = 8;
    };
};

struct Ps2xSio2EmuConfigShim
{
    struct
    {
        bool IsMultitapPortEnabled(u32) const { return false; }   // no multitap attached (ps2x_sio2.cpp)
    } Pad;
};
inline Ps2xSio2EmuConfigShim EmuConfig;
