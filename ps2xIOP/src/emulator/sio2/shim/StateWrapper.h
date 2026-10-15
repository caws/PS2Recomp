// ps2x SIO2 port shim: PCSX2 StateWrapper.h. ps2x has no save states, so every DoState/Freeze is a no-op.
#pragma once
#include <cstddef>
class StateWrapper
{
public:
    template <typename T> void Do(T *) {}
    void DoBytes(void *, size_t) {}
    bool DoMarker(const char *) { return true; }
    bool HasError() const { return false; }
    bool IsGood() const { return true; }
    bool IsReading() const { return false; }
    bool IsWriting() const { return false; }
};
