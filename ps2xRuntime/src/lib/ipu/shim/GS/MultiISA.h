// ps2xRuntime shim for PCSX2's GS/MultiISA.h. PCSX2 compiles its "unshared" translation units once per
// host ISA and selects at runtime; this runtime targets one ISA (SSE4.1), so every ISA-scoped symbol
// lives in ONE namespace, CurrentISA, exactly as PCSX2's single-ISA fallback does.
#pragma once
#define MULTI_ISA_COMPILE_ONCE 1
#define MULTI_ISA_UNSHARED_START namespace CurrentISA {
#define MULTI_ISA_UNSHARED_END }
#define MULTI_ISA_DEF(...) namespace CurrentISA { __VA_ARGS__ }
#define MULTI_ISA_SELECT(fname) (CurrentISA::fname)
