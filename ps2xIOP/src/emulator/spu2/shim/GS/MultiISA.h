// ps2x SPU2 port shim: PCSX2 compiles some files once per ISA into an ISA namespace and picks one at runtime
// (GS/MultiISA.h). Here there is ONE build (the runtime's -msse4.1): one namespace, and MULTI_ISA_SELECT names it.
#pragma once
#define MULTI_ISA_COMPILE_ONCE 1
#define MULTI_ISA_DEF(...) namespace isa_native { __VA_ARGS__ }
#define MULTI_ISA_UNSHARED_START namespace isa_native {
#define MULTI_ISA_UNSHARED_END }
#define MULTI_ISA_SELECT(fn) (&isa_native::fn)
