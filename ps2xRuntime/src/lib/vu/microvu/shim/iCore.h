// ps2xRuntime shim for PCSX2's x86/iCore.h: the EE recompiler's host-register tables that microVU's
// allocator consults in COP2 mode only (mVU.cop2 is never set here). Tables are empty, the
// allocation calls abort if ever reached.
#pragma once
#include "common/Pcsx2Defs.h"
#include "common/emitter/x86emitter.h"
#include <cstdlib>
#include <cstdio>

#define MODE_READ 1
#define MODE_WRITE 2
#define MODE_CALLEESAVED 0x20
#define MODE_COP2 0x40

enum { X86TYPE_TEMP = 0, X86TYPE_GPR = 1, X86TYPE_FPRC = 2, X86TYPE_VIREG = 3, X86TYPE_PCWRITEBACK = 4, X86TYPE_PSX = 5, X86TYPE_PSX_PCWRITEBACK = 6 };
#define XMMTYPE_TEMP 0
#define XMMTYPE_VFREG 8
#define XMMTYPE_ACC 9
#define XMMTYPE_FPREG 10
#define XMMTYPE_FPACC 11
#define XMMTYPE_GPRREG 12

struct _x86regs
{
	u8 inuse;
	s8 reg;
	u8 mode;
	u8 needed;
	u8 type;
	u16 counter;
	u32 extra;
};
struct _xmmregs
{
	u8 inuse;
	s8 reg;
	u8 type;
	u8 mode;
	u8 needed;
	u16 counter;
};
extern _x86regs x86regs[iREGCNT_GPR];
extern _xmmregs xmmregs[iREGCNT_XMM];
extern u16 g_x86AllocCounter;

inline int _allocX86reg(int, int, int) { std::fprintf(stderr, "[microvu] COP2-mode register allocation reached\n"); std::abort(); }
inline void _freeX86regWithoutWriteback(int) { std::abort(); }
inline void _freeXMMreg(int) { std::abort(); }
inline int _allocVFtoXMMreg(int, int) { std::abort(); }
