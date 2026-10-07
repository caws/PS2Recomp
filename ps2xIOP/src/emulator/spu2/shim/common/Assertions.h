// ps2x SPU2 port shim: PCSX2's assertions (release build: no checks, as PCSX2's release config).
#pragma once
#define pxAssert(cond) ((void)0)
#define pxAssertMsg(cond, msg) ((void)0)
#define pxAssertRel(cond, msg) ((void)0)
#define pxAssume(cond) ((void)0)
#define pxFail(msg) ((void)0)
#define pxFailRel(msg) ((void)0)
