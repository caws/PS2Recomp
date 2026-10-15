// ps2x SIO2 port shim: PCSX2 common/Assertions.h (release behaviour: assertions compiled out, pxFail logs).
#pragma once
#include "common/Console.h"
#define pxAssert(cond) ((void)0)
#define pxAssertMsg(cond, msg) ((void)0)
#define pxFail(msg) Console.Error("%s", msg)
