// ps2xRuntime shim for PCSX2's GS/MultiISA.h: the host ISA level (microVU only tests for AVX2).
#pragma once
struct ProcessorFeatures
{
	enum class VectorISA { SSE4, AVX, AVX2 };
	VectorISA vectorISA = VectorISA::SSE4;
};
extern ProcessorFeatures g_cpu;
