#include "runtime/gs/gs_gpu_device.h"
#include "runtime/gs/ps2_gs_memory.h"
// cont.356: ps2xGsPresentAspect() / ps2xWindowAspect() -- the natural and presented ratios the
// 2D/HUD counter-scale is derived from.
#include "runtime/gs/gs_cpu_backend.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <array>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

// GPU presentation-decode device (cont.166, phase 0 of the GPU-backend arc).
//
// EGL GL 4.3 core context + one compute kernel replicating the CPU display decode
// (GSCpuBackend::CopyFrameToHostRgbaCpu, useLocalMemoryLayout path) bit-exactly. The
// page lookup tables uploaded to the GPU are built with the SAME PixelStorageTraits<>
// template code as ps2_gs_memory.cpp, from the same hardware block/column layout
// constants (ps2tek "GS local memory"; PCSX2 GSLocalMemory.cpp holds the equivalent
// tables), so the address math cannot drift from the CPU path. PS2X_GS_GPU_VERIFY in
// gs_cpu_backend.cpp additionally proves equality empirically per present.
//
// THREADING: all EGL/GL work happens on a dedicated device thread. The presenting main
// thread already holds raylib's GLX context current, and a thread has ONE current GL
// context regardless of binding API -- eglMakeCurrent there fails with EGL_BAD_ACCESS
// (observed 0x3002, run171 of build gpupres0-171). Callers marshal one synchronous job
// at a time (a present decodes at most two rects); this is also the shape the future
// GPU raster backend needs (its GL work rides the GS worker's command stream).
//
// GL is loaded via eglGetProcAddress with local typedefs -- no GL headers, so this file
// stays inert to the rest of the build (raylib's GL 3.3 window context is untouched;
// results return via buffer readback, cheap on this UMA iGPU class of host).

namespace
{
    // ---- minimal GL declarations (loaded at runtime; no system GL headers) ----
    using GLenum = unsigned int;
    using GLuint = unsigned int;
    using GLint = int;
    using GLsizei = int;
    using GLbitfield = unsigned int;
    using GLchar = char;
    using GLubyte = unsigned char;
    using GLsizeiptr = ptrdiff_t;
    using GLintptr = ptrdiff_t;

    constexpr GLenum GL_COMPUTE_SHADER = 0x91B9;
    constexpr GLenum GL_SHADER_STORAGE_BUFFER = 0x90D2;
    constexpr GLenum GL_STATIC_DRAW = 0x88E4;
    constexpr GLenum GL_DYNAMIC_DRAW = 0x88E8;
    constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
    constexpr GLenum GL_LINK_STATUS = 0x8B82;
    constexpr GLbitfield GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000;
    constexpr GLbitfield GL_BUFFER_UPDATE_BARRIER_BIT = 0x00000200;
    constexpr GLbitfield GL_SHADER_IMAGE_ACCESS_BARRIER_BIT = 0x00000020;
    constexpr GLbitfield GL_TEXTURE_FETCH_BARRIER_BIT = 0x00000008;
    constexpr GLbitfield GL_FRAMEBUFFER_BARRIER_BIT = 0x00000400;
    constexpr GLenum GL_R32F = 0x822E;
    constexpr GLenum GL_RED = 0x1903;
    constexpr GLenum GL_READ_WRITE = 0x88BA;
    constexpr GLenum GL_TIME_ELAPSED = 0x88BF;
    constexpr GLenum GL_QUERY_RESULT = 0x8866;
    constexpr GLenum GL_SAMPLES_PASSED = 0x8914;
    constexpr GLenum GL_NEAREST_MIPMAP_NEAREST = 0x2700;
    constexpr GLenum GL_LINEAR_MIPMAP_NEAREST = 0x2701;
    constexpr GLenum GL_NEAREST_MIPMAP_LINEAR = 0x2702;
    constexpr GLenum GL_LINEAR_MIPMAP_LINEAR = 0x2703;
    constexpr GLenum GL_TEXTURE_MAX_LEVEL = 0x813D;
    constexpr GLenum GL_RENDERER = 0x1F01;
    constexpr GLenum GL_VERSION = 0x1F02;
    // Hardware-raster path (cont.233): a vertex+fragment program on a no-attachment FBO.
    constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
    constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
    constexpr GLenum GL_TRIANGLES = 0x0004;
    constexpr GLenum GL_FRAMEBUFFER = 0x8D40;
    constexpr GLenum GL_FRAMEBUFFER_DEFAULT_WIDTH = 0x9310;
    constexpr GLenum GL_FRAMEBUFFER_DEFAULT_HEIGHT = 0x9311;
    constexpr GLenum GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
    constexpr GLenum GL_CONSERVATIVE_RASTERIZATION_INTEL = 0x83FE;
    constexpr GLenum GL_CULL_FACE = 0x0B44;
    constexpr GLenum GL_DEPTH_TEST = 0x0B71;
    constexpr GLenum GL_STENCIL_TEST = 0x0B90;
    constexpr GLenum GL_SCISSOR_TEST = 0x0C11;
    constexpr GLenum GL_BLEND = 0x0BE2;
    constexpr GLenum GL_MULTISAMPLE = 0x809D;
    // ★ cont.329 phase 1 (the GL renderer arc): a REAL colour+depth render target, resolved on
    // the GPU to display size before readback -- so readback cost is flat in the upscale factor.
    constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
    constexpr GLenum GL_RGBA8 = 0x8058;
    constexpr GLenum GL_RGBA16F = 0x881A;   // cont.343b: the HDR colour buffer for COLCLAMP=0 targets
    constexpr GLenum GL_HALF_FLOAT = 0x140B;
    constexpr GLenum GL_RGBA = 0x1908;
    constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
    constexpr GLenum GL_UNSIGNED_SHORT = 0x1403;  // cont.358: GsGlVertex::hudAnchor's attribute type
    constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
    constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
    constexpr GLenum GL_NEAREST = 0x2600;
    constexpr GLenum GL_LINEAR = 0x2601;
    constexpr GLenum GL_COLOR_ATTACHMENT0 = 0x8CE0;
    constexpr GLenum GL_DEPTH_ATTACHMENT = 0x8D00;
    constexpr GLenum GL_RENDERBUFFER = 0x8D41;
    constexpr GLenum GL_DEPTH_COMPONENT24 = 0x81A6;
    // cont.332j: the depth attachment is a texture now, so the upload triple is needed too.
    constexpr GLenum GL_DEPTH_COMPONENT = 0x1902;
    constexpr GLenum GL_UNSIGNED_INT = 0x1405;
    constexpr GLenum GL_TEXTURE1 = 0x84C1;
    constexpr GLenum GL_TEXTURE0_CONT332J = 0x84C0;
    constexpr GLbitfield GL_COLOR_BUFFER_BIT = 0x00004000;
    constexpr GLbitfield GL_DEPTH_BUFFER_BIT = 0x00000100;
    constexpr GLenum GL_READ_FRAMEBUFFER = 0x8CA8;
    constexpr GLenum GL_DRAW_FRAMEBUFFER = 0x8CA9;
    constexpr GLenum GL_PACK_ALIGNMENT = 0x0D05;
    // ★ cont.329 phase 2a: geometry into the target.
    constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
    constexpr GLenum GL_FLOAT = 0x1406;
    constexpr GLenum GL_STREAM_DRAW = 0x88E0;
    constexpr GLenum GL_GEQUAL = 0x0206;
    constexpr GLenum GL_GREATER = 0x0204;
    constexpr GLenum GL_ALWAYS = 0x0207;
    constexpr GLenum GL_NEVER = 0x0200;
    // ★ cont.329 phase 2b: blend factors and equations.
    constexpr GLenum GL_ZERO = 0x0000;
    constexpr GLenum GL_ONE = 0x0001;
    constexpr GLenum GL_SRC_ALPHA = 0x0302;
    constexpr GLenum GL_ONE_MINUS_SRC_ALPHA = 0x0303;
    constexpr GLenum GL_DST_ALPHA = 0x0304;
    constexpr GLenum GL_ONE_MINUS_DST_ALPHA = 0x0305;
    constexpr GLenum GL_CONSTANT_COLOR = 0x8001;
    constexpr GLenum GL_CONSTANT_ALPHA = 0x8003;
    constexpr GLenum GL_ONE_MINUS_CONSTANT_ALPHA = 0x8004;
    constexpr GLenum GL_FUNC_ADD = 0x8006;
    constexpr GLenum GL_FUNC_SUBTRACT = 0x800A;
    // ★ cont.329 phase 3: sampler state.
    constexpr GLenum GL_TEXTURE_WRAP_S = 0x2802;
    constexpr GLenum GL_TEXTURE_WRAP_T = 0x2803;
    constexpr GLenum GL_REPEAT = 0x2901;
    constexpr GLenum GL_CLAMP_TO_EDGE = 0x812F;

    struct GlApi
    {
        const GLubyte *(*GetString)(GLenum) = nullptr;
        GLenum (*GetError)() = nullptr;
        GLuint (*CreateShader)(GLenum) = nullptr;
        void (*ShaderSource)(GLuint, GLsizei, const GLchar *const *, const GLint *) = nullptr;
        void (*CompileShader)(GLuint) = nullptr;
        void (*GetShaderiv)(GLuint, GLenum, GLint *) = nullptr;
        void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *) = nullptr;
        GLuint (*CreateProgram)() = nullptr;
        void (*AttachShader)(GLuint, GLuint) = nullptr;
        void (*LinkProgram)(GLuint) = nullptr;
        void (*GetProgramiv)(GLuint, GLenum, GLint *) = nullptr;
        void (*GetProgramInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *) = nullptr;
        void (*DeleteShader)(GLuint) = nullptr;
        void (*GenBuffers)(GLsizei, GLuint *) = nullptr;
        void (*BindBuffer)(GLenum, GLuint) = nullptr;
        void (*BindBufferBase)(GLenum, GLuint, GLuint) = nullptr;
        void (*BufferData)(GLenum, GLsizeiptr, const void *, GLenum) = nullptr;
        void (*BufferSubData)(GLenum, GLintptr, GLsizeiptr, const void *) = nullptr;
        void (*GetBufferSubData)(GLenum, GLintptr, GLsizeiptr, void *) = nullptr;
        void (*UseProgram)(GLuint) = nullptr;
        GLint (*GetUniformLocation)(GLuint, const GLchar *) = nullptr;
        void (*Uniform1ui)(GLint, GLuint) = nullptr;
        void (*DispatchCompute)(GLuint, GLuint, GLuint) = nullptr;
        void (*MemoryBarrier)(GLbitfield) = nullptr;
        // ★★★★★ cont.332l PS2X_GS_GLR_AFAIL=6: the software-Z AFAIL path binds the depth copy as
        // a read/write image so the fragment shader can do the depth test itself. Loaded
        // NON-fatally (see the loader) -- a driver without it must not disable the renderer.
        void (*BindImageTexture)(GLuint, GLuint, GLint, GLubyte, GLint, GLenum, GLenum) = nullptr;
        void (*Finish)() = nullptr;
        void (*GenQueries)(GLsizei, GLuint *) = nullptr;
        void (*BeginQuery)(GLenum, GLuint) = nullptr;
        void (*EndQuery)(GLenum) = nullptr;
        void (*GetQueryObjectui64v)(GLuint, GLenum, uint64_t *) = nullptr;
        // Hardware-raster path (cont.233).
        void (*GenVertexArrays)(GLsizei, GLuint *) = nullptr;
        void (*BindVertexArray)(GLuint) = nullptr;
        void (*DrawArrays)(GLenum, GLint, GLsizei) = nullptr;
        void (*Enable)(GLenum) = nullptr;
        void (*Disable)(GLenum) = nullptr;
        void (*Viewport)(GLint, GLint, GLsizei, GLsizei) = nullptr;
        void (*GenFramebuffers)(GLsizei, GLuint *) = nullptr;
        void (*BindFramebuffer)(GLenum, GLuint) = nullptr;
        void (*FramebufferParameteri)(GLenum, GLenum, GLint) = nullptr;
        GLenum (*CheckFramebufferStatus)(GLenum) = nullptr;
        void (*Uniform1f)(GLint, float) = nullptr;
        // ★ cont.329 phase 1. The table above is COMPUTE-ONLY plus the cont.233 no-attachment
        // FBO; a renderer needs real textures, attachments, clears, a resolve blit and a readback.
        void (*GenTextures)(GLsizei, GLuint *) = nullptr;
        void (*DeleteTextures)(GLsizei, const GLuint *) = nullptr;
        void (*BindTexture)(GLenum, GLuint) = nullptr;
        void (*ActiveTexture)(GLenum) = nullptr;
        void (*TexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum,
                           const void *) = nullptr;
        void (*TexParameteri)(GLenum, GLenum, GLint) = nullptr;
        void (*FramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint) = nullptr;
        void (*GenRenderbuffers)(GLsizei, GLuint *) = nullptr;
        void (*DeleteRenderbuffers)(GLsizei, const GLuint *) = nullptr;
        void (*BindRenderbuffer)(GLenum, GLuint) = nullptr;
        void (*RenderbufferStorage)(GLenum, GLenum, GLsizei, GLsizei) = nullptr;
        void (*FramebufferRenderbuffer)(GLenum, GLenum, GLenum, GLuint) = nullptr;
        void (*DeleteFramebuffers)(GLsizei, const GLuint *) = nullptr;
        void (*ClearColor)(float, float, float, float) = nullptr;
        void (*Clear)(GLbitfield) = nullptr;
        void (*BlitFramebuffer)(GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint,
                                GLbitfield, GLenum) = nullptr;
        void (*ReadPixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *) = nullptr;
        void (*PixelStorei)(GLenum, GLint) = nullptr;
        // ★ cont.329 phase 2a: vertex submission + the per-draw state a renderer sets.
        void (*VertexAttribPointer)(GLuint, GLint, GLenum, GLubyte, GLsizei, const void *) = nullptr;
        void (*EnableVertexAttribArray)(GLuint) = nullptr;
        void (*DepthFunc)(GLenum) = nullptr;
        void (*DepthMask)(GLubyte) = nullptr;
        void (*ClearDepth)(double) = nullptr;
        void (*ColorMask)(GLubyte, GLubyte, GLubyte, GLubyte) = nullptr;
        void (*Scissor)(GLint, GLint, GLsizei, GLsizei) = nullptr;
        void (*Uniform2f)(GLint, float, float) = nullptr;
        void (*Uniform3f)(GLint, float, float, float) = nullptr;
        void (*DeleteBuffers)(GLsizei, const GLuint *) = nullptr;
        void (*BlendFunc)(GLenum, GLenum) = nullptr;
        void (*BlendEquation)(GLenum) = nullptr;
        void (*BlendColor)(float, float, float, float) = nullptr;
        void (*Uniform1i)(GLint, GLint) = nullptr;
        void (*TexSubImage2D)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum,
                              const void *) = nullptr;
    };

    template <typename Fn>
    bool loadGlFn(Fn &slot, const char *name)
    {
        slot = reinterpret_cast<Fn>(eglGetProcAddress(name));
        return slot != nullptr;
    }

    // ---- GS local-memory layout constants ----
    // Verbatim copies of ps2_gs_memory.cpp's tables (file-static there). These are frozen
    // hardware constants -- the GS block/column raster order (ps2tek; PCSX2 GSLocalMemory.cpp
    // blockTable32/16/16S + columnTable32/16) -- fed to the same InitPageLookupTable template.
    using GSMem::PixelStorageTraits;

    constexpr PixelStorageTraits<GSMem::C32>::BlockLookupTableT kBlockTableC32{{
        {0, 1, 4, 5, 16, 17, 20, 21},
        {2, 3, 6, 7, 18, 19, 22, 23},
        {8, 9, 12, 13, 24, 25, 28, 29},
        {10, 11, 14, 15, 26, 27, 30, 31},
    }};

    constexpr PixelStorageTraits<GSMem::C32>::ColumnLookupTableT kColumnTable32{{
        {0, 1, 4, 5, 8, 9, 12, 13},
        {2, 3, 6, 7, 10, 11, 14, 15},
        {16, 17, 20, 21, 24, 25, 28, 29},
        {18, 19, 22, 23, 26, 27, 30, 31},
        {32, 33, 36, 37, 40, 41, 44, 45},
        {34, 35, 38, 39, 42, 43, 46, 47},
        {48, 49, 52, 53, 56, 57, 60, 61},
        {50, 51, 54, 55, 58, 59, 62, 63},
    }};

    constexpr PixelStorageTraits<GSMem::C16>::BlockLookupTableT kBlockTableC16{{
        {0, 2, 8, 10},
        {1, 3, 9, 11},
        {4, 6, 12, 14},
        {5, 7, 13, 15},
        {16, 18, 24, 26},
        {17, 19, 25, 27},
        {20, 22, 28, 30},
        {21, 23, 29, 31},
    }};

    constexpr PixelStorageTraits<GSMem::C16S>::BlockLookupTableT kBlockTableC16S{{
        {0, 2, 16, 18},
        {1, 3, 17, 19},
        {8, 10, 24, 26},
        {9, 11, 25, 27},
        {4, 6, 20, 22},
        {5, 7, 21, 23},
        {12, 14, 28, 30},
        {13, 15, 29, 31},
    }};

    // Z32/Z24 do NOT share the C32 block order (ps2_gs_memory.cpp BlockTableZ32); the
    // column order IS shared with C32.
    constexpr PixelStorageTraits<GSMem::Z32>::BlockLookupTableT kBlockTableZ32{{
        {24, 25, 28, 29, 8, 9, 12, 13},
        {26, 27, 30, 31, 10, 11, 14, 15},
        {16, 17, 20, 21, 0, 1, 4, 5},
        {18, 19, 22, 23, 2, 3, 6, 7},
    }};

    constexpr PixelStorageTraits<GSMem::P8>::BlockLookupTableT kBlockTableP8{{
        {0, 1, 4, 5, 16, 17, 20, 21},
        {2, 3, 6, 7, 18, 19, 22, 23},
        {8, 9, 12, 13, 24, 25, 28, 29},
        {10, 11, 14, 15, 26, 27, 30, 31},
    }};

    constexpr PixelStorageTraits<GSMem::P4>::BlockLookupTableT kBlockTableP4{{
        {0, 2, 8, 10},
        {1, 3, 9, 11},
        {4, 6, 12, 14},
        {5, 7, 13, 15},
        {16, 18, 24, 26},
        {17, 19, 25, 27},
        {20, 22, 28, 30},
        {21, 23, 29, 31},
    }};

    constexpr PixelStorageTraits<GSMem::P8>::ColumnLookupTableT kColumnTable8{{
        {0, 4, 16, 20, 32, 36, 48, 52, 2, 6, 18, 22, 34, 38, 50, 54},
        {8, 12, 24, 28, 40, 44, 56, 60, 10, 14, 26, 30, 42, 46, 58, 62},
        {33, 37, 49, 53, 1, 5, 17, 21, 35, 39, 51, 55, 3, 7, 19, 23},
        {41, 45, 57, 61, 9, 13, 25, 29, 43, 47, 59, 63, 11, 15, 27, 31},
        {96, 100, 112, 116, 64, 68, 80, 84, 98, 102, 114, 118, 66, 70, 82, 86},
        {104, 108, 120, 124, 72, 76, 88, 92, 106, 110, 122, 126, 74, 78, 90, 94},
        {65, 69, 81, 85, 97, 101, 113, 117, 67, 71, 83, 87, 99, 103, 115, 119},
        {73, 77, 89, 93, 105, 109, 121, 125, 75, 79, 91, 95, 107, 111, 123, 127},
        {128, 132, 144, 148, 160, 164, 176, 180, 130, 134, 146, 150, 162, 166, 178, 182},
        {136, 140, 152, 156, 168, 172, 184, 188, 138, 142, 154, 158, 170, 174, 186, 190},
        {161, 165, 177, 181, 129, 133, 145, 149, 163, 167, 179, 183, 131, 135, 147, 151},
        {169, 173, 185, 189, 137, 141, 153, 157, 171, 175, 187, 191, 139, 143, 155, 159},
        {224, 228, 240, 244, 192, 196, 208, 212, 226, 230, 242, 246, 194, 198, 210, 214},
        {232, 236, 248, 252, 200, 204, 216, 220, 234, 238, 250, 254, 202, 206, 218, 222},
        {193, 197, 209, 213, 225, 229, 241, 245, 195, 199, 211, 215, 227, 231, 243, 247},
        {201, 205, 217, 221, 233, 237, 249, 253, 203, 207, 219, 223, 235, 239, 251, 255},
    }};

    constexpr PixelStorageTraits<GSMem::P4>::ColumnLookupTableT kColumnTable4{{
        {0, 8, 32, 40, 64, 72, 96, 104, 2, 10, 34, 42, 66, 74, 98, 106, 4, 12, 36, 44, 68, 76, 100, 108, 6, 14, 38, 46, 70, 78, 102, 110},
        {16, 24, 48, 56, 80, 88, 112, 120, 18, 26, 50, 58, 82, 90, 114, 122, 20, 28, 52, 60, 84, 92, 116, 124, 22, 30, 54, 62, 86, 94, 118, 126},
        {65, 73, 97, 105, 1, 9, 33, 41, 67, 75, 99, 107, 3, 11, 35, 43, 69, 77, 101, 109, 5, 13, 37, 45, 71, 79, 103, 111, 7, 15, 39, 47},
        {81, 89, 113, 121, 17, 25, 49, 57, 83, 91, 115, 123, 19, 27, 51, 59, 85, 93, 117, 125, 21, 29, 53, 61, 87, 95, 119, 127, 23, 31, 55, 63},
        {192, 200, 224, 232, 128, 136, 160, 168, 194, 202, 226, 234, 130, 138, 162, 170, 196, 204, 228, 236, 132, 140, 164, 172, 198, 206, 230, 238, 134, 142, 166, 174},
        {208, 216, 240, 248, 144, 152, 176, 184, 210, 218, 242, 250, 146, 154, 178, 186, 212, 220, 244, 252, 148, 156, 180, 188, 214, 222, 246, 254, 150, 158, 182, 190},
        {129, 137, 161, 169, 193, 201, 225, 233, 131, 139, 163, 171, 195, 203, 227, 235, 133, 141, 165, 173, 197, 205, 229, 237, 135, 143, 167, 175, 199, 207, 231, 239},
        {145, 153, 177, 185, 209, 217, 241, 249, 147, 155, 179, 187, 211, 219, 243, 251, 149, 157, 181, 189, 213, 221, 245, 253, 151, 159, 183, 191, 215, 223, 247, 255},
        {256, 264, 288, 296, 320, 328, 352, 360, 258, 266, 290, 298, 322, 330, 354, 362, 260, 268, 292, 300, 324, 332, 356, 364, 262, 270, 294, 302, 326, 334, 358, 366},
        {272, 280, 304, 312, 336, 344, 368, 376, 274, 282, 306, 314, 338, 346, 370, 378, 276, 284, 308, 316, 340, 348, 372, 380, 278, 286, 310, 318, 342, 350, 374, 382},
        {321, 329, 353, 361, 257, 265, 289, 297, 323, 331, 355, 363, 259, 267, 291, 299, 325, 333, 357, 365, 261, 269, 293, 301, 327, 335, 359, 367, 263, 271, 295, 303},
        {337, 345, 369, 377, 273, 281, 305, 313, 339, 347, 371, 379, 275, 283, 307, 315, 341, 349, 373, 381, 277, 285, 309, 317, 343, 351, 375, 383, 279, 287, 311, 319},
        {448, 456, 480, 488, 384, 392, 416, 424, 450, 458, 482, 490, 386, 394, 418, 426, 452, 460, 484, 492, 388, 396, 420, 428, 454, 462, 486, 494, 390, 398, 422, 430},
        {464, 472, 496, 504, 400, 408, 432, 440, 466, 474, 498, 506, 402, 410, 434, 442, 468, 476, 500, 508, 404, 412, 436, 444, 470, 478, 502, 510, 406, 414, 438, 446},
        {385, 393, 417, 425, 449, 457, 481, 489, 387, 395, 419, 427, 451, 459, 483, 491, 389, 397, 421, 429, 453, 461, 485, 493, 391, 399, 423, 431, 455, 463, 487, 495},
        {401, 409, 433, 441, 465, 473, 497, 505, 403, 411, 435, 443, 467, 475, 499, 507, 405, 413, 437, 445, 469, 477, 501, 509, 407, 415, 439, 447, 471, 479, 503, 511},
    }};

    constexpr PixelStorageTraits<GSMem::C16>::ColumnLookupTableT kColumnTable16{{
        {0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27},
        {4, 6, 12, 14, 20, 22, 28, 30, 5, 7, 13, 15, 21, 23, 29, 31},
        {32, 34, 40, 42, 48, 50, 56, 58, 33, 35, 41, 43, 49, 51, 57, 59},
        {36, 38, 44, 46, 52, 54, 60, 62, 37, 39, 45, 47, 53, 55, 61, 63},
        {64, 66, 72, 74, 80, 82, 88, 90, 65, 67, 73, 75, 81, 83, 89, 91},
        {68, 70, 76, 78, 84, 86, 92, 94, 69, 71, 77, 79, 85, 87, 93, 95},
        {96, 98, 104, 106, 112, 114, 120, 122, 97, 99, 105, 107, 113, 115, 121, 123},
        {100, 102, 108, 110, 116, 118, 124, 126, 101, 103, 109, 111, 117, 119, 125, 127},
    }};

    constexpr uint32_t kVramBytes = 4u * 1024u * 1024u;
    // A GS page is 8KB for EVERY psm (64x32x4 = 64x64x2 = 128x64x1 = 128x128x0.5), so
    // page granularity is the natural attribution unit for the full-mirror compare.
    constexpr uint32_t kVramPages = kVramBytes / 8192u;
    constexpr uint32_t kHostW = 640u;
    constexpr uint32_t kHostH = 512u;
    // Flat halfword offsets of each page table in the concatenated table SSBO. Phase 0/1
    // needed only the display formats; the phase-2 rasterizer also samples paletted
    // textures (P8/P4 have their own page geometry AND column order) and reads/writes the
    // Z buffer (Z32/Z24 share C32's column order but NOT its block order).
    constexpr uint32_t kTabOffC32 = 0u;
    constexpr uint32_t kTabOffC16 = 32u * 64u * 32u;
    constexpr uint32_t kTabOffC16S = kTabOffC16 + 32u * 64u * 64u;
    constexpr uint32_t kTabOffZ32 = kTabOffC16S + 32u * 64u * 64u;
    constexpr uint32_t kTabOffP8 = kTabOffZ32 + 32u * 64u * 32u;
    constexpr uint32_t kTabOffP4 = kTabOffP8 + 32u * 128u * 64u;
    constexpr uint32_t kTabHalfwords = kTabOffP4 + 32u * 128u * 128u;

    // The kernel: one invocation per host output pixel. Mirrors, in order:
    // PixelStorageTraits<>::PageId/Address (page tables uploaded), ::Read's
    // byte-address mask, and CopyFrameToHostRgbaCpu's CT32/CT24/CT16(S) unpack.
    const char *kDecodeShader = R"GLSL(#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Vram { uint vram[]; };
layout(std430, binding = 1) readonly buffer PageTab { uint ptab[]; };
layout(std430, binding = 2) writeonly buffer OutBuf { uint outPix[]; };
uniform uint uMode;          // 0=CT32 1=CT24 2=CT16 3=CT16S
uniform uint uBlockBase;     // frame base in BLOCKS
uniform uint uBw;            // effective FBW (64-pixel units)
uniform uint uOriginX;
uniform uint uOriginY;
uniform uint uWidth;
uniform uint uHeight;
uniform uint uPreserveAlpha;
uniform uint uTabOff;        // halfword offset of this mode's page table

uint tab16(uint idx)
{
    uint h = uTabOff + idx;
    uint w = ptab[h >> 1u];
    return ((h & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
}

void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= 640u || y >= 512u)
        return;
    uint o = y * 640u + x;
    if (x >= uWidth || y >= uHeight)
    {
        outPix[o] = 0u; // CPU path zero-fills the full host buffer first
        return;
    }
    uint sx = uOriginX + x;
    uint sy = uOriginY + y;
    bool wide = (uMode >= 2u);            // CT16 family: 64x64 page, 16bpp storage
    uint pageH = wide ? 64u : 32u;
    uint pixPerPage = 64u * pageH;
    uint page = (uBlockBase >> 5u) + (sy / pageH) * uBw + (sx >> 6u);
    uint lidx = (uBlockBase & 31u) * pixPerPage + (sy % pageH) * 64u + (sx & 63u);
    uint pixAddr = page * pixPerPage + tab16(lidx);
    uint value;
    if (!wide)
    {
        uint byteAddr = (pixAddr * 4u) & 0x3FFFFCu;
        value = vram[byteAddr >> 2u];
    }
    else
    {
        uint byteAddr = (pixAddr * 2u) & 0x3FFFFEu;
        uint w = vram[byteAddr >> 2u];
        value = ((byteAddr & 2u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
    }
    uint r, g, b, a;
    if (uMode <= 1u)
    {
        r = value & 0xFFu;
        g = (value >> 8u) & 0xFFu;
        b = (value >> 16u) & 0xFFu;
        a = (uMode == 0u && uPreserveAlpha != 0u) ? ((value >> 24u) & 0xFFu) : 255u;
    }
    else
    {
        uint r5 = value & 31u;
        uint g5 = (value >> 5u) & 31u;
        uint b5 = (value >> 10u) & 31u;
        r = (r5 << 3u) | (r5 >> 2u);
        g = (g5 << 3u) | (g5 >> 2u);
        b = (b5 << 3u) | (b5 >> 2u);
        a = (uPreserveAlpha != 0u) ? (((value & 0x8000u) != 0u) ? 0x80u : 0u) : 255u;
    }
    outPix[o] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
)GLSL";

    // Phase-1 upload kernel: one invocation per chunk pixel, replicating
    // UploadImageUnlocked's CT32 path (cursor x = dsax + p%rrw, y = dsay + p/rrw;
    // WriteCT32's page walk + byte-address mask). Each pixel owns its word -- no
    // atomics needed within a chunk.
    const char *kUploadShader = R"GLSL(#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Mirror { uint vram[]; };
layout(std430, binding = 1) readonly buffer Payload { uint pay[]; };
layout(std430, binding = 2) readonly buffer PageTab { uint ptab[]; };
uniform uint uDbp;        // destination base, BLOCKS
uniform uint uDbw;        // buffer width, 64-pixel units
uniform uint uDsax;
uniform uint uDsay;
uniform uint uRrw;
uniform uint uStartPixel;
uniform uint uCount;
uniform uint uTabOff;     // C32 table = 0

uint tab16(uint idx)
{
    uint h = uTabOff + idx;
    uint w = ptab[h >> 1u];
    return ((h & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
}

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= uCount)
        return;
    uint p = uStartPixel + i;
    uint x = uDsax + (p % uRrw);
    uint y = uDsay + (p / uRrw);
    uint page = (uDbp >> 5u) + (y >> 5u) * uDbw + (x >> 6u);
    uint lidx = (uDbp & 31u) * 2048u + (y & 31u) * 64u + (x & 63u);
    uint pixAddr = page * 2048u + tab16(lidx);
    uint byteAddr = (pixAddr * 4u) & 0x3FFFFCu;
    vram[byteAddr >> 2u] = pay[i];
}
)GLSL";

    // Phase-1 verify kernel: compare the mirror at the chunk's addresses against the
    // CPU-read post-upload values; atomically tally mismatches + capture the first
    // divergent index/value pair (16-byte readback instead of the 4MB mirror).
    const char *kVerifyShader = R"GLSL(#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Mirror { uint vram[]; };
layout(std430, binding = 1) readonly buffer Expected { uint expv[]; };
layout(std430, binding = 2) readonly buffer PageTab { uint ptab[]; };
layout(std430, binding = 3) buffer Result { uint mism; uint firstIdx; uint firstGot; uint firstExp; };
uniform uint uDbp;
uniform uint uDbw;
uniform uint uDsax;
uniform uint uDsay;
uniform uint uRrw;
uniform uint uStartPixel;
uniform uint uCount;
uniform uint uTabOff;

uint tab16(uint idx)
{
    uint h = uTabOff + idx;
    uint w = ptab[h >> 1u];
    return ((h & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
}

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= uCount)
        return;
    uint p = uStartPixel + i;
    uint x = uDsax + (p % uRrw);
    uint y = uDsay + (p / uRrw);
    uint page = (uDbp >> 5u) + (y >> 5u) * uDbw + (x >> 6u);
    uint lidx = (uDbp & 31u) * 2048u + (y & 31u) * 64u + (x & 63u);
    uint pixAddr = page * 2048u + tab16(lidx);
    uint byteAddr = (pixAddr * 4u) & 0x3FFFFCu;
    uint got = vram[byteAddr >> 2u];
    if (got != expv[i])
    {
        atomicAdd(mism, 1u);
        if (atomicCompSwap(firstIdx, 0xFFFFFFFFu, i) == 0xFFFFFFFFu)
        {
            firstGot = got;
            firstExp = expv[i];
        }
    }
}
)GLSL";

    // Phase-1b full-mirror compare kernel (cont.168): the whole 4MB mirror against a
    // 4MB upload of CPU-authoritative VRAM, word for word. Besides the total and the
    // first divergence it tallies a per-8KB-PAGE histogram (2048 words/page, 512 pages,
    // the GS page granularity) -- the page numbers name which writer is still
    // unmirrored, which a scalar mismatch count cannot.
    const char *kFullCompareShader = R"GLSL(#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer Mirror { uint mir[]; };
layout(std430, binding = 1) readonly buffer Ref { uint refv[]; };
layout(std430, binding = 2) buffer Result { uint mism; uint firstIdx; uint firstGot; uint firstExp; };
layout(std430, binding = 3) buffer Hist { uint pageHist[]; };
uniform uint uWords;
uniform uint uBase;   // word offset into the mirror (0 = whole-VRAM compare)

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= uWords)
        return;
    uint g = mir[uBase + i];
    uint e = refv[i];
    if (g != e)
    {
        atomicAdd(mism, 1u);
        atomicAdd(pageHist[(uBase + i) >> 11u], 1u);
        if (atomicCompSwap(firstIdx, 0xFFFFFFFFu, i) == 0xFFFFFFFFu)
        {
            firstGot = g;
            firstExp = e;
        }
    }
}
)GLSL";

    // Phase-2a rasterizer (cont.169): ONE primitive per dispatch, one invocation per pixel
    // of its scissor-clipped bbox. Race-free by construction -- within a single primitive
    // each pixel is visited once, and a pixel owns its frame word and its Z word.
    //
    // This is a line-by-line replica of the CPU oracle's pixel pipeline: DrawTriangle's
    // barycentric setup and interpolation, SampleTexture (wrap, bilinear, TEXA, CLUT),
    // combineTexture, classifyAlphaTest, the Z test, the ALPHA blend equation and
    // WritePixel's FBMSK/FBA/preserve-alpha write-out. Where the CPU uses double (the Z
    // interpolation) so does this; where the CPU divides, the host pre-divided (see
    // GsGpuState/GsGpuGeom) -- except SampleTexture's per-pixel 1/q, which is in double and
    // rounded once, since GLSL only guarantees ~2.5 ULP for float division.
    // Shared pixel pipeline: one source of truth for both entry points below.
    const char *kRasterHeader = R"GLSL(#version 430
#extension GL_ARB_gpu_shader_fp64 : enable
)GLSL";
    // The hardware-raster fragment stage (cont.233) prepends `#define MIRROR_QUAL coherent`:
    // ARB_fragment_shader_interlock only guarantees that a critical section's stores are
    // visible to the next critical section on that pixel through `coherent` variables. The
    // compute entry points leave it empty, so their codegen is untouched.
    const char *kRasterCommon = R"GLSL(
#ifndef MIRROR_QUAL
#define MIRROR_QUAL
#endif
// HW_ABL (cont.233 hardware-raster ABLATIONS, never correctness): 1 = coverage only (count +
// discard), 2 = float Z (no fp64), 4 = untextured, 8 = no frame/Z reads, 16 = count covered.
#ifndef HW_ABL
#define HW_ABL 0
#endif
layout(std430, binding = 0) MIRROR_QUAL buffer Mirror { uint vram[]; };
layout(std430, binding = 7) buffer HwCount { uint hwCount[]; };
layout(std430, binding = 1) readonly buffer PageTab { uint ptab[]; };

// Mirrors GsGpuState / GsGpuGeom field-for-field (all 4-byte scalars, so the std430
// layout is unambiguous and matches the C++ structs byte for byte).
struct GsState
{
    uint fbp, fbw, fpsm, fbmsk;
    uint zbp, zpsm, zmsk;
    uint scissorX0, scissorY0, scissorX1, scissorY1;
    uint primType, iip, tme, fge, fst, abe;
    uint ate, atst, aref, afail, zte, ztst;
    uint blendA, blendB, blendC, blendD, blendFix;
    uint fba;
    uint fogR, fogG, fogB;
    uint tbp, tbw, tpsm, tcc, tfx;
    uint cbp, cpsm, csm, csa;
    uint texW, texH, linear;
    uint wms, wmt, minU, maxU, minV, maxV;
    uint ta0, ta1, aem;
    float fstDiv;
    uint clutShadowHave;
    uint linTexOff, pad1, pad2;
    uint clutShadow[16];
};

struct GsGeom
{
    float x0, y0, x1, y1, x2, y2;
    uint z0lo, z0hi, z1lo, z1hi, z2lo, z2hi;
    uint rgba0, rgba1, rgba2;
    float q0, q1, q2;
    float s0, s1, s2;
    float t0, t1, t2;
    uint uv0, uv1, uv2;
    uint fog0, fog1, fog2;
    float winding, invAbsDenom;
    uint minX, minY, maxX, maxY;
    uint spriteZ;
    uint unclipX0, unclipY0;
    float spriteW, spriteH;
    float su0, sv0, su1, sv1;
    uint stateIndex;
    uint pad0, pad1, pad2;
};

layout(std430, binding = 2) readonly buffer StateBuf { GsState states[]; };
// ★ cont.250 PS2X_GS_LINTEX: textures decoded host-side into TEXEL-FINAL form (swizzle resolved,
// CLUT applied, TEXA applied). Only read when a state's linTexOff is not the sentinel, which the
// host sets only for states it actually decoded -- so programs that never bind this are unaffected.
layout(std430, binding = 8) readonly buffer LinTex { uint linTexels[]; };
layout(std430, binding = 3) readonly buffer GeomBuf { GsGeom prims[]; };
layout(std430, binding = 4) readonly buffer BinOff { uint binOffsets[]; };
layout(std430, binding = 5) readonly buffer BinEnt { uint binEntries[]; };
layout(std430, binding = 6) readonly buffer TileLst { uint tileList[]; };

// Set once per primitive; the S./G. macros read through them so the pixel pipeline below
// is written exactly once and serves both the single-primitive and the tiled entry point.
uint gStateIdx = 0u;
uint gPrimIdx = 0u;
#define S states[gStateIdx]
#define G prims[gPrimIdx]

// Page-table offsets, in halfwords, matching the host's kTabOff* constants.
const uint TAB_C32  = 0u;
const uint TAB_C16  = 65536u;
const uint TAB_C16S = 196608u;
const uint TAB_Z32  = 327680u;
const uint TAB_P8   = 393216u;
const uint TAB_P4   = 655360u;

uint tab16(uint tabOff, uint idx)
{
    uint h = tabOff + idx;
    uint w = ptab[h >> 1u];
    return ((h & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
}

// PixelStorageTraits<>::Address: page id (monotonic in x/y) * pixels-per-page + the
// in-page lookup. bw is in 64-pixel units; T8/T4's wider pages divide it down.
uint pixelAddr(uint tabOff, uint bp, uint bw, uint x, uint y, uint pw, uint ph)
{
    uint pixPerPage = pw * ph;
    uint page = (bp >> 5u) + (y / ph) * ((max(bw, 1u) * 64u) / pw) + (x / pw);
    uint lidx = (bp & 31u) * pixPerPage + (y % ph) * pw + (x % pw);
    return page * pixPerPage + tab16(tabOff, lidx);
}

uint loadByte(uint byteAddr)
{
    return (vram[byteAddr >> 2u] >> ((byteAddr & 3u) * 8u)) & 0xFFu;
}

uint rgba5551to8888(uint c)
{
    uint r = (c & 31u) << 3u;
    uint g = ((c >> 5u) & 31u) << 3u;
    uint b = ((c >> 10u) & 31u) << 3u;
    uint a = ((c >> 15u) & 1u) << 7u;
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

// GSMem::Read<psm> for every format the sampler and the frame/Z paths can see.
uint readVram(uint psm, uint bp, uint bw, uint x, uint y)
{
    if (psm == 2u || psm == 10u || psm == 50u || psm == 58u) // CT16 / CT16S / Z16 / Z16S
    {
        uint tab = (psm == 10u || psm == 58u) ? TAB_C16S : TAB_C16;
        uint pa = pixelAddr(tab, bp, bw, x, y, 64u, 64u);
        uint byteAddr = (pa * 2u) & 0x3FFFFEu;
        uint w = vram[byteAddr >> 2u];
        return ((byteAddr & 2u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
    }
    if (psm == 19u) // T8
    {
        uint pa = pixelAddr(TAB_P8, bp, bw, x, y, 128u, 64u);
        return loadByte(pa & 0x3FFFFFu);
    }
    if (psm == 20u) // T4
    {
        uint pa = pixelAddr(TAB_P4, bp, bw, x, y, 128u, 128u);
        uint bits = pa * 4u;
        uint byteAddr = (bits >> 3u) & 0x3FFFFFu;
        return (loadByte(byteAddr) >> (bits & 7u)) & 0x0Fu;
    }
    // Everything else rides the C32 page layout (Z32/Z24 use their own block order).
    uint tab = (psm == 48u || psm == 49u) ? TAB_Z32 : TAB_C32;
    uint pa = pixelAddr(tab, bp, bw, x, y, 64u, 32u);
    if (psm == 27u) // T8H: the high byte of the word
        return loadByte((pa * 4u + 3u) & 0x3FFFFFu);
    if (psm == 36u) // T4HL: low nibble of the high byte
        return loadByte((pa * 4u + 3u) & 0x3FFFFFu) & 0x0Fu;
    if (psm == 44u) // T4HH: high nibble of the high byte
        return (loadByte((pa * 4u + 3u) & 0x3FFFFFu) >> 4u) & 0x0Fu;
    uint word = vram[((pa * 4u) & 0x3FFFFCu) >> 2u];
    if (psm == 1u || psm == 49u) // CT24 / Z24
        return word & 0x00FFFFFFu;
    return word;
}

// GSMem::Write<psm> for the frame/Z formats. Only one invocation ever targets a given
// pixel within a single-primitive dispatch, so a plain read-modify-write is safe for the
// sub-word formats -- EXCEPT the 16-bit ones, where two pixels share a word and the RMW
// needs an atomic. (This game's census has CT32 frames and Z24 only; the 16-bit arm is
// there so an unexpected format is still correct rather than silently wrong.)
void writeVramPsm(uint psm, uint bp, uint bw, uint x, uint y, uint value)
{
    if (psm == 2u || psm == 10u || psm == 50u || psm == 58u) // CT16 / CT16S / Z16 / Z16S
    {
        uint tab = (psm == 10u || psm == 58u) ? TAB_C16S : TAB_C16;
        uint pa = pixelAddr(tab, bp, bw, x, y, 64u, 64u);
        uint byteAddr = (pa * 2u) & 0x3FFFFEu;
        uint wi = byteAddr >> 2u;
        if ((byteAddr & 2u) == 0u)
            atomicAnd(vram[wi], 0xFFFF0000u), atomicOr(vram[wi], value & 0xFFFFu);
        else
            atomicAnd(vram[wi], 0x0000FFFFu), atomicOr(vram[wi], (value & 0xFFFFu) << 16u);
        return;
    }
    uint tab = (psm == 48u || psm == 49u) ? TAB_Z32 : TAB_C32;
    uint pa = pixelAddr(tab, bp, bw, x, y, 64u, 32u);
    uint wi = ((pa * 4u) & 0x3FFFFCu) >> 2u;
    if (psm == 1u || psm == 49u) // CT24 / Z24 keep the top byte (RMW)
        vram[wi] = (vram[wi] & 0xFF000000u) | (value & 0x00FFFFFFu);
    else
        vram[wi] = value;
}

uint applyTexa(uint psm, uint texel)
{
    if (psm == 0u) // CT32 passes through
        return texel;
    bool rgbZero = (texel & 0x00FFFFFFu) == 0u;
    uint a = (texel >> 24u) & 0xFFu;
    if (psm == 1u) // CT24
        a = (S.aem != 0u && rgbZero) ? 0u : S.ta0;
    else if (psm == 2u || psm == 10u) // CT16 / CT16S
        a = ((a & 0x80u) != 0u) ? S.ta1 : ((S.aem != 0u && rgbZero) ? 0u : S.ta0);
    return (texel & 0x00FFFFFFu) | (a << 24u);
}

uint resolveClutIndex(uint index)
{
    if (S.csm != 0u)
        return (S.tpsm == 20u || S.tpsm == 44u || S.tpsm == 36u) ? (index & 0x0Fu) : index;
    bool is16 = (S.cpsm == 2u || S.cpsm == 10u);
    uint csaMask = is16 ? 0x1Fu : 0x0Fu;
    uint idxMask = is16 ? 0x1FFu : 0x0FFu;
    uint clutBase = (S.csa & csaMask) << 4u;
    uint ci;
    if (S.tpsm == 20u || S.tpsm == 44u || S.tpsm == 36u)
        ci = clutBase + (index & 0x0Fu);
    else if (S.tpsm == 19u || S.tpsm == 27u)
        ci = clutBase + index;
    else
        return index;
    ci = ci & idxMask;
    // CSM1 swaps address bits 3 and 4.
    return (ci & ~0x18u) | ((ci & 0x08u) << 1u) | ((ci & 0x10u) >> 1u);
}

uint lookupClut(uint index)
{
    uint ci = resolveClutIndex(index);
    bool csm2 = (S.csm != 0u);
    uint clutWidth = 1u; // CSM1 ignores TEXCLUT; CSM2 does not occur in this game
    uint clutX = ci & 0x0Fu;
    uint clutY = ci >> 4u;
    if (S.cpsm == 0u)
    {
        uint entry = readVram(0u, S.cbp, clutWidth, clutX, clutY);
        // CLUT shadow: an entry whose RGB was zeroed by a colliding render target while
        // its alpha still matches the uploaded payload is served from the upload.
        if (S.clutShadowHave != 0u && !csm2 && (entry & 0x00FFFFFFu) == 0u &&
            clutX < 8u && clutY < 2u)
        {
            uint up = S.clutShadow[clutY * 8u + clutX];
            if ((up & 0xFF000000u) == (entry & 0xFF000000u) && (up & 0x00FFFFFFu) != 0u)
                entry = up;
        }
        return applyTexa(0u, entry);
    }
    if (S.cpsm == 1u)
        return applyTexa(1u, readVram(1u, S.cbp, clutWidth, clutX, clutY));
    if (S.cpsm == 2u || S.cpsm == 10u)
        return applyTexa(S.cpsm, rgba5551to8888(readVram(S.cpsm, S.cbp, clutWidth, clutX, clutY)));
    return 0xFFFF00FFu;
}

int wrapCoord(int c, int size, uint mode, uint rMin, uint rMax)
{
    if (mode == 0u)
        return int(uint(c) & uint(size - 1));
    if (mode == 1u)
        return clamp(c, 0, size - 1);
    if (mode == 2u)
        return min(max(c, int(rMin)), int(rMax));
    return int((uint(c) & rMin) | rMax);
}

uint samplePoint(int su, int sv)
{
    su = wrapCoord(su, int(S.texW), S.wms, S.minU, S.maxU);
    sv = wrapCoord(sv, int(S.texH), S.wmt, S.minV, S.maxV);
    // ★ cont.250 PS2X_GS_LINTEX: one direct indexed read instead of a swizzle-table lookup plus a
    // DEPENDENT CLUT read. Bit-exact by construction -- the decode reproduces exactly what the lines
    // below would return, and wrapCoord/lerpChan are untouched, so vramHash still verifies this.
    // The host only sets linTexOff when wms<2 && wmt<2, where wrapCoord provably lands in [0,size).
    if (S.linTexOff != 0xFFFFFFFFu)
        return linTexels[S.linTexOff + uint(sv) * S.texW + uint(su)];
    uint raw = readVram(S.tpsm, S.tbp, S.tbw, uint(su), uint(sv));
    if (S.tpsm == 0u || S.tpsm == 48u || S.tpsm == 1u || S.tpsm == 49u)
        return applyTexa(S.tpsm, raw);
    if (S.tpsm == 2u || S.tpsm == 10u || S.tpsm == 50u || S.tpsm == 58u)
        return applyTexa(S.tpsm, rgba5551to8888(raw));
    if (S.tpsm == 19u || S.tpsm == 27u || S.tpsm == 20u || S.tpsm == 36u || S.tpsm == 44u)
        return lookupClut(raw);
    return applyTexa(0u, raw); // undefined psm decodes as CT32
}

// GLSL only guarantees ~2.5 ULP for float division, but the CPU oracle divides with
// correctly-rounded IEEE. Rounding a correctly-rounded fp64 quotient to float reproduces
// the float divide exactly (double rounding is innocuous for division: 53 >= 2*24+2).
float exactDiv(float a, float b)
{
    return float(double(a) / double(b));
}

uint lerpChan(uint c00, uint c10, uint c01, uint c11, float fx, float fy)
{
    // `precise` is load-bearing, not decoration: GLSL lets an implementation contract
    // a*b+c into an FMA, and the CPU oracle (built -msse4.1, so no FMA at all) does not.
    // A sub-ULP difference here lands on the other side of floor(x+0.5) and shows up as a
    // +-1 LSB colour channel -- exactly the 634-word divergence run181b measured.
    precise float top = float(c00) + (float(c10) - float(c00)) * fx;
    precise float bot = float(c01) + (float(c11) - float(c01)) * fx;
    precise float mid = top + (bot - top) * fy;
    // std::lround = half away from zero; GLSL round() picks its direction at .5, so spell
    // it out. Channel lerps of 0..255 endpoints with fx,fy in [0,1) never go negative.
    return uint(clamp(int(floor(mid + 0.5)), 0, 255));
}

uint sampleTexture(float s, float t, float q, uint uu, uint vv)
{
    precise float texUf, texVf;
    if (S.fst != 0u)
    {
        texUf = float(uu) / S.fstDiv;
        texVf = float(vv) / S.fstDiv;
    }
    else
    {
        float aq = (abs(q) > 1.0e-8) ? q : 1.0;
        float invQ = exactDiv(1.0, aq);
        texUf = s * invQ * float(S.texW);
        texVf = t * invQ * float(S.texH);
        // (both already `precise` by declaration -- see lerpChan's note)
    }
    if (S.linear == 0u)
        return samplePoint(int(texUf), int(texVf));

    precise float su = texUf - 0.5;
    precise float sv = texVf - 0.5;
    int u0 = int(floor(su));
    int v0 = int(floor(sv));
    precise float fx = su - float(u0);
    precise float fy = sv - float(v0);
    uint c00 = samplePoint(u0, v0);
    uint c10 = samplePoint(u0 + 1, v0);
    uint c01 = samplePoint(u0, v0 + 1);
    uint c11 = samplePoint(u0 + 1, v0 + 1);
    uint r = lerpChan(c00 & 0xFFu, c10 & 0xFFu, c01 & 0xFFu, c11 & 0xFFu, fx, fy);
    uint g = lerpChan((c00 >> 8u) & 0xFFu, (c10 >> 8u) & 0xFFu, (c01 >> 8u) & 0xFFu, (c11 >> 8u) & 0xFFu, fx, fy);
    uint b = lerpChan((c00 >> 16u) & 0xFFu, (c10 >> 16u) & 0xFFu, (c01 >> 16u) & 0xFFu, (c11 >> 16u) & 0xFFu, fx, fy);
    uint a = lerpChan((c00 >> 24u) & 0xFFu, (c10 >> 24u) & 0xFFu, (c01 >> 24u) & 0xFFu, (c11 >> 24u) & 0xFFu, fx, fy);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

bool passesAlphaTest(uint a)
{
    if (S.ate == 0u)
        return true;
    if (S.atst == 0u) return false;
    if (S.atst == 1u) return true;
    if (S.atst == 2u) return a < S.aref;
    if (S.atst == 3u) return a <= S.aref;
    if (S.atst == 4u) return a == S.aref;
    if (S.atst == 5u) return a >= S.aref;
    if (S.atst == 6u) return a > S.aref;
    return a != S.aref;
}

int pickRGB(uint sel, int cs, int cd)
{
    if (sel == 0u) return cs;
    if (sel == 1u) return cd;
    return 0;
}

// GSCpuBackend::WritePixel -- shared by the triangle and sprite paths. The caller has
// already resolved the pixel's colour, its fog value (interpolated for a triangle, flat
// from v1 for a sprite) and its integer Z (rounded +0.5 for a triangle, truncated for a
// sprite), because those three are the only places the two primitive types differ once
// coverage is decided.
void writePixel(uint px_i, uint py_i, uint r, uint g, uint b, uint a, uint fog, uint zval)
{
    if (px_i < S.scissorX0 || px_i > S.scissorX1 || py_i < S.scissorY0 || py_i > S.scissorY1)
        return;

    if (S.fge != 0u)
    {
        uint inv = 255u - fog;
        r = (((fog * r) >> 8u) + ((inv * S.fogR) >> 8u)) & 0xFFu;
        g = (((fog * g) >> 8u) + ((inv * S.fogG) >> 8u)) & 0xFFu;
        b = (((fog * b) >> 8u) + ((inv * S.fogB) >> 8u)) & 0xFFu;
    }

    // classifyAlphaTest
    bool writeRgb = true, writeAlpha = true, writeDepth = true;
    if (!passesAlphaTest(a))
    {
        if (S.afail == 1u)      { writeRgb = true;  writeAlpha = true;  writeDepth = false; }
        else if (S.afail == 2u) { writeRgb = false; writeAlpha = false; writeDepth = true;  }
        else if (S.afail == 3u)
        {
            if (S.fpsm == 0u)   { writeRgb = true;  writeAlpha = false; writeDepth = false; }
            else                { writeRgb = true;  writeAlpha = true;  writeDepth = false; }
        }
        else                    { writeRgb = false; writeAlpha = false; writeDepth = false; }
    }
    bool writesFb = writeRgb || writeAlpha;
    if (!writesFb && !writeDepth)
        return;

    bool preserveDstAlpha = writeRgb && !writeAlpha && S.fpsm == 0u;
    bool frmw = writesFb && ((S.fbmsk != 0u) || S.abe != 0u || preserveDstAlpha);
#if (HW_ABL & 8)
    frmw = false;
#endif
    uint fbrgba = 0u;
    if (frmw)
    {
        uint raw = readVram(S.fpsm, S.fbp, S.fbw, px_i, py_i);
        fbrgba = raw;
        if (S.fpsm == 2u || S.fpsm == 10u)
            fbrgba = rgba5551to8888(raw);
        else if (S.fpsm == 1u)
            fbrgba = raw | 0x80000000u; // the GS supplies 0x80 dest alpha for RGB24
    }

    bool zpass = false;
    if (S.ztst == 1u)
        zpass = true;
    else if (S.ztst == 2u || S.ztst == 3u)
    {
#if (HW_ABL & 8)
        zpass = true;
#else
        uint stored = readVram(S.zpsm, S.zbp, S.fbw, px_i, py_i);
        zpass = (S.ztst == 2u) ? (zval >= stored) : (zval > stored);
#endif
    }
    if (!zpass)
        return;

    if (writesFb)
    {
        if (S.abe != 0u)
        {
            int dr = int(fbrgba & 0xFFu), dg = int((fbrgba >> 8u) & 0xFFu);
            int db = int((fbrgba >> 16u) & 0xFFu), da = int((fbrgba >> 24u) & 0xFFu);
            int cAlpha = (S.blendC == 0u) ? int(a) : ((S.blendC == 1u) ? da : int(S.blendFix));
            int sr = int(r), sg = int(g), sb = int(b);
            r = uint(clamp(((pickRGB(S.blendA, sr, dr) - pickRGB(S.blendB, sr, dr)) * cAlpha >> 7) + pickRGB(S.blendD, sr, dr), 0, 255));
            g = uint(clamp(((pickRGB(S.blendA, sg, dg) - pickRGB(S.blendB, sg, dg)) * cAlpha >> 7) + pickRGB(S.blendD, sg, dg), 0, 255));
            b = uint(clamp(((pickRGB(S.blendA, sb, db) - pickRGB(S.blendB, sb, db)) * cAlpha >> 7) + pickRGB(S.blendD, sb, db), 0, 255));
        }
        if (writeAlpha && (S.fba & 1u) != 0u && S.fpsm != 1u)
            a |= 0x80u;
        uint pixel = r | (g << 8u) | (b << 16u) | (a << 24u);
        if (S.fbmsk != 0u)
            pixel = (pixel & ~S.fbmsk) | (fbrgba & S.fbmsk);
        if (preserveDstAlpha)
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        if (S.fpsm == 2u || S.fpsm == 10u)
        {
            uint pr = ((pixel      ) & 0xFFu) >> 3u;
            uint pg = ((pixel >>  8) & 0xFFu) >> 3u;
            uint pb = ((pixel >> 16) & 0xFFu) >> 3u;
            uint pa = ((pixel >> 24) & 0xFFu) >> 7u;
            pixel = pr | (pg << 5u) | (pb << 10u) | (pa << 15u);
        }
        writeVramPsm(S.fpsm, S.fbp, S.fbw, px_i, py_i, pixel);
    }

    if (writeDepth && S.zmsk == 0u)
        writeVramPsm(S.zpsm, S.zbp, S.fbw, px_i, py_i, zval);
}


// Shade one pixel of the primitive currently selected by gPrimIdx / gStateIdx. The caller
// has already established that the pixel lies inside that primitive's bbox. Written once
// and shared by every entry point -- a duplicated pixel pipeline would drift, and
// bit-exactness against the CPU oracle is the whole point.
//
// cont.233: split in two. shadePrimColor is everything up to the framebuffer read-modify-
// write -- coverage, interpolation, the texture sample, the combine, fog input -- and needs
// no ordering at all (a batch never samples pages an earlier primitive of the same batch
// wrote: the accumulator flushes on that hazard). writePixel is the RMW that must run in
// primitive order per pixel. The hardware-raster fragment stage locks ONLY the latter.
bool shadePrimColor(uint px_i, uint py_i, out uint oR, out uint oG, out uint oB, out uint oA,
                    out uint oFog, out uint oZ)
{
    // ---- SPRITE (DrawSprite): axis-aligned, flat shaded, screen-space UV lerp ----
    // The host already clamped minX/minY/maxX/maxY to the scissor-intersected rect and
    // rejected a fully-off-screen sprite, so every invocation here is covered.
    if (S.primType == 6u)
    {
        uint sr, sg, sb, sa;
        sr = G.rgba1 & 0xFFu;
        sg = (G.rgba1 >> 8u) & 0xFFu;
        sb = (G.rgba1 >> 16u) & 0xFFu;
        sa = (G.rgba1 >> 24u) & 0xFFu;
#if (HW_ABL & 4)
        if (false)
#else
        if (S.tme != 0u)
#endif
        {
            precise float tx = exactDiv(float(px_i - G.unclipX0) + 0.5, G.spriteW);
            precise float ty = exactDiv(float(py_i - G.unclipY0) + 0.5, G.spriteH);
            precise float texUf = G.su0 + (G.su1 - G.su0) * tx;
            precise float texVf = G.sv0 + (G.sv1 - G.sv0) * ty;
            float is = 0.0, it = 0.0, iq = 1.0;
            uint iu = 0u, iv = 0u;
            if (S.fst != 0u)
            {
                // DrawSprite re-quantizes to 4.4 fixed point and hands SampleTexture a UV,
                // which divides by 16 again -- replicate the round trip, not a shortcut.
                iu = uint(clamp(int(texUf * 16.0 + 0.5), 0, 0xFFFF));
                iv = uint(clamp(int(texVf * 16.0 + 0.5), 0, 0xFFFF));
            }
            else
            {
                // ...and in the non-FST case it divides by the texture size, which
                // SampleTexture then multiplies back. Not identity in float: do both.
                is = exactDiv(texUf, float(S.texW));
                it = exactDiv(texVf, float(S.texH));
            }
            uint texel = sampleTexture(is, it, iq, iu, iv);
            uint tr = texel & 0xFFu, tg = (texel >> 8u) & 0xFFu;
            uint tb = (texel >> 16u) & 0xFFu, ta = (texel >> 24u) & 0xFFu;
            bool hasAlpha = S.tcc != 0u;
            uint nr, ng, nb, na;
            if (S.tfx == 0u)
            {
                nr = uint(clamp(int((tr * sr) >> 7u), 0, 255));
                ng = uint(clamp(int((tg * sg) >> 7u), 0, 255));
                nb = uint(clamp(int((tb * sb) >> 7u), 0, 255));
                na = hasAlpha ? uint(clamp(int((ta * sa) >> 7u), 0, 255)) : sa;
            }
            else if (S.tfx == 1u)
            {
                nr = tr; ng = tg; nb = tb; na = hasAlpha ? ta : sa;
            }
            else if (S.tfx == 2u)
            {
                nr = uint(clamp(int(((tr * sr) >> 7u) + sa), 0, 255));
                ng = uint(clamp(int(((tg * sg) >> 7u) + sa), 0, 255));
                nb = uint(clamp(int(((tb * sb) >> 7u) + sa), 0, 255));
                na = hasAlpha ? uint(clamp(int(ta + sa), 0, 255)) : sa;
            }
            else
            {
                nr = uint(clamp(int(((tr * sr) >> 7u) + sa), 0, 255));
                ng = uint(clamp(int(((tg * sg) >> 7u) + sa), 0, 255));
                nb = uint(clamp(int(((tb * sb) >> 7u) + sa), 0, 255));
                na = hasAlpha ? ta : sa;
            }
            sr = nr; sg = ng; sb = nb; sa = na;
        }
        // Flat Z and fog from v1; note DrawSprite truncates v1.z with no +0.5 rounding.
        oR = sr; oG = sg; oB = sb; oA = sa; oFog = G.fog1; oZ = G.spriteZ;
        return true;
    }

    float px = float(px_i) + 0.5;
    float py = float(py_i) + 0.5;

    // DrawTriangle's barycentric coverage test, on the same floats in the same order.
    precise float w0 = (((G.y1 - G.y2) * (px - G.x2) + (G.x2 - G.x1) * (py - G.y2)) * G.winding) * G.invAbsDenom;
    precise float w1 = (((G.y2 - G.y0) * (px - G.x2) + (G.x0 - G.x2) * (py - G.y2)) * G.winding) * G.invAbsDenom;
    precise float w2 = 1.0 - w0 - w1;
    const float kEdgeEpsilon = 1.0e-4;
    if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
        return false;

    // The oracle interpolates Z in double from float weights; match exactly.
#if (HW_ABL & 2)
    precise float zd = float(G.z0hi) * w0 + float(G.z1hi) * w1 + float(G.z2hi) * w2; // garbage, cheap
#else
    double zd = packDouble2x32(uvec2(G.z0lo, G.z0hi)) * double(w0) +
                packDouble2x32(uvec2(G.z1lo, G.z1hi)) * double(w1) +
                packDouble2x32(uvec2(G.z2lo, G.z2hi)) * double(w2);
#endif

    uint r, g, b, a;
    if (S.iip != 0u)
    {
        precise float ir = float((G.rgba0      ) & 0xFFu) * w0 + float((G.rgba1      ) & 0xFFu) * w1 + float((G.rgba2      ) & 0xFFu) * w2;
        precise float ig = float((G.rgba0 >>  8u) & 0xFFu) * w0 + float((G.rgba1 >>  8u) & 0xFFu) * w1 + float((G.rgba2 >>  8u) & 0xFFu) * w2;
        precise float ib = float((G.rgba0 >> 16u) & 0xFFu) * w0 + float((G.rgba1 >> 16u) & 0xFFu) * w1 + float((G.rgba2 >> 16u) & 0xFFu) * w2;
        precise float ia = float((G.rgba0 >> 24u) & 0xFFu) * w0 + float((G.rgba1 >> 24u) & 0xFFu) * w1 + float((G.rgba2 >> 24u) & 0xFFu) * w2;
        r = uint(clamp(int(ir), 0, 255));
        g = uint(clamp(int(ig), 0, 255));
        b = uint(clamp(int(ib), 0, 255));
        a = uint(clamp(int(ia), 0, 255));
    }
    else
    {
        r = G.rgba2 & 0xFFu;
        g = (G.rgba2 >> 8u) & 0xFFu;
        b = (G.rgba2 >> 16u) & 0xFFu;
        a = (G.rgba2 >> 24u) & 0xFFu;
    }

#if (HW_ABL & 4)
    if (false)
#else
    if (S.tme != 0u)
#endif
    {
        precise float is = 0.0, it = 0.0, iq = 1.0;
        uint iu = 0u, iv = 0u;
        if (S.fst != 0u)
        {
            precise float fu = float(G.uv0 & 0xFFFFu) * w0 + float(G.uv1 & 0xFFFFu) * w1 + float(G.uv2 & 0xFFFFu) * w2;
            precise float fv = float(G.uv0 >> 16u) * w0 + float(G.uv1 >> 16u) * w1 + float(G.uv2 >> 16u) * w2;
            iu = uint(fu) & 0xFFFFu;
            iv = uint(fv) & 0xFFFFu;
        }
        else
        {
            // The GS DDA interpolates homogeneous S/T/Q; texel coords come from S/Q, T/Q
            // only AFTER interpolation.
            is = G.s0 * w0 + G.s1 * w1 + G.s2 * w2;
            it = G.t0 * w0 + G.t1 * w1 + G.t2 * w2;
            iq = G.q0 * w0 + G.q1 * w1 + G.q2 * w2;
        }
        uint texel = sampleTexture(is, it, iq, iu, iv);
        uint tr = texel & 0xFFu, tg = (texel >> 8u) & 0xFFu;
        uint tb = (texel >> 16u) & 0xFFu, ta = (texel >> 24u) & 0xFFu;
        bool hasAlpha = S.tcc != 0u;
        uint nr, ng, nb, na;
        if (S.tfx == 0u) // MODULATE
        {
            nr = uint(clamp(int((tr * r) >> 7u), 0, 255));
            ng = uint(clamp(int((tg * g) >> 7u), 0, 255));
            nb = uint(clamp(int((tb * b) >> 7u), 0, 255));
            na = hasAlpha ? uint(clamp(int((ta * a) >> 7u), 0, 255)) : a;
        }
        else if (S.tfx == 1u) // DECAL
        {
            nr = tr; ng = tg; nb = tb; na = hasAlpha ? ta : a;
        }
        else if (S.tfx == 2u) // HIGHLIGHT
        {
            nr = uint(clamp(int(((tr * r) >> 7u) + a), 0, 255));
            ng = uint(clamp(int(((tg * g) >> 7u) + a), 0, 255));
            nb = uint(clamp(int(((tb * b) >> 7u) + a), 0, 255));
            na = hasAlpha ? uint(clamp(int(ta + a), 0, 255)) : a;
        }
        else // HIGHLIGHT2 (tfx is 2 bits, so this is the only remaining case)
        {
            nr = uint(clamp(int(((tr * r) >> 7u) + a), 0, 255));
            ng = uint(clamp(int(((tg * g) >> 7u) + a), 0, 255));
            nb = uint(clamp(int(((tb * b) >> 7u) + a), 0, 255));
            na = hasAlpha ? ta : a;
        }
        r = nr; g = ng; b = nb; a = na;
    }

    // Fog is interpolated per pixel for a triangle (a sprite takes v1's flat value), and
    // Z carries the +0.5 rounding DrawTriangle applies at its WritePixel call site.
    uint fogv = 0u;
    if (S.fge != 0u)
    {
        precise float ifog = float(G.fog0) * w0 + float(G.fog1) * w1 + float(G.fog2) * w2;
        fogv = uint(clamp(int(ifog), 0, 255));
    }
#if (HW_ABL & 2)
    oZ = uint(zd + 0.5);
#else
    oZ = uint(zd + 0.5lf);
#endif
    oR = r; oG = g; oB = b; oA = a; oFog = fogv;
    return true;
}

void shadePrimPixel(uint px_i, uint py_i)
{
    uint r, g, b, a, fog, z;
    if (shadePrimColor(px_i, py_i, r, g, b, a, fog, z))
        writePixel(px_i, py_i, r, g, b, a, fog, z);
}
)GLSL";

    // Entry point A -- ONE primitive, dispatched over its bbox (the phase-2a shadow-verify
    // path). A batch of one.
    const char *kRasterMainPrim = R"GLSL(
layout(local_size_x = 8, local_size_y = 8) in;
void main()
{
    gPrimIdx = 0u;
    gStateIdx = prims[0].stateIndex;
    uint px_i = G.minX + gl_GlobalInvocationID.x;
    uint py_i = G.minY + gl_GlobalInvocationID.y;
    if (px_i > G.maxX || py_i > G.maxY)
        return;
    shadePrimPixel(px_i, py_i);
}
)GLSL";

    // Entry point B -- a BATCH, one workgroup per non-empty 16x16 tile, one thread per
    // pixel. Each thread walks its tile's primitive list in SUBMISSION ORDER, which is the
    // only ordering the GS actually requires: pixels are independent of one another, so
    // there is no barrier here at all. Tiles are VRAM-disjoint because a batch never spans
    // a render-target change (enforced host-side).
    const char *kRasterMainTile = R"GLSL(
layout(local_size_x = 16, local_size_y = 16) in;
uniform uint uTilesX;
void main()
{
    uint tile = tileList[gl_WorkGroupID.x];
    uint px_i = (tile % uTilesX) * 16u + gl_LocalInvocationID.x;
    uint py_i = (tile / uTilesX) * 16u + gl_LocalInvocationID.y;
    uint b = binOffsets[tile];
    uint e = binOffsets[tile + 1u];
    for (uint i = b; i < e; ++i)
    {
        gPrimIdx = binEntries[i];
        if (px_i < G.minX || px_i > G.maxX || py_i < G.minY || py_i > G.maxY)
            continue;
        gStateIdx = G.stateIndex;
        shadePrimPixel(px_i, py_i);
    }
}
)GLSL";

    // ---- Hardware rasterization (cont.233, PS2X_GS_HWRASTER) ----
    //
    // The tile kernel above is bit-exact but slow BY DESIGN: every thread of a 16x16 tile
    // walks every primitive binned there (~91M thread-iterations per level frame for ~1.2M
    // covered pixels). Here the GPU's own rasterizer does the walking: one glDrawArrays per
    // batch, one hardware triangle per primitive (no vertex buffers -- the vertex stage reads
    // the same GsGeom SSBO by gl_VertexID/3), so fragment invocations run only for pixels the
    // primitive touches. The fragment stage is the SAME shadePrimPixel/writePixel pipeline,
    // wrapped in an ARB_fragment_shader_interlock critical section with the
    // pixel_interlock_ordered qualifier: the spec guarantees that overlapping fragments run
    // their critical sections in primitive order, across draw calls too, which is exactly the
    // per-pixel submission order the GS requires -- the only ordering it requires.
    //
    // Coverage stays the CPU's decision, not the hardware's. The CPU oracle covers a pixel
    // when its centre passes the barycentric test with a 1e-4 epsilon (a band of up to
    // ~0.2 px OUTSIDE the true edges, and a longer spike past an acute apex), so the
    // hardware must merely produce a fragment for every such pixel and the fragment stage
    // re-runs the exact test: GL_INTEL_conservative_rasterization ("all fragments for which
    // any part of their squares are inside the polygon, after expanding the polygon by
    // 1/512th of a pixel") covers the edge band; pushing each vertex uDilate px away from
    // the centroid covers the apex spike. Pixels the CPU would not cover discard BEFORE the
    // interlock (explicitly allowed by the spec) and never join the pixel's ordering chain.
    //
    // A SPRITE is one enclosing right triangle over its clipped rect (hypotenuse through the
    // far corner, so every rect pixel centre is strictly inside); the extra half discards on
    // the bbox test. gl_FragCoord is the pixel centre (px+0.5) and the vertex stage maps GS
    // pixel space 1:1 onto the window (window x = GS x, no flip), so uint(gl_FragCoord) is
    // the CPU's pixel index. The no-attachment framebuffer only sizes the viewport; every
    // byte the pipeline writes goes to the mirror SSBO exactly as the compute kernels do.
    const char *kHwVertex = R"GLSL(
uniform float uDilate;    // px to push each triangle vertex away from the centroid
uniform float uInvHalfW;  // 2 / viewport width
uniform float uInvHalfH;  // 2 / viewport height
void main()
{
    uint vid = uint(gl_VertexID);
    uint p = vid / 3u;
    uint c = vid - p * 3u;
    gPrimIdx = p;
    gStateIdx = G.stateIndex;
    vec2 pos;
    if (S.primType == 6u)
    {
        float w = float(G.maxX + 1u - G.minX);
        float h = float(G.maxY + 1u - G.minY);
        vec2 o = vec2(float(G.minX), float(G.minY));
        pos = (c == 0u) ? o : ((c == 1u) ? o + vec2(2.0 * w, 0.0) : o + vec2(0.0, 2.0 * h));
    }
    else
    {
        vec2 v0 = vec2(G.x0, G.y0), v1 = vec2(G.x1, G.y1), v2 = vec2(G.x2, G.y2);
        vec2 v = (c == 0u) ? v0 : ((c == 1u) ? v1 : v2);
        if (uDilate > 0.0)
        {
            vec2 cen = (v0 + v1 + v2) * (1.0 / 3.0);
            vec2 d = v - cen;
            float len = length(d);
            if (len > 1.0e-6)
                v += d * (uDilate / len);
        }
        pos = v;
    }
    gl_Position = vec4(pos.x * uInvHalfW - 1.0, pos.y * uInvHalfH - 1.0, 0.0, 1.0);
}
)GLSL";

    const char *kHwFragment = R"GLSL(
#if HW_LOCK == 2
layout(pixel_interlock_ordered) in;
#elif HW_LOCK == 1
layout(pixel_interlock_unordered) in;
#endif

// shadePrimPixel's own coverage decision, hoisted (same floats, same order, `precise`) so an
// uncovered fragment discards before the interlock instead of occupying a slot in the
// pixel's ordering chain. The bbox test is load-bearing for the triangle path too: the CPU
// walks only [minX, maxX] x [minY, maxY], so a pixel outside it that the epsilon band would
// admit must NOT be shaded.
bool hwCovers(uint px_i, uint py_i)
{
    if (px_i < G.minX || px_i > G.maxX || py_i < G.minY || py_i > G.maxY)
        return false;
    if (S.primType == 6u)
        return true;
    float px = float(px_i) + 0.5;
    float py = float(py_i) + 0.5;
    precise float w0 = (((G.y1 - G.y2) * (px - G.x2) + (G.x2 - G.x1) * (py - G.y2)) * G.winding) * G.invAbsDenom;
    precise float w1 = (((G.y2 - G.y0) * (px - G.x2) + (G.x0 - G.x2) * (py - G.y2)) * G.winding) * G.invAbsDenom;
    precise float w2 = 1.0 - w0 - w1;
    const float kEdgeEpsilon = 1.0e-4;
    return !(w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon);
}

void main()
{
    gPrimIdx = uint(gl_PrimitiveID);
    gStateIdx = G.stateIndex;
    uint px_i = uint(gl_FragCoord.x);
    uint py_i = uint(gl_FragCoord.y);
    if (!hwCovers(px_i, py_i))
        discard;
#if (HW_ABL & 17)
    atomicAdd(hwCount[0], 1u);
#endif
#if (HW_ABL & 1)
    discard;
#endif
    // Colour, texture, fog and Z need no ordering: compute them before the critical section
    // so the interlock covers only the framebuffer read-modify-write.
    uint r, g, b, a, fog, z;
    if (!shadePrimColor(px_i, py_i, r, g, b, a, fog, z))
        discard;
#if HW_LOCK != 0
    beginInvocationInterlockARB();
#endif
    writePixel(px_i, py_i, r, g, b, a, fog, z);
#if HW_LOCK != 0
    endInvocationInterlockARB();
#endif
}
)GLSL";


#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
#ifndef EGL_CONTEXT_MAJOR_VERSION
#define EGL_CONTEXT_MAJOR_VERSION 0x3098
#endif
#ifndef EGL_CONTEXT_MINOR_VERSION
#define EGL_CONTEXT_MINOR_VERSION 0x30FB
#endif
#ifndef EGL_CONTEXT_OPENGL_PROFILE_MASK
#define EGL_CONTEXT_OPENGL_PROFILE_MASK 0x30FD
#endif
#ifndef EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT 0x00000001
#endif
#ifndef EGL_NO_CONFIG_KHR
#define EGL_NO_CONFIG_KHR ((EGLConfig)0)
#endif
} // namespace

struct GsGpuPresentDevice::Impl
{
    enum class State : uint8_t
    {
        Untried,
        Ready,
        Failed
    };

    // Job marshalling: a bounded FIFO. Decode jobs are synchronous (the caller parks on
    // cvDone until its stack flags are set); mirror upload/verify jobs are self-contained
    // fire-and-forget. Multiple producers (present thread decodes, GS worker mirrors).
    std::mutex mx;
    std::condition_variable cv;     // wakes the device thread
    std::condition_variable cvDone; // wakes callers (init done / decode done / queue space)
    State state = State::Untried;   // written by the device thread under mx
    bool stop = false;
    struct DecodeDesc
    {
        const uint8_t *vram = nullptr;
        uint32_t vramSize = 0;
        uint64_t sourceId = 0;
        GSFrameReg frame{};
        uint32_t width = 0, height = 0;
        bool frameBaseIsPages = false;
        uint32_t originX = 0, originY = 0;
        bool preserveAlpha = false;
        std::vector<uint8_t> *out = nullptr;
    };
    struct Job
    {
        enum class Kind : uint8_t
        {
            Decode,
            Upload,
            Verify,
            PatchRaw,
            VerifyFull,
            Readback,
            RasterBatch,
            VerifyRange,
            GlFrame, // cont.329 phase 1: render+resolve+read back one frame (synchronous)
            GlDraw,  // cont.329 phase 2a: queue geometry into the target (async)
            GlSnap   // cont.345: resolve a target at the flip point, keep it under a seq (async)
        } kind = Kind::Decode;
        uint64_t seq = 0;         // GlFrame: the flip snapshot wanted (0 = live); GlSnap: the seq to file under
        DecodeDesc decode{};      // Decode / VerifyFull (vram+vramSize; out unused for the latter)
        GsGpuUploadChunk chunk{}; // Upload/Verify only
        std::vector<uint8_t> raw; // PatchRaw / VerifyRange: CPU VRAM bytes
        uint32_t rawOffset = 0;   // PatchRaw / VerifyRange: mirror byte offset
        GsGpuBatch batch{};       // RasterBatch only
        GsGlBatch glBatch{};      // GlDraw only
        uint64_t tag = 0;         // VerifyRange only: echoed on divergence
        const uint8_t *drawPageMask = nullptr; // VerifyFull only (caller-owned, 512 bytes)
        bool quiet = false;                    // VerifyFull only: suppress the report
        uint32_t *outFirstWord = nullptr;      // VerifyFull only: where it diverged
        uint32_t *outStaleWords = nullptr;
        std::vector<uint8_t> *readOut = nullptr; // Readback only
        bool *done = nullptr;     // Decode / VerifyFull (caller-stack completion flags)
        bool *result = nullptr;
    };
    std::deque<Job> jobs;
    size_t pendingBytes = 0; // queued payload bytes (producer backpressure)
    // ★ cont.329 phase 3b: a job's weight for the backpressure budget. EVERY payload a job can
    // carry must be counted here. The GL renderer's batches were not, so only the 8192-JOB cap
    // applied to them: the queue grew to 8192 batches of ~118 KB, RSS went 215 MB -> 1.19 GB in
    // twenty seconds, presenting stopped dead, and the game looked HUNG rather than slow (the EE
    // kept running and filling the log, which is what made it read as a deadlock). A byte budget
    // that silently ignores a payload type is worse than no budget, because it looks like it works.
    static size_t jobBytes(const Job &j)
    {
        size_t n = j.chunk.payload.size() * 4u + j.raw.size();
        n += j.glBatch.verts.size() * sizeof(GsGlVertex);
        n += j.glBatch.groups.size() * sizeof(GsGlGroup);
        for (const GsGlTexUpload &u : j.glBatch.texUploads)
            n += u.rgba.size();
        return n;
    }
    static constexpr size_t kMaxPendingBytes = 64u * 1024u * 1024u;
    static constexpr size_t kMaxPendingJobs = 8192u;
    std::thread worker;

    // Device-thread-only state.
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE; // pbuffer fallback only
    GlApi gl{};
    GLuint program = 0;
    GLuint vramBuf = 0; // phase-0 present-snapshot buffer (upload cache per sourceId)
    GLuint tabBuf = 0;
    GLuint outBuf = 0;
    GLint locMode = -1, locBlockBase = -1, locBw = -1, locOriginX = -1, locOriginY = -1;
    GLint locWidth = -1, locHeight = -1, locPreserveAlpha = -1, locTabOff = -1;
    uint64_t lastSourceId = 0;
    bool haveUpload = false;
    unsigned errorLogs = 0;
    // Phase-1 VRAM mirror state (device thread only).
    GLuint mirrorBuf = 0;  // the persistent 4MB guest-VRAM mirror
    GLuint payloadBuf = 0; // per-chunk payload / expected-values scratch (orphaned per job)
    GLuint resultBuf = 0;  // verify result: {mism, firstIdx, firstGot, firstExp}
    GLuint uploadProgram = 0;
    GLuint verifyProgram = 0;
    struct ChunkLocs
    {
        GLint dbp = -1, dbw = -1, dsax = -1, dsay = -1, rrw = -1, startPixel = -1, count = -1, tabOff = -1;
    } upLocs, vfLocs;
    unsigned long xvChunks = 0, xvMismChunks = 0;
    unsigned long long xvWords = 0, xvMismWords = 0;
    // Phase-1b: full-mirror compare (lazily created -- a second 4MB buffer only exists
    // once PS2X_GS_GPU_MIRRORVERIFY asks for it).
    GLuint refBuf = 0;
    GLuint histBuf = 0;
    GLuint fullProgram = 0;
    GLint locFullWords = -1;
    GLint locFullBase = -1;
    unsigned long mvRuns = 0;
    unsigned long prPatches = 0;
    unsigned long long prBytes = 0;
    // Phase-2a raster state (lazily created with the first raster job).
    GLuint rasterPrimProgram = 0;  // one primitive, dispatched over its bbox
    GLuint rasterTileProgram = 0;  // a batch, one workgroup per 16x16 tile
    GLint locTilesX = -1;
    GLuint stateBuf = 0, geomBuf = 0, binOffBuf = 0, binEntBuf = 0, tileListBuf = 0;
    GLuint linTexBuf = 0; // cont.250 PS2X_GS_LINTEX: host-decoded texels
    // Hardware-raster path (cont.233, PS2X_GS_HWRASTER=1; default OFF = the tile kernel).
    // PS2X_GS_HWRASTER_CONS (default 1) enables GL_INTEL_conservative_rasterization;
    // PS2X_GS_HWRASTER_DILATE=<px> (default 1) is the per-vertex centroid push.
    static bool envFlag(const char *name, bool def)
    {
        const char *e = std::getenv(name);
        if (!e || !e[0])
            return def;
        return e[0] != '0';
    }
    const bool hwRaster = envFlag("PS2X_GS_HWRASTER", false);
    const bool hwCons = envFlag("PS2X_GS_HWRASTER_CONS", true);
    // ABLATIONS (never correctness runs): PS2X_GS_HWRASTER_LOCK = 2 ordered interlock
    // (default, the only faithful mode) / 1 unordered / 0 no critical section at all;
    // PS2X_GS_HWRASTER_COHERENT=0 drops the mirror's `coherent` qualifier.
    const int hwLock = []
    {
        const char *e = std::getenv("PS2X_GS_HWRASTER_LOCK");
        if (!e || !e[0])
            return 2;
        const int v = std::atoi(e);
        return (v < 0 || v > 2) ? 2 : v;
    }();
    const bool hwCoherent = envFlag("PS2X_GS_HWRASTER_COHERENT", true);
    // PS2X_GS_HWRASTER_ABL=<bits>: the HW_ABL shader ablations (see kRasterCommon).
    const int hwAbl = []
    {
        const char *e = std::getenv("PS2X_GS_HWRASTER_ABL");
        return (e && e[0]) ? std::atoi(e) : 0;
    }();
    GLuint hwCountBuf = 0;
    const float hwDilate = []
    {
        const char *e = std::getenv("PS2X_GS_HWRASTER_DILATE");
        return (e && e[0]) ? static_cast<float>(std::atof(e)) : 1.0f;
    }();
    static constexpr uint32_t kHwW = 2048u, kHwH = 2048u; // GS scissor space is 11 bits
    GLuint hwProgram = 0, hwVao = 0, hwFbo = 0;
    GLint locHwDilate = -1, locHwInvHalfW = -1, locHwInvHalfH = -1;
    bool hwFailed = false;
    unsigned hwErrLogs = 0;
    // ★ cont.329 phase 1 (the GL renderer arc) -- device-thread-only state.
    // A SCENE target is a real RGBA8 colour texture + a 24-bit depth renderbuffer, at
    // native resolution x `glrScale`. `glrResolveFbo` is a native-sized (= display-sized) colour
    // target that the scene is BLIT-DOWNSAMPLED into before readback -- which is why the readback
    // cost does not grow with the upscale factor, and why the whole existing present path
    // (640-stride repack -> UpdateTexture -> aspect blit) is reused untouched.
    // PHASE 1 draws nothing into the scene target yet; it only clears it. That is deliberate: it
    // proves context + FBO + resolve + readback + present integration in one testable step,
    // before any GS state is translated.
    // ★ cont.329 phase 4: ONE TARGET PER FRAMEBUFFER ADDRESS. Each carries its own clear state:
    // a target starts a new frame the first time it is drawn into AFTER it was last displayed,
    // which is exactly the back-buffer/front-buffer alternation of double buffering.
    struct GlTarget
    {
        GLuint fbo = 0, color = 0, depth = 0;
        uint32_t w = 0, h = 0;
        uint32_t refW = 0, refH = 0; // rotk row 257: the logical (unscaled) extent the guest draws it at
        bool needClear = true;
        // ★★★★ cont.344: this target's depth texture comes from glrDepthPool (shared by ZBP) -- never delete it here.
        bool depthShared = false;
        // ★★★★ cont.343b: this target's colour buffer is RGBA16F. Set once a COLCLAMP=0 ADDITIVE
        // draw hits it (this game's shadow volume adds +1/-1 into fbp 0x180 and its composite only
        // asks "non-zero?"): float blending neither saturates nor wraps, so the count comes out
        // right for that consumer where an RGBA8 target would have saturated every touched pixel
        // to 255 -- the whole sheared volume darkened the wall (user, 2026-09-18). PCSX2's HW
        // renderer does the same thing with its "colclip HDR" target. PS2X_GS_GLR_COLCLIP=0 disables.
        bool hdr = false;
        uint64_t depthPoolKey = 0; // cont.344: the glrDepthPool entry this target's depth came from
        // ★★★★ cont.345: this target ATTACHED to a pooled depth that other targets had already
        // written (depthPre at alloc, or a rebind). Its own creation clear must then leave the depth
        // alone: clearing it wiped the scene's Z mid-frame when the shadow-volume target 0x180 was
        // rebuilt as HDR at the first cutscene cut -- the invert-back pass then drove every pixel to
        // "nearest", the characters and fog failed their depth test, and one flip presented the new
        // shot's background alone (the user's "flicker soon after selecting a player").
        bool depthJoined = false;
    };
    // cont.345 PS2X_GS_GLR_ZREBIND (default ON; =0 restores): (1) a joined pooled depth is never
    // cleared by a target's own creation clear, and (2) an EXISTING target whose batch names a
    // different ZBUF than the one its depth came from re-attaches the pooled depth for that ZBUF --
    // target 0x80 was created at boot by a draw with ZBP=0 and kept a private depth for the whole
    // game, so its frames never shared the scene's Z (the invert pass only ever touched 0x0's).
    static bool glrZRebind()
    { static const bool v = []{ const char *e = std::getenv("PS2X_GS_GLR_ZREBIND"); return !(e && e[0] == '0'); }(); return v; }
    static bool glrColclipHdr()
    {
        static const bool v = []{ const char *e = std::getenv("PS2X_GS_GLR_COLCLIP"); return !(e && e[0] == '0'); }();
        return v;
    }
    std::unordered_map<uint64_t, GlTarget> glrTargets;
    // ★★★★ cont.344 PS2X_GS_GLR_ZSHARE (default ON): ONE DEPTH TEXTURE PER Z ADDRESS, shared by every
    // colour target that names it. The target key carries (fbp, zbp) and each target owned a private
    // depth texture, so the scene (fbp 0/0x80, zbp 0x100) wrote depth that the shadow volume (fbp 0x180,
    // zbp 0x100) never tested against: every volume face passed and the +1/-1 z-fail count cancelled to
    // ZERO everywhere -- the GL frame was pixel-identical with the composite dropped (cont.344 A/B). On
    // the GS the Z buffer is a VRAM page, one per ZBP, whoever draws. PCSX2 GSTextureCache keys its
    // DepthStencil targets by ZBP for the same reason. Pool key = zbp | zpsm<<20 | allocW<<32 | allocH<<48.
    struct GlDepthPoolEntry { GLuint tex = 0; unsigned long long lastClearFrame = ~0ull; };
    std::unordered_map<uint64_t, GlDepthPoolEntry> glrDepthPool;
    std::unordered_map<uint32_t, std::vector<uint64_t>> glrDepthByZbp; // zbp -> pool keys
    static bool glrZShare()
    { static const bool v = []{ const char *e = std::getenv("PS2X_GS_GLR_ZSHARE"); return !(e && e[0] == '0'); }(); return v; }
    // ★★★★ cont.344 PS2X_GS_GLR_ZALIAS (default ON): a COLOUR draw onto a page that is a live Z buffer.
    // This game inverts its Z24 buffer in place before the shadow-volume pass -- a full-screen white
    // sprite onto fbp 0x100 (= the scene's ZBP) with ALPHA (Cs-Cd)*FIX(0x80)+0 = 255-Cd per lane, i.e.
    // z' = 0xFFFFFF - z -- and inverts it back afterwards. In GL that sprite landed in a colour target
    // keyed fbp=0x100 that no depth test reads. Here the same draw is applied to the shared depth
    // texture(s) of that ZBP as d' = 1 - d (Z24 maps to depth as z/0xFFFFFF, so the complement is
    // exact). PCSX2's HW renderer handles frame-aliases-depth by converting the depth target to colour
    // and back (GSTextureCache, ShaderConvert FLOAT32_TO_RGBA8 / RGBA8_TO_FLOAT32); this is the one
    // formula this game uses, done in the depth domain.
    // cont.344 PS2X_GS_GLR_TEXFLIP (default ON; =0 restores the mirrored read for an A/B).
    static bool glrTexFlipY()
    { static const bool v = []{ const char *e = std::getenv("PS2X_GS_GLR_TEXFLIP"); return !(e && e[0] == '0'); }(); return v; }
    static bool glrZAlias()
    { static const bool v = []{ const char *e = std::getenv("PS2X_GS_GLR_ZALIAS"); return !(e && e[0] == '0'); }(); return v; }
    GLuint glrZInvProgram = 0, glrZAliasFbo = 0;
    GLint locZiSrc = -1;
    unsigned long long glrZAliasPasses = 0, glrZAliasSkipped = 0;
    static uint64_t glrDepthPoolKey(uint32_t zbp, uint32_t zpsm, uint32_t allocW, uint32_t allocH)
    {
        return uint64_t(zbp & 0xFFFFFu) | (uint64_t(zpsm & 0xFFu) << 20) | (uint64_t(allocW) << 32) | (uint64_t(allocH) << 48);
    }
    bool glrEnsureZInv()
    {
        if (glrZInvProgram && glrZAliasFbo)
            return true;
        static constexpr const char *kZiVertex = R"(#version 430
void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
        static constexpr const char *kZiFragment = R"(#version 430
uniform sampler2D uSrcDepth;
void main() { gl_FragDepth = 1.0 - texelFetch(uSrcDepth, ivec2(gl_FragCoord.xy), 0).r; }
)";
        if (!glrZInvProgram)
        {
            glrZInvProgram = buildGraphicsProgram(kZiVertex, kZiFragment, "glr-zinvert");
            if (!glrZInvProgram)
            {
                std::fprintf(stderr, "[gsgpu] ZALIAS DISABLED: the depth-invert program failed\n");
                return false;
            }
            locZiSrc = gl.GetUniformLocation(glrZInvProgram, "uSrcDepth");
        }
        if (!glrZAliasFbo)
            gl.GenFramebuffers(1, &glrZAliasFbo);
        return glrZAliasFbo != 0u;
    }
    // d' = 1 - d over one shared depth texture (w x h), through the AFAIL depth copy (a texture cannot
    // be sampled while it is the bound depth attachment). Restores the batch's framebuffer/viewport/
    // program/colour mask; depth, blend and scissor are re-established per group by the caller's loop.
    bool glrComplementDepth(GLuint depthTex, uint32_t w, uint32_t h, GlTarget &restoreTo, uint32_t sceneW, uint32_t sceneH)
    {
        if (!depthTex || !glrEnsureZInv() || !ensureDepthCopy(w, h))
            return false;
        const bool diag = glrZAliasPasses < 6ull;
        if (diag) (void)gl.GetError(); // drain earlier errors so the ones below are this pass's own
        gl.BindFramebuffer(GL_FRAMEBUFFER, glrZAliasFbo);
        gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);
        float dSrc = -1.f;
        if (diag)
            gl.ReadPixels(static_cast<GLint>(w / 2u), static_cast<GLint>(h / 2u), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &dSrc);
        const GLenum errAttach = diag ? gl.GetError() : 0u;
        gl.BindFramebuffer(GL_READ_FRAMEBUFFER, glrZAliasFbo);
        gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, glrDepthCopyFbo);
        gl.BlitFramebuffer(0, 0, static_cast<GLint>(w), static_cast<GLint>(h),
                           0, 0, static_cast<GLint>(w), static_cast<GLint>(h),
                           GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        const GLenum errBlit = diag ? gl.GetError() : 0u;
        gl.BindFramebuffer(GL_FRAMEBUFFER, glrZAliasFbo);
        gl.BindVertexArray(glrVao); // core profile: no VAO bound = INVALID_OPERATION on the draw
        gl.Viewport(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h));
        gl.Disable(GL_SCISSOR_TEST);
        gl.Disable(GL_BLEND);
        gl.Enable(GL_DEPTH_TEST);
        gl.DepthFunc(GL_ALWAYS);
        gl.DepthMask(1);
        gl.ColorMask(0, 0, 0, 0);
        gl.ActiveTexture(GL_TEXTURE1);
        gl.BindTexture(GL_TEXTURE_2D, glrDepthCopyTex);
        gl.ActiveTexture(GL_TEXTURE0_CONT332J);
        gl.UseProgram(glrZInvProgram);
        if (locZiSrc >= 0)
            gl.Uniform1i(locZiSrc, 1);
        // cont.344 diagnostics (first passes only): framebuffer status, GL error, and the depth at one
        // pixel before/after -- "the pass ran" is not "the pass landed".
        float dBefore = -1.f, dAfter = -1.f;
        if (diag)
        {
            gl.BindFramebuffer(GL_READ_FRAMEBUFFER, glrDepthCopyFbo);
            gl.ReadPixels(static_cast<GLint>(w / 2u), static_cast<GLint>(h / 2u), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &dBefore);
            gl.BindFramebuffer(GL_FRAMEBUFFER, glrZAliasFbo);
        }
        const GLenum fbStatus = gl.CheckFramebufferStatus(GL_FRAMEBUFFER);
        const GLenum errPre = diag ? gl.GetError() : 0u;
        gl.DrawArrays(GL_TRIANGLES, 0, 3);
        if (diag)
        {
            const GLenum errDraw = gl.GetError();
            gl.BindFramebuffer(GL_READ_FRAMEBUFFER, glrZAliasFbo);
            gl.ReadPixels(static_cast<GLint>(w / 2u), static_cast<GLint>(h / 2u), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &dAfter);
            std::fprintf(stderr, "[gsgpu:glr] ZALIAS diag: fbo-status=0x%x err{attach=0x%x blit=0x%x pre=0x%x draw=0x%x} depthTex=%u copyTex=%u centre: src %.6f copy %.6f -> after %.6f\n",
                         static_cast<unsigned>(fbStatus), static_cast<unsigned>(errAttach), static_cast<unsigned>(errBlit), static_cast<unsigned>(errPre), static_cast<unsigned>(errDraw), depthTex, glrDepthCopyTex, dSrc, dBefore, dAfter);
        }
        gl.UseProgram(glrProgram);
        gl.BindFramebuffer(GL_FRAMEBUFFER, restoreTo.fbo);
        gl.Viewport(0, 0, static_cast<GLsizei>(sceneW), static_cast<GLsizei>(sceneH));
        gl.ColorMask(1, 1, 1, 0);
        gl.Enable(GL_SCISSOR_TEST);
        ++glrZAliasPasses;
        return true;
    }
    // The invert-sprite detector: a batch whose COLOUR address is a pooled Z address, carrying a
    // group with ALPHA A=Cs B=Cd C=FIX(0x80) D=0 and white vertices. Applied to every pooled depth of
    // that ZBP at the batch's scene size; the colour draw itself still runs afterwards (harmless).
    void glrZAliasBatch(const GsGlBatch &b, GlTarget &tgt, uint32_t sceneW, uint32_t sceneH)
    {
        if (!glrZAlias() || !glrZShare())
            return;
        const uint32_t bFbp = static_cast<uint32_t>(b.targetKey & 0xFFFFFu);
        auto it = glrDepthByZbp.find(bFbp);
        if (it == glrDepthByZbp.end())
            return;
        for (const GsGlGroup &g : b.groups)
        {
            if (!(g.abe && g.ba == 0u && g.bb == 1u && g.bc == 2u && g.bd == 2u && g.bfix == 0x80u && g.count >= 3u))
                continue;
            const GsGlVertex &v = b.verts[g.first];
            if (!(v.r == 255u && v.g == 255u && v.b == 255u))
                continue;
            bool any = false;
            // PS2X_GS_GLR_ZALIASLOG=1 (cont.345): every pass and every skipped pool entry, with the
            // depth at the centre pixel before/after, so a complement that never lands (or lands on
            // the wrong pooled depth) is visible against the flip snapshots.
            static const bool s_zaLog = []
            { const char *e = std::getenv("PS2X_GS_GLR_ZALIASLOG"); return e && e[0] && e[0] != '0'; }();
            for (uint64_t pk : it->second)
            {
                const uint32_t w = static_cast<uint32_t>((pk >> 32) & 0xFFFFu), h = static_cast<uint32_t>((pk >> 48) & 0xFFFFu);
                if (w != sceneW || h != sceneH)
                {
                    ++glrZAliasSkipped;
                    if (s_zaLog)
                        std::fprintf(stderr, "[gsgpu:zalias] SKIP pool %ux%u (batch scene %ux%u) zbp=0x%x batch-target=0x%llx verts=%zu\n",
                                     w, h, sceneW, sceneH, bFbp, (unsigned long long)b.targetKey, b.verts.size());
                    continue;
                }
                auto pe = glrDepthPool.find(pk);
                if (pe == glrDepthPool.end())
                    continue;
                float dBefore = -1.f, dAfter = -1.f;
                if (s_zaLog)
                {
                    gl.BindFramebuffer(GL_FRAMEBUFFER, glrZAliasFbo ? glrZAliasFbo : tgt.fbo);
                    if (glrZAliasFbo) gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, pe->second.tex, 0);
                    gl.ReadPixels(static_cast<GLint>(w / 2u), static_cast<GLint>(h / 2u), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &dBefore);
                    gl.BindFramebuffer(GL_FRAMEBUFFER, tgt.fbo);
                }
                if (glrComplementDepth(pe->second.tex, w, h, tgt, sceneW, sceneH))
                    any = true;
                if (s_zaLog)
                {
                    gl.BindFramebuffer(GL_FRAMEBUFFER, glrZAliasFbo);
                    gl.ReadPixels(static_cast<GLint>(w / 2u), static_cast<GLint>(h / 2u), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &dAfter);
                    gl.BindFramebuffer(GL_FRAMEBUFFER, tgt.fbo);
                    std::fprintf(stderr, "[gsgpu:zalias] PASS #%llu zbp=0x%x depthTex=%u %ux%u batch-target=0x%llx groups=%zu verts=%zu centre %.4f -> %.4f\n",
                                 glrZAliasPasses, bFbp, pe->second.tex, w, h, (unsigned long long)b.targetKey, b.groups.size(), b.verts.size(), dBefore, dAfter);
                }
            }
            if (any && glrZAliasPasses <= 4ull)
                std::fprintf(stderr, "[gsgpu:glr] ZALIAS: Z-page 0x%x invert sprite -> depth complement (pass #%llu, %ux%u)\n",
                             bFbp, glrZAliasPasses, sceneW, sceneH);
        }
    }
    // ★★★ cont.330: scratch texture + FBO used to blit a VRAM seed rect into a target (see
    // GsGlTargetSeed). A blit, not glTexSubImage2D, because the target is at glrScale x native.
    GLuint glrSeedTex = 0, glrSeedFbo = 0;
    // ★★★★ cont.332j PS2X_GS_GLR_AFAIL=5: the depth AS IT STANDS BEFORE an AFAIL group, copied
    // so the shader can read it. A fragment that FAILS the alpha test under AFAIL=FB_ONLY writes
    // colour but must leave Z untouched, and GL has no per-fragment depth-write mask -- so it
    // writes back the value already there, which makes the write a no-op. Sampling the target's
    // own depth attachment while writing it would be a feedback loop, hence the copy.
    GLuint glrDepthCopyTex = 0, glrDepthCopyFbo = 0;
    uint32_t glrDepthCopyW = 0, glrDepthCopyH = 0;
    unsigned long long glrAfailKeepZ = 0, glrDepthCopies = 0;
    bool ensureDepthCopy(uint32_t w, uint32_t h)
    {
        if (glrDepthCopyTex && glrDepthCopyW == w && glrDepthCopyH == h)
            return true;
        if (glrDepthCopyTex) gl.DeleteTextures(1, &glrDepthCopyTex);
        if (glrDepthCopyFbo) gl.DeleteFramebuffers(1, &glrDepthCopyFbo);
        gl.GenTextures(1, &glrDepthCopyTex);
        gl.BindTexture(GL_TEXTURE_2D, glrDepthCopyTex);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        gl.TexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_DEPTH_COMPONENT24),
                      static_cast<GLsizei>(w), static_cast<GLsizei>(h), 0,
                      GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
        gl.BindTexture(GL_TEXTURE_2D, 0);
        gl.GenFramebuffers(1, &glrDepthCopyFbo);
        gl.BindFramebuffer(GL_FRAMEBUFFER, glrDepthCopyFbo);
        gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, glrDepthCopyTex, 0);
        const bool ok = gl.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
        if (!ok)
        { gl.DeleteTextures(1, &glrDepthCopyTex); gl.DeleteFramebuffers(1, &glrDepthCopyFbo);
          glrDepthCopyTex = glrDepthCopyFbo = 0; return false; }
        glrDepthCopyW = w; glrDepthCopyH = h;
        return true;
    }
    // Snapshot the target's depth, then restore it as the draw framebuffer.
    bool snapshotDepth(GlTarget &t, uint32_t w, uint32_t h)
    {
        if (!ensureDepthCopy(w, h))
            return false;
        gl.BindFramebuffer(GL_READ_FRAMEBUFFER, t.fbo);
        gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, glrDepthCopyFbo);
        gl.BlitFramebuffer(0, 0, static_cast<GLint>(w), static_cast<GLint>(h),
                           0, 0, static_cast<GLint>(w), static_cast<GLint>(h),
                           GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        gl.BindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        ++glrDepthCopies;
        return true;
    }
    // ★★★★★ cont.332l PS2X_GS_GLR_AFAIL=6 -- THE ACCURATE PATH'S DEPTH IMAGE.
    // The shader has to READ and WRITE one depth value per pixel inside the interlock, and GL
    // image objects cannot have a depth format, so the target's depth is copied into an R32F
    // image before each AFAIL group (scissored to that group's own rect, which also bounds the
    // group's draw). The fixed-function write still runs -- the shader hands it the same value
    // it stored -- so the real depth buffer stays authoritative for every later draw.
    GLuint glrDepthImgTex = 0, glrDepthImgFbo = 0;
    uint32_t glrDepthImgW = 0, glrDepthImgH = 0;
    GLuint glrDepthSeedProgram = 0;
    GLint locDsSrc = -1;
    unsigned long long glrAfailSwZ = 0, glrDepthImgSeeds = 0;
    bool ensureDepthImage(uint32_t w, uint32_t h)
    {
        if (glrDepthImgTex && glrDepthImgW == w && glrDepthImgH == h && glrDepthSeedProgram)
            return true;
        if (glrDepthImgTex) { gl.DeleteTextures(1, &glrDepthImgTex); glrDepthImgTex = 0; }
        if (glrDepthImgFbo) { gl.DeleteFramebuffers(1, &glrDepthImgFbo); glrDepthImgFbo = 0; }
        gl.GenTextures(1, &glrDepthImgTex);
        gl.BindTexture(GL_TEXTURE_2D, glrDepthImgTex);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        gl.TexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_R32F),
                      static_cast<GLsizei>(w), static_cast<GLsizei>(h), 0,
                      GL_RED, GL_FLOAT, nullptr);
        gl.BindTexture(GL_TEXTURE_2D, 0);
        gl.GenFramebuffers(1, &glrDepthImgFbo);
        gl.BindFramebuffer(GL_FRAMEBUFFER, glrDepthImgFbo);
        gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, glrDepthImgTex, 0);
        const bool ok = gl.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
        if (!ok)
        {
            gl.DeleteTextures(1, &glrDepthImgTex); gl.DeleteFramebuffers(1, &glrDepthImgFbo);
            glrDepthImgTex = glrDepthImgFbo = 0;
            std::fprintf(stderr, "[gsgpu] AFAIL=6 DISABLED: the R32F depth image is incomplete\n");
            return false;
        }
        if (!glrDepthSeedProgram)
        {
            // One attribute-less full-screen triangle; the scissor decides what it touches.
            static constexpr const char *kDsVertex = R"(#version 430
void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
            static constexpr const char *kDsFragment = R"(#version 430
uniform sampler2D uSrcDepth;
out float oDepth;
void main() { oDepth = texelFetch(uSrcDepth, ivec2(gl_FragCoord.xy), 0).r; }
)";
            glrDepthSeedProgram = buildGraphicsProgram(kDsVertex, kDsFragment, "glr-depthseed");
            if (!glrDepthSeedProgram)
            {
                std::fprintf(stderr, "[gsgpu] AFAIL=6 DISABLED: the depth-seed program failed\n");
                return false;
            }
            locDsSrc = gl.GetUniformLocation(glrDepthSeedProgram, "uSrcDepth");
        }
        glrDepthImgW = w; glrDepthImgH = h;
        return true;
    }
    // Runs BEFORE the group's own state is set, so it only has to restore the framebuffer, the
    // viewport, the program and the batch's colour mask -- depth, blend, scissor and every
    // uniform are re-established by the group loop immediately afterwards.
    bool seedDepthImage(GlTarget &t, uint32_t w, uint32_t h)
    {
        if (!t.depth || !ensureDepthImage(w, h))
            return false;
        gl.BindFramebuffer(GL_FRAMEBUFFER, glrDepthImgFbo);
        gl.Viewport(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h));
        gl.Disable(GL_BLEND);
        gl.Disable(GL_DEPTH_TEST);
        gl.DepthMask(0);
        gl.ColorMask(1, 1, 1, 1);
        gl.ActiveTexture(GL_TEXTURE1);
        gl.BindTexture(GL_TEXTURE_2D, t.depth);
        gl.ActiveTexture(GL_TEXTURE0_CONT332J);
        gl.UseProgram(glrDepthSeedProgram);
        if (locDsSrc >= 0)
            gl.Uniform1i(locDsSrc, 1);
        gl.DrawArrays(GL_TRIANGLES, 0, 3);
        gl.UseProgram(glrProgram);
        gl.BindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        gl.Viewport(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h));
        gl.ColorMask(1, 1, 1, 0);
        ++glrDepthImgSeeds;
        return true;
    }
    uint32_t glrSeedW = 0, glrSeedH = 0;
    unsigned long long glrSeedRects = 0, glrSeedPixels = 0;
    unsigned long long glrSeedBatches = 0, glrSeedBatchesWith = 0;
    GLuint glrResolveFbo = 0, glrResolveTex = 0;
    uint32_t glrResolveW = 0, glrResolveH = 0; // resolve target size (= display size)
    bool glrFailed = false;
    unsigned glrErrLogs = 0;
    unsigned long long glrFrames = 0;
    // ★ cont.329 phase 2a: the geometry program and its vertex stream.
    GLuint glrProgram = 0, glrVao = 0, glrVbo = 0;
    GLint locGlrRef = -1, locGlrTexEnable = -1, locGlrTfx = -1, locGlrTcc = -1, locGlrTex = -1;
    GLint locGlrHudScale = -1; // cont.356: the 2D/HUD counter-scale, 1.0 = identity
    GLint locGlrAtst = -1, locGlrAref = -1, locGlrAfail = -1;
    GLint locGlrSrcMode = -1, locGlrTexa = -1, locGlrAem = -1, locGlrSignedAdd = -1;
    GLint locGlrTexFlipY = -1; // cont.344
    GLint locGlrTgtMap = -1;   // rotk row 257
    GLint locGlrPrevDepth = -1, locGlrAfailKeepZ = -1;   // cont.332j
    GLint locGlrAfailSw = -1, locGlrZtstSw = -1;        // cont.332l
    GLint locGlrDebugOut = -1;                          // cont.332l
    const int glrDebugOut = []
    { const char *e = std::getenv("PS2X_GS_GLR_DEBUGOUT"); return (e && e[0]) ? std::atoi(e) : 0; }();
    GLint locGlrFge = -1, locGlrFogCol = -1;
    GLint locGlrLodMode = -1, locGlrLodL = -1, locGlrLodK = -1;
    // ★★ cont.329j, both default OFF while they are measured:
    //   PS2X_GS_GLR_FOG        -- implement PRIM.FGE / FOGCOL (the GL renderer had NO fog).
    //   PS2X_GS_GLR_NOPERSPCOL -- interpolate gouraud colour screen-linearly, as the GS does,
    //                             instead of GL's perspective-correct default.
    const bool glrFog = []
    { const char *e = std::getenv("PS2X_GS_GLR_FOG"); return !(e && e[0] == '0'); }();
    const bool glrNoPerspCol = []
    { const char *e = std::getenv("PS2X_GS_GLR_NOPERSPCOL"); return !(e && e[0] == '0'); }();
    // ★★★★★ cont.332h PS2X_GS_GLR_MIPS, DEVICE HALF -- DEFAULT ON (was OFF).
    // ⚠ THE SAME ENV VAR WAS READ TWICE WITH OPPOSITE DEFAULTS. The backend half
    // (gs_cpu_backend.cpp `s_glMips`) defaults ON and decodes + uploads the whole TEX1.MXL chain;
    // this half selects the mipmap MIN filter and defaulted OFF, so at pure defaults the runtime
    // paid for four levels per texture and then sampled level 0 for every draw -- while the CPU
    // reference samples the level the GS selects on 74.8% of textured draws. It also made an
    // ablation of this flag half-void: `=0` only turned off the half that was already inert.
    // ⚠⚠ THE FIDELITY EFFECT IS WITHIN NOISE -- this is a COHERENCE fix, not a measured win.
    // A first measurement (one run per config) read +0.052 on `a` and was RETRACTED by repeats:
    // on tools/harness/tonefit.py (`GL = a*CPU + b` vs the CPU reference, same 7 gameplay flips)
    //     old default (level 0 only), 2 runs: a = 0.623, 0.613
    //     mips selected, 3 runs:            a = 0.649, 0.620, 0.613
    // The distributions overlap. ⚠ AND THE REASON MATTERS: under `PS2X_VIRTUAL_TIME=1` the CPU arm
    // is BYTE-IDENTICAL run to run (a=1.000 b=-0.0 against another CPU run), but the GL arm is NOT
    // -- its own spread on `a` is ~0.04. **Every GL-arm measurement needs at least three runs.**
    // What this change does buy: the flag becomes honest (an A/B of mipmapping was impossible while
    // `=0` and unset differed only in the half that was already inert), and the runtime stops
    // decoding + uploading four levels per texture that nothing ever samples.
    // PCSX2 reference: GSRendererHW drives mipmapping from TEX1 (MXL/MMIN) the same way, and its
    // HW renderer enables mipmapping by default.
    const bool glrMips = []
    { const char *e = std::getenv("PS2X_GS_GLR_MIPS"); return !(e && e[0] == '0'); }();
    unsigned long long glrTwoPass = 0;
    const bool glrNoTex = []
    { const char *e = std::getenv("PS2X_GS_GLR_NOTEX"); return e && e[0] && e[0] != '0'; }();
    const bool glrNoDepth = []
    { const char *e = std::getenv("PS2X_GS_GLR_NODEPTH"); return e && e[0] && e[0] != '0'; }();
    // ★★ cont.329h PS2X_GS_GLR_FBMSK: honour FRAME.FBMSK per draw through glColorMask instead of
    // writing RGB unconditionally. A byte of 0xFF keeps that channel, 0x00 writes it; anything
    // between needs a read-modify-write in the shader (PCSX2 GSRendererHW's fbmask "mix" path) and
    // is COUNTED here rather than approximated.
    //
    // ★★★★★ cont.331q DEFAULT ON (was OFF "while it is being validated"). Ignoring FBMSK was not a
    // fidelity detail -- it was DESTROYING THE FRAME, and it is the cause of the whole cont.331l-p
    // arc ("every seed placed before the resolve lands and is gone by resolve time"). The device-
    // order trace (PS2X_GS_GLR_ORDER) caught it in one frame: this game opens each title-era frame
    // with a full-screen 6-vertex quad carrying `FBMSK=ffffffff zmsk=0 abe=0` -- a DEPTH-ONLY pass
    // that writes Z and, by the mask, no colour at all. With FBMSK ignored, GL drew its raw colour
    // over everything, taking the target from the clear (mean 19.67) to EXACTLY ZERO. Every seed
    // placed before it therefore died, and the present-resolve seed "worked" only because it lands
    // after the wipe. Honouring the mask: the same draw leaves the target at 92.47 and the UI
    // glyphs then composite ON TOP of the background -> resolve 93.47 against the CPU reference's
    // 93.5. Census over the run: 2,345 masked draws, `partial=0` -- every mask in this game is
    // whole-byte, so the glColorMask mapping is EXACT here, not an approximation.
    // ps2tek FRAME.FBMSK: "prevent the specified bits in the framebuffer from being updated:
    // final_color = (final_color & ~mask) | (frame_color & mask)" -- a set bit is PROTECTED, so an
    // all-ones mask writes nothing. PCSX2 GSRendererHW::IsRTWritten() computes the same quantity as
    // `written_bits = ~FRAME.FBMSK & GSLocalMemory::m_psm[FRAME.PSM].fmsk`.
    // `=0` restores the old ignore-FBMSK behaviour; it only reads under PS2X_GS_RENDERER=gl, so a
    // CPU-renderer run is unaffected either way.
    const bool glrFbmsk = []
    { const char *e = std::getenv("PS2X_GS_GLR_FBMSK"); return !(e && e[0] == '0'); }();
    unsigned long long glrFbmskPartial = 0, glrFbmskMasked = 0;
    // PS2X_GS_GLR_FRAMECLEAR (default 1): clear a target when it starts a new frame. The game
    // draws its OWN full-screen clear quad (the big-primitive census shows a 100%-cover black
    // quad), so this may be redundant -- and clearing depth at the wrong moment would let the
    // game's clear quad fail its own depth test. Flagged so the two can be told apart.
    const bool glrFrameClear = []
    { const char *e = std::getenv("PS2X_GS_GLR_FRAMECLEAR"); return !(e && e[0] == '0'); }();
    // cont.345: whether a RESOLVE re-arms that clear (see glFrameOnThread). Default OFF.
    const bool glrResolveClear = []
    { const char *e = std::getenv("PS2X_GS_GLR_RESOLVECLEAR"); return e && e[0] && e[0] != '0'; }();
    // cont.360: the colour a target is cleared to when the runtime (not the guest) clears it --
    // at creation/realloc, and at a frame-start clear. BLACK by default, as PCSX2 clears a target
    // it has no VRAM for (GSTextureCache::LookupTarget, ClearRenderTarget(dst->m_texture, 0)) and
    // as the CPU arm presents zeroed VRAM. The dark navy (0.04,0.05,0.14) was the phase-1 test
    // marker ("did anything draw?"); in real time the boot loading screen presented it for its
    // first flips. PS2X_GS_GLR_CLEARNAVY=1 restores it as a diagnostic. Alpha is unchanged (1.0).
    const bool glrClearNavy = []
    { const char *e = std::getenv("PS2X_GS_GLR_CLEARNAVY"); return e && e[0] && e[0] != '0'; }();
    void glrSetClearColor()
    {
        if (glrClearNavy)
            gl.ClearColor(0.04f, 0.05f, 0.14f, 1.0f);
        else
            gl.ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    }
    // ★ cont.329 phase 3: decoded textures, keyed exactly as the CPU texture cache keys them.
    // The device only OWNS the GL objects; deciding what to (re)decode is the backend's job, which
    // is what keeps the decode off this thread.
    struct GlTex { GLuint id = 0; uint32_t w = 0, h = 0; uint8_t levels = 0; };
    std::unordered_map<uint64_t, GlTex> glrTextures;
    unsigned long long glrTexUploads = 0, glrTexBinds = 0, glrTexAlloc = 0;
    // The frame boundary is the PRESENT: once RenderFrameGl has resolved and read a frame back,
    // the next draw starts a fresh one. Set true initially so the first draw clears.
    unsigned long long glrDrawn = 0, glrVerts = 0, glrBlendFallback = 0;
    // Draws since the last resolve. CopyFrameToHostRgba is called MORE THAN ONCE per
    // present (the preferred-source attempt, then the display-frame fallback), so arming
    // the clear on every resolve wiped the frame and the second call presented an empty
    // target -- build 801 alternated real frames with pure clears for exactly this reason.
    unsigned long long glrDrawsSinceResolve = 0;
    // ★ cont.329 phase 3c: where the device thread's time actually goes. Default OFF (it adds two
    // clock reads per section). Three cycles of inferring device behaviour from counters have now
    // misled me twice, so this measures inside the thread instead.
    const bool glrProf = []
    { const char *e = std::getenv("PS2X_GS_GLR_PROF"); return e && e[0] && e[0] != '0'; }();
    unsigned long long tDraw = 0, tTexUp = 0, tVbo = 0, tLoop = 0;
    unsigned long long tFrame = 0, tClear = 0, tBlit = 0, tRead = 0, tFlip = 0;
    unsigned long long nBatch = 0, nResolve = 0, nFrameCacheHit = 0;
    // ★ cont.371 PS2X_GS_GLR_SPLIT=1 (default OFF; instrument only): WHERE the device thread's
    // wall goes, per window of 256 resolves. The synchronous resolve readback (cont.365: ~19.7 ms
    // at 4x) mixes two costs -- waiting for the GPU to finish the frame's queued draws, and the
    // blit + copy themselves -- so a glFinish just before the blit splits them (the readback
    // drains the pipeline anyway, so the extra sync point moves the wait, it does not add one).
    // Alongside: device-thread idle vs busy by job kind (is this thread the wall at all?), GPU
    // elapsed per draw job from GL_TIME_ELAPSED queries, and the time producers spend parked
    // on a full job queue (the only way this thread's cost reaches the EE).
    const bool glrSplit = []
    { const char *e = std::getenv("PS2X_GS_GLR_SPLIT"); return e && e[0] && e[0] != '0'; }();
    struct SplitAcc
    {
        unsigned long long idle = 0, draw = 0, snap = 0, frame = 0, other = 0;
        unsigned long long finishWait = 0, blitRead = 0, hires = 0, gpu = 0, producerWait = 0;
        unsigned long long resolves = 0, drawJobs = 0, gpuQueries = 0, producerWaits = 0;
    } splitAcc;
    std::chrono::steady_clock::time_point splitT0{};
    unsigned long long splitProducerWait = 0, splitProducerWaits = 0; // guarded by mx
    std::vector<GLuint> splitQFree, splitQLive;
    void splitQueryBegin()
    {
        if (!gl.GenQueries || !gl.BeginQuery || !gl.EndQuery || !gl.GetQueryObjectui64v)
            return;
        if (splitQFree.empty())
        {
            GLuint q[64] = {};
            gl.GenQueries(64, q);
            splitQFree.assign(q, q + 64);
        }
        const GLuint q = splitQFree.back();
        splitQFree.pop_back();
        gl.BeginQuery(GL_TIME_ELAPSED, q);
        splitQLive.push_back(q);
    }
    void splitQueryEnd()
    {
        if (gl.EndQuery)
            gl.EndQuery(GL_TIME_ELAPSED);
    }
    // Called right after a resolve: its readback drained the pipeline, so every query issued
    // before it has a result and GL_QUERY_RESULT does not block.
    void splitQueryCollect()
    {
        for (GLuint q : splitQLive)
        {
            uint64_t ns = 0;
            gl.GetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
            splitAcc.gpu += ns;
            ++splitAcc.gpuQueries;
            splitQFree.push_back(q);
        }
        splitQLive.clear();
    }
    void splitReport()
    {
        const auto now = std::chrono::steady_clock::now();
        if (splitT0 == std::chrono::steady_clock::time_point{})
        { splitT0 = now; splitAcc = SplitAcc{}; return; }
        if (splitAcc.resolves < 256ull)
            return;
        const double wall = double(std::chrono::duration_cast<std::chrono::nanoseconds>(now - splitT0).count());
        const double r = double(splitAcc.resolves);
        auto pct = [&](unsigned long long v) { return 100.0 * double(v) / wall; };
        auto ms = [&](unsigned long long v) { return double(v) * 1e-6 / r; };
        std::fprintf(stderr,
                     "[gsgpu:split] wall=%.2fs resolves=%llu (%.1f/s) scale=x%u | device%%: idle=%.1f draw=%.1f "
                     "snap=%.1f frame=%.1f other=%.1f | per-resolve ms: gpu-wait=%.2f blit+read=%.2f hires=%.2f "
                     "gpu-draw=%.2f (draw-jobs=%.1f) | producer-parked=%.1f%% (%llu waits)\n",
                     wall * 1e-9, splitAcc.resolves, r * 1e9 / wall, glrScale,
                     pct(splitAcc.idle), pct(splitAcc.draw), pct(splitAcc.snap), pct(splitAcc.frame),
                     pct(splitAcc.other), ms(splitAcc.finishWait), ms(splitAcc.blitRead), ms(splitAcc.hires),
                     ms(splitAcc.gpu), double(splitAcc.drawJobs) / r, pct(splitAcc.producerWait),
                     splitAcc.producerWaits);
        splitT0 = now;
        splitAcc = SplitAcc{};
    }
    // ★ The resolved frame, kept so a second request within the same present does not redo the
    // downsample and a SYNCHRONOUS readback. CopyFrameToHostRgba is called more than once per
    // displayed frame (preferred-source attempt, then the display-frame fallback), and a readback
    // stalls the whole GPU pipeline, so this removes more than half of them.
    const bool glrFrameCache = []
    { const char *e = std::getenv("PS2X_GS_GLR_FRAMECACHE"); return !(e && e[0] == '0'); }();
    std::vector<uint8_t> glrLastFrame;
    uint32_t glrLastW = 0, glrLastH = 0;
    uint64_t glrLastKey = ~0ull;
    // ★★★★ cont.345 flip-point snapshots (PS2X_GS_GLR_SNAPRESOLVE, default ON; =0 restores the
    // presenter's live resolve). The live resolve is a job the PRESENTER thread enqueues when it
    // wakes for a new flip snapshot; by then the GS worker may already have forwarded the guest's
    // NEXT frame's first draws into the same buffer (its clear quad, the far scene), so the
    // presenter showed a half-drawn next frame -- a one-flip flash of a different camera at every
    // cutscene cut, and (with the old resolve-armed clear, row 172) the solid dark-blue frame.
    // The CPU arm never had this: it copies VRAM ON THE WORKER at the flip marker, in FIFO order.
    // The GL arm now does the same: the worker enqueues a GlSnap at that marker, the device
    // resolves the display target behind everything queued before the flip and files the pixels
    // under the snapshot's seq; the presenter's GlFrame for that seq returns them. A miss (no
    // snapshot for that seq/key/size) falls back to the live resolve and is counted.
    const bool glrSnapResolve = []
    { const char *e = std::getenv("PS2X_GS_GLR_SNAPRESOLVE"); return !(e && e[0] == '0'); }();
    struct GlSnap { uint64_t seq = 0, key = ~0ull; uint32_t w = 0, h = 0; std::vector<uint8_t> px; };
    std::array<GlSnap, 6> glrSnaps{};
    size_t glrSnapNext = 0;
    unsigned long long glrSnapMade = 0, glrSnapNoTarget = 0, glrSnapHit = 0, glrSnapMissSeq = 0,
                       glrSnapMissKey = 0, glrSnapMissSize = 0, glrSnapLive = 0;
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> glrDispSize; // last live (w,h) per key
    // PS2X_GS_SCALE (default 1): the internal resolution multiplier. Phase 1 honours it for the
    // scene target's ALLOCATION only -- with nothing drawn, 1x and 4x differ solely in how much
    // the resolve blit shrinks. Kept here from the start so phase 5 does not have to retrofit it.
    const uint32_t glrScale = []
    {
        const char *e = std::getenv("PS2X_GS_SCALE");
        const int v = (e && e[0]) ? std::atoi(e) : 1;
        return static_cast<uint32_t>(v < 1 ? 1 : (v > 8 ? 8 : v));
    }();
    // ★★★★ cont.332c PS2X_GS_PRESENT_HIRES (default OFF): PRESENT the scene target at its OWN
    // resolution instead of the display-sized resolve. The scene is rendered at dispW*glrScale x
    // dispH*glrScale and then blitted DOWN to display size (phase 1's choice, which keeps the
    // readback flat in the scale) -- so with PS2X_GS_SCALE=4 the window has been showing a 640x448
    // image with 4x supersampled edges, never a sharper one. This reads the scene target itself
    // into a second latch that the presenter uploads to a matching raylib texture.
    // ⚠ It is a SECOND readback (16x the pixels at scale 4), which is exactly what phase 1 avoided;
    // it is default OFF until measured. The display-sized path is untouched, so the present cache,
    // field presentation, PS2X_GS_PRESENT_SAVE and every other consumer keep their 640-stride
    // buffer.
    const bool glrHires = []
    { const char *e = std::getenv("PS2X_GS_PRESENT_HIRES"); return e && e[0] && e[0] != '0'; }();
    // ★★★ cont.332c PS2X_GS_PRESENT_HIRES_SAVE=<dir> (+ _EVERY, default 256): dump BOTH images of
    // the SAME present -- the hi-res scene read and the display-sized resolve the window used to
    // show -- so the comparison needs no cross-run alignment at all. Two live runs cannot be
    // compared frame to frame (a 45 s screenshot burst caught the two arms at different beats:
    // best cross-arm RMSE 0.072), and that is the only honest way to ask "is it sharper".
    const char *glrHiresSaveDir = std::getenv("PS2X_GS_PRESENT_HIRES_SAVE");
    const unsigned long long glrHiresSaveEvery = []
    { const char *e = std::getenv("PS2X_GS_PRESENT_HIRES_SAVE_EVERY");
      return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 256ull; }();
    std::mutex hiresMx;
    std::vector<uint8_t> hiresBuf;
    uint32_t hiresW = 0, hiresH = 0;
    bool hiresFresh = false;
    unsigned long long hiresFrames = 0, hiresNs = 0;
    // ★★ cont.329g THE PIPELINE CENSUS (PS2X_GS_GLR_CENSUS=1, default OFF) -- the DEVICE half.
    // ~40% of the scene is missing and SEVEN candidate causes have been tested one at a time;
    // four eliminations are permanent, which says the MODEL of the draw path has a hole in it
    // rather than that the next hypothesis will be luckier. So count what survives EVERY stage of
    // ONE run -- batches in, batches dropped BY REASON, groups in, groups skipped BY REASON, draw
    // calls issued, fragments that actually passed the depth test, and draws per target between
    // clears -- and read off the stage where the count collapses. Every early return and every
    // `continue` on this path is counted, because a silent drop is exactly the kind of gap an
    // error counter never shows (the same shape as cont.231's null ablation).
    // ⚠ COUNTING ONLY: nothing here changes what is drawn, so a census run and a normal run
    // render the same frames.
    const bool glrCensus = []
    { const char *e = std::getenv("PS2X_GS_GLR_CENSUS"); return e && e[0] && e[0] != '0'; }();
    // Per RENDER TARGET, because the accumulation hypothesis is a per-target question: how many
    // draws land between two clears of the same buffer? A target that is drawn into every frame
    // but only cleared occasionally composites frames on top of each other -- which is what dark
    // vertex colours reaching a 208/255 frame would require.
    struct GlcTarget
    {
        unsigned long long batches = 0, groups = 0, verts = 0; // since THIS target's last clear
        unsigned long long lifeBatches = 0, lifeGroups = 0, lifeVerts = 0;
        unsigned long long clears = 0, resolves = 0, resolvesNoMark = 0;
        unsigned long long groupsAtClearSum = 0, groupsAtClearMax = 0, clearSamples = 0;
    };
    std::unordered_map<uint64_t, GlcTarget> glcTargets;
    unsigned long long glcBatchIn = 0, glcBatchDrawn = 0, glcBatchDropEmpty = 0,
                       glcBatchDropRef = 0, glcBatchDropTarget = 0, glcBatchDropProgram = 0;
    unsigned long long glcBatchVertsIn = 0, glcBatchGroupsIn = 0;
    unsigned long long glcGroupIn = 0, glcGroupEmpty = 0, glcGroupScissor = 0;
    unsigned long long glcDrawCalls = 0, glcVertsDrawn = 0;
    unsigned long long glcTexMissing = 0, glcTexTargetMissing = 0, glcGroupUntextured = 0;
    unsigned long long glcResolves = 0, glcResolvesNoMark = 0, glcClears = 0;
    // Fragments that PASSED the depth test, via GL_SAMPLES_PASSED around each batch's draw loop.
    // ⚠ Two counting notes, so the number is not over-read: a discarded fragment (the alpha test)
    // never reaches the depth test and is not counted, and the AFAIL two-pass path draws its
    // group TWICE, so its samples are counted twice -- `two-pass` is reported alongside so the
    // correction is available. Results are collected through a 4-deep ring so the query never
    // stalls the device thread on the frame it was issued in.
    unsigned long long glcSamples = 0, glcQueriesDone = 0;
    GLuint glcQ[4] = {0, 0, 0, 0};
    bool glcQLive[4] = {false, false, false, false};
    unsigned glcQHead = 0;

    void glcQueryBegin()
    {
        if (!glrCensus || !gl.GenQueries || !gl.BeginQuery || !gl.GetQueryObjectui64v)
            return;
        const unsigned slot = glcQHead & 3u;
        if (glcQ[slot] == 0u)
            gl.GenQueries(1, &glcQ[slot]);
        if (glcQ[slot] == 0u)
            return;
        if (glcQLive[slot])
        {
            uint64_t n = 0;
            gl.GetQueryObjectui64v(glcQ[slot], GL_QUERY_RESULT, &n); // 4 batches old: ready
            glcSamples += n;
            ++glcQueriesDone;
            glcQLive[slot] = false;
        }
        gl.BeginQuery(GL_SAMPLES_PASSED, glcQ[slot]);
    }
    void glcQueryEnd()
    {
        if (!glrCensus || !gl.EndQuery || glcQ[glcQHead & 3u] == 0u)
            return;
        gl.EndQuery(GL_SAMPLES_PASSED);
        glcQLive[glcQHead & 3u] = true;
        ++glcQHead;
    }
    // A clear ENDS a target's frame: sample how much was drawn into it since the last one.
    void glcNoteClear(uint64_t key)
    {
        if (!glrCensus)
            return;
        ++glcClears;
        GlcTarget &t = glcTargets[key];
        ++t.clears;
        ++t.clearSamples;
        t.groupsAtClearSum += t.groups;
        if (t.groups > t.groupsAtClearMax)
            t.groupsAtClearMax = t.groups;
        t.batches = t.groups = t.verts = 0;
    }
    // ★★ cont.329h: read the CURRENTLY BOUND target back and report what is on it. Used by the
    // per-draw trace, so it reads the scene target directly (no resolve, no present path).
    unsigned long long glcTraceSeq = 0;
    void glcTraceShot(const char *what, uint32_t w, uint32_t h, uint64_t tgtKey)
    {
        static std::vector<uint8_t> px;
        px.resize(static_cast<size_t>(w) * h * 4u);
        gl.PixelStorei(GL_PACK_ALIGNMENT, 1);
        gl.ReadPixels(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h), GL_RGBA,
                      GL_UNSIGNED_BYTE, px.data());
        static std::vector<uint64_t> seen(1u << 18);
        std::fill(seen.begin(), seen.end(), 0ull);
        unsigned long long sum = 0, distinct = 0, n = 0, white = 0;
        for (size_t i = 0; i + 3 < px.size(); i += 4)
        {
            const unsigned r = px[i], g2 = px[i + 1], b2 = px[i + 2];
            sum += r + g2 + b2;
            if (r > 250u && g2 > 250u && b2 > 250u) ++white;
            const uint32_t c = (uint32_t(r) << 16) | (uint32_t(g2) << 8) | b2;
            uint64_t &wd = seen[c >> 6];
            const uint64_t bit = 1ull << (c & 63u);
            if (!(wd & bit)) { wd |= bit; ++distinct; }
            ++n;
        }
        const double fn = n ? double(n) : 1.0;
        std::fprintf(stderr, "[gsgpu:trace] tgt=0x%-4llx %-96s luma=%6.1f distinct=%6llu white=%5.1f%%\n",
                     (unsigned long long)tgtKey, what, double(sum) / (3.0 * fn), distinct,
                     100.0 * double(white) / fn);
        // ★ cont.329i: the film strip. One PPM per traced draw call, named by sequence and target,
        // so a frame can be watched target by target instead of inferred from luma.
        const char *dir = std::getenv("PS2X_GS_DRAWCMP_DIR");
        if (dir && dir[0] && glcTraceSeq < 400ull)
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/t%04llu_fbp%03llx.ppm", dir, glcTraceSeq,
                          (unsigned long long)tgtKey);
            if (FILE *f = std::fopen(path, "wb"))
            {
                std::fprintf(f, "P6\n%u %u\n255\n", w, h);
                // GL reads bottom-up; write top-down so the file looks like the screen.
                for (uint32_t y = 0; y < h; ++y)
                {
                    const uint8_t *row = px.data() + static_cast<size_t>(h - 1u - y) * w * 4u;
                    for (uint32_t x = 0; x < w; ++x)
                    { std::fputc(row[x*4], f); std::fputc(row[x*4+1], f); std::fputc(row[x*4+2], f); }
                }
                std::fclose(f);
            }
        }
        ++glcTraceSeq;
    }
    // ★★★★ cont.331q PS2X_GS_GLR_ORDER -- TRACE ONE TARGET IN DEVICE ORDER.
    // Five hypotheses for "every seed placed before the resolve lands and is gone by resolve
    // time" have now been formed and killed from AGGREGATES around a sequence nobody has
    // actually watched (cont.331l page-ownership, 331m tgtFresh, 331n source-side/order,
    // 331o+331p the darken pass). This stops inferring: every clear / seed / group draw /
    // resolve that touches one target key is printed IN THE ORDER THE DEVICE THREAD EXECUTES
    // IT, each followed by that target's own read-back mean. Every one of those steps runs on
    // this single device thread (Job::Kind::GlDraw -> drawBatchGlOnThread, Job::Kind::GlFrame
    // -> glFrameOnThread), so a counter incremented here IS device order -- no inference.
    //
    //   PS2X_GS_GLR_ORDER=<fbp>   the target to read back: a number (hex with 0x, else decimal,
    //                             so `0` really means fbp=0x0), or `disp` to latch whichever key
    //                             the first armed resolve selects. Unset = OFF (the default).
    //   PS2X_GS_GLR_ORDER_AT=<n>  arm at resolve n (default 0 = from the start).
    //   PS2X_GS_GLR_ORDER_N=<n>   trace n GLOBAL resolves, then stop for good (default 6 -- the
    //                             display alternates between two buffers, so ~6 covers three
    //                             frames of the traced one). ⚠ The window is counted in global
    //                             resolves, NOT in resolves of the traced key: a key the display
    //                             stops selecting would never close a key-counted window, and the
    //                             trace would print for the rest of the run. PS2X_GS_GLR_ORDER_MAX
    //                             (default 6000) is the second belt -- a hard line cap, because an
    //                             instrument that floods the log costs a cycle exactly like one
    //                             that is silent.
    //
    // Steps touching OTHER targets are printed too, as one cheap line with no read-back, so the
    // interleaving is visible (a draw into 0x180 between 0x0's seed and its resolve is exactly
    // the kind of thing an aggregate cannot show). The read-back is a full glReadPixels of the
    // scene target, which is why this is default OFF and bounded to a couple of frames.
    const char *const glrOrderEnv = std::getenv("PS2X_GS_GLR_ORDER");
    const bool glrOrderDisp = glrOrderEnv && glrOrderEnv[0] &&
                              (glrOrderEnv[0] == 'd' || glrOrderEnv[0] == 'D');
    uint64_t glrOrderKey = [this]
    {
        if (!glrOrderEnv || !glrOrderEnv[0] || glrOrderDisp)
            return ~0ull;
        return std::strtoull(glrOrderEnv, nullptr, 0);
    }();
    const unsigned long long glrOrderAt = []
    { const char *e = std::getenv("PS2X_GS_GLR_ORDER_AT"); return (e && e[0]) ? std::strtoull(e, nullptr, 0) : 0ull; }();
    const unsigned long long glrOrderN = []
    { const char *e = std::getenv("PS2X_GS_GLR_ORDER_N"); return (e && e[0]) ? std::strtoull(e, nullptr, 0) : 6ull; }();
    const unsigned long long glrOrderMax = []
    { const char *e = std::getenv("PS2X_GS_GLR_ORDER_MAX"); return (e && e[0]) ? std::strtoull(e, nullptr, 0) : 6000ull; }();
    // ★★★ cont.331q PS2X_GS_GLR_ORDER_FLIP=<n>: arm at the GUEST DISPLAY FLIP n instead of at a
    // device resolve. Every era in this arc is named by a flip, because the present captures are
    // flip-indexed (presentation is host-paced; only the flip is a guest-driven instant). The
    // device's own glrFrames is a resolve count whose ratio to flips is NOT fixed -- it moves with
    // frame-cache hits and with how much the trace itself slows the device thread -- so aiming a
    // trace at "the regression at flip 900" through glrFrames is guesswork. This makes it exact.
    const unsigned long long glrOrderFlip = []
    { const char *e = std::getenv("PS2X_GS_GLR_ORDER_FLIP"); return (e && e[0]) ? std::strtoull(e, nullptr, 0) : 0ull; }();
    unsigned long long glrOrderSeq = 0;
    unsigned long long glrOrderResolves = 0;
    // Latched the first time the flip arm trips, so the window still closes on a fixed count of
    // GLOBAL resolves (the flip counter keeps moving while we trace).
    unsigned long long glrOrderArmFrame = ~0ull;
    bool glrOrderArmed()
    {
        if (!glrOrderEnv || !glrOrderEnv[0] || glrOrderSeq >= glrOrderMax)
            return false;
        if (glrOrderFlip)
        {
            if (glrOrderArmFrame == ~0ull)
            {
                if (gs2CurrentFlip() < glrOrderFlip)
                    return false;
                glrOrderArmFrame = glrFrames;
                std::fprintf(stderr,
                             "[gsgpu:order] armed at guest flip %llu (>= %llu), device frame %llu\n",
                             gs2CurrentFlip(), glrOrderFlip, glrFrames);
            }
            return glrFrames < glrOrderArmFrame + glrOrderN;
        }
        return glrFrames >= glrOrderAt && glrFrames < glrOrderAt + glrOrderN;
    }
    // Read the target back and print one device-order line. `what` describes the step that JUST
    // ran, so the mean is the state AFTER it (the resolve prints before it reads, on purpose).
    void glrOrderShot(const char *what, GlTarget &t, uint64_t key)
    {
        if (!glrOrderArmed() || key != glrOrderKey || !t.fbo || !t.w || !t.h)
            return;
        static std::vector<uint8_t> px;
        px.resize(static_cast<size_t>(t.w) * t.h * 4u);
        gl.BindFramebuffer(GL_READ_FRAMEBUFFER, t.fbo);
        gl.PixelStorei(GL_PACK_ALIGNMENT, 1);
        gl.ReadPixels(0, 0, static_cast<GLsizei>(t.w), static_cast<GLsizei>(t.h), GL_RGBA,
                      GL_UNSIGNED_BYTE, px.data());
        unsigned long long sum = 0, nz = 0, n = 0;
        unsigned mx = 0;
        for (size_t i = 0; i + 3 < px.size(); i += 4)
        {
            const unsigned r = px[i], g2 = px[i + 1], b2 = px[i + 2];
            sum += r + g2 + b2;
            if (r | g2 | b2) ++nz;
            if (r > mx) mx = r;
            if (g2 > mx) mx = g2;
            if (b2 > mx) mx = b2;
            ++n;
        }
        std::fprintf(stderr,
                     "[gsgpu:order] #%05llu flip=%llu f=%llu tgt=0x%llx fbo=%u %ux%u | mean=%8.4f max=%3u "
                     "nonzero=%llu/%llu | %s\n",
                     ++glrOrderSeq, gs2CurrentFlip(), glrFrames, (unsigned long long)key, t.fbo, t.w, t.h,
                     n ? double(sum) / (3.0 * double(n)) : 0.0, mx, nz, n, what);
        gl.BindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    }
    // A step on a target we are NOT reading back: one line, no GL work, so the interleaving is
    // on the record without paying for it.
    void glrOrderNote(const char *what, uint64_t key)
    {
        if (!glrOrderArmed() || key == glrOrderKey)
            return;
        std::fprintf(stderr,
                     "[gsgpu:order] #%05llu flip=%llu f=%llu tgt=0x%llx | (other target)      | %s\n",
                     ++glrOrderSeq, gs2CurrentFlip(), glrFrames, (unsigned long long)key, what);
    }
    // ★★★★★ cont.332i PS2X_GS_GLPIXWATCH=<x>:<y> (default OFF, read-only): THE GL-SIDE
    // COUNTERPART OF PS2X_GS_PIXWATCH. cont.332h asked one screen pixel of the CPU rasterizer
    // "which draws paint you" and got a list of ~13 layered alpha-blended T4 draws; the same
    // question cannot be asked of the GL arm, because that watch lives in the CPU pixel write
    // path, which a PS2X_GS_RENDERER=gl run never enters. This asks it here: read the one pixel
    // back after every group, and print the group whenever the pixel MOVED. The two traces then
    // line up row by row and "which layer does GL lose or weaken" is a table, not an argument.
    //
    // Aimed by the SAME env as the CPU arm (PS2X_GS_PROBEFLIP / _PROBEFLIPW), so one flip number
    // arms both renderers and the two traces are of the same guest instant -- which under
    // PS2X_VIRTUAL_TIME=1 is the same guest instant in both arms (cont.332h).
    //
    // ⚠ Coordinates are the GS/display ones the CPU watch prints (y from the TOP). The target is
    // at glrScale x native with GL's origin at the BOTTOM, so the row flips exactly as the
    // scissor's does. The read is one pixel, but it is still a pipeline stall per group, hence
    // the flip window and the line cap.
    // ★★★★★ cont.332i PS2X_GS_GLR_AFAIL (default 1 = the PCSX2-faithful selector; =0 restores
    // the pre-932 behaviour for an A/B; =2 forces pass/fail; =3 forces a plain discard).
    //
    // THE BUG THIS FIXES. AFAIL=FB_ONLY means "a fragment that fails the alpha test still writes
    // COLOUR, but not Z". We emulated that as PCSX2's SIMPLE_FB_ONLY: pass 1 draws every fragment
    // with depth writes OFF, pass 2 re-draws the passers depth-only. ⚠ PCSX2 only picks that split
    // when `independent_z` holds (GSRendererHW.cpp EmulateATST):
    //     independent_z = (ZTST == GEQUAL && m_vt.m_eq.z) || ZTST == ALWAYS || !zwe ||
    //                     m_prim_overlap == PRIM_OVERLAP_NO
    // i.e. only when the Z result cannot depend on overlapping Z writes INSIDE the draw. We applied
    // it unconditionally, so for a character model -- one draw, hundreds of overlapping triangles,
    // ZTST=GEQUAL with Z writes on -- pass 1 let every triangle write colour with NO occlusion
    // between them, and the pixel kept whichever triangle was submitted last instead of the
    // nearest. Measured (PS2X_GS_GLPIXWATCH at 275:319, flip 7000): the one draw that paints the
    // character (128x128 PSMT8, ate=1 atst=GEQUAL aref=128, afail=FB_ONLY, ztst=GEQUAL zmsk=0)
    // moves the CPU pixel to exactly its source colour (86..183 per channel) and the GL pixel only
    // to 33..53.
    //
    // The fix is PCSX2's own fallback for !independent_z, PASS_THEN_FAIL:
    //   pass 1 -- normal alpha test, failures DISCARDED, colour AND Z written (ordering correct)
    //   pass 2 -- the test INVERTED, only failures survive, colour written, Z write disabled
    // (EmulateATST's `else` branch + EmulateAlphaTestSecondPass's AFAIL_FB_ONLY case).
    const int glrAfailMode = []
    { const char *e = std::getenv("PS2X_GS_GLR_AFAIL"); return (e && e[0]) ? std::atoi(e) : 1; }();
    unsigned long long glrAfailSimple = 0, glrAfailPassFail = 0, glrAfailOnePass = 0;
    const int glPixWatchX = []
    { const char *e = std::getenv("PS2X_GS_GLPIXWATCH"); return (e && e[0]) ? std::atoi(e) : -1; }();
    const int glPixWatchY = []
    { const char *e = std::getenv("PS2X_GS_GLPIXWATCH");
      const char *c = (e && e[0]) ? std::strchr(e, ':') : nullptr; return c ? std::atoi(c + 1) : -1; }();
    const unsigned long long glPixWatchN = []
    { const char *e = std::getenv("PS2X_GS_GLPIXWATCH_N"); return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 256ull; }();
    // The same window probeFlipOk() applies on the CPU side: [PROBEFLIP, PROBEFLIP + PROBEFLIPW).
    const unsigned long long glPixWatchFlip = []
    { const char *e = std::getenv("PS2X_GS_PROBEFLIP"); return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 0ull; }();
    const unsigned long long glPixWatchFlipW = []
    { const char *e = std::getenv("PS2X_GS_PROBEFLIPW"); unsigned long long v = (e && e[0]) ? std::strtoull(e, nullptr, 10) : 2ull; return v ? v : 2ull; }();
    unsigned long long glPixWatchShown = 0, glPixWatchQuiet = 0, glPixWatchBatches = 0;
    bool glPixWatchOn() const { return glPixWatchX >= 0 && glPixWatchY >= 0; }
    bool glPixWatchArmed()
    {
        if (!glPixWatchOn() || glPixWatchShown >= glPixWatchN)
            return false;
        if (glPixWatchFlip == 0ull)
            return true;
        const unsigned long long f = gs2CurrentFlip();
        return f >= glPixWatchFlip && f < glPixWatchFlip + glPixWatchFlipW;
    }
    // One pixel out of `t`, packed 0xRRGGBBAA. `ok` is false when the watched pixel is outside
    // this target -- a batch drawing into a smaller buffer must not be reported as "unchanged".
    uint32_t glPixWatchRead(GlTarget &t, uint32_t refH, bool &ok)
    {
        ok = false;
        if (!t.fbo || !t.w || !t.h)
            return 0u;
        const int sc = static_cast<int>(glrScale);
        const int gx = glPixWatchX * sc;
        const int gy = (static_cast<int>(refH) - 1 - glPixWatchY) * sc;
        if (gx < 0 || gy < 0 || gx >= static_cast<int>(t.w) || gy >= static_cast<int>(t.h))
            return 0u;
        uint8_t px[4] = { 0, 0, 0, 0 };
        gl.BindFramebuffer(GL_READ_FRAMEBUFFER, t.fbo);
        gl.PixelStorei(GL_PACK_ALIGNMENT, 1);
        gl.ReadPixels(gx, gy, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        gl.BindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        ok = true;
        return (uint32_t(px[0]) << 24) | (uint32_t(px[1]) << 16) | (uint32_t(px[2]) << 8) | uint32_t(px[3]);
    }
    void glcReport()
    {
        const double f = glcResolves ? double(glcResolves) : 1.0;
        std::fprintf(stderr,
                     "[gsgpu:glcensus] resolves=%llu | batches in=%llu drawn=%llu "
                     "drop{empty=%llu ref=%llu target=%llu program=%llu} | "
                     "groups in=%llu (arrived=%llu) skip{empty=%llu scissor=%llu} | "
                     "drawcalls=%llu two-pass=%llu verts{in=%llu drawn=%llu} | "
                     "tex{binds=%llu missing=%llu tgt-missing=%llu untextured-groups=%llu} | "
                     "depth-passed-samples=%llu (q=%llu) | fbmsk{masked-draws=%llu partial=%llu}\n",
                     glcResolves, glcBatchIn, glcBatchDrawn, glcBatchDropEmpty, glcBatchDropRef,
                     glcBatchDropTarget, glcBatchDropProgram, glcGroupIn, glcBatchGroupsIn,
                     glcGroupEmpty, glcGroupScissor, glcDrawCalls, glrTwoPass, glcBatchVertsIn,
                     glcVertsDrawn, glrTexBinds, glcTexMissing, glcTexTargetMissing,
                     glcGroupUntextured, glcSamples, glcQueriesDone, glrFbmskMasked, glrFbmskPartial);
        std::fprintf(stderr,
                     "[gsgpu:glcensus] PER RESOLVE: batches=%.1f groups=%.1f verts=%.0f "
                     "drawcalls=%.1f depth-passed=%.0f (a 640x448 frame is 286720 px) | "
                     "clears=%llu (%.2f/resolve) targets=%zu\n",
                     double(glcBatchDrawn) / f, double(glcGroupIn) / f, double(glcVertsDrawn) / f,
                     double(glcDrawCalls) / f, double(glcSamples) / f, glcClears,
                     double(glcClears) / f, glcTargets.size());
        for (const auto &kv : glcTargets)
        {
            const GlcTarget &t = kv.second;
            std::fprintf(stderr,
                         "[gsgpu:glcensus/tgt] fbp=0x%llx life{batches=%llu groups=%llu verts=%llu} "
                         "clears=%llu resolves=%llu resolve-without-clear-mark=%llu "
                         "groups-between-clears{mean=%.1f max=%llu} pending-now{groups=%llu}\n",
                         (unsigned long long)kv.first, t.lifeBatches, t.lifeGroups, t.lifeVerts,
                         t.clears, t.resolves, t.resolvesNoMark,
                         t.clearSamples ? double(t.groupsAtClearSum) / double(t.clearSamples) : 0.0,
                         t.groupsAtClearMax, t.groups);
        }
    }
    unsigned long hwDraws = 0;
    unsigned long long hwPrims = 0;
    // The FIFO puts a VerifyRange immediately behind the batch it checks, so the last
    // rastered primitive IS the one being verified -- enough to name a divergence precisely
    // for the single-primitive gate.
    GsGpuState lastState{};
    GsGpuGeom lastGeom{};
    GLuint timeQuery = 0;
    bool perfTiming = false;
    unsigned long long gpuNs = 0, gpuPrims = 0;
    unsigned long gpuDispatches = 0;
    unsigned long rasterBatches = 0;
    unsigned long long rasterBatchPrims = 0;
    unsigned long rasterPrims = 0, rasterVerifies = 0, rasterMismRanges = 0;
    unsigned long long rasterMismWords = 0;

    bool fail(const char *stage)
    {
        std::fprintf(stderr, "[gsgpu] DISABLED: %s (egl=0x%x)\n", stage,
                     static_cast<unsigned>(eglGetError()));
        return false;
    }

    void threadLoop()
    {
        const bool ok = initOnThread();
        {
            std::lock_guard<std::mutex> lk(mx);
            state = ok ? State::Ready : State::Failed;
        }
        cvDone.notify_all();
        if (!ok)
            return; // callers see Failed and never post a job
        for (;;)
        {
            Job j;
            const auto w0 = glrSplit ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            {
                std::unique_lock<std::mutex> lk(mx);
                cv.wait(lk, [&] { return stop || !jobs.empty(); });
                if (stop)
                    return;
                j = std::move(jobs.front());
                jobs.pop_front();
                pendingBytes -= jobBytes(j);
                if (glrSplit)
                {
                    // The producers' parked time is accumulated under mx (enqueue); move it here.
                    splitAcc.producerWait += splitProducerWait;
                    splitAcc.producerWaits += splitProducerWaits;
                    splitProducerWait = 0;
                    splitProducerWaits = 0;
                }
            }
            // cont.371 PS2X_GS_GLR_SPLIT: idle = parked on the empty queue; busy by job kind below.
            std::chrono::steady_clock::time_point w1{};
            if (glrSplit)
            {
                w1 = std::chrono::steady_clock::now();
                splitAcc.idle += (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count();
                if (j.kind == Job::Kind::GlDraw)
                {
                    splitQueryBegin();
                    ++splitAcc.drawJobs;
                }
            }
            const Job::Kind splitKind = j.kind;
            struct SplitEnd
            {
                Impl &d;
                Job::Kind kind;
                std::chrono::steady_clock::time_point w1;
                ~SplitEnd()
                {
                    if (!d.glrSplit)
                        return;
                    if (kind == Job::Kind::GlDraw)
                        d.splitQueryEnd();
                    const unsigned long long ns = (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - w1).count();
                    switch (kind)
                    {
                    case Job::Kind::GlDraw: d.splitAcc.draw += ns; break;
                    case Job::Kind::GlSnap: d.splitAcc.snap += ns; break;
                    case Job::Kind::GlFrame: d.splitAcc.frame += ns; break;
                    default: d.splitAcc.other += ns; break;
                    }
                    if (kind == Job::Kind::GlSnap || kind == Job::Kind::GlFrame)
                    {
                        d.splitQueryCollect();
                        d.splitReport();
                    }
                }
            } splitEnd{*this, splitKind, w1};
            switch (j.kind)
            {
            case Job::Kind::Decode:
            {
                // The caller parks on cvDone until *done, so decode.out and the flags
                // (caller-stack) are stable while we run unlocked.
                const bool r = decodeOnThread(j.decode);
                std::lock_guard<std::mutex> lk(mx);
                if (j.result)
                    *j.result = r;
                if (j.done)
                    *j.done = true;
                break;
            }
            case Job::Kind::Upload:
                uploadOnThread(j.chunk);
                break;
            case Job::Kind::Verify:
                verifyOnThread(j.chunk);
                break;
            case Job::Kind::PatchRaw:
                patchRawOnThread(j.rawOffset, j.raw);
                break;
            case Job::Kind::Readback:
            {
                readbackOnThread(j.rawOffset, j.decode.vramSize, j.readOut);
                std::lock_guard<std::mutex> lk(mx);
                if (j.done)
                    *j.done = true;
                break;
            }
            case Job::Kind::RasterBatch:
                rasterBatchOnThread(j.batch);
                break;
            case Job::Kind::VerifyRange:
                verifyRangeOnThread(j.rawOffset, j.raw, j.tag);
                break;
            case Job::Kind::GlSnap:
                glSnapOnThread(j.tag, j.seq);
                break;
            case Job::Kind::GlDraw:
                drawBatchGlOnThread(j.glBatch);
                break;
            case Job::Kind::GlFrame:
            {
                // Synchronous, like Decode: the caller parks on cvDone, so decode.out and the
                // caller-stack flags are stable while this runs unlocked.
                const bool r = glFrameForPresent(j.tag, j.seq, j.decode.width, j.decode.height, j.decode.out);
                std::lock_guard<std::mutex> lk(mx);
                if (j.result)
                    *j.result = r;
                if (j.done)
                    *j.done = true;
                break;
            }
            case Job::Kind::VerifyFull:
            {
                const bool r = verifyFullOnThread(j.decode.vram, j.decode.vramSize,
                                                  j.drawPageMask, j.quiet,
                                                  j.outFirstWord, j.outStaleWords);
                std::lock_guard<std::mutex> lk(mx);
                if (j.result)
                    *j.result = r;
                if (j.done)
                    *j.done = true;
                break;
            }
            }
            cvDone.notify_all(); // decode completion AND queue-space waiters
        }
    }

    void enqueue(Job &&j)
    {
        std::unique_lock<std::mutex> lk(mx);
        auto hasSpace = [&] { return stop || (jobs.size() < kMaxPendingJobs &&
                                              pendingBytes < kMaxPendingBytes); };
        if (glrSplit && !hasSpace())
        {
            // cont.371 PS2X_GS_GLR_SPLIT: a producer parked on a full queue (back-pressure).
            const auto p0 = std::chrono::steady_clock::now();
            cvDone.wait(lk, hasSpace);
            splitProducerWait += (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - p0).count();
            ++splitProducerWaits;
        }
        cvDone.wait(lk, hasSpace);
        if (stop)
            return;
        pendingBytes += jobBytes(j);
        jobs.push_back(std::move(j));
        cv.notify_one();
    }

    GLuint buildProgram(const char *src, const char *name)
    {
        const GLuint shader = gl.CreateShader(GL_COMPUTE_SHADER);
        gl.ShaderSource(shader, 1, &src, nullptr);
        gl.CompileShader(shader);
        GLint status = 0;
        gl.GetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (!status)
        {
            char log[1024] = {};
            gl.GetShaderInfoLog(shader, sizeof(log) - 1, nullptr, log);
            std::fprintf(stderr, "[gsgpu] %s shader compile log:\n%s\n", name, log);
            return 0;
        }
        const GLuint prog = gl.CreateProgram();
        gl.AttachShader(prog, shader);
        gl.LinkProgram(prog);
        gl.GetProgramiv(prog, GL_LINK_STATUS, &status);
        gl.DeleteShader(shader);
        if (!status)
        {
            char log[1024] = {};
            gl.GetProgramInfoLog(prog, sizeof(log) - 1, nullptr, log);
            std::fprintf(stderr, "[gsgpu] %s program link log:\n%s\n", name, log);
            return 0;
        }
        return prog;
    }

    GLuint compileStage(GLenum type, const char *src, const char *name)
    {
        const GLuint shader = gl.CreateShader(type);
        gl.ShaderSource(shader, 1, &src, nullptr);
        gl.CompileShader(shader);
        GLint status = 0;
        gl.GetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (!status)
        {
            char log[2048] = {};
            gl.GetShaderInfoLog(shader, sizeof(log) - 1, nullptr, log);
            std::fprintf(stderr, "[gsgpu] %s shader compile log:\n%s\n", name, log);
            return 0;
        }
        return shader;
    }

    GLuint buildGraphicsProgram(const char *vsSrc, const char *fsSrc, const char *name)
    {
        const GLuint vs = compileStage(GL_VERTEX_SHADER, vsSrc, name);
        const GLuint fs = compileStage(GL_FRAGMENT_SHADER, fsSrc, name);
        if (!vs || !fs)
            return 0;
        const GLuint prog = gl.CreateProgram();
        gl.AttachShader(prog, vs);
        gl.AttachShader(prog, fs);
        gl.LinkProgram(prog);
        GLint status = 0;
        gl.GetProgramiv(prog, GL_LINK_STATUS, &status);
        gl.DeleteShader(vs);
        gl.DeleteShader(fs);
        if (!status)
        {
            char log[2048] = {};
            gl.GetProgramInfoLog(prog, sizeof(log) - 1, nullptr, log);
            std::fprintf(stderr, "[gsgpu] %s program link log:\n%s\n", name, log);
            return 0;
        }
        return prog;
    }

    bool initOnThread()
    {
        // Surfaceless platform display first (no window system involvement at all);
        // fall back to the default (X11) display, still never drawing to a surface.
        auto getPlatformDisplay = reinterpret_cast<EGLDisplay (*)(EGLenum, void *, const EGLint *)>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
        if (getPlatformDisplay)
            display = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        if (display == EGL_NO_DISPLAY)
            display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY)
            return fail("no EGL display");
        EGLint maj = 0, min = 0;
        if (!eglInitialize(display, &maj, &min))
            return fail("eglInitialize");
        if (!eglBindAPI(EGL_OPENGL_API))
            return fail("eglBindAPI(OPENGL)");

        const EGLint ctxAttribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 3,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE};
        // No-config context + surfaceless current (Mesa supports both); pbuffer fallback.
        context = eglCreateContext(display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctxAttribs);
        if (context == EGL_NO_CONTEXT)
        {
            const EGLint cfgAttribs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                         EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
            EGLConfig cfg = nullptr;
            EGLint n = 0;
            if (!eglChooseConfig(display, cfgAttribs, &cfg, 1, &n) || n == 0)
                return fail("eglChooseConfig");
            context = eglCreateContext(display, cfg, EGL_NO_CONTEXT, ctxAttribs);
            if (context == EGL_NO_CONTEXT)
                return fail("eglCreateContext(4.3 core)");
            const EGLint pbAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
            surface = eglCreatePbufferSurface(display, cfg, pbAttribs);
            if (surface == EGL_NO_SURFACE)
                return fail("eglCreatePbufferSurface");
        }
        if (!eglMakeCurrent(display, surface, surface, context))
            return fail("eglMakeCurrent");

        bool ok = true;
        ok &= loadGlFn(gl.GetString, "glGetString");
        ok &= loadGlFn(gl.GetError, "glGetError");
        ok &= loadGlFn(gl.CreateShader, "glCreateShader");
        ok &= loadGlFn(gl.ShaderSource, "glShaderSource");
        ok &= loadGlFn(gl.CompileShader, "glCompileShader");
        ok &= loadGlFn(gl.GetShaderiv, "glGetShaderiv");
        ok &= loadGlFn(gl.GetShaderInfoLog, "glGetShaderInfoLog");
        ok &= loadGlFn(gl.CreateProgram, "glCreateProgram");
        ok &= loadGlFn(gl.AttachShader, "glAttachShader");
        ok &= loadGlFn(gl.LinkProgram, "glLinkProgram");
        ok &= loadGlFn(gl.GetProgramiv, "glGetProgramiv");
        ok &= loadGlFn(gl.GetProgramInfoLog, "glGetProgramInfoLog");
        ok &= loadGlFn(gl.DeleteShader, "glDeleteShader");
        ok &= loadGlFn(gl.GenBuffers, "glGenBuffers");
        ok &= loadGlFn(gl.BindBuffer, "glBindBuffer");
        ok &= loadGlFn(gl.BindBufferBase, "glBindBufferBase");
        ok &= loadGlFn(gl.BufferData, "glBufferData");
        ok &= loadGlFn(gl.BufferSubData, "glBufferSubData");
        ok &= loadGlFn(gl.GetBufferSubData, "glGetBufferSubData");
        ok &= loadGlFn(gl.UseProgram, "glUseProgram");
        ok &= loadGlFn(gl.GetUniformLocation, "glGetUniformLocation");
        ok &= loadGlFn(gl.Uniform1ui, "glUniform1ui");
        ok &= loadGlFn(gl.DispatchCompute, "glDispatchCompute");
        ok &= loadGlFn(gl.MemoryBarrier, "glMemoryBarrier");
        ok &= loadGlFn(gl.Finish, "glFinish");
        ok &= loadGlFn(gl.GenQueries, "glGenQueries");
        ok &= loadGlFn(gl.BeginQuery, "glBeginQuery");
        ok &= loadGlFn(gl.EndQuery, "glEndQuery");
        ok &= loadGlFn(gl.GetQueryObjectui64v, "glGetQueryObjectui64v");
        if (!ok)
            return fail("eglGetProcAddress(GL 4.3 core fn)");
        // The hardware-raster entry points: all GL 3.0-4.3 core, but their absence must only
        // disable that path, never the device.
        bool hwOk = true;
        hwOk &= loadGlFn(gl.GenVertexArrays, "glGenVertexArrays");
        hwOk &= loadGlFn(gl.BindVertexArray, "glBindVertexArray");
        hwOk &= loadGlFn(gl.DrawArrays, "glDrawArrays");
        hwOk &= loadGlFn(gl.Enable, "glEnable");
        hwOk &= loadGlFn(gl.Disable, "glDisable");
        hwOk &= loadGlFn(gl.Viewport, "glViewport");
        hwOk &= loadGlFn(gl.GenFramebuffers, "glGenFramebuffers");
        hwOk &= loadGlFn(gl.BindFramebuffer, "glBindFramebuffer");
        hwOk &= loadGlFn(gl.FramebufferParameteri, "glFramebufferParameteri");
        hwOk &= loadGlFn(gl.CheckFramebufferStatus, "glCheckFramebufferStatus");
        hwOk &= loadGlFn(gl.Uniform1f, "glUniform1f");
        if (!hwOk)
        {
            hwFailed = true;
            if (hwRaster)
                std::fprintf(stderr, "[gsgpu] hwraster DISABLED: a GL entry point is missing\n");
        }

        // ★ cont.329 phase 1: the renderer's own entry points, tracked SEPARATELY from hwOk so a
        // missing one disables the GL renderer without touching the hw-raster spike (and vice
        // versa). All are GL 3.0/3.3 core -- the context is 4.3, so a failure here means a driver
        // problem, not a version problem, and is worth saying out loud.
        bool glrOk = true;
        glrOk &= loadGlFn(gl.GenTextures, "glGenTextures");
        glrOk &= loadGlFn(gl.DeleteTextures, "glDeleteTextures");
        glrOk &= loadGlFn(gl.BindTexture, "glBindTexture");
        glrOk &= loadGlFn(gl.ActiveTexture, "glActiveTexture");
        glrOk &= loadGlFn(gl.TexImage2D, "glTexImage2D");
        glrOk &= loadGlFn(gl.TexParameteri, "glTexParameteri");
        glrOk &= loadGlFn(gl.FramebufferTexture2D, "glFramebufferTexture2D");
        glrOk &= loadGlFn(gl.GenRenderbuffers, "glGenRenderbuffers");
        glrOk &= loadGlFn(gl.DeleteRenderbuffers, "glDeleteRenderbuffers");
        glrOk &= loadGlFn(gl.BindRenderbuffer, "glBindRenderbuffer");
        glrOk &= loadGlFn(gl.RenderbufferStorage, "glRenderbufferStorage");
        glrOk &= loadGlFn(gl.FramebufferRenderbuffer, "glFramebufferRenderbuffer");
        glrOk &= loadGlFn(gl.DeleteFramebuffers, "glDeleteFramebuffers");
        glrOk &= loadGlFn(gl.ClearColor, "glClearColor");
        glrOk &= loadGlFn(gl.Clear, "glClear");
        glrOk &= loadGlFn(gl.BlitFramebuffer, "glBlitFramebuffer");
        glrOk &= loadGlFn(gl.ReadPixels, "glReadPixels");
        glrOk &= loadGlFn(gl.PixelStorei, "glPixelStorei");
        glrOk &= loadGlFn(gl.VertexAttribPointer, "glVertexAttribPointer");
        glrOk &= loadGlFn(gl.EnableVertexAttribArray, "glEnableVertexAttribArray");
        glrOk &= loadGlFn(gl.DepthFunc, "glDepthFunc");
        glrOk &= loadGlFn(gl.DepthMask, "glDepthMask");
        glrOk &= loadGlFn(gl.ClearDepth, "glClearDepth");
        glrOk &= loadGlFn(gl.ColorMask, "glColorMask");
        glrOk &= loadGlFn(gl.Scissor, "glScissor");
        glrOk &= loadGlFn(gl.Uniform2f, "glUniform2f");
        glrOk &= loadGlFn(gl.Uniform3f, "glUniform3f");
        glrOk &= loadGlFn(gl.DeleteBuffers, "glDeleteBuffers");
        glrOk &= loadGlFn(gl.BlendFunc, "glBlendFunc");
        glrOk &= loadGlFn(gl.BlendEquation, "glBlendEquation");
        glrOk &= loadGlFn(gl.BlendColor, "glBlendColor");
        glrOk &= loadGlFn(gl.Uniform1i, "glUniform1i");
        // ⚠ cont.356: glUniform1f was loaded ONLY under the hardware-raster path (hwOk), so on a
        // plain PS2X_GS_RENDERER=gl run the pointer was null. The HUD counter-scale needs it.
        glrOk &= loadGlFn(gl.Uniform1f, "glUniform1f");
        glrOk &= loadGlFn(gl.TexSubImage2D, "glTexSubImage2D");
        // ★ cont.332l: ONLY PS2X_GS_GLR_AFAIL=6 needs this, so a miss must not take the whole
        // renderer down -- mode 6 checks the pointer and falls back to the default split.
        loadGlFn(gl.BindImageTexture, "glBindImageTexture");
        if (!glrOk)
        {
            glrFailed = true;
            std::fprintf(stderr, "[gsgpu] GL renderer DISABLED: a GL entry point is missing\n");
        }

        program = buildProgram(kDecodeShader, "decode");
        uploadProgram = buildProgram(kUploadShader, "upload");
        verifyProgram = buildProgram(kVerifyShader, "verify");
        if (!program || !uploadProgram || !verifyProgram)
            return fail("compute program build");
        locMode = gl.GetUniformLocation(program, "uMode");
        locBlockBase = gl.GetUniformLocation(program, "uBlockBase");
        locBw = gl.GetUniformLocation(program, "uBw");
        locOriginX = gl.GetUniformLocation(program, "uOriginX");
        locOriginY = gl.GetUniformLocation(program, "uOriginY");
        locWidth = gl.GetUniformLocation(program, "uWidth");
        locHeight = gl.GetUniformLocation(program, "uHeight");
        locPreserveAlpha = gl.GetUniformLocation(program, "uPreserveAlpha");
        locTabOff = gl.GetUniformLocation(program, "uTabOff");
        auto chunkLocs = [&](GLuint prog) -> ChunkLocs
        {
            ChunkLocs l;
            l.dbp = gl.GetUniformLocation(prog, "uDbp");
            l.dbw = gl.GetUniformLocation(prog, "uDbw");
            l.dsax = gl.GetUniformLocation(prog, "uDsax");
            l.dsay = gl.GetUniformLocation(prog, "uDsay");
            l.rrw = gl.GetUniformLocation(prog, "uRrw");
            l.startPixel = gl.GetUniformLocation(prog, "uStartPixel");
            l.count = gl.GetUniformLocation(prog, "uCount");
            l.tabOff = gl.GetUniformLocation(prog, "uTabOff");
            return l;
        };
        upLocs = chunkLocs(uploadProgram);
        vfLocs = chunkLocs(verifyProgram);

        // Build the page lookup tables with the same template code as ps2_gs_memory.cpp
        // (heap: ~640KB total) and upload the concatenation once.
        std::vector<uint16_t> flat(kTabHalfwords);
        {
            auto t32 = std::make_unique<PixelStorageTraits<GSMem::C32>::PageLookupTableT>();
            PixelStorageTraits<GSMem::C32>::InitPageLookupTable(*t32, kBlockTableC32, kColumnTable32);
            std::memcpy(flat.data() + kTabOffC32, t32->data(), sizeof(*t32));
            auto t16 = std::make_unique<PixelStorageTraits<GSMem::C16>::PageLookupTableT>();
            PixelStorageTraits<GSMem::C16>::InitPageLookupTable(*t16, kBlockTableC16, kColumnTable16);
            std::memcpy(flat.data() + kTabOffC16, t16->data(), sizeof(*t16));
            auto t16s = std::make_unique<PixelStorageTraits<GSMem::C16S>::PageLookupTableT>();
            PixelStorageTraits<GSMem::C16S>::InitPageLookupTable(*t16s, kBlockTableC16S, kColumnTable16);
            std::memcpy(flat.data() + kTabOffC16S, t16s->data(), sizeof(*t16s));
            auto tz32 = std::make_unique<PixelStorageTraits<GSMem::Z32>::PageLookupTableT>();
            PixelStorageTraits<GSMem::Z32>::InitPageLookupTable(*tz32, kBlockTableZ32, kColumnTable32);
            std::memcpy(flat.data() + kTabOffZ32, tz32->data(), sizeof(*tz32));
            auto tp8 = std::make_unique<PixelStorageTraits<GSMem::P8>::PageLookupTableT>();
            PixelStorageTraits<GSMem::P8>::InitPageLookupTable(*tp8, kBlockTableP8, kColumnTable8);
            std::memcpy(flat.data() + kTabOffP8, tp8->data(), sizeof(*tp8));
            auto tp4 = std::make_unique<PixelStorageTraits<GSMem::P4>::PageLookupTableT>();
            PixelStorageTraits<GSMem::P4>::InitPageLookupTable(*tp4, kBlockTableP4, kColumnTable4);
            std::memcpy(flat.data() + kTabOffP4, tp4->data(), sizeof(*tp4));
        }

        gl.GenBuffers(1, &vramBuf);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, vramBuf);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, kVramBytes, nullptr, GL_DYNAMIC_DRAW);
        gl.GenBuffers(1, &tabBuf);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, tabBuf);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(flat.size() * 2u),
                      flat.data(), GL_STATIC_DRAW);
        gl.GenBuffers(1, &outBuf);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, outBuf);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, kHostW * kHostH * 4u, nullptr, GL_DYNAMIC_DRAW);
        gl.GenBuffers(1, &mirrorBuf);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, mirrorBuf);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, kVramBytes, nullptr, GL_DYNAMIC_DRAW);
        gl.GenBuffers(1, &payloadBuf);
        gl.GenBuffers(1, &resultBuf);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuf);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, 16, nullptr, GL_DYNAMIC_DRAW);
        if (gl.GetError() != 0u)
            return fail("buffer setup");

        std::fprintf(stderr, "[gsgpu] ready: %s | %s%s\n",
                     reinterpret_cast<const char *>(gl.GetString(GL_RENDERER)),
                     reinterpret_cast<const char *>(gl.GetString(GL_VERSION)),
                     surface == EGL_NO_SURFACE ? " (surfaceless)" : " (pbuffer)");
        return true;
    }

    bool decodeOnThread(DecodeDesc &j)
    {
        if (!j.vram || j.vramSize < kVramBytes || !j.out)
            return false;
        if (j.width == 0u || j.height == 0u || j.width > kHostW || j.height > kHostH)
            return false;

        uint32_t mode = 0, tabOff = kTabOffC32;
        switch (j.frame.psm)
        {
        case GS_PSM_CT32:
            mode = 0u;
            break;
        case GS_PSM_CT24:
            mode = 1u;
            break;
        case GS_PSM_CT16:
            mode = 2u;
            tabOff = kTabOffC16;
            break;
        case GS_PSM_CT16S:
            mode = 3u;
            tabOff = kTabOffC16S;
            break;
        default:
            return false; // CPU path also rejects non-CT display formats
        }

        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, vramBuf);
        if (!haveUpload || j.sourceId == 0u || j.sourceId != lastSourceId)
        {
            gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kVramBytes, j.vram);
            lastSourceId = j.sourceId;
            haveUpload = true;
        }

        // Same defaults as CopyFrameToHostRgbaCpu: FBW 0 reads as the host width,
        // page-unit bases convert to blocks (framePageBaseToBlock).
        const uint32_t effFbw = j.frame.fbw ? j.frame.fbw : kHostW / 64u;
        const uint32_t blockBase = j.frameBaseIsPages ? (j.frame.fbp << 5u) : j.frame.fbp;

        gl.UseProgram(program);
        gl.Uniform1ui(locMode, mode);
        gl.Uniform1ui(locBlockBase, blockBase);
        gl.Uniform1ui(locBw, effFbw);
        gl.Uniform1ui(locOriginX, j.originX);
        gl.Uniform1ui(locOriginY, j.originY);
        gl.Uniform1ui(locWidth, j.width);
        gl.Uniform1ui(locHeight, j.height);
        gl.Uniform1ui(locPreserveAlpha, j.preserveAlpha ? 1u : 0u);
        gl.Uniform1ui(locTabOff, tabOff);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vramBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tabBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, outBuf);
        gl.DispatchCompute(kHostW / 8u, kHostH / 8u, 1u);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        j.out->resize(kHostW * kHostH * 4u);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, outBuf);
        gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kHostW * kHostH * 4u, j.out->data());

        const GLenum err = gl.GetError();
        if (err != 0u)
        {
            if (errorLogs < 8u)
            {
                ++errorLogs;
                std::fprintf(stderr,
                             "[gsgpu] GL error 0x%x during decode (psm=0x%x fbp=%u) -> CPU fallback\n",
                             err, j.frame.psm, j.frame.fbp);
            }
            haveUpload = false; // buffer state suspect; re-upload next time
            return false;
        }
        return true;
    }

    void setChunkUniforms(const ChunkLocs &l, const GsGpuUploadChunk &c, uint32_t count)
    {
        gl.Uniform1ui(l.dbp, c.dbp);
        gl.Uniform1ui(l.dbw, c.dbw);
        gl.Uniform1ui(l.dsax, c.dsax);
        gl.Uniform1ui(l.dsay, c.dsay);
        gl.Uniform1ui(l.rrw, c.rrw);
        gl.Uniform1ui(l.startPixel, c.startPixel);
        gl.Uniform1ui(l.count, count);
        gl.Uniform1ui(l.tabOff, kTabOffC32);
    }

    void uploadOnThread(const GsGpuUploadChunk &c)
    {
        const uint32_t count = static_cast<uint32_t>(c.payload.size());
        if (count == 0u || c.rrw == 0u)
            return;
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, payloadBuf);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(count) * 4,
                      c.payload.data(), GL_DYNAMIC_DRAW); // orphan + fill per chunk
        gl.UseProgram(uploadProgram);
        setChunkUniforms(upLocs, c, count);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mirrorBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, payloadBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tabBuf);
        gl.DispatchCompute((count + 63u) / 64u, 1u, 1u);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); // later verify/raster reads see it
    }

    void verifyOnThread(const GsGpuUploadChunk &c)
    {
        const uint32_t count = static_cast<uint32_t>(c.payload.size());
        if (count == 0u || c.rrw == 0u)
            return;
        const uint32_t init[4] = {0u, 0xFFFFFFFFu, 0u, 0u};
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuf);
        gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, init);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, payloadBuf);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(count) * 4,
                      c.payload.data(), GL_DYNAMIC_DRAW); // expected values
        gl.UseProgram(verifyProgram);
        setChunkUniforms(vfLocs, c, count);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mirrorBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, payloadBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tabBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, resultBuf);
        gl.DispatchCompute((count + 63u) / 64u, 1u, 1u);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        uint32_t res[4] = {0u, 0u, 0u, 0u};
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuf);
        gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, res);
        ++xvChunks;
        xvWords += count;
        if (res[0] != 0u)
        {
            ++xvMismChunks;
            xvMismWords += res[0];
            if (xvMismChunks <= 8u)
                std::fprintf(stderr,
                             "[gsgpu:xverify] MISMATCH chunk#%lu dbp=%u dbw=%u dsa=(%u,%u) rrw=%u "
                             "start=%u n=%u: %u words differ, first i=%u mirror=%08x cpu=%08x\n",
                             xvMismChunks, c.dbp, c.dbw, c.dsax, c.dsay, c.rrw,
                             c.startPixel, count, res[0], res[1], res[2], res[3]);
        }
        if ((xvChunks % 2048u) == 0u)
            std::fprintf(stderr,
                         "[gsgpu:xverify] chunks=%lu words=%llu mismatch-chunks=%lu mismatch-words=%llu\n",
                         xvChunks, xvWords, xvMismChunks, xvMismWords);
        const GLenum err = gl.GetError();
        if (err != 0u && errorLogs < 8u)
        {
            ++errorLogs;
            std::fprintf(stderr, "[gsgpu] GL error 0x%x during xverify\n", err);
        }
    }

    // Phase-1b: a straight authoritative-bytes copy into the mirror. No kernel, no
    // swizzle -- the caller already resolved WHAT the CPU wrote by reading VRAM back, so
    // this cannot diverge from the CPU by construction.
    void readbackOnThread(uint32_t offset, uint32_t len, std::vector<uint8_t> *out)
    {
        if (!out || offset >= kVramBytes)
            return;
        len = std::min<uint32_t>(len, kVramBytes - offset);
        out->resize(len);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, mirrorBuf);
        gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(offset),
                            static_cast<GLsizeiptr>(len), out->data());
        if ((hwAbl & 17) != 0 && hwCountBuf != 0u)
        {
            uint32_t cnt[4] = {0u, 0u, 0u, 0u};
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, hwCountBuf);
            gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, cnt);
            std::fprintf(stderr, "[gsgpu:hwcount] covered-fragments=%u draws=%lu prims=%llu\n",
                         cnt[0], hwDraws, hwPrims);
            const uint32_t zeros[4] = {0u, 0u, 0u, 0u};
            gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, zeros);
        }
    }

    void patchRawOnThread(uint32_t offset, const std::vector<uint8_t> &bytes)
    {
        if (bytes.empty() || offset >= kVramBytes)
            return;
        const uint32_t len = std::min<uint32_t>(static_cast<uint32_t>(bytes.size()),
                                                kVramBytes - offset);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, mirrorBuf);
        gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(offset),
                         static_cast<GLsizeiptr>(len), bytes.data());
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        ++prPatches;
        prBytes += len;
    }

    bool ensureRaster()
    {
        if (rasterPrimProgram != 0u)
            return true;
        // One shared pixel-pipeline source, two entry points -- a duplicated pipeline
        // would drift, and bit-exactness is the whole point.
        const std::string common = std::string(kRasterHeader) + kRasterCommon;
        rasterPrimProgram = buildProgram((common + kRasterMainPrim).c_str(), "raster-prim");
        rasterTileProgram = buildProgram((common + kRasterMainTile).c_str(), "raster-tile");
        if (!rasterPrimProgram || !rasterTileProgram)
            return false;
        locTilesX = gl.GetUniformLocation(rasterTileProgram, "uTilesX");
        gl.GenBuffers(1, &linTexBuf);
        gl.GenBuffers(1, &stateBuf);
        gl.GenBuffers(1, &geomBuf);
        gl.GenBuffers(1, &binOffBuf);
        gl.GenBuffers(1, &binEntBuf);
        gl.GenBuffers(1, &tileListBuf);
        return gl.GetError() == 0u;
    }

    // The hardware-raster program + its no-attachment framebuffer (cont.233). Any failure
    // logs once and leaves the tile kernel in charge for the run.
    // ★ cont.329 phase 2a: the geometry program. UNTEXTURED and UNBLENDED on purpose -- this
    // phase proves the vertex path, the GS-pixel-space transform and the depth test in isolation.
    //
    // Transform: the readback flips vertically (GL's origin is bottom-left, the presenter's rows
    // run top-down), so GS y=0 must land at the TOP of the GL framebuffer, i.e. NDC y = +1. Hence
    // `1.0 - 2*y/refH` rather than `2*y/refH - 1`.
    //
    // Depth: the PS2 treats a BIGGER Z as nearer, so Z is passed through unflipped, cleared to 0
    // (farthest) and compared with GEQUAL/GREATER. Phase 2a maps raw Z to [0,1] on the host.
    static constexpr const char *kGlVertex = R"(#version 430
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aCol;
layout(location = 2) in vec3 aSTQ;
layout(location = 3) in float aFog;
// ★★★★★ cont.356: 1 = the backend classified this draw as 2D/HUD (q == 1 && ZTST == ALWAYS &&
// not full-screen). Classified there, not here, because the width term is a PER-PRIMITIVE extent
// that a vertex shader cannot see.
layout(location = 4) in float aHud2d;
// ★★★★★ cont.358: WHERE the counter-scale is anchored, as the fixed point described on
// GsGlVertex::hudAnchor. 32768 decodes to exactly 0.5 -- the screen centre, and the value every
// draw carries unless the game's anchor hook asked for something else.
layout(location = 5) in float aHudAnchor;
uniform vec2 uRef;
// ★★★★★ cont.356: naturalAspect / presentedAspect, 1.0 = identity. Anamorphic widescreen renders
// a wider frustum into the game's OWN raster and stretches the frame at presentation, so the 3D
// world comes out right (it was pre-compressed by the FOV change) and 2D does NOT, because nothing
// compressed it. Squeezing the 2D layer about the screen centre by exactly the inverse of the
// stretch restores its authored proportions. Deriving it from the PRESENTED ratio is what makes it
// structurally unable to drift from the stretch it cancels -- a separate "widescreen is on" flag
// would be a second statement of the same ratio.
uniform float uHudScale;
// ★★ cont.329j: the GS interpolates gouraud colour LINEARLY IN SCREEN SPACE, like everything else
// it interpolates -- there is no perspective correction anywhere in the GS except the explicit
// per-pixel q divide for texture coordinates. GL's default `smooth` divides by clip w, which bends
// the gradient across large polygons. PS2X_GS_GLR_NOPERSPCOL selects between the two so the
// difference is measured rather than argued.
#ifdef NOPERSP_COL
noperspective out vec4 vCol;
noperspective out float vFog;
#else
out vec4 vCol;
noperspective out float vFog;
#endif
// ★ NOT perspective-correct on purpose. The GS interpolates s, t and q LINEARLY IN SCREEN SPACE
// and divides per pixel; GL's default would divide by clip w instead and give a different curve.
noperspective out vec3 vSTQ;
void main()
{
    vec2 p = aPos.xy / uRef;
    // ★ cont.358: about the anchor the GAME chose for this draw, defaulting to the screen CENTRE.
    // Centre keeps every element's SHAPE correct and needs no per-element knowledge, but pulls
    // corner elements inward so the HUD ends up pillarboxed inside a natural-aspect region while
    // the world fills the wider frame. A game that knows its own layout says otherwise per draw
    // through ps2xGsSetHudAnchorFn; anything it declines still arrives here as exactly 0.5.
    if (aHud2d > 0.5)
    {
        float anchor = (aHudAnchor - 1.0) / 65534.0;
        p.x = anchor + (p.x - anchor) * uHudScale;
    }
    gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, aPos.z * 2.0 - 1.0, 1.0);
    vCol = aCol;
    vFog = aFog;
    vSTQ = aSTQ;
}
)";
    static constexpr const char *kGlFragment = R"(#version 430
#ifdef NOPERSP_COL
noperspective in vec4 vCol;
#else
in vec4 vCol;
#endif
noperspective in float vFog;
noperspective in vec3 vSTQ;
uniform sampler2D uTex;
uniform int uTexEnable;
uniform int uTfx;
uniform int uTcc;
uniform int uAtst;
uniform int uSrcMode;
uniform int uTexFlipY;   // cont.344: the sampled texture is a GL render target (stored bottom-up): mirror V
uniform vec3 uTgtMap;    // rotk row 257: target read: st = (s*x, z - t*y); (1,1,1) = the cont.344 mirror
uniform float uTexa;
uniform int uAem;
uniform int uSignedAdd; // cont.343b: COLCLAMP=0 additive into an HDR target -- 128..255 count as -128..-1
uniform int uLodMode;    // 0 = no mips (plain texture()), 1 = GS formula, 2 = fixed K
uniform float uLodL;     // 2^TEX1.L
uniform float uLodK;     // TEX1.K / 16
uniform int uFge;        // PRIM.FGE
uniform vec3 uFogCol;    // FOGCOL, normalised
// uAtst: 0 NEVER 1 ALWAYS 2 LESS 3 LEQUAL 4 EQUAL 5 GEQUAL 6 GREATER 7 NOTEQUAL
uniform float uAref;    // PS2 units, 0..255
// 0 = test decides the fragment (AFAIL=KEEP, or no test)
// 1 = FB_ONLY colour pass: EVERY fragment writes colour, none writes depth
// 2 = FB_ONLY depth pass: only PASSING fragments survive, and they write depth only
// 3 = the pass/fail SECOND pass: only FAILING fragments survive (PCSX2 GSRendererHW.cpp
//     EmulateAlphaTestSecondPass -- GetAlphaTestConfigPS(..., invert=true) + zwe=false)
uniform int uAfailPass;
// ★★★★ cont.332j AFAIL=FB_ONLY done as the hardware does it: DEPTH IS WRITTEN ONLY BY FRAGMENTS
// THAT PASS THE ALPHA TEST. GL has no per-fragment depth-write mask, so a failing fragment writes
// back the depth already at its pixel (read from uPrevDepth, a copy taken before this group) --
// the store is then a no-op, while the fragment still writes colour and is still depth-tested.
// uAfailKeepZ != 0 selects it; the depth test and the write stay on the fixed-function path.
uniform sampler2D uPrevDepth;
uniform int uAfailKeepZ;
// ★★★★★ cont.332l PS2X_GS_GLR_DEBUGOUT (default 0 = off): make the fragment write a DIAGNOSTIC
// value instead of its shaded colour, so PS2X_GS_GLPIXWATCH -- which reads the framebuffer --
// reports what the shader actually sampled at that pixel rather than the composite.
// 1 = the sampled TEXEL rgb, 2 = the texture coordinates (s,t in the red/green channels, each
// wrapped into 0..1), 3 = the alpha the alpha test sees, as grey (aPs2/255).
// Blending is forced off by the host in this mode, so the value read back is the value written.
uniform int uDebugOut;
#ifdef AFAIL_SW
// ★★★★★ cont.332l PS2X_GS_GLR_AFAIL=6 -- THE ACCURATE AFAIL=FB_ONLY PATH (PCSX2's FEEDBACK
// shape). Every two-pass split reorders the draw, and cont.332j proved no split can be right:
// the hardware decides, PER FRAGMENT AND IN SUBMISSION ORDER, whether Z is written, and GL has
// no per-fragment depth-write mask. Mode 5 tried to express it with gl_FragDepth and was refuted
// because GL then performs the depth TEST with the written value too.
// So: do the depth test HERE, against a copy of the target's depth held in an image, inside a
// pixel-ordered interlock so that the read-modify-write of one pixel is ordered across the
// draw's overlapping triangles. GL's own test is ALWAYS for these groups; the fixed-function
// write is kept in sync by writing gl_FragDepth (own Z for a passer, the old value otherwise),
// so every LATER draw still depth-tests against the hardware buffer as usual.
// Colour stays on the hardware blender -- this is depth-only, NOT cont.233's full software ROP.
layout(pixel_interlock_ordered) in;
layout(r32f, binding = 0) coherent uniform image2D uDepthImg;
uniform int uAfailSw;   // 1 = this group takes the software Z test (uniform: interlock needs it)
uniform int uZtst;      // TEST.ZTST: 0 NEVER, 1 ALWAYS, 2 GEQUAL, 3 GREATER
#endif
out vec4 oCol;
void main()
{
    vec4 c = vCol;
    if (uTexEnable != 0)
    {
        // The GS divides the screen-linear s,t by the screen-linear q per pixel.
        vec2 st = vSTQ.xy / vSTQ.z;
        // ★★★★ cont.344: a PROMOTED render target is drawn with GL's bottom-up rows (the vertex
        // transform flips y), but the guest's texture coordinates address it top-down. Every
        // spatially-varying target read was mirrored: the shadow composite darkened the frame at
        // the vertical mirror of the count blob (measured: fix-vs-dropped luma 26.1 vs 32.3 inside
        // the mirrored mask, equal inside the true one). Full-screen uniform passes never showed it.
        if (uTexFlipY != 0)
            st = vec2(st.x * uTgtMap.x, uTgtMap.z - st.y * uTgtMap.y);
        // ★★★ cont.329j THE GS PICKS THE MIP LEVEL ITSELF: LOD = (log2(1/|Q|) << L) + K, or a
        // fixed K when LCM=1 (PCSX2 GSRendererHW's manual_lod path, with K/16). GL's own rule is
        // the maximum screen-space derivative, which is far more aggressive on a grazing surface.
        vec4 t;
        if (uLodMode == 0)
            t = texture(uTex, st);
        else
        {
            float lod = (uLodMode == 2)
                            ? uLodK
                            : (-log2(max(abs(vSTQ.z), 1e-8)) * uLodL + uLodK);
            t = textureLod(uTex, st, lod);
        }
        // ★★★ A PROMOTED RENDER TARGET IS NOT NECESSARILY RGBA. The guest names a pixel format
        // for the read; these two matter here (cont.329i):
        //   1 = PSMCT24 -- 24-bit colour, so the alpha of the sampled texel comes from TEXA and
        //       NOT from the buffer. Ours was the target's own alpha, which is 1.0 everywhere
        //       (cleared, never written), and the composite's blend is Cd*(1-As): As=1 wiped the
        //       whole visible buffer to BLACK.
        //   2 = a PALETTED read of a target (T4HL here: a 4-bit index from the ALPHA byte through
        //       the CLUT -- PCSX2's "channel shuffle"). Until that exists, sampling neutral white
        //       is the honest placeholder for a MODULATE, and it is a DIAGNOSTIC, not a fix.
        // CT24 has no alpha: it is TEXA.TA0, expanded to 0 where RGB is zero if AEM is set --
        // exactly what the CPU sampler does (gs_cpu_backend.cpp: `a = (texa.aem && rgbZero) ? 0 :
        // texa.ta0`), so both renderers read a 24-bit source the same way.
        if (uSrcMode == 1)
            t.a = (uAem != 0 && all(lessThan(t.rgb, vec3(0.5 / 255.0)))) ? 0.0 : uTexa;
        else if (uSrcMode == 2) t = vec4(1.0, 1.0, 1.0, 1.0);
        // ★ TFX. MODULATE is (Ct * Cv) >> 7 -- a divide by 128, not 255 -- so in normalised terms
        // the factor is 255/128, NOT 2.0. Using 2.0 would brighten every textured draw by 0.4%,
        // which is invisible per-draw and wrong everywhere.
        const float kMod = 255.0 / 128.0;
        if (uTfx == 0)        // MODULATE -- 96.1% of this game's draws
        {
            c.rgb = clamp(t.rgb * vCol.rgb * kMod, 0.0, 1.0);
            c.a = (uTcc != 0) ? clamp(t.a * vCol.a * kMod, 0.0, 1.0) : vCol.a;
        }
        else if (uTfx == 1)   // DECAL -- the texture replaces the fragment
        {
            c.rgb = t.rgb;
            c.a = (uTcc != 0) ? t.a : vCol.a;
        }
        else                  // HIGHLIGHT / HIGHLIGHT2
        {
            c.rgb = clamp(t.rgb * vCol.rgb * kMod + vCol.a, 0.0, 1.0);
            c.a = (uTcc == 0) ? vCol.a : (uTfx == 2 ? clamp(t.a + vCol.a, 0.0, 1.0) : t.a);
        }
    }
    // ★★ cont.329j FOG, exactly as the CPU rasterizer's writePixelT does it:
    //     C = ((F * C) >> 8) + (((255 - F) * FOGCOL) >> 8)
    // with F and C in 0..255. In normalised terms that is (255/256) * mix(FOGCOL, C, F) -- the
    // >>8-instead-of-/255 is the hardware's, and dropping it would brighten every fogged pixel by
    // 0.4%. Alpha is NOT fogged, so the alpha test below is unaffected either way.
    if (uFge != 0)
        c.rgb = (255.0 / 256.0) * mix(uFogCol, c.rgb, vFog);
    // ★★★★ cont.343b: a COLCLAMP=0 additive draw relies on the GS wrapping the sum mod 256, so a
    // face coloured 255 is really -1. Into an HDR (float) target that is just a signed add: map
    // 128..255 to -128..-1 here and let the blend add it. (0,1,1)+(0,255,255) then cancels to 0,
    // as it does on the GS -- this game's shadow-volume count (see GlTarget::hdr).
    if (uSignedAdd != 0)
        c.rgb -= step(vec3(128.5 / 255.0), c.rgb) * (256.0 / 255.0);

    // ★ cont.332l: the diagnostic outputs, taken BEFORE the alpha test so that what is reported
    // is what this fragment sampled, not what survived.
    vec3 dbgTexel = vec3(0.0);
    vec2 dbgSt = vec2(0.0);
    vec3 dbgFixed = vec3(0.0), dbgFixed2 = vec3(0.0);
    if (uTexEnable != 0)
    {
        dbgSt = vSTQ.xy / vSTQ.z;
        // ⚠ the SAME fetch the shaded path uses -- an instrument that samples differently from
        // the arm under test measures itself (cont.332i's lesson, applied to a shader).
        if (uLodMode == 0)
            dbgTexel = texture(uTex, dbgSt).rgb;
        else
        {
            float dlod = (uLodMode == 2)
                             ? uLodK
                             : (-log2(max(abs(vSTQ.z), 1e-8)) * uLodL + uLodK);
            dbgTexel = textureLod(uTex, dbgSt, dlod).rgb;
        }
        // Two FIXED texels of whatever object is bound: one in the dark robe field, one in the
        // pale collar. They identify the CONTENT independently of where this fragment samples.
        dbgFixed = texelFetch(uTex, ivec2(40, 41), 0).rgb;
        dbgFixed2 = texelFetch(uTex, ivec2(80, 2), 0).rgb;
    }
    // ★ THE ALPHA TEST, against the fragment's alpha in PS2 UNITS -- i.e. BEFORE the x255/128
    // blend rescale below, because AREF is compared against the raw 8-bit alpha.
    float aPs2 = c.a * 255.0;
    bool passes = true;
    if (uAtst == 0) passes = false;
    else if (uAtst == 1) passes = true;
    else if (uAtst == 2) passes = aPs2 < uAref;
    else if (uAtst == 3) passes = aPs2 <= uAref;
    else if (uAtst == 4) passes = aPs2 == uAref;
    else if (uAtst == 5) passes = aPs2 >= uAref;
    else if (uAtst == 6) passes = aPs2 > uAref;
    else passes = aPs2 != uAref;
    // uAfailPass == 1 is the colour half of FB_ONLY: failures still write colour, so no discard.
    // uAfailPass == 3 is its mirror: the pass/fail second pass keeps ONLY the failures.
    // ⚠⚠ gl_FragDepth MUST be assigned on EVERY path. GLSL leaves it UNDEFINED for any path that
    // does not write it once a shader assigns it anywhere -- build 936 wrote it only under
    // uAfailKeepZ and every other draw got garbage depth (gameplay tone fit a 0.75 -> 0.40 on BOTH
    // modes, which is how this was caught; a narrow per-pixel check on one menu screen had
    // "cleared" it). The unconditional assignment costs early-Z on this program; that is a
    // measured trade-off, not an oversight.
    gl_FragDepth = gl_FragCoord.z;
#ifdef AFAIL_SW
    // ⚠ The interlock functions must be reached under control flow that is dynamically uniform.
    // uAfailSw is a uniform and every branch above it (uTexEnable, uFge) is uniform too.
    // ⚠⚠ THE INTERLOCK CALLS MAY NOT SIT IN CONTROL FLOW. Mesa enforces the ARB spec to the
    // letter -- "beginInvocationInterlockARB() may not be used in control flow" -- and it means
    // ANY control flow, uniform or not (build 938's first cut put them under `if (uAfailSw)` and
    // the program failed to compile, which disabled the whole renderer). So every fragment of
    // this program enters the section and the work inside is what is conditional.
    // That makes the interlock's own cost a property of the PROGRAM, which is why mode 7 exists:
    // the same program with uAfailSw never set, i.e. mode 1's behaviour paying only the
    // interlock. mode 7 - mode 1 is the interlock's price; mode 6 - mode 7 is the software Z's.
    // ⚠ Nothing above this point may discard, or an invocation could die before the section.
    bool swZpass = true;
    float swOld = 0.0;
    beginInvocationInterlockARB();
    if (uAfailSw != 0)
    {
        swOld = imageLoad(uDepthImg, ivec2(gl_FragCoord.xy)).r;
        if (uZtst == 0) swZpass = false;
        else if (uZtst == 1) swZpass = true;
        else if (uZtst == 2) swZpass = gl_FragCoord.z >= swOld;
        else swZpass = gl_FragCoord.z > swOld;
        // AFAIL=FB_ONLY: a fragment that FAILS the alpha test writes colour but NOT depth.
        if (swZpass && passes)
            imageStore(uDepthImg, ivec2(gl_FragCoord.xy), vec4(gl_FragCoord.z));
    }
    endInvocationInterlockARB();
    if (uAfailSw != 0)
    {
        if (!swZpass)
            discard;
        gl_FragDepth = passes ? gl_FragCoord.z : swOld;
        oCol = vec4(c.rgb, clamp(c.a * (255.0 / 128.0), 0.0, 1.0));
        return;
    }
#endif
    if (uAfailKeepZ != 0)
    {
        // One pass, submission order preserved: nothing is discarded by the alpha test, and the
        // fragment's own depth is written only when it PASSED. A failure re-writes what is
        // already there, which is how "write colour, leave Z" is expressed in GL.
        if (!passes)
            gl_FragDepth = texelFetch(uPrevDepth, ivec2(gl_FragCoord.xy), 0).r;
    }
    else if (uAfailPass == 3)
    {
        if (passes) discard;
    }
    else if (uAfailPass != 1 && !passes)
        discard;

    // ★ THE ALPHA SCALE. The PS2 divides blending by 128, so an alpha byte of 0x80 means 1.0.
    // GL's GL_SRC_ALPHA divides by 255 and would read that same byte as 0.502 -- every blended
    // draw at HALF strength, which reads as "a bit dark" rather than as a bug. vCol.a arrives
    // normalised by 255, so x255/128 == x(255/128) restores the PS2 meaning; saturating is safe
    // because the census found no sampled alpha above 0x80. Alpha itself is never WRITTEN (FBMSK
    // is 'A' on 100% of this game's draws), so this value only ever feeds the blend factor.
    oCol = vec4(c.rgb, clamp(c.a * (255.0 / 128.0), 0.0, 1.0));
    if (uDebugOut == 1)
        oCol = vec4(dbgTexel, 1.0);
    else if (uDebugOut == 2)
        oCol = vec4(fract(dbgSt.x), fract(dbgSt.y), 0.0, 1.0);
    else if (uDebugOut == 3)
        oCol = vec4(vec3(clamp(c.a, 0.0, 1.0)), 1.0);
    else if (uDebugOut == 4)
        // red/green = fract(s),fract(t); blue = RANGE FLAGS, because fract() hides the sign and a
        // negative coordinate under CLAMP samples a completely different texel than 0.3 does:
        //   0x80 s<0, 0x40 t<0, 0x20 s>1, 0x10 t>1
        oCol = vec4(fract(dbgSt.x), fract(dbgSt.y),
                    (dbgSt.x < 0.0 ? 0.5 : 0.0) + (dbgSt.y < 0.0 ? 0.25 : 0.0) +
                    (dbgSt.x > 1.0 ? 0.125 : 0.0) + (dbgSt.y > 1.0 ? 0.0625 : 0.0), 1.0);
    else if (uDebugOut == 5)
        oCol = vec4(dbgFixed, 1.0);
    else if (uDebugOut == 7)
    {
        // cont.332m: the fragment's DEPTH as 24 bits spread over rgb (r = high byte), so the
        // pixel watch can compare two overlapping fragments of one draw against the CPU's z.
        float zq = clamp(gl_FragCoord.z, 0.0, 1.0) * 16777215.0;
        float hi = floor(zq / 65536.0);
        float mid = floor((zq - hi * 65536.0) / 256.0);
        float lo = zq - hi * 65536.0 - mid * 256.0;
        oCol = vec4(hi / 255.0, mid / 255.0, lo / 255.0, 1.0);
    }
    else if (uDebugOut == 6)
        oCol = vec4(dbgFixed2, 1.0);
}
)";

    bool ensureGlProgram()
    {
        if (glrProgram != 0u)
            return true;
        if (glrFailed)
            return false;
        // The colour's interpolation qualifier is a compile-time choice in GLSL, so the define
        // is injected after the #version line (which must stay first).
        // ★ cont.332l: the AFAIL=6 program declares the fragment-shader interlock. It is a
        // FRAGMENT-only extension, so the vertex source never gets it -- and the declaration is
        // compiled in only for mode 6, which keeps the shipped default program byte-identical.
        auto withDefines = [](const char *src, bool nopersp, bool afailSw)
        {
            std::string t(src);
            std::string ins;
            if (afailSw)
                ins += "#extension GL_ARB_fragment_shader_interlock : require\n#define AFAIL_SW 1\n";
            if (nopersp)
                ins += "#define NOPERSP_COL 1\n";
            if (!ins.empty())
            {
                const size_t nl = t.find('\n');
                if (nl != std::string::npos)
                    t.insert(nl + 1, ins);
            }
            return t;
        };
        const bool wantAfailSw = (glrAfailMode == 6 || glrAfailMode == 7) &&
                                 gl.BindImageTexture != nullptr;
        const std::string vsrc = withDefines(kGlVertex, glrNoPerspCol, false);
        const std::string fsrc = withDefines(kGlFragment, glrNoPerspCol, wantAfailSw);
        glrProgram = buildGraphicsProgram(vsrc.c_str(), fsrc.c_str(), "glrenderer");
        if (!glrProgram)
        {
            glrFailed = true;
            std::fprintf(stderr, "[gsgpu] GL renderer DISABLED: geometry program build failed\n");
            return false;
        }
        locGlrRef = gl.GetUniformLocation(glrProgram, "uRef");
        locGlrHudScale = gl.GetUniformLocation(glrProgram, "uHudScale");
        locGlrTexEnable = gl.GetUniformLocation(glrProgram, "uTexEnable");
        locGlrTfx = gl.GetUniformLocation(glrProgram, "uTfx");
        locGlrTcc = gl.GetUniformLocation(glrProgram, "uTcc");
        locGlrTex = gl.GetUniformLocation(glrProgram, "uTex");
        locGlrAtst = gl.GetUniformLocation(glrProgram, "uAtst");
        locGlrAref = gl.GetUniformLocation(glrProgram, "uAref");
        locGlrAfail = gl.GetUniformLocation(glrProgram, "uAfailPass");
        locGlrSrcMode = gl.GetUniformLocation(glrProgram, "uSrcMode");
        locGlrTexFlipY = gl.GetUniformLocation(glrProgram, "uTexFlipY");
        locGlrTgtMap = gl.GetUniformLocation(glrProgram, "uTgtMap");
        locGlrTexa = gl.GetUniformLocation(glrProgram, "uTexa");
        locGlrPrevDepth = gl.GetUniformLocation(glrProgram, "uPrevDepth");
        locGlrAfailKeepZ = gl.GetUniformLocation(glrProgram, "uAfailKeepZ");
        locGlrDebugOut = gl.GetUniformLocation(glrProgram, "uDebugOut"); // cont.332l
        locGlrAfailSw = gl.GetUniformLocation(glrProgram, "uAfailSw");   // cont.332l, -1 unless mode 6
        locGlrZtstSw = gl.GetUniformLocation(glrProgram, "uZtst");
        // cont.332j: the pre-draw depth copy lives on unit 1; unit 0 stays the draw's texture.
        gl.UseProgram(glrProgram);
        if (locGlrPrevDepth >= 0)
            gl.Uniform1i(locGlrPrevDepth, 1);
        if (locGlrAfailKeepZ >= 0)
            gl.Uniform1i(locGlrAfailKeepZ, 0);
        locGlrAem = gl.GetUniformLocation(glrProgram, "uAem");
        locGlrSignedAdd = gl.GetUniformLocation(glrProgram, "uSignedAdd");
        locGlrFge = gl.GetUniformLocation(glrProgram, "uFge");
        locGlrFogCol = gl.GetUniformLocation(glrProgram, "uFogCol");
        locGlrLodMode = gl.GetUniformLocation(glrProgram, "uLodMode");
        locGlrLodL = gl.GetUniformLocation(glrProgram, "uLodL");
        locGlrLodK = gl.GetUniformLocation(glrProgram, "uLodK");
        gl.GenVertexArrays(1, &glrVao);
        gl.GenBuffers(1, &glrVbo);
        gl.BindVertexArray(glrVao);
        gl.BindBuffer(GL_ARRAY_BUFFER, glrVbo);
        gl.EnableVertexAttribArray(0);
        gl.VertexAttribPointer(0, 3, GL_FLOAT, 0, static_cast<GLsizei>(sizeof(GsGlVertex)),
                               reinterpret_cast<const void *>(0));
        gl.EnableVertexAttribArray(1);
        gl.VertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, 1, static_cast<GLsizei>(sizeof(GsGlVertex)),
                               reinterpret_cast<const void *>(offsetof(GsGlVertex, r)));
        gl.EnableVertexAttribArray(2);
        gl.VertexAttribPointer(2, 3, GL_FLOAT, 0, static_cast<GLsizei>(sizeof(GsGlVertex)),
                               reinterpret_cast<const void *>(offsetof(GsGlVertex, s)));
        gl.EnableVertexAttribArray(3);
        gl.VertexAttribPointer(3, 1, GL_UNSIGNED_BYTE, 1, static_cast<GLsizei>(sizeof(GsGlVertex)),
                               reinterpret_cast<const void *>(offsetof(GsGlVertex, f)));
        // cont.356: the 2D/HUD flag. NOT normalised -- it is a 0/1 selector, and normalising a
        // ubyte would divide it by 255 and make the `> 0.5` test in the shader never fire.
        gl.EnableVertexAttribArray(4);
        gl.VertexAttribPointer(4, 1, GL_UNSIGNED_BYTE, 0, static_cast<GLsizei>(sizeof(GsGlVertex)),
                               reinterpret_cast<const void *>(offsetof(GsGlVertex, hud2d)));
        // cont.358: the anchor that correction is applied about, as a 16-bit fixed point of the
        // normalised x. NOT normalised here either -- GL would divide a ushort by 65535, and the
        // decode wants /65534 so that 32768 lands on exactly 0.5 (the centre, i.e. cont.356's
        // behaviour). The shader does that divide.
        gl.EnableVertexAttribArray(5);
        gl.VertexAttribPointer(5, 1, GL_UNSIGNED_SHORT, 0, static_cast<GLsizei>(sizeof(GsGlVertex)),
                               reinterpret_cast<const void *>(offsetof(GsGlVertex, hudAnchor)));
        gl.BindVertexArray(0);
        return true;
    }

    // Clear a target if it is starting a new frame. PER TARGET now: with double buffering, one
    // buffer is being displayed while the other is drawn into, so "a new frame" is per target and
    // a single global flag merged the two (cont.329 phase 4).
    // ★★★ cont.330: write VRAM seed rects into the target before any of this batch's draws.
    // The target lives at glrScale x native, so each rect is uploaded to a native-size scratch
    // texture and BLITTED (GL_NEAREST) into place -- glTexSubImage2D would only be correct at
    // scale 1. ⚠ GS y=0 is the TOP; a GL framebuffer's y=0 is the BOTTOM, so the destination row
    // range flips exactly as the scissor's does (see glY below).
    void glrApplyTargetSeeds(GsGlBatch &b, GlTarget &t)
    {
        // ★ cont.330b: the backend proved it produces correct, non-empty seed rects for target
        // 0x180, yet DUMPTGT shows that target still black -- so the fault is on this side. Report
        // arrivals, framebuffer completeness and GL errors rather than guessing which of the three.
        ++glrSeedBatches;
        if (b.targetSeeds.empty())
            return;
        ++glrSeedBatchesWith;
        if (glrSeedFbo == 0u)
            gl.GenFramebuffers(1, &glrSeedFbo);
        if (glrSeedTex == 0u)
            gl.GenTextures(1, &glrSeedTex);
        for (const GsGlTargetSeed &sd : b.targetSeeds)
        {
            if (sd.w == 0u || sd.h == 0u ||
                sd.rgba.size() < static_cast<size_t>(sd.w) * sd.h * 4u)
                continue;
            gl.BindTexture(GL_TEXTURE_2D, glrSeedTex);
            if (sd.w != glrSeedW || sd.h != glrSeedH)
            {
                gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(sd.w),
                              static_cast<GLsizei>(sd.h), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                              sd.rgba.data());
                gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glrSeedW = sd.w;
                glrSeedH = sd.h;
            }
            else
            {
                gl.TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(sd.w),
                                 static_cast<GLsizei>(sd.h), GL_RGBA, GL_UNSIGNED_BYTE,
                                 sd.rgba.data());
            }
            gl.BindFramebuffer(GL_READ_FRAMEBUFFER, glrSeedFbo);
            gl.FramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                    glrSeedTex, 0);
            gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, t.fbo);
            // Guest rows [y, y+h) counted from the top become GL rows [refH-(y+h), refH-y).
            const int glY0 = static_cast<int>(b.refH) - static_cast<int>(sd.y + sd.h);
            const int sc = static_cast<int>(glrScale);
            gl.Disable(GL_SCISSOR_TEST);
            // ★ cont.331q: core GL says a blit is affected only by pixel ownership, the scissor and
            // sRGB -- not the colour mask -- and the seed measures identically with FBMSK on and
            // off here. Set it anyway: PS2X_GS_GLR_FBMSK is now default ON, so a per-draw mask is
            // live state on this thread, and a driver that does apply it to a blit would silently
            // drop seed channels. One call per seed batch.
            gl.ColorMask(1, 1, 1, 1);
            // ★★★★ cont.331f: FLIP THE SOURCE ROWS. glY0 places the 32-row page band at the right
            // height, but the rows WITHIN the band were not inverted -- so every band arrived
            // internally upside-down and the FMV presented as horizontal banding with the title
            // text doubled and mirrored. Reading the source bottom-to-top (src y1 > y0 swapped)
            // makes BlitFramebuffer invert it, matching the guest's top-down row order to GL's
            // bottom-up target.
            gl.BlitFramebuffer(0, static_cast<GLint>(sd.h), static_cast<GLint>(sd.w), 0,
                               static_cast<GLint>(sd.x) * sc, glY0 * sc,
                               static_cast<GLint>(sd.x + sd.w) * sc,
                               (glY0 + static_cast<int>(sd.h)) * sc, GL_COLOR_BUFFER_BIT,
                               GL_NEAREST);
            ++glrSeedRects;
            glrSeedPixels += static_cast<unsigned long long>(sd.w) * sd.h;
            // ★★★★ cont.331e PS2X_GS_GLR_SEEDVERIFY=1: read the DESTINATION back IMMEDIATELY, in
            // this same device job, and compare against the SOURCE rect we just uploaded. Every
            // check in this arc so far ("read-fbo-status=COMPLETE", "gl-err=0x0") only shows the
            // call did not ERROR -- not that pixels moved. cont.330c retracted the blit as a
            // suspect on exactly that evidence. This is the test that distinguishes "the blit
            // moved nothing" from "something later overwrote it".
            {
                static const bool seedVerify = []
                { const char *e = std::getenv("PS2X_GS_GLR_SEEDVERIFY"); return e && e[0] && e[0] != '0'; }();
                if (seedVerify && glrSeedVerifyDone < 6ull)
                {
                    ++glrSeedVerifyDone;
                    // Source: what we handed the blit.
                    unsigned long long srcNz = 0; double srcSum = 0.0;
                    for (size_t i = 0; i + 3 < sd.rgba.size(); i += 4)
                    { const unsigned v = sd.rgba[i] + sd.rgba[i+1] + sd.rgba[i+2];
                      if (v) ++srcNz; srcSum += v; }
                    // Destination: read straight back out of the target we just blitted into.
                    const uint32_t rw = sd.w * glrScale, rh = sd.h * glrScale;
                    std::vector<uint8_t> back(size_t(rw) * rh * 4u, 0u);
                    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, t.fbo);
                    gl.PixelStorei(GL_PACK_ALIGNMENT, 1);
                    gl.ReadPixels(static_cast<GLint>(sd.x) * sc, glY0 * sc,
                                  static_cast<GLsizei>(rw), static_cast<GLsizei>(rh),
                                  GL_RGBA, GL_UNSIGNED_BYTE, back.data());
                    const GLenum rerr = gl.GetError ? gl.GetError() : 0u;
                    unsigned long long dstNz = 0; double dstSum = 0.0;
                    for (size_t i = 0; i + 3 < back.size(); i += 4)
                    { const unsigned v = back[i] + back[i+1] + back[i+2];
                      if (v) ++dstNz; dstSum += v; }
                    std::fprintf(stderr,
                                 "[gsgpu:seedverify] #%llu tgt=0x%llx rect=(%u,%u) %ux%u -> dst(%d,%d) %ux%u | "
                                 "SRC nonzero=%llu mean=%.2f | DST-AFTER-BLIT nonzero=%llu mean=%.2f | "
                                 "read-err=0x%x\n",
                                 glrSeedVerifyDone, (unsigned long long)b.targetKey, sd.x, sd.y,
                                 sd.w, sd.h, static_cast<int>(sd.x) * sc, glY0 * sc, rw, rh,
                                 srcNz, sd.rgba.empty() ? 0.0 : srcSum / double(sd.rgba.size() / 4),
                                 dstNz, back.empty() ? 0.0 : dstSum / double(back.size() / 4),
                                 (unsigned)rerr);
                    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, t.fbo);
                }
            }
            if (glrSeedRects <= 4ull || (glrSeedRects % 4096ull) == 0ull)
            {
                const GLenum rs = gl.CheckFramebufferStatus
                                      ? gl.CheckFramebufferStatus(GL_READ_FRAMEBUFFER) : 0u;
                const GLenum err = gl.GetError ? gl.GetError() : 0u;
                std::fprintf(stderr,
                             "[gsgpu:seed] rect#%llu tgt=0x%llx fbo=%u src=%ux%u dst=(%d,%d) "
                             "refH=%u scale=%u read-fbo-status=0x%x gl-err=0x%x\n",
                             glrSeedRects, (unsigned long long)b.targetKey, t.fbo, sd.w, sd.h,
                             static_cast<int>(sd.x) * sc, glY0 * sc, b.refH, glrScale,
                             (unsigned)rs, (unsigned)err);
            }
        }
        gl.BindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    }

    void glrClearIfNeeded(GlTarget &t, uint64_t key, uint32_t sceneW, uint32_t sceneH)
    {
        if (!t.needClear || !glrFrameClear)
            return;
        t.needClear = false;
        glcNoteClear(key);
        gl.Viewport(0, 0, static_cast<GLsizei>(sceneW), static_cast<GLsizei>(sceneH));
        gl.Disable(GL_SCISSOR_TEST);
        gl.ColorMask(1, 1, 1, 1);
        gl.DepthMask(1);
        glrSetClearColor();
        gl.ClearDepth(0.0); // PS2: bigger Z is nearer, so "nothing drawn" is 0
        // cont.344: a shared depth is cleared by the FIRST target of the frame that names it (the
        // scene's frame start); a later target joining the same Z buffer mid-frame (the shadow volume's
        // 0x180, the invert sprite's 0x100) clears only its own colour.
        GLbitfield bits = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
        if (t.depthShared && t.depthJoined && glrZRebind())
            bits = GL_COLOR_BUFFER_BIT; // cont.345 (1): other targets' Z lives in it
        else if (t.depthShared)
        {
            auto pe = glrDepthPool.find(t.depthPoolKey);
            if (pe != glrDepthPool.end())
            {
                if (pe->second.lastClearFrame == glrFrames) bits = GL_COLOR_BUFFER_BIT;
                else pe->second.lastClearFrame = glrFrames;
            }
        }
        gl.Clear(bits);
        glrOrderShot("CLEAR (frame start)", t, key);
        glrOrderNote("CLEAR (frame start)", key);
    }

    void drawBatchGlOnThread(GsGlBatch &b)
    {
        if (glrCensus)
        {
            ++glcBatchIn;
            glcBatchVertsIn += b.verts.size();
            glcBatchGroupsIn += b.groups.size();
        }
        // ★★★★ cont.331f: a SEED-ONLY batch (no verts/groups, only targetSeeds) must NOT be
        // dropped here -- this guard ran BEFORE glrApplyTargetSeeds, so every seed batch the
        // backend emitted (264,498 rects in one run) was counted as "dropped empty" and discarded,
        // and the device's seed path never executed at all. That, not the blit, is why phase A
        // read as a null: [gsgpu:seed] printed ZERO lines while the backend reported thousands of
        // events.
        if ((b.verts.empty() || b.groups.empty()) && b.targetSeeds.empty())
        {
            if (glrCensus) ++glcBatchDropEmpty;
            return;
        }
        if (b.refW == 0u || b.refH == 0u)
        {
            if (glrCensus) ++glcBatchDropRef;
            return;
        }
        // Draws go to the target named by the batch, not to a single shared one.
        glrDrawsPerKey[b.targetKey] += b.groups.size();
        // cont.343b: a COLCLAMP=0 ADDITIVE group (ABCD=0?21 / 0?01: B=0, D=Cd) needs a target that
        // does not saturate -- the shadow volume's +1/-1 count. Other colclamp=0 draws in this game
        // (the CT24 composite, the Z-invert sprites) never leave [0,255] and stay as they are.
        bool wantHdr = false;
        if (glrColclipHdr())
            for (const GsGlGroup &g : b.groups)
                if (g.abe && g.colclamp == 0u && g.bb == 2u && g.bd == 1u) { wantHdr = true; break; }
        GlTarget *tgt = ensureGlTarget(b.targetKey, b.refW, b.refH, wantHdr, b.zbp, b.zpsm);
        if (!tgt)
        {
            if (glrCensus) ++glcBatchDropTarget;
            return;
        }
        if (!ensureGlProgram())
        {
            if (glrCensus) ++glcBatchDropProgram;
            return;
        }
        const auto t0 = glrProf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const uint32_t sceneW = b.refW * glrScale, sceneH = b.refH * glrScale;
        gl.BindFramebuffer(GL_FRAMEBUFFER, tgt->fbo);
        // ★ cont.331q: the batch's state as it ARRIVES, before this batch's clear/seed/draws --
        // so the device-order log opens each batch with what the target already held.
        char orderWhat[288];
        if (glrOrderArmed())
        {
            std::snprintf(orderWhat, sizeof(orderWhat),
                          "BATCH-IN verts=%zu groups=%zu seeds=%zu tex=%zu ref=%ux%u",
                          b.verts.size(), b.groups.size(), b.targetSeeds.size(),
                          b.texUploads.size(), b.refW, b.refH);
            glrOrderShot(orderWhat, *tgt, b.targetKey);
            glrOrderNote(orderWhat, b.targetKey);
        }
        const bool clearArmed = tgt->needClear;
        glrClearIfNeeded(*tgt, b.targetKey, sceneW, sceneH);
        if (b.trace && clearArmed)
            std::fprintf(stderr, "[gsgpu:trace] *** CLEAR of target fbp=0x%llx before this batch\n",
                         (unsigned long long)b.targetKey);
        // ★★★ cont.330: the VRAM seed goes in AFTER the clear (a clear would wipe it) and BEFORE
        // every draw in this batch, so the game's own draws composite on top of the uploaded image
        // exactly as they do in VRAM on the CPU path.
        glrApplyTargetSeeds(b, *tgt);
        if (glrOrderArmed() && !b.targetSeeds.empty())
        {
            std::snprintf(orderWhat, sizeof(orderWhat), "SEEDS applied n=%zu",
                          b.targetSeeds.size());
            glrOrderShot(orderWhat, *tgt, b.targetKey);
            glrOrderNote(orderWhat, b.targetKey);
        }
        // ★ cont.331: seed-only batch -- the dirty rects are in, there is nothing to draw.
        // ★★★★ cont.331d: but the resolve's FRAME CACHE returns the PREVIOUS frame whenever
        // glrDrawsSinceResolve == 0, so a seed that changes the target while no draw follows was
        // applied and then ignored -- the target was never read back. Count the seed as work so
        // the next resolve actually reads this target.
        if (b.verts.empty())
        {
            glrDrawsSinceResolve += b.targetSeeds.size();
            return;
        }
        glrZAliasBatch(b, *tgt, sceneW, sceneH);
        gl.Viewport(0, 0, static_cast<GLsizei>(sceneW), static_cast<GLsizei>(sceneH));
        gl.UseProgram(glrProgram);
        if (locGlrRef >= 0)
            gl.Uniform2f(locGlrRef, static_cast<float>(b.refW), static_cast<float>(b.refH));
        // ★★★★★ cont.356: the 2D/HUD counter-scale. Applied ONLY when an explicit presentation
        // ratio is in force (PS2X_WINDOW_ASPECT > 1.1, i.e. the user asked for widescreen): with no
        // override the presented ratio IS the natural one, so the scale is 1.0 and this is a no-op
        // on every title. The pixel-square mode (`=0`) is deliberately left alone -- it is not an
        // anamorphic stretch, so there is nothing for this to cancel.
        if (locGlrHudScale >= 0 && gl.Uniform1f)
        {
            const double presented = ps2xWindowAspect();
            const float natural = ps2xGsPresentAspect();
            // ★ cont.356e: on a 2D screen the PRESENTER pillarboxes the whole framebuffer, so the
            // HUD in it is already correct -- counter-scaling the draws as well would correct the
            // same pixels twice and shrink the text into the middle of an already-pillarboxed image.
            const bool twoD = ps2xGsIs2dScreen();
            const float scale = (!twoD && presented > 1.1 && natural > 0.1f && natural < 4.0f)
                                    ? static_cast<float>(double(natural) / presented)
                                    : 1.0f;
            gl.Uniform1f(locGlrHudScale, scale);
            // ★★★★★ cont.358v: announce every CHANGE of this value. It is the one knob that can
            // alter the whole frame's geometry mid-gameplay -- `twoD` suppresses the 2D counter-scale
            // AND makes the presenter pillarbox -- so if it toggles, everything on screen jumps at
            // once. A flicker tied to a camera switch, with the HUD anchoring, the FOV field and the
            // scissor all measured stable, leaves this as the remaining candidate, and it has never
            // been logged: the existing `[gsgpu:hudscale]` line prints only the first three calls,
            // which is exactly the window in which nothing has gone wrong yet.
            {
                static float s_lastScale = -1.f;
                static bool  s_lastTwoD  = false;
                if (scale != s_lastScale || twoD != s_lastTwoD)
                {
                    if (s_lastScale >= 0.f)
                        std::fprintf(stderr, "[gsgpu:hudscale] CHANGED scale %.4f -> %.4f  2dScreen %d -> %d\n",
                                     double(s_lastScale), double(scale), int(s_lastTwoD), int(twoD));
                    s_lastScale = scale; s_lastTwoD = twoD;
                }
            }
            // cont.356d: the value, once, so "the backend tagged draws but nothing moved" can be
            // split between a bad uniform and an attribute that never reaches the shader.
            static int s_hsShown = 0;
            if (s_hsShown < 3)
            {
                ++s_hsShown;
                std::fprintf(stderr, "[gsgpu:hudscale] natural=%.4f presented=%.4f -> scale=%.4f "
                             "(loc=%d)\n", double(natural), presented, double(scale), locGlrHudScale);
            }
        }
        gl.BindVertexArray(glrVao);
        gl.BindBuffer(GL_ARRAY_BUFFER, glrVbo);
        gl.BufferData(GL_ARRAY_BUFFER,
                      static_cast<GLsizeiptr>(b.verts.size() * sizeof(GsGlVertex)),
                      b.verts.data(), GL_STREAM_DRAW);
        const auto tB = glrProf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // ★ Upload whatever textures this batch introduced or refreshed, BEFORE any draw binds
        // them. The backend ships a key only when it is new or its contents changed, so this is
        // not a per-batch re-upload -- that was the PS2X_GS_LINTEX spike's fatal mistake.
        for (GsGlTexUpload &u : b.texUploads)
        {
            if (u.w == 0u || u.h == 0u || u.rgba.size() < static_cast<size_t>(u.w) * u.h * 4u)
                continue;
            // ★ ONE GL OBJECT PER TEXTURE, CONTENTS REPLACED IN PLACE.
            // Keying the device map by a per-change epoch created a brand new texture object on
            // every content change and never freed the old one: 5866 objects, a leak, and a full
            // reallocation per upload measured at ~21 ms each -- 77% of the device thread. The
            // arrival of an upload is itself the "contents changed" signal, so no epoch is needed
            // in the key; glTexSubImage2D then reuses the existing storage.
            GlTex &t = glrTextures[u.key];
            if (t.id == 0u)
                gl.GenTextures(1, &t.id);
            gl.BindTexture(GL_TEXTURE_2D, t.id);
            // ★★★ cont.329j: levels arrive as separate uploads sharing one key. Level 0 owns the
            // object's shape; a level whose size already matches is refilled in place.
            const uint32_t expectW = u.level == 0u ? u.w : (t.w >> u.level ? t.w >> u.level : 1u);
            const uint32_t expectH = u.level == 0u ? u.h : (t.h >> u.level ? t.h >> u.level : 1u);
            const bool sameShape = (u.level == 0u) ? (t.w == u.w && t.h == u.h)
                                                   : (expectW == u.w && expectH == u.h &&
                                                      t.levels > u.level);
            if (sameShape)
            {
                gl.TexSubImage2D(GL_TEXTURE_2D, static_cast<GLint>(u.level), 0, 0,
                                 static_cast<GLsizei>(u.w), static_cast<GLsizei>(u.h), GL_RGBA,
                                 GL_UNSIGNED_BYTE, u.rgba.data());
            }
            else
            {
                gl.TexImage2D(GL_TEXTURE_2D, static_cast<GLint>(u.level),
                              static_cast<GLint>(GL_RGBA8), static_cast<GLsizei>(u.w),
                              static_cast<GLsizei>(u.h), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                              u.rgba.data());
                if (u.level == 0u) { t.w = u.w; t.h = u.h; t.levels = 1u; }
                else if (u.level + 1u > t.levels) t.levels = static_cast<uint8_t>(u.level + 1u);
                ++glrTexAlloc;
            }
            ++glrTexUploads;
        }
        gl.BindTexture(GL_TEXTURE_2D, 0);
        if (locGlrTex >= 0)
            gl.Uniform1i(locGlrTex, 0); // sampler on texture unit 0
        const auto tA = glrProf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        gl.Disable(GL_CULL_FACE);
        // FBMSK is 'A' (write RGB, keep alpha) on 100% of this game's draws (cont.329 census).
        gl.ColorMask(1, 1, 1, 0);
        gl.Enable(GL_SCISSOR_TEST);

        // ★★★★★ cont.332i: the per-layer trace for ONE pixel (PS2X_GS_GLPIXWATCH). Read the
        // watched pixel ONCE here -- after this batch's clear and its VRAM seeds -- and again
        // after each group, so every line in the log is a draw that actually MOVED this pixel.
        const bool pwArmed = glPixWatchArmed();
        bool pwOk = false;
        uint32_t pwPrev = 0u;
        if (pwArmed)
        {
            pwPrev = glPixWatchRead(*tgt, b.refH, pwOk);
            if (pwOk)
                std::fprintf(stderr,
                             "[gsgpu:glpixwatch] BATCH #%llu flip=%llu f=%llu tgt=0x%llx ref=%ux%u "
                             "(%d,%d) in=%02x%02x%02x%02x groups=%zu verts=%zu\n",
                             ++glPixWatchBatches, gs2CurrentFlip(), glrFrames,
                             (unsigned long long)b.targetKey, b.refW, b.refH,
                             glPixWatchX, glPixWatchY,
                             (pwPrev >> 24) & 0xFFu, (pwPrev >> 16) & 0xFFu,
                             (pwPrev >> 8) & 0xFFu, pwPrev & 0xFFu,
                             b.groups.size(), b.verts.size());
        }
        glcQueryBegin();
        for (const GsGlGroup &g : b.groups)
        {
            if (glrCensus) ++glcGroupIn;
            if (g.count == 0u)
            {
                if (glrCensus) ++glcGroupEmpty;
                continue;
            }
            // ⚠ GS scissor bounds are INCLUSIVE; glScissor takes width/height. And GS y=0 is the
            // TOP while glScissor's y is measured from the BOTTOM, so the row range flips.
            const int w = static_cast<int>(g.sx1) - static_cast<int>(g.sx0) + 1;
            const int h = static_cast<int>(g.sy1) - static_cast<int>(g.sy0) + 1;
            if (w <= 0 || h <= 0)
            {
                if (glrCensus) ++glcGroupScissor;
                continue;
            }
            const int glY = static_cast<int>(b.refH) - 1 - static_cast<int>(g.sy1);
            gl.Scissor(static_cast<GLint>(g.sx0 * glrScale), static_cast<GLint>(glY * static_cast<int>(glrScale)),
                       static_cast<GLsizei>(w * static_cast<int>(glrScale)),
                       static_cast<GLsizei>(h * static_cast<int>(glrScale)));
            // ★★★★★ cont.332l AFAIL=6: this group's depth image, seeded under the scissor just
            // set. The predicate is `twoPass` computed below, restated here because the seed has
            // to happen before any of the group's own GL state is established.
            const bool afailSwGroup = glrAfailMode == 6 && gl.BindImageTexture != nullptr &&
                                      locGlrAfailSw >= 0 && !glrNoDepth &&
                                      g.ate != 0u && g.atst != 1u && g.afail == 1u &&
                                      g.zte != 0u && !g.zmsk;
            const bool afailSwReady = afailSwGroup && seedDepthImage(*tgt, sceneW, sceneH);
            // ★ cont.329 phase 4d ABLATION (PS2X_GS_GLR_NODEPTH=1, default off): force the depth
            // test off. A matched-frame comparison showed most of the scene missing in GL while
            // the software path drew it, and there are only two explanations -- the geometry never
            // arrives, or it arrives and is rejected. This separates them in one run instead of a
            // third round of reasoning from screenshots.
            if (glrNoDepth)
            {
                gl.Disable(GL_DEPTH_TEST);
                gl.DepthMask(0);
            }
            else if (g.zte == 0u)
            {
                // TEST.ZTE=0 means the depth test is OFF and Z is NOT written (PCSX2's rule).
                gl.Disable(GL_DEPTH_TEST);
                gl.DepthMask(0);
            }
            else
            {
                gl.Enable(GL_DEPTH_TEST);
                gl.DepthFunc(g.ztst == 0u ? GL_NEVER : g.ztst == 1u ? GL_ALWAYS
                                                     : g.ztst == 2u ? GL_GEQUAL
                                                                    : GL_GREATER);
                gl.DepthMask(g.zmsk ? 0 : 1);
            }
            // ★ Texturing. A key of 0, or one the device has never been given pixels for, draws
            // UNTEXTURED rather than sampling a stale or absent object.
            GLuint tex = 0u;
            int lodMode = 0;
            if (glrCensus && g.texKey == 0u)
                ++glcGroupUntextured;
            if (g.texIsTarget)
            {
                // ★ phase 4b: sample the GL target the game rendered into, not a decode of memory
                // this path never writes. The target texture lives at scale x resolution while the
                // coordinates are normalised, so upscaling needs no special handling here.
                auto it = glrTargets.find(g.texKey);
                if (it != glrTargets.end())
                    tex = it->second.color;
                else if (glrCensus)
                    ++glcTexTargetMissing;
            }
            else if (g.texKey != 0u)
            {
                auto it = glrTextures.find(g.texKey);
                if (it != glrTextures.end())
                    tex = it->second.id;
                else if (glrCensus)
                    ++glcTexMissing;
            }
            if (tex != 0u)
            {
                gl.BindTexture(GL_TEXTURE_2D, tex);
                // Wrap and filter are per-DRAW on the PS2 but per-OBJECT in GL, so set them on
                // each bind: cheap (95.4% of draws are REPEAT/REPEAT) and correct when one texture
                // is sampled with two different wrap modes.
                const GLint wu = g.wrapU ? static_cast<GLint>(GL_CLAMP_TO_EDGE) : static_cast<GLint>(GL_REPEAT);
                const GLint wv = g.wrapV ? static_cast<GLint>(GL_CLAMP_TO_EDGE) : static_cast<GLint>(GL_REPEAT);
                const GLint fl = g.lin ? static_cast<GLint>(GL_LINEAR) : static_cast<GLint>(GL_NEAREST);
                gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wu);
                gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wv);
                // ★★★ cont.329j: TEX1.MMIN chooses the minification filter, exactly as the GS
                // does (0 NEAREST, 1 LINEAR, 2 NEAREST_MIPMAP_NEAREST, 3 NEAREST_MIPMAP_LINEAR,
                // 4 LINEAR_MIPMAP_NEAREST, 5 LINEAR_MIPMAP_LINEAR). A mipmap mode is only asked
                // for when the chain was actually uploaded, or GL would sample an incomplete
                // texture and draw nothing.
                GLint minF = fl;
                lodMode = 0;
                if (glrMips && !g.texIsTarget)
                {
                    auto ti = glrTextures.find(g.texKey);
                    const unsigned have = (ti != glrTextures.end()) ? ti->second.levels : 1u;
                    if (have > 1u && g.minFilter >= 2u)
                    {
                        switch (g.minFilter)
                        {
                        case 2: minF = static_cast<GLint>(GL_NEAREST_MIPMAP_NEAREST); break;
                        case 3: minF = static_cast<GLint>(GL_NEAREST_MIPMAP_LINEAR); break;
                        case 4: minF = static_cast<GLint>(GL_LINEAR_MIPMAP_NEAREST); break;
                        default: minF = static_cast<GLint>(GL_LINEAR_MIPMAP_LINEAR); break;
                        }
                        gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL,
                                         static_cast<GLint>(have - 1u));
                        lodMode = g.lcm ? 2 : 1;
                    }
                }
                gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minF);
                gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fl);
                ++glrTexBinds;
            }
            // ★ cont.329 phase 4g ABLATION (PS2X_GS_GLR_NOTEX=1): draw everything with flat vertex
            // colour. Every primitive IS translated (census: 0 dropped of 88 M), so the missing
            // scene is either geometry that never reaches the screen or geometry that reaches it
            // and samples wrongly. Untextured, present-but-mis-sampled geometry becomes visible
            // as flat shapes; absent geometry stays absent. One run, two answers.
            if (locGlrTexEnable >= 0)
                gl.Uniform1i(locGlrTexEnable, (tex != 0u && !glrNoTex) ? 1 : 0);
            if (locGlrTfx >= 0)
                gl.Uniform1i(locGlrTfx, static_cast<GLint>(g.tfx));
            if (locGlrTcc >= 0)
                gl.Uniform1i(locGlrTcc, static_cast<GLint>(g.tcc));
            if (locGlrSignedAdd >= 0) // cont.343b: see the shader
                gl.Uniform1i(locGlrSignedAdd, (tgt->hdr && g.abe && g.colclamp == 0u && g.bb == 2u && g.bd == 1u) ? 1 : 0);
            if (locGlrSrcMode >= 0)
                gl.Uniform1i(locGlrSrcMode, tex != 0u ? static_cast<GLint>(g.srcMode) : 0);
            if (locGlrTexFlipY >= 0)
                gl.Uniform1i(locGlrTexFlipY, (tex != 0u && g.texIsTarget && glrTexFlipY()) ? 1 : 0);
            if (locGlrTgtMap >= 0)
            {
                // ★★★★ rotk row 257 PS2X_GS_GLR_TGTMAP (default ON; `=0` = the cont.344 mirror `v = 1 - t`).
                // The guest's s,t are normalised by its TEXTURE size (2^TW x 2^TH), but a promoted target is
                // a GL texture sized by the TARGET's extent x scale (and only ever grows), with guest row y
                // stored at GL row (refH - y) * scale (DrawBatchGl: "rows [y, y+h) become GL rows
                // [refH-(y+h), refH-y)"). So texel (s,t) is at u = s*texW*scale/allocW, v = (refH -
                // t*texH)*scale/allocH. `1 - t` is that only when texH == refH and allocH == refH*scale:
                // true for a 512-line PAL frame (2^9), FALSE for a 448-line NTSC one -- every target read on
                // the USA disc was stretched by 512/448 and mirrored about the wrong centre (user, 2026-10-04:
                // the hero's shadow composite drawn below his feet on Min02; CPU renderer and PCSX2 correct).
                static const bool s_tgtMap = [] { const char *e = std::getenv("PS2X_GS_GLR_TGTMAP"); return !(e && e[0] == '0'); }();
                float mx = 1.f, my = 1.f, mz = 1.f;
                if (s_tgtMap && tex != 0u && g.texIsTarget && g.texW && g.texH)
                {
                    auto itT = glrTargets.find(g.texKey);
                    if (itT != glrTargets.end() && itT->second.w && itT->second.h && itT->second.refH)
                    {
                        const float sc = static_cast<float>(glrScale);
                        mx = static_cast<float>(g.texW) * sc / static_cast<float>(itT->second.w);
                        my = static_cast<float>(g.texH) * sc / static_cast<float>(itT->second.h);
                        mz = static_cast<float>(itT->second.refH) * sc / static_cast<float>(itT->second.h);
                    }
                }
                gl.Uniform3f(locGlrTgtMap, mx, my, mz);
            }
            if (locGlrTexa >= 0)
                gl.Uniform1f(locGlrTexa, static_cast<float>(g.srcTexa) / 255.0f);
            if (locGlrAem >= 0)
                gl.Uniform1i(locGlrAem, static_cast<GLint>(g.srcAem));
            if (locGlrLodMode >= 0)
                gl.Uniform1i(locGlrLodMode, lodMode);
            if (locGlrLodL >= 0)
                gl.Uniform1f(locGlrLodL, static_cast<float>(1u << (g.lodL & 3u)));
            if (locGlrLodK >= 0)
                gl.Uniform1f(locGlrLodK, static_cast<float>(g.lodK) / 16.0f);
            if (locGlrFge >= 0)
                gl.Uniform1i(locGlrFge, (glrFog && g.fge) ? 1 : 0);
            if (locGlrFogCol >= 0)
                gl.Uniform3f(locGlrFogCol, static_cast<float>(g.fogR) / 255.0f,
                             static_cast<float>(g.fogG) / 255.0f,
                             static_cast<float>(g.fogB) / 255.0f);

            // ★ Blending. PS2: Cv = ((A - B) * C) >> 7 + D, with A/B/D in {Cs, Cd, 0} and
            // C in {As, Ad, FIX}. Only the four combinations the cont.329 census found live are
            // mapped; anything else falls back to straight alpha-over rather than guessing, and is
            // counted so an unmapped mode shows up as a number instead of a silent wrong picture.
            if (locGlrDebugOut >= 0)
                gl.Uniform1i(locGlrDebugOut, glrDebugOut);
            if (!g.abe || glrDebugOut != 0)
            {
                gl.Disable(GL_BLEND);
            }
            else
            {
                gl.Enable(GL_BLEND);
                const float f = static_cast<float>(g.bfix) / 128.0f; // FIX is PS2 units: 0x80 = 1.0
                gl.BlendColor(f, f, f, f);
                const uint32_t abcd = (uint32_t(g.ba) << 12) | (uint32_t(g.bb) << 8) |
                                      (uint32_t(g.bc) << 4) | uint32_t(g.bd);
                switch (abcd)
                {
                case 0x0101: // (Cs - Cd)*As + Cd  -- standard alpha-over, 94.3% of draws
                    gl.BlendEquation(GL_FUNC_ADD);
                    gl.BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    break;
                case 0x0201: // (Cs - 0)*As + Cd   -- additive by source alpha
                    gl.BlendEquation(GL_FUNC_ADD);
                    gl.BlendFunc(GL_SRC_ALPHA, GL_ONE);
                    break;
                case 0x0221: // (Cs - 0)*FIX + Cd  -- additive by a constant
                    gl.BlendEquation(GL_FUNC_ADD);
                    gl.BlendFunc(GL_CONSTANT_COLOR, GL_ONE);
                    break;
                case 0x0121: // (Cs - Cd)*FIX + Cd -- constant-alpha lerp
                    gl.BlendEquation(GL_FUNC_ADD);
                    gl.BlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA);
                    break;
                case 0x2101: // (0 - Cd)*As + Cd   -- darken
                    gl.BlendEquation(GL_FUNC_ADD);
                    gl.BlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
                    break;
                case 0x0122: // (Cs - Cd)*FIX + 0  -- subtract, no destination term
                    gl.BlendEquation(GL_FUNC_SUBTRACT);
                    gl.BlendFunc(GL_CONSTANT_COLOR, GL_CONSTANT_COLOR);
                    break;
                default:
                    ++glrBlendFallback;
                    gl.BlendEquation(GL_FUNC_ADD);
                    gl.BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    break;
                }
            }
            // ★ THE ALPHA TEST, and the two-pass AFAIL=FB_ONLY path.
            // GL cannot mask depth per fragment, so "failures write colour but not depth" is not
            // a discard. PCSX2 splits it into two passes (GSRendererHW.cpp: "First pass is to
            // update color; second pass is to update Z"), and so do we:
            //   pass 1 -- no discard, colour only, depth writes OFF
            //   pass 2 -- discard failures, colour masked OFF, depth writes ON
            // ⚠ Ours is not PCSX2's `independent_z` case (Z varies and primitives overlap), so the
            // split is an approximation of the ordering. That is precisely the kind of divergence
            // dropping bit-exactness bought, and it is far closer than ignoring AFAIL entirely --
            // which drew failing texels solid AND let them occlude what was behind them.
            const bool testActive = g.ate != 0u && g.atst != 1u;
            const bool depthOn = (g.zte != 0u) && !g.zmsk;
            const bool twoPass = testActive && g.afail == 1u && depthOn;
            // PCSX2's independent_z, with the terms we can evaluate here: twoPass already implies
            // zwe, so ZTST=ALWAYS is the only remaining case in which the colour-then-Z split is
            // exact. Everything else gets the pass/fail fallback.
            const bool independentZ = (g.ztst == 1u);
            const bool passThenFail = twoPass && glrAfailMode != 0 && glrAfailMode != 4 &&
                                      glrAfailMode != 5 &&
                                      (glrAfailMode == 2 || !independentZ);
            // ★★★★ cont.332j mode 4: ONE pass, no discard, colour AND depth written. Both
            // two-pass splits reorder the draw, and each loses a different thing: colour-then-Z
            // loses intra-draw occlusion (cont.332i's gameplay bug), pass-then-fail depth-rejects
            // failing fragments that hardware would have drawn (the character-select cloak, which
            // FAILS the alpha test and is rejected against the Z pass 1 wrote for the skin behind
            // it). A single pass keeps submission ORDER exactly; its only error is that failing
            // fragments also write Z, so they can occlude something farther that follows.
            const bool afailOnePass = twoPass && glrAfailMode == 4;
            // ★★★★★ cont.332j mode 5: the hardware rule, expressed in one pass -- depth written
            // ONLY by fragments that pass the alpha test, failures writing back the depth already
            // there. Correct in both the gameplay case (passers occlude, as mode 1 got right) and
            // the character-select case (all fragments fail, nothing writes Z, so the result is
            // submission order, as mode 0 got right).
            const bool afailKeepZ = twoPass && glrAfailMode == 5;
            if (locGlrAtst >= 0)
                gl.Uniform1i(locGlrAtst, testActive ? static_cast<GLint>(g.atst) : 1);
            if (locGlrAref >= 0)
                gl.Uniform1f(locGlrAref, static_cast<float>(g.aref));

            const size_t traceIdx = &g - b.groups.data();
            // ★★ cont.329h: FRAME.FBMSK -> glColorMask. Alpha is never written either way.
            bool wr = true, wg = true, wb = true;
            if (glrFbmsk)
            {
                const uint32_t m = g.fbmsk;
                wr = (m & 0x000000FFu) != 0x000000FFu;
                wg = (m & 0x0000FF00u) != 0x0000FF00u;
                wb = (m & 0x00FF0000u) != 0x00FF0000u;
                const bool partial =
                    ((m & 0xFFu) != 0u && (m & 0xFFu) != 0xFFu) ||
                    (((m >> 8) & 0xFFu) != 0u && ((m >> 8) & 0xFFu) != 0xFFu) ||
                    (((m >> 16) & 0xFFu) != 0u && ((m >> 16) & 0xFFu) != 0xFFu);
                if (partial) ++glrFbmskPartial;
                if (!wr || !wg || !wb) ++glrFbmskMasked;
                gl.ColorMask(wr, wg, wb, 0);
            }
            if (afailSwReady)
            {
                // ★★★★★ cont.332l: ONE pass, submission order intact, the depth test done in the
                // shader against the image and the per-fragment depth write decided there too.
                // GL's own test is ALWAYS (the shader already rejected what must not survive) and
                // the fixed-function write stores what the shader computed, so the hardware depth
                // buffer stays correct for every later draw.
                ++glrAfailSwZ;
                gl.BindImageTexture(0, glrDepthImgTex, 0, 0, 0, GL_READ_WRITE, GL_R32F);
                gl.Uniform1i(locGlrAfailSw, 1);
                if (locGlrZtstSw >= 0)
                    gl.Uniform1i(locGlrZtstSw, static_cast<GLint>(g.ztst));
                if (locGlrAfail >= 0)
                    gl.Uniform1i(locGlrAfail, 0);
                gl.DepthFunc(GL_ALWAYS);
                gl.DepthMask(1);
                gl.ColorMask(wr, wg, wb, 0);
                if (glrCensus) { ++glcDrawCalls; glcVertsDrawn += g.count; }
                gl.DrawArrays(GL_TRIANGLES, static_cast<GLint>(g.first), static_cast<GLsizei>(g.count));
                gl.Uniform1i(locGlrAfailSw, 0);
                // The next group re-seeds the image through the framebuffer and samples this
                // target's depth as a texture, so both hazards are named.
                gl.MemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT |
                                 GL_TEXTURE_FETCH_BARRIER_BIT);
            }
            else if (afailKeepZ)
            {
                ++glrAfailKeepZ;
                const bool haveDepth = snapshotDepth(*tgt, sceneW, sceneH);
                if (haveDepth)
                {
                    gl.ActiveTexture(GL_TEXTURE1);
                    gl.BindTexture(GL_TEXTURE_2D, glrDepthCopyTex);
                    gl.ActiveTexture(GL_TEXTURE0_CONT332J);
                }
                if (locGlrAfailKeepZ >= 0)
                    gl.Uniform1i(locGlrAfailKeepZ, haveDepth ? 1 : 0);
                if (locGlrAfail >= 0)
                    gl.Uniform1i(locGlrAfail, 0);
                gl.DepthMask(1);
                gl.ColorMask(wr, wg, wb, 0);
                if (glrCensus) { ++glcDrawCalls; glcVertsDrawn += g.count; }
                gl.DrawArrays(GL_TRIANGLES, static_cast<GLint>(g.first), static_cast<GLsizei>(g.count));
                if (locGlrAfailKeepZ >= 0)
                    gl.Uniform1i(locGlrAfailKeepZ, 0);
            }
            else if (afailOnePass)
            {
                ++glrAfailOnePass;
                if (locGlrAfail >= 0)
                    gl.Uniform1i(locGlrAfail, 1); // no discard: every fragment writes colour
                gl.DepthMask(1);
                gl.ColorMask(wr, wg, wb, 0);
                if (glrCensus) { ++glcDrawCalls; glcVertsDrawn += g.count; }
                gl.DrawArrays(GL_TRIANGLES, static_cast<GLint>(g.first), static_cast<GLsizei>(g.count));
            }
            else if (!twoPass || glrAfailMode == 3)
            {
                if (locGlrAfail >= 0)
                    gl.Uniform1i(locGlrAfail, 0);
                if (glrCensus) { ++glcDrawCalls; glcVertsDrawn += g.count; }
                gl.DrawArrays(GL_TRIANGLES, static_cast<GLint>(g.first), static_cast<GLsizei>(g.count));
            }
            else if (passThenFail)
            {
                // PCSX2 PASS_THEN_FAIL (GSRendererHW.cpp EmulateATST `else` branch).
                // 1: the alpha test decides; survivors write colour AND depth, so the draw's own
                //    overlapping triangles occlude each other exactly as they do on hardware.
                ++glrAfailPassFail;
                if (locGlrAfail >= 0)
                    gl.Uniform1i(locGlrAfail, 0);
                gl.DepthMask(1);
                gl.ColorMask(wr, wg, wb, 0);
                if (glrCensus) { ++glcDrawCalls; glcVertsDrawn += g.count; }
                gl.DrawArrays(GL_TRIANGLES, static_cast<GLint>(g.first), static_cast<GLsizei>(g.count));
                // 2: the test inverted -- only the FAILING fragments -- colour only, no Z write.
                //    (EmulateAlphaTestSecondPass: AFAIL_FB_ONLY => alpha_second_pass.depth.zwe = false.)
                if (locGlrAfail >= 0)
                    gl.Uniform1i(locGlrAfail, 3);
                gl.DepthMask(0);
                if (glrCensus) ++glcDrawCalls;
                gl.DrawArrays(GL_TRIANGLES, static_cast<GLint>(g.first), static_cast<GLsizei>(g.count));
                gl.DepthMask(1);
                ++glrTwoPass;
            }
            else
            {
                ++glrAfailSimple;
                // 1: colour, no depth write, no discard.
                if (locGlrAfail >= 0)
                    gl.Uniform1i(locGlrAfail, 1);
                gl.DepthMask(0);
                gl.ColorMask(wr, wg, wb, 0);
                if (glrCensus) { ++glcDrawCalls; glcVertsDrawn += g.count; }
                gl.DrawArrays(GL_TRIANGLES, static_cast<GLint>(g.first), static_cast<GLsizei>(g.count));
                // 2: depth only, discarding failures.
                if (locGlrAfail >= 0)
                    gl.Uniform1i(locGlrAfail, 2);
                gl.DepthMask(1);
                gl.ColorMask(0, 0, 0, 0);
                if (glrCensus) ++glcDrawCalls;
                gl.DrawArrays(GL_TRIANGLES, static_cast<GLint>(g.first), static_cast<GLsizei>(g.count));
                gl.ColorMask(wr, wg, wb, 0);
                ++glrTwoPass;
            }
            // ★ cont.331q: the same per-draw state string feeds the device-order trace, so a draw
            // that moves the traced target's mean is named exactly (its blend, its source, its
            // alpha test) instead of being guessed at from an aggregate afterwards.
            // ★ cont.332i: built for the pixel watch too, so a line that moves the watched
            // pixel is NAMED by the same state string the device-order trace uses. glrOrderShot
            // still gates itself on the traced key, so widening the condition prints nothing new.
            const bool wantWhat = b.trace || (glrOrderArmed() && b.targetKey == glrOrderKey) || pwArmed;
            char what[288];
            what[0] = '\0';
            if (wantWhat)
            {
                std::snprintf(what, sizeof(what),
                              "ref=%ux%u g%03zu n=%5u key=%016llx tbp=%llu(%ux%u) tex=%s%s tfx=%u tcc=%u lin=%u abe=%u abcd=%x%x%x%x fix=%3u "
                              "ate=%u atst=%u aref=%3u afail=%u zte=%u ztst=%u zmsk=%u FBMSK=%08x src=%u/%02x aem=%u fge=%u/%02x%02x%02x",
                              b.refW, b.refH, traceIdx, g.count,
                              (unsigned long long)g.texKey,
                              (unsigned long long)(g.texKey & 0x3FFFull),   // cont.332l: tbp is now bits 0-13
                              (glrTextures.count(g.texKey) ? glrTextures[g.texKey].w : 0u),
                              (glrTextures.count(g.texKey) ? glrTextures[g.texKey].h : 0u),
                              g.texKey ? "y" : "n", g.texIsTarget ? "(tgt)" : "",
                              g.tfx, g.tcc, g.lin, g.abe, g.ba, g.bb, g.bc, g.bd, g.bfix,
                              g.ate, g.atst, g.aref, g.afail, g.zte, g.ztst, g.zmsk, g.fbmsk,
                              g.srcMode, g.srcTexa, g.srcAem, g.fge, g.fogR, g.fogG, g.fogB);
                if (b.trace)
                    glcTraceShot(what, sceneW, sceneH, b.targetKey);
                glrOrderShot(what, *tgt, b.targetKey);
            }
            // ★★★★★ cont.332i: did THIS group move the watched pixel? Only a change is printed;
            // the draws that left it alone are counted, so the log is the layer stack and nothing
            // else. `quiet` is the running count of groups that touched neither.
            if (pwArmed && pwOk)
            {
                bool pwNowOk = false;
                const uint32_t pwNow = glPixWatchRead(*tgt, b.refH, pwNowOk);
                if (pwNowOk && pwNow != pwPrev)
                {
                    ++glPixWatchShown;
                    std::fprintf(stderr,
                                 "[gsgpu:glpixwatch] #%llu flip=%llu tgt=0x%llx %02x%02x%02x%02x -> "
                                 "%02x%02x%02x%02x quiet=%llu | %s\n",
                                 glPixWatchShown, gs2CurrentFlip(), (unsigned long long)b.targetKey,
                                 (pwPrev >> 24) & 0xFFu, (pwPrev >> 16) & 0xFFu,
                                 (pwPrev >> 8) & 0xFFu, pwPrev & 0xFFu,
                                 (pwNow >> 24) & 0xFFu, (pwNow >> 16) & 0xFFu,
                                 (pwNow >> 8) & 0xFFu, pwNow & 0xFFu,
                                 glPixWatchQuiet, what);
                    pwPrev = pwNow;
                }
                else if (pwNowOk)
                    ++glPixWatchQuiet;
            }
        }
        glcQueryEnd();
        if (glrCensus)
        {
            ++glcBatchDrawn;
            GlcTarget &ts = glcTargets[b.targetKey];
            ++ts.batches; ++ts.lifeBatches;
            ts.groups += b.groups.size(); ts.lifeGroups += b.groups.size();
            ts.verts += b.verts.size();   ts.lifeVerts += b.verts.size();
        }
        gl.BindVertexArray(0);
        gl.Disable(GL_SCISSOR_TEST);
        gl.ColorMask(1, 1, 1, 1);
        gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
        if (glrProf)
        {
            const auto tC = std::chrono::steady_clock::now();
            auto ns = [](auto a, auto b2) { return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(b2 - a).count(); };
            tVbo += ns(t0, tB); tTexUp += ns(tB, tA); tLoop += ns(tA, tC); tDraw += ns(t0, tC);
            ++nBatch;
        }
        glrDrawn += b.groups.size();
        glrVerts += b.verts.size();
        ++glrDrawsSinceResolve;
        const GLenum err = gl.GetError();
        if (err != 0u && glrErrLogs < 8u)
        {
            ++glrErrLogs;
            std::fprintf(stderr, "[gsgpu:glr] GL error 0x%x in draw (groups=%zu verts=%zu)\n",
                         static_cast<unsigned>(err), b.groups.size(), b.verts.size());
        }
    }

    // ★ cont.329 phase 4: create/resize the scene target for `key`, plus the shared resolve
    // target. GROW-ONLY: the presenter's display height alternates 512/511 every frame, and an
    // exact-size test rebuilt everything twice per frame -- 2168 reallocations in one 150 s run.
    GlTarget *ensureGlTarget(uint64_t key, uint32_t dispW, uint32_t dispH, bool wantHdr = false, uint32_t zbp = 0u, uint32_t zpsm = 0u)
    {
        if (glrFailed || dispW == 0u || dispH == 0u)
            return nullptr;
        auto giveUp = [&](const char *why) -> GlTarget *
        {
            glrFailed = true;
            std::fprintf(stderr, "[gsgpu] GL renderer DISABLED: %s (gl=0x%x)\n", why,
                         static_cast<unsigned>(gl.GetError()));
            return nullptr;
        };
        auto makeColorTex = [&](GLuint &tex, uint32_t w, uint32_t h, bool hdr)
        {
            gl.GenTextures(1, &tex);
            gl.BindTexture(GL_TEXTURE_2D, tex);
            if (hdr)
                gl.TexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_RGBA16F), static_cast<GLsizei>(w),
                              static_cast<GLsizei>(h), 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
            else
                gl.TexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_RGBA8), static_cast<GLsizei>(w),
                              static_cast<GLsizei>(h), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
            gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
            gl.BindTexture(GL_TEXTURE_2D, 0);
        };

        const uint32_t wantW = dispW * glrScale, wantH = dispH * glrScale;
        GlTarget &t = glrTargets[key];
        t.refW = dispW; t.refH = dispH; // rotk row 257: where the guest's rows sit in it (see uTgtMap)
        // cont.343b: an HDR request on an RGBA8 target rebuilds it (the content is lost -- in this
        // game the request arrives right after the guest's own black clear of that target, and the
        // fresh target is cleared below). A target never goes back to RGBA8.
        const bool toHdr = wantHdr && !t.hdr;
        if (t.fbo == 0u || t.w < wantW || t.h < wantH || toHdr)
        {
            if (toHdr)
                std::fprintf(stderr, "[gsgpu:glr] target key=%llx -> HDR (RGBA16F) for COLCLAMP=0 additive draws\n",
                             static_cast<unsigned long long>(key));
            t.hdr = t.hdr || wantHdr;
            const uint32_t allocW = wantW > t.w ? wantW : t.w;
            const uint32_t allocH = wantH > t.h ? wantH : t.h;
            if (t.color) gl.DeleteTextures(1, &t.color);
            if (t.depth && !t.depthShared) gl.DeleteTextures(1, &t.depth);
            if (t.fbo) gl.DeleteFramebuffers(1, &t.fbo);
            t.color = t.depth = t.fbo = 0u;
            t.depthShared = false;
            makeColorTex(t.color, allocW, allocH, t.hdr);
            // cont.344: a depth texture shared by ZBP (see glrDepthPool). Reusing an existing one means
            // this target joins a Z buffer other targets already wrote this frame -- it must NOT be
            // cleared below.
            bool depthPre = false;
            const uint64_t poolKey = glrDepthPoolKey(zbp, zpsm, allocW, allocH);
            t.depthPoolKey = poolKey;
            if (glrZShare())
            {
                auto pe = glrDepthPool.find(poolKey);
                if (pe != glrDepthPool.end() && pe->second.tex)
                {
                    t.depth = pe->second.tex;
                    t.depthShared = true;
                    depthPre = true;
                }
            }
            if (!depthPre)
            {
            // ★★★★ cont.332j: a TEXTURE, not a renderbuffer. AFAIL=FB_ONLY needs the depth
            // already at a pixel to be READABLE (see PS2X_GS_GLR_AFAIL=5): a fragment that fails
            // the alpha test must write colour but leave Z alone, and the only way to leave Z
            // alone in GL is to write back the value that is already there. A renderbuffer can be
            // neither sampled nor copied into a sampleable object.
            gl.GenTextures(1, &t.depth);
            gl.BindTexture(GL_TEXTURE_2D, t.depth);
            gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            gl.TexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_DEPTH_COMPONENT24),
                          static_cast<GLsizei>(allocW), static_cast<GLsizei>(allocH), 0,
                          GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
            gl.BindTexture(GL_TEXTURE_2D, 0);
            if (glrZShare())
            {
                GlDepthPoolEntry &pe = glrDepthPool[poolKey];
                if (std::getenv("PS2X_GS_GLR_ZALIASLOG"))
                    std::fprintf(stderr, "[gsgpu:zalias] POOL lookup zbp=0x%x zpsm=0x%x alloc %ux%u tex=%u (target key=0x%llx)\n",
                                 zbp, zpsm, allocW, allocH, pe.tex, (unsigned long long)key);
                pe.tex = t.depth;
                pe.lastClearFrame = glrFrames;
                glrDepthByZbp[static_cast<uint32_t>(poolKey & 0xFFFFFu)].push_back(poolKey);
                t.depthShared = true;
            }
            }
            gl.GenFramebuffers(1, &t.fbo);
            gl.BindFramebuffer(GL_FRAMEBUFFER, t.fbo);
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.color, 0);
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, t.depth, 0);
            if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                return giveUp("scene framebuffer incomplete");
            // ★ Clear a FRESH target immediately. A resolve of a never-drawn target must return a
            // defined picture; without this, build 815 presented pure black (not even the clear
            // colour) whenever the display pointed at a target no draw had reached yet.
            gl.Viewport(0, 0, static_cast<GLsizei>(allocW), static_cast<GLsizei>(allocH));
            gl.Disable(GL_SCISSOR_TEST);
            gl.ColorMask(1, 1, 1, 1);
            gl.DepthMask(1);
            glrSetClearColor();
            gl.ClearDepth(0.0);
            gl.Clear(GL_COLOR_BUFFER_BIT | (depthPre ? 0u : GL_DEPTH_BUFFER_BIT));
            t.w = allocW; t.h = allocH; t.needClear = true;
            t.depthJoined = depthPre; // cont.345
            // ★ cont.331q: a REALLOC destroys the colour texture and clears the new one, so it is
            // a way a target can silently lose everything placed in it. In the device-order log it
            // must be visible as its own step, not inferred from the ALLOC line below.
            glrOrderShot("ALLOC+CLEAR (target (re)created)", t, key);
            glrOrderNote("ALLOC+CLEAR (target (re)created)", key);
            gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
            std::fprintf(stderr, "[gsgpu:glr] target key=%llx ALLOC %ux%u (display %ux%u x%u) zbp=0x%x zpsm=0x%x depthTex=%u shared=%d pre=%d\n",
                         static_cast<unsigned long long>(key), allocW, allocH, dispW, dispH, glrScale, zbp, zpsm, t.depth, t.depthShared ? 1 : 0, depthPre ? 1 : 0);
        }

        else if (glrZShare() && glrZRebind() && zbp != 0u && t.fbo != 0u)
        {
            // cont.345 (2): the batch names a ZBUF this target's depth did not come from.
            const uint64_t want = glrDepthPoolKey(zbp, zpsm, t.w, t.h);
            if (want != t.depthPoolKey)
            {
                GlDepthPoolEntry &pe = glrDepthPool[want];
                bool existed = pe.tex != 0u;
                if (!existed)
                {
                    gl.GenTextures(1, &pe.tex);
                    gl.BindTexture(GL_TEXTURE_2D, pe.tex);
                    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    gl.TexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_DEPTH_COMPONENT24),
                                  static_cast<GLsizei>(t.w), static_cast<GLsizei>(t.h), 0,
                                  GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
                    gl.BindTexture(GL_TEXTURE_2D, 0);
                    pe.lastClearFrame = glrFrames;
                    glrDepthByZbp[static_cast<uint32_t>(want & 0xFFFFFu)].push_back(want);
                }
                const GLuint oldDepth = t.depth;
                const bool oldShared = t.depthShared;
                t.depth = pe.tex;
                t.depthShared = true;
                t.depthPoolKey = want;
                t.depthJoined = existed;
                gl.BindFramebuffer(GL_FRAMEBUFFER, t.fbo);
                gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, t.depth, 0);
                const GLenum st = gl.CheckFramebufferStatus(GL_FRAMEBUFFER);
                if (!existed)
                {
                    // a fresh pooled depth: define it (0 = nothing drawn, PS2 bigger-is-nearer)
                    gl.Viewport(0, 0, static_cast<GLsizei>(t.w), static_cast<GLsizei>(t.h));
                    gl.Disable(GL_SCISSOR_TEST);
                    gl.DepthMask(1);
                    gl.ClearDepth(0.0);
                    gl.Clear(GL_DEPTH_BUFFER_BIT);
                }
                gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
                if (oldDepth && !oldShared)
                    gl.DeleteTextures(1, &oldDepth);
                std::fprintf(stderr, "[gsgpu:glr] target key=%llx REBIND depth -> zbp=0x%x zpsm=0x%x tex=%u (%s) fbo-status=0x%x\n",
                             static_cast<unsigned long long>(key), zbp, zpsm, t.depth, existed ? "joined" : "fresh", static_cast<unsigned>(st));
                if (st != GL_FRAMEBUFFER_COMPLETE)
                    return giveUp("scene framebuffer incomplete after depth rebind");
            }
        }

        if (glrResolveFbo == 0u || glrResolveW < dispW || glrResolveH < dispH)
        {
            const uint32_t rw = dispW > glrResolveW ? dispW : glrResolveW;
            const uint32_t rh = dispH > glrResolveH ? dispH : glrResolveH;
            if (glrResolveTex) gl.DeleteTextures(1, &glrResolveTex);
            if (glrResolveFbo) gl.DeleteFramebuffers(1, &glrResolveFbo);
            glrResolveTex = glrResolveFbo = 0u;
            makeColorTex(glrResolveTex, rw, rh, false);
            gl.GenFramebuffers(1, &glrResolveFbo);
            gl.BindFramebuffer(GL_FRAMEBUFFER, glrResolveFbo);
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                    glrResolveTex, 0);
            if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                return giveUp("resolve framebuffer incomplete");
            gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
            glrResolveW = rw; glrResolveH = rh;
        }
        return &t;
    }

    // Render one frame and hand it back as display-sized RGBA8 rows.
    // PHASE 1: the scene target is only CLEARED -- nothing is drawn into it yet. A distinctly
    // non-black clear is deliberate: it makes "the GL path presented this frame" unmistakable in
    // the window, and distinguishes it from the stale-VRAM picture the CPU decode would produce.
    // ★★★ cont.330f PS2X_GS_GLR_PRESENTLOG=<n> (default 0 = OFF, read-only): every n-th GL
    // present, print WHICH target key the DISPLAY selects and how many draws have ever landed in
    // it, plus the keys that did receive draws. ensureGlTarget CREATES a missing key and hands
    // back a cleared target, so presenting an address GL never drew into is a silently BLACK
    // frame with no GL error -- indistinguishable in a log from "the scene really is black".
    unsigned long long glrPresentN = 0;
    unsigned long long glrSeedVerifyDone = 0;
    std::unordered_map<uint64_t, unsigned long long> glrDrawsPerKey;
    // cont.345: resolve `key` at the flip point and file the pixels under `seq` (see glrSnapResolve).
    void glSnapOnThread(uint64_t key, uint64_t seq)
    {
        if (!glrSnapResolve)
            return;
        auto it = glrTargets.find(key);
        if (it == glrTargets.end())
        { ++glrSnapNoTarget; return; }
        // Resolve at the display width the presenter last asked for and the target's FULL
        // allocated height: the output buffer is kHostW x kHostH with the image top-left, so a
        // taller resolve serves every shorter request (this game alternates 512x512 / 512x511).
        uint32_t w = it->second.w / std::max(1u, glrScale), h = it->second.h / std::max(1u, glrScale);
        auto ds = glrDispSize.find(key);
        if (ds != glrDispSize.end())
        { w = ds->second.first; h = std::max(h, ds->second.second); }
        if (w == 0u || h == 0u)
        { ++glrSnapNoTarget; return; }
        GlSnap &slot = glrSnaps[glrSnapNext];
        glrSnapNext = (glrSnapNext + 1u) % glrSnaps.size();
        slot.seq = 0; slot.key = ~0ull;
        if (!glFrameOnThread(key, w, h, &slot.px))
            return;
        slot.seq = seq; slot.key = key; slot.w = w; slot.h = h;
        ++glrSnapMade;
        // PS2X_GS_GLR_SNAPLOG=2: one line per snapshot -- the draws that landed in this key since
        // its previous snapshot (a half-drawn frame at a cut shows as an anomalous count).
        static const int s_snapLog = []
        { const char *e = std::getenv("PS2X_GS_GLR_SNAPLOG"); return e ? std::atoi(e) : 0; }();
        if (s_snapLog >= 2)
        {
            static std::unordered_map<uint64_t, unsigned long long> s_lastDraws;
            const unsigned long long now = glrDrawsPerKey.count(key) ? glrDrawsPerKey[key] : 0ull;
            unsigned long long &prev = s_lastDraws[key];
            std::fprintf(stderr, "[gsgpu:glsnap] SNAP seq=%llu key=0x%llx %ux%u draws-into-key-since-its-last-snap=%llu (total %llu) draws-since-any-resolve=%llu\n",
                         (unsigned long long)seq, (unsigned long long)key, w, h, now - prev, now,
                         (unsigned long long)glrDrawsSinceResolve);
            prev = now;
        }
    }

    void glSnapReport()
    {
        const unsigned long long n = glrSnapHit + glrSnapLive;
        if (n == 1ull || (n % 256ull) == 0ull)
            std::fprintf(stderr, "[gsgpu:glsnap] made=%llu hits=%llu live=%llu (no-seq=%llu no-key=%llu size=%llu) no-target=%llu\n",
                         glrSnapMade, glrSnapHit, glrSnapLive, glrSnapMissSeq, glrSnapMissKey, glrSnapMissSize, glrSnapNoTarget);
    }
    // cont.345: the presenter's resolve -- a filed snapshot for (seq, key) when there is one.
    bool glFrameForPresent(uint64_t key, uint64_t seq, uint32_t dispW, uint32_t dispH,
                           std::vector<uint8_t> *out)
    {
        if (glrSnapResolve && seq != 0ull && out)
        {
            bool seqSeen = false, keySeen = false;
            for (const GlSnap &sn : glrSnaps)
            {
                if (sn.seq != seq) continue;
                seqSeen = true;
                if (sn.key != key) continue;
                keySeen = true;
                // The output buffer is kHostW x kHostH with the image top-left, so a taller
                // snapshot serves a shorter request (this game alternates 512x512 / 512x511).
                if (sn.w == dispW && sn.h >= dispH)
                {
                    *out = sn.px;
                    ++glrSnapHit;
                    glSnapReport();
                    return true;
                }
            }
            if (!seqSeen) ++glrSnapMissSeq; else if (!keySeen) ++glrSnapMissKey; else ++glrSnapMissSize;
            static unsigned s_missShown = 0;
            static const bool s_missLog = []
            { const char *e = std::getenv("PS2X_GS_GLR_SNAPLOG"); return e && e[0] && e[0] != '0'; }();
            if (s_missShown < 12u || s_missLog)
            {
                ++s_missShown;
                std::string slots;
                for (const GlSnap &sn : glrSnaps)
                {
                    if (s_missShown > 12u) break; // PS2X_GS_GLR_SNAPLOG=1: every miss, one short line
                    char b2[80];
                    std::snprintf(b2, sizeof(b2), " [seq=%llu key=0x%llx %ux%u]", (unsigned long long)sn.seq,
                                  (unsigned long long)sn.key, sn.w, sn.h);
                    slots += b2;
                }
                std::fprintf(stderr, "[gsgpu:glsnap] MISS seq=%llu key=0x%llx %ux%u (%s) slots:%s\n",
                             (unsigned long long)seq, (unsigned long long)key, dispW, dispH,
                             !seqSeen ? "no-seq" : !keySeen ? "no-key" : "size", slots.c_str());
            }
        }
        if (seq != 0ull) ++glrSnapLive;
        glSnapReport();
        glrDispSize[key] = std::make_pair(dispW, dispH);
        return glFrameOnThread(key, dispW, dispH, out);
    }

    bool glFrameOnThread(uint64_t targetKey, uint32_t dispW, uint32_t dispH,
                         std::vector<uint8_t> *out)
    {
        if (!out)
            return false;
        // ★ cont.331q PS2X_GS_GLR_ORDER=disp: latch the key the DISPLAY actually selects, so the
        // trace can be aimed without first running a PRESENTLOG pass to look it up.
        if (glrOrderDisp && glrOrderKey == ~0ull && glrFrames >= glrOrderAt)
        {
            glrOrderKey = targetKey;
            std::fprintf(stderr, "[gsgpu:order] latched DISPLAY target key=0x%llx at frame %llu\n",
                         (unsigned long long)targetKey, glrFrames);
        }
        // Resolve the target the DISPLAY points at. With double buffering this alternates, and
        // resolving a single shared target is what merged two frames together.
        const bool keyExisted = glrTargets.find(targetKey) != glrTargets.end();
        GlTarget *tgt = ensureGlTarget(targetKey, dispW, dispH);
        if (!tgt)
            return false;
        {
            static const unsigned long long presentLog = []
            { const char *e = std::getenv("PS2X_GS_GLR_PRESENTLOG"); return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 0ull; }();
            ++glrPresentN;
            if (presentLog && (glrPresentN % presentLog) == 0ull)
            {
                std::string keys;
                for (const auto &kv : glrDrawsPerKey)
                {
                    char b2[64];
                    std::snprintf(b2, sizeof(b2), " 0x%llx=%llu",
                                  (unsigned long long)kv.first, kv.second);
                    keys += b2;
                }
                auto it = glrDrawsPerKey.find(targetKey);
                std::fprintf(stderr,
                             "[gsgpu:present] #%llu DISPLAY key=0x%llx existed=%d %ux%u | draws-ever-into-it=%llu"
                             " | draws-since-resolve=%llu | all keys:%s\n",
                             glrPresentN, (unsigned long long)targetKey, keyExisted ? 1 : 0,
                             dispW, dispH,
                             it == glrDrawsPerKey.end() ? 0ull : it->second,
                             (unsigned long long)glrDrawsSinceResolve, keys.c_str());
            }
        }
        // ★ Nothing drawn since the last resolve, and we already have that frame: hand it back
        // instead of repeating the downsample and a synchronous readback. The present decode is
        // called more than once per displayed frame, and each readback stalls the GPU pipeline.
        if (glrFrameCache && glrDrawsSinceResolve == 0ull && !glrLastFrame.empty() &&
            glrLastW == dispW && glrLastH == dispH && glrLastKey == targetKey)
        {
            *out = glrLastFrame;
            ++nFrameCacheHit;
            // ★ cont.331q: a cache hit does NOT read the target, so the picture presented here is
            // the previous resolve's. In device order that is a real step and has to show.
            glrOrderShot("RESOLVE SKIPPED (frame-cache hit, target not read)", *tgt, targetKey);
            glrOrderNote("RESOLVE SKIPPED (frame-cache hit)", targetKey);
            return true;
        }
        const auto f0 = glrProf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // The allocation is a high-water mark (see ensureGlTarget); everything below works on the
        // REQUESTED rect, anchored at the origin.
        const uint32_t sceneW = dispW * glrScale, sceneH = dispH * glrScale;
        gl.BindFramebuffer(GL_FRAMEBUFFER, tgt->fbo);
        gl.Viewport(0, 0, static_cast<GLsizei>(sceneW), static_cast<GLsizei>(sceneH));
        gl.Disable(GL_SCISSOR_TEST);
        // ★ cont.331q: THE step the whole trace exists for -- what the target holds at the instant
        // the resolve reads it, on the same line numbering as every clear/seed/draw before it. The
        // shot is taken BEFORE the blit because the blit is the read.
        if (glrOrderArmed() && targetKey == glrOrderKey)
        {
            char w2[160];
            std::snprintf(w2, sizeof(w2),
                          "RESOLVE reads target here (draws-since-resolve=%llu, disp %ux%u)",
                          (unsigned long long)glrDrawsSinceResolve, dispW, dispH);
            glrOrderShot(w2, *tgt, targetKey);
            ++glrOrderResolves;
        }
        const unsigned long long orderLast =
            (glrOrderFlip ? (glrOrderArmFrame == ~0ull ? ~0ull : glrOrderArmFrame) : glrOrderAt) +
            glrOrderN;
        if (glrOrderEnv && glrOrderEnv[0] && glrFrames + 1ull == orderLast)
            std::fprintf(stderr,
                         "[gsgpu:order] window closes: %llu lines, %llu resolves of the traced "
                         "key 0x%llx over %llu global frames\n",
                         glrOrderSeq, glrOrderResolves, (unsigned long long)glrOrderKey,
                         glrOrderN);
        // ★ RESOLVE NEVER CLEARS. CopyFrameToHostRgba is called more than once per present (the
        // preferred-source attempt, then the display-frame fallback), so clearing here made the
        // SECOND call of a pair resolve a freshly-wiped target and present an empty frame --
        // builds 801 and 802 both interleaved real frames with pure background for this reason.
        // The clear belongs to the start of a frame, which is the first DRAW after a resolve
        // (glrClearIfNeeded, called from drawBatchGlOnThread). The target is also cleared once at
        // creation, so a resolve with no draws at all still returns a defined picture.

        std::chrono::steady_clock::time_point s0{}, s1{};
        if (glrSplit)
        {
            s0 = std::chrono::steady_clock::now();
            gl.Finish(); // cont.371 split: the wait for the frame's queued GPU work, alone
            s1 = std::chrono::steady_clock::now();
        }
        const auto f1 = glrProf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // Resolve: scene -> display size, on the GPU. This is the step that keeps the readback
        // below constant in `glrScale`.
        gl.BindFramebuffer(GL_READ_FRAMEBUFFER, tgt->fbo);
        gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, glrResolveFbo);
        gl.BlitFramebuffer(0, 0, static_cast<GLint>(sceneW), static_cast<GLint>(sceneH),
                           0, 0, static_cast<GLint>(dispW), static_cast<GLint>(dispH),
                           GL_COLOR_BUFFER_BIT, GL_LINEAR);

        // ★ THE OUTPUT BUFFER SHAPE IS AN INTERFACE, NOT A CHOICE. Everything downstream of
        // CopyFrameToHostRgba -- the present cache, applyFieldPresentation, the two-CRT composite,
        // PS2X_GS_PRESENT_SAVE and copyLatchedHostPresentationFrame -- indexes rows as
        // `y * kHostW * 4` into a kHostW x kHostH buffer, with the live image in the TOP-LEFT
        // sub-rect. Returning a tightly-packed dispW*dispH buffer instead made the presenter read
        // 640-stride rows out of a smaller allocation: build 799 showed exactly the valid fraction
        // (512*511)/(640*512) = 79.8% of the frame, and out-of-bounds garbage below it.
        const auto f2 = glrProf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const uint32_t copyW = dispW < kHostW ? dispW : kHostW;
        const uint32_t copyH = dispH < kHostH ? dispH : kHostH;
        out->assign(static_cast<size_t>(kHostW) * kHostH * 4u, 0u);

        thread_local std::vector<uint8_t> scratch;
        scratch.resize(static_cast<size_t>(copyW) * copyH * 4u);
        gl.BindFramebuffer(GL_READ_FRAMEBUFFER, glrResolveFbo);
        gl.PixelStorei(GL_PACK_ALIGNMENT, 1); // rows are width*4; do not rely on the 4-byte default
        gl.ReadPixels(0, 0, static_cast<GLsizei>(copyW), static_cast<GLsizei>(copyH), GL_RGBA,
                      GL_UNSIGNED_BYTE, scratch.data());
        gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
        const auto f3 = glrProf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        std::chrono::steady_clock::time_point s2{};
        if (glrSplit)
        {
            s2 = std::chrono::steady_clock::now();
            auto ns = [](auto a, auto b2) { return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(b2 - a).count(); };
            splitAcc.finishWait += ns(s0, s1);
            splitAcc.blitRead += ns(s1, s2);
        }

        // GL's origin is bottom-left; the presenter's rows run top-down. Flip while scattering
        // into the 640-stride layout, so this costs nothing extra.
        const size_t srcRow = static_cast<size_t>(copyW) * 4u;
        const size_t dstRow = static_cast<size_t>(kHostW) * 4u;
        for (uint32_t y = 0; y < copyH; ++y)
            std::memcpy(out->data() + static_cast<size_t>(y) * dstRow,
                        scratch.data() + static_cast<size_t>(copyH - 1u - y) * srcRow, srcRow);

        // ★★★★ cont.332c: the HI-RES present latch. Read the scene target at its own size, flip
        // the rows into the latch, and let the presenter upload that instead of the 640x448 image.
        // Done after the display-sized readback so the shared path is byte-for-byte what it was.
        if (glrHires && sceneW != 0u && sceneH != 0u)
        {
            const auto h0 = std::chrono::steady_clock::now();
            thread_local std::vector<uint8_t> hiScratch;
            hiScratch.resize(static_cast<size_t>(sceneW) * sceneH * 4u);
            gl.BindFramebuffer(GL_READ_FRAMEBUFFER, tgt->fbo);
            gl.PixelStorei(GL_PACK_ALIGNMENT, 1);
            gl.ReadPixels(0, 0, static_cast<GLsizei>(sceneW), static_cast<GLsizei>(sceneH),
                          GL_RGBA, GL_UNSIGNED_BYTE, hiScratch.data());
            gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
            const size_t hiRow = static_cast<size_t>(sceneW) * 4u;
            {
                std::lock_guard<std::mutex> lk(hiresMx);
                hiresBuf.resize(hiScratch.size());
                for (uint32_t y = 0; y < sceneH; ++y)
                    std::memcpy(hiresBuf.data() + static_cast<size_t>(y) * hiRow,
                                hiScratch.data() + static_cast<size_t>(sceneH - 1u - y) * hiRow,
                                hiRow);
                hiresW = sceneW;
                hiresH = sceneH;
                hiresFresh = true;
            }
            ++hiresFrames;
            if (glrHiresSaveDir && glrHiresSaveDir[0] && glrHiresSaveEvery &&
                (hiresFrames % glrHiresSaveEvery) == 0ull)
            {
                char path[512];
                const auto ppm = [&](const char *name, const uint8_t *px, uint32_t w, uint32_t h,
                                     uint32_t stridePx)
                {
                    std::snprintf(path, sizeof(path), "%s/%s_%06llu_%ux%u.ppm", glrHiresSaveDir,
                                  name, hiresFrames, w, h);
                    if (FILE *f = std::fopen(path, "wb"))
                    {
                        std::fprintf(f, "P6\n%u %u\n255\n", w, h);
                        for (uint32_t y = 0; y < h; ++y)
                            for (uint32_t x = 0; x < w; ++x)
                            {
                                const uint8_t *p = px + (static_cast<size_t>(y) * stridePx + x) * 4u;
                                std::fputc(p[0], f); std::fputc(p[1], f); std::fputc(p[2], f);
                            }
                        std::fclose(f);
                    }
                };
                // The pair, from this one present: what the window shows with the feature ON, and
                // what it showed before (the display-sized resolve, still computed above).
                ppm("hires", hiresBuf.data(), sceneW, sceneH, sceneW);
                ppm("disp", out->data(), copyW, copyH, kHostW);
                std::fprintf(stderr, "[gsgpu:hires] saved pair #%llu %ux%u + %ux%u -> %s\n",
                             hiresFrames, sceneW, sceneH, copyW, copyH, glrHiresSaveDir);
            }
            hiresNs += static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - h0).count());
            if ((hiresFrames % 256ull) == 0ull)
                std::fprintf(stderr,
                             "[gsgpu:hires] frames=%llu %ux%u readback+flip=%.2f ms/frame\n",
                             hiresFrames, sceneW, sceneH,
                             double(hiresNs) * 1e-6 / double(hiresFrames));
        }

        if (glrSplit)
        {
            // Everything after the display readback: its row flip + the hi-res readback/flip.
            splitAcc.hires += (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - s2).count();
            ++splitAcc.resolves;
        }
        const GLenum err = gl.GetError();
        if (err != 0u && glrErrLogs < 8u)
        {
            ++glrErrLogs;
            std::fprintf(stderr, "[gsgpu:glr] GL error 0x%x after frame %llu\n",
                         static_cast<unsigned>(err), glrFrames);
        }
        if (glrProf)
        {
            const auto f4 = std::chrono::steady_clock::now();
            auto ns = [](auto a, auto b2) { return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(b2 - a).count(); };
            tClear += ns(f0, f1); tBlit += ns(f1, f2); tRead += ns(f2, f3); tFlip += ns(f3, f4);
            tFrame += ns(f0, f4);
            ++nResolve;
        }
        // Keep the resolved frame for a repeat request within the same present.
        if (glrFrameCache)
        {
            glrLastFrame = *out;
            glrLastW = dispW;
            glrLastH = dispH;
            glrLastKey = targetKey;
        }
        // ★★★ cont.329k PS2X_GS_GLR_DUMPTGT=<dir>: save EVERY live GL target once every
        // PS2X_GS_GLR_DUMPEVERY resolves (default 512). The background screens present a vista
        // that is TILED 4x in GL while the sampling draw's UVs are provably correct, so the
        // question is what the sampled TARGET actually holds.
        {
            static const char *dumpDir = std::getenv("PS2X_GS_GLR_DUMPTGT");
            static const unsigned dumpEvery = []
            { const char *e = std::getenv("PS2X_GS_GLR_DUMPEVERY"); unsigned v = (e && e[0]) ? unsigned(std::strtoul(e, nullptr, 10)) : 512u; return v ? v : 512u; }();
            if (dumpDir && dumpDir[0] && (glrFrames % dumpEvery) == 0u)
            {
                static std::vector<uint8_t> px;
                for (auto &kv : glrTargets)
                {
                    GlTarget &t = kv.second;
                    if (!t.fbo || !t.w || !t.h) continue;
                    px.resize(size_t(t.w) * t.h * 4u);
                    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, t.fbo);
                    gl.PixelStorei(GL_PACK_ALIGNMENT, 1);
                    gl.ReadPixels(0, 0, GLsizei(t.w), GLsizei(t.h), GL_RGBA, GL_UNSIGNED_BYTE, px.data());
                    // ★ cont.330b: name the FBO id too. The seed blits log fbo=5 for target
                    // 0x180 and report a COMPLETE framebuffer with no GL error, yet this dump
                    // reads that target back as exactly 0 while its neighbours read non-zero --
                    // so "is the object we blit into the object we read back" is the last
                    // unverified link, and it is answered by printing the id on both sides.
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/tgt_%06llu_fbp%03llx_fbo%u_%ux%u.ppm",
                                  dumpDir, (unsigned long long)glrFrames,
                                  (unsigned long long)kv.first, t.fbo, t.w, t.h);
                    if (FILE *f = std::fopen(path, "wb"))
                    {
                        std::fprintf(f, "P6\n%u %u\n255\n", t.w, t.h);
                        for (uint32_t y = 0; y < t.h; ++y)
                        {
                            const uint8_t *row = px.data() + size_t(t.h - 1u - y) * t.w * 4u;
                            for (uint32_t x = 0; x < t.w; ++x)
                            { std::fputc(row[x*4], f); std::fputc(row[x*4+1], f); std::fputc(row[x*4+2], f); }
                        }
                        std::fclose(f);
                    }
                }
                std::fprintf(stderr,
                             "[gsgpu:seedtot] batches=%llu with-seeds=%llu rects=%llu pixels=%llu\n",
                             glrSeedBatches, glrSeedBatchesWith, glrSeedRects, glrSeedPixels);
                gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
            }
        }
        ++glrFrames;
        // Only treat this resolve as a frame boundary if something was actually drawn since the
        // last one. Otherwise a second resolve within the same present would clear the target and
        // hand back an empty frame.
        // This target has now been displayed, so the next draw into it begins its next frame.
        // ★★ cont.329g census: a resolve that does NOT arm the clear leaves its target holding
        // everything already drawn into it -- the next frame's draws land ON TOP. That is the
        // accumulation hypothesis stated as a counter, so count it here rather than argue it.
        if (glrCensus)
        {
            ++glcResolves;
            GlcTarget &ts = glcTargets[targetKey];
            ++ts.resolves;
            if (glrDrawsSinceResolve == 0ull)
            {
                ++glcResolvesNoMark;
                ++ts.resolvesNoMark;
            }
        }
        if (glrDrawsSinceResolve != 0ull)
        {
            // ★★★★ cont.345 PS2X_GS_GLR_RESOLVECLEAR (default OFF; =1 restores): re-arm a frame-start
            // clear on every resolve. That clear wiped the target on the FIRST draw after a present --
            // and this game draws onto its DISPLAYED buffer after the flip (UI glyphs, the loading
            // screen, the menu text), so a second resolve of the same buffer before its next full
            // frame presented the clear colour (0.04,0.05,0.14 = the "solid dark-blue frame" the
            // user reported on the loading screen) plus whatever few draws had landed. Real time
            // only: under PS2X_VIRTUAL_TIME=1 presents and flips never double up. Measured on a
            // real-time replay (shadow_spot.pad, every flip saved): 26 dark-blue frames + 27
            // partial hero-select frames in 1,583 flips at default, 0 + 0 with the re-arm off.
            // PCSX2 never clears a render target on its own (GSTextureCache creates one from VRAM
            // and only the guest's own clear sprite clears it -- GSRendererHW::TryTargetClear);
            // the guest's full-screen clear quad + its FBMSK=ffffffff depth-only quad do that work
            // here too. A target is still cleared once when it is created or reallocated.
            if (glrResolveClear)
                tgt->needClear = true;
            glrDrawsSinceResolve = 0ull;
        }
        if (glrCensus && (glcResolves % 256ull) == 0ull)
            glcReport();
        if ((glrFrames % 512ull) == 0ull)
            std::fprintf(stderr, "[gsgpu:glr] frames=%llu groups=%llu verts=%llu blend-fallback=%llu tex{cached=%zu up=%llu binds=%llu} (scale x%u)\n",
                         glrFrames, glrDrawn, glrVerts, glrBlendFallback, glrTextures.size(), glrTexUploads, glrTexBinds, glrScale);
        if (glrProf)
            std::fprintf(stderr, "[gsgpu:glrtexobj] objects=%zu uploads=%llu allocations=%llu two-pass-groups=%llu\n",
                         glrTextures.size(), glrTexUploads, glrTexAlloc, glrTwoPass);
        // ★ cont.332i: the AFAIL split, reported on the SAME cadence as [gsgpu:glr] -- a path
        // that is never taken is indistinguishable from a path that does nothing, and this arc has
        // paid for that. ⚠ It rides the resolve, so it must be gated: ungated it printed 12,607
        // lines in one run AND the FIRST line reads {0,0} (no AFAIL group has been drawn yet),
        // which is exactly how a working path reads as a null. Take the LAST line.
        if ((glrFrames % 512ull) == 0ull)
            std::fprintf(stderr, "[gsgpu:glrafail] mode=%d groups{colour-then-Z=%llu pass-then-fail=%llu one-pass=%llu keep-z=%llu sw-z=%llu} depth-copies=%llu img-seeds=%llu\n",
                         glrAfailMode, glrAfailSimple, glrAfailPassFail, glrAfailOnePass,
                         glrAfailKeepZ, glrAfailSwZ, glrDepthCopies, glrDepthImgSeeds);
        if (glrProf && nBatch && nResolve)
        {
            auto us = [](unsigned long long n, unsigned long long c) { return double(n) / double(c) / 1000.0; };
            std::fprintf(stderr,
                         "[gsgpu:glrprof] batch n=%llu tot=%.1fus {texup=%.1f vbo=%.1f loop=%.1f} | "
                         "resolve n=%llu tot=%.1fus {clear=%.1f blit=%.1f READ=%.1f flip=%.1f} | "
                         "framecache-hits=%llu\n",
                         nBatch, us(tDraw, nBatch), us(tTexUp, nBatch), us(tVbo, nBatch), us(tLoop, nBatch),
                         nResolve, us(tFrame, nResolve), us(tClear, nResolve), us(tBlit, nResolve),
                         us(tRead, nResolve), us(tFlip, nResolve), nFrameCacheHit);
        }
        return true;
    }

    bool ensureHwRaster()
    {
        if (hwProgram != 0u)
            return true;
        if (hwFailed)
            return false;
        auto giveUp = [&](const char *why)
        {
            hwFailed = true;
            std::fprintf(stderr, "[gsgpu] hwraster DISABLED: %s (gl=0x%x) -- tile kernel in charge\n",
                         why, static_cast<unsigned>(gl.GetError()));
            return false;
        };
        // Both stages must declare the Mirror block identically (the linker rejects a
        // memory-qualifier mismatch), so the vertex stage carries `coherent` too.
        const std::string qual = hwCoherent ? "#define MIRROR_QUAL coherent\n" : "#define MIRROR_QUAL\n";
        const std::string lock = "#define HW_LOCK " + std::to_string(hwLock) + "\n" +
                                 "#define HW_ABL " + std::to_string(hwAbl) + "\n";
        const std::string vs = std::string(kRasterHeader) + qual + kRasterCommon + kHwVertex;
        const std::string fs = std::string(kRasterHeader) +
                               (hwLock != 0 ? "#extension GL_ARB_fragment_shader_interlock : require\n" : "") +
                               qual + lock + kRasterCommon + kHwFragment;
        hwProgram = buildGraphicsProgram(vs.c_str(), fs.c_str(), "hwraster");
        if (!hwProgram)
            return giveUp("program build");
        locHwDilate = gl.GetUniformLocation(hwProgram, "uDilate");
        locHwInvHalfW = gl.GetUniformLocation(hwProgram, "uInvHalfW");
        locHwInvHalfH = gl.GetUniformLocation(hwProgram, "uInvHalfH");
        gl.GenVertexArrays(1, &hwVao);
        gl.GenBuffers(1, &hwCountBuf);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, hwCountBuf);
        const uint32_t zeros[4] = {0u, 0u, 0u, 0u};
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, 16, zeros, GL_DYNAMIC_DRAW);
        gl.GenFramebuffers(1, &hwFbo);
        gl.BindFramebuffer(GL_FRAMEBUFFER, hwFbo);
        gl.FramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, static_cast<GLint>(kHwW));
        gl.FramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, static_cast<GLint>(kHwH));
        if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            return giveUp("no-attachment framebuffer incomplete");
        // Nothing but the fragment stage's SSBO writes may filter fragments or reorder them.
        gl.Disable(GL_CULL_FACE);
        gl.Disable(GL_DEPTH_TEST);
        gl.Disable(GL_STENCIL_TEST);
        gl.Disable(GL_SCISSOR_TEST);
        gl.Disable(GL_BLEND);
        gl.Disable(GL_MULTISAMPLE);
        if (hwCons)
            gl.Enable(GL_CONSERVATIVE_RASTERIZATION_INTEL);
        const GLenum err = gl.GetError();
        if (err != 0u)
        {
            std::fprintf(stderr, "[gsgpu] hwraster: GL error 0x%x during setup\n", static_cast<unsigned>(err));
            return giveUp("raster state setup");
        }
        std::fprintf(stderr, "[gsgpu] hwraster READY: conservative=%d dilate=%.2f px lock=%d coherent=%d abl=%d viewport=%ux%u\n",
                     hwCons ? 1 : 0, static_cast<double>(hwDilate), hwLock, hwCoherent ? 1 : 0, hwAbl, kHwW, kHwH);
        return true;
    }

    // Rasterize a batch into the mirror. The caller seeded the mirror with the CPU's
    // PRE-batch bytes (FIFO order guarantees those patches landed), so what this leaves
    // behind is directly comparable with the CPU's post-batch VRAM.
    // Bin a batch into 16x16 tiles for the tile kernel (the same CSR layout the host built
    // before cont.233; now the host skips it whenever the hardware path is requested, and this
    // fallback runs only if that path is unavailable).
    static void binBatch(GsGpuBatch &b)
    {
        uint32_t maxX = 0, maxY = 0;
        for (const GsGpuGeom &g : b.prims)
        {
            maxX = std::max(maxX, g.maxX);
            maxY = std::max(maxY, g.maxY);
        }
        b.tilesX = (maxX >> 4) + 1u;
        b.tilesY = (maxY >> 4) + 1u;
        const size_t tiles = size_t(b.tilesX) * b.tilesY;
        std::vector<uint32_t> counts(tiles + 1u, 0u);
        auto tileSpan = [&](const GsGpuGeom &g, uint32_t &tx0, uint32_t &ty0, uint32_t &tx1, uint32_t &ty1)
        {
            tx0 = g.minX >> 4; ty0 = g.minY >> 4;
            tx1 = std::min<uint32_t>(g.maxX >> 4, b.tilesX - 1u);
            ty1 = std::min<uint32_t>(g.maxY >> 4, b.tilesY - 1u);
        };
        for (const GsGpuGeom &g : b.prims)
        {
            uint32_t tx0, ty0, tx1, ty1;
            tileSpan(g, tx0, ty0, tx1, ty1);
            for (uint32_t ty = ty0; ty <= ty1; ++ty)
                for (uint32_t tx = tx0; tx <= tx1; ++tx)
                    ++counts[size_t(ty) * b.tilesX + tx];
        }
        b.binOffsets.assign(tiles + 1u, 0u);
        b.tileList.clear();
        uint32_t running = 0;
        for (size_t t = 0; t < tiles; ++t)
        {
            b.binOffsets[t] = running;
            if (counts[t] != 0u)
                b.tileList.push_back(static_cast<uint32_t>(t));
            running += counts[t];
        }
        b.binOffsets[tiles] = running;
        b.binEntries.assign(running, 0u);
        std::vector<uint32_t> cursor(b.binOffsets.begin(), b.binOffsets.end() - 1);
        for (size_t i = 0; i < b.prims.size(); ++i)
        {
            uint32_t tx0, ty0, tx1, ty1;
            tileSpan(b.prims[i], tx0, ty0, tx1, ty1);
            for (uint32_t ty = ty0; ty <= ty1; ++ty)
                for (uint32_t tx = tx0; tx <= tx1; ++tx)
                    b.binEntries[cursor[size_t(ty) * b.tilesX + tx]++] = static_cast<uint32_t>(i);
        }
    }

    void rasterBatchOnThread(GsGpuBatch &b)
    {
        if (!ensureRaster() || b.prims.empty() || b.states.empty())
            return;
        // Perf mode times the dispatch by WALL CLOCK around an explicit glFinish. A
        // GL_TIME_ELAPSED query was tried first and reported ~200ns per dispatch -- i.e.
        // essentially nothing for millions of primitives, which is not believable, so it is
        // not trusted. Finish forces the work to actually complete before the clock stops.
        // This serializes the device thread, which is fine for a measurement mode and is
        // why it is not the default.
        const bool timing = perfTiming;
        const auto t0 = timing ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
        // ★★★ cont.250 PS2X_GS_LINTEX -- FAILED SPIKE, DEFAULT OFF, KNOWN INCORRECT. Do NOT enable it
        // expecting either correctness or speed; it is kept only as scaffolding for a future cached
        // implementation. Two defects, both measured (progress.md cont.250 §15):
        //   1. WRONG OUTPUT: vramHash goes aa9cd72b600021ce -> 9d41ddef65d0b2a3 (deterministic, same on
        //      the tile and hardware paths), so the decode below does not reproduce samplePoint exactly.
        //      Not yet bisected; the CLUT index path and mip-level selection are the prime suspects.
        //   2. SLOWER, and structurally so: this rebuilds the texel buffer EVERY BATCH, and the bench
        //      capture replays 630 batches, so each texture is re-decoded 630x per rep. Tile
        //      238.76 -> 242.37 ms, hardware 210.70 -> 253.02 ms.
        // The lesson: decode-once-per-batch cannot beat per-texel reads without a PERSISTENT cache
        // keyed on content + VRAM generation -- and building that is most of the real texture cache,
        // so this shortcut does not avoid the work it was meant to avoid.
        //
        // (original intent) decode each state's texture ONCE, host-side, into
        // texel-final form so the shader does a single indexed read instead of a swizzle-table lookup
        // plus a DEPENDENT CLUT read. The census says T4 82.6% + T8 11.1% + CT24 6.0% = 99.6% of texel
        // fetches, every CLUT is CT32/CSM1, and wrap is only REPEAT/CLAMP -- so this narrow set covers
        // essentially everything. BIT-EXACT by construction: it reuses the same GSMem readers the CPU
        // rasterizer uses, and wrapCoord/lerpChan are untouched, so vramHash still verifies it.
        static const bool linTexOn = []
        { const char *e = std::getenv("PS2X_GS_LINTEX"); return e && e[0] && e[0] != '0'; }();
        static const size_t kLinTexBudget = 4u * 1024u * 1024u; // texels (16 MB), plenty for a batch
        static std::vector<uint32_t> linTexels;
        linTexels.clear();
        if (linTexOn && b.vram != nullptr && b.vramSize >= kVramBytes && !b.states.empty())
        {
            uint8_t *const vr = const_cast<uint8_t *>(b.vram);
            // states are already dedup'd but several can share one texture; key the decode so a
            // repeated texture is decoded once per batch.
            struct Key { uint32_t tbp, tbw, tpsm, w, h, cbp, csa, ta0, aem; };
            std::vector<std::pair<Key, uint32_t>> seen;
            for (GsGpuState &q : b.states)
            {
                if (q.tme == 0u || q.wms >= 2u || q.wmt >= 2u) continue;      // region modes keep the old path
                const bool pal = (q.tpsm == 19u || q.tpsm == 20u);
                if (!(q.tpsm == 0u || q.tpsm == 1u || pal)) continue;          // CT32/CT24/T8/T4 only
                if (pal && (q.cpsm != 0u || q.csm != 0u)) continue;            // CT32 CSM1 palettes only
                if (q.texW == 0u || q.texH == 0u || q.texW > 1024u || q.texH > 1024u) continue;
                const size_t texels = size_t(q.texW) * size_t(q.texH);
                const Key k{q.tbp, q.tbw, q.tpsm, q.texW, q.texH, q.cbp, q.csa, q.ta0, q.aem};
                bool hit = false;
                for (const auto &pr : seen)
                    if (std::memcmp(&pr.first, &k, sizeof k) == 0) { q.linTexOff = pr.second; hit = true; break; }
                if (hit) continue;
                if (linTexels.size() + texels > kLinTexBudget) continue;       // over budget: old path
                const uint32_t off = static_cast<uint32_t>(linTexels.size());
                linTexels.resize(off + texels);
                for (uint32_t y = 0; y < q.texH; ++y)
                {
                    for (uint32_t x = 0; x < q.texW; ++x)
                    {
                        uint32_t texel;
                        if (q.tpsm == 0u)
                            texel = GSMem::ReadCT32(vr, q.tbp, q.tbw, x, y);   // CT32 passes through applyTexa
                        else if (q.tpsm == 1u)
                        {
                            const uint32_t raw = GSMem::ReadCT24(vr, q.tbp, q.tbw, x, y);
                            const uint32_t a = (q.aem != 0u && (raw & 0x00FFFFFFu) == 0u) ? 0u : q.ta0;
                            texel = (raw & 0x00FFFFFFu) | (a << 24u);
                        }
                        else
                        {
                            const uint32_t idx = (q.tpsm == 20u) ? GSMem::ReadP4(vr, q.tbp, q.tbw, x, y)
                                                                 : GSMem::ReadP8(vr, q.tbp, q.tbw, x, y);
                            // resolveClutIndex, CSM1 / CT32 palette: bits 3 and 4 swap.
                            uint32_t ci = ((q.csa & 0x0Fu) << 4u) + ((q.tpsm == 20u) ? (idx & 0x0Fu) : idx);
                            ci &= 0xFFu;
                            ci = (ci & ~0x18u) | ((ci & 0x08u) << 1u) | ((ci & 0x10u) >> 1u);
                            const uint32_t cx = ci & 0x0Fu, cy = ci >> 4u;
                            uint32_t entry = GSMem::ReadCT32(vr, q.cbp, 1u, cx, cy);
                            if (q.clutShadowHave != 0u && (entry & 0x00FFFFFFu) == 0u && cx < 8u && cy < 2u)
                            {
                                const uint32_t up = q.clutShadow[cy * 8u + cx];
                                if ((up & 0xFF000000u) == (entry & 0xFF000000u) && (up & 0x00FFFFFFu) != 0u)
                                    entry = up;
                            }
                            texel = entry; // applyTexa(CT32) is a pass-through
                        }
                        linTexels[off + size_t(y) * q.texW + x] = texel;
                    }
                }
                q.linTexOff = off;
                seen.push_back({k, off});
            }
        }

        auto upload = [&](GLuint buf, const void *data, size_t bytes)
        {
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
            gl.BufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), data,
                          GL_DYNAMIC_DRAW); // orphan + refill per batch
        };
        if (!linTexels.empty())
            upload(linTexBuf, linTexels.data(), linTexels.size() * sizeof(uint32_t));
        upload(stateBuf, b.states.data(), b.states.size() * sizeof(GsGpuState));
        upload(geomBuf, b.prims.data(), b.prims.size() * sizeof(GsGpuGeom));
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mirrorBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tabBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, stateBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, geomBuf);
        if (!linTexels.empty())
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, linTexBuf);

        if (hwRaster && ensureHwRaster())
        {
            // Hardware raster (cont.233): one draw per batch, three vertices per primitive
            // (the vertex stage reads the geometry SSBO), fragments only where the primitive
            // touches, ordered per pixel by the interlock. Serves the single-primitive
            // (verify) path as a batch of one, so the live per-prim oracle covers it too.
            gl.BindFramebuffer(GL_FRAMEBUFFER, hwFbo);
            gl.Viewport(0, 0, static_cast<GLsizei>(kHwW), static_cast<GLsizei>(kHwH));
            gl.BindVertexArray(hwVao);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, hwCountBuf);
            gl.UseProgram(hwProgram);
            gl.Uniform1f(locHwDilate, hwDilate);
            gl.Uniform1f(locHwInvHalfW, 2.0f / static_cast<float>(kHwW));
            gl.Uniform1f(locHwInvHalfH, 2.0f / static_cast<float>(kHwH));
            gl.DrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(3u * b.prims.size()));
            ++hwDraws;
            hwPrims += b.prims.size();
            if (hwErrLogs < 4u)
            {
                const GLenum err = gl.GetError();
                if (err != 0u)
                {
                    ++hwErrLogs;
                    std::fprintf(stderr, "[gsgpu] hwraster: GL error 0x%x after draw %lu (%zu prims)\n",
                                 static_cast<unsigned>(err), hwDraws, b.prims.size());
                }
            }
        }
        else
        {
            // The host skips binning whenever the hardware path is requested; if that path
            // is unavailable the tile kernel still needs bins, so build them here.
            if (b.tileList.empty() && b.prims.size() > 1u)
                binBatch(b);
            if (b.tileList.empty())
            {
                // Single-primitive path: dispatch straight over the bbox, no binning.
                const GsGpuGeom &g = b.prims[0];
                if (g.maxX < g.minX || g.maxY < g.minY)
                    return;
                gl.UseProgram(rasterPrimProgram);
                const uint32_t w = g.maxX - g.minX + 1u;
                const uint32_t h = g.maxY - g.minY + 1u;
                gl.DispatchCompute((w + 7u) / 8u, (h + 7u) / 8u, 1u);
            }
            else
            {
                upload(binOffBuf, b.binOffsets.data(), b.binOffsets.size() * 4u);
                upload(binEntBuf, b.binEntries.data(), b.binEntries.size() * 4u);
                upload(tileListBuf, b.tileList.data(), b.tileList.size() * 4u);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, binOffBuf);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, binEntBuf);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, tileListBuf);
                gl.UseProgram(rasterTileProgram);
                gl.Uniform1ui(locTilesX, b.tilesX);
                gl.DispatchCompute(static_cast<GLuint>(b.tileList.size()), 1u, 1u);
            }
        }
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        if (timing)
        {
            gl.Finish(); // the dispatch must be DONE, not merely submitted
            gpuNs += static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count());
            ++gpuDispatches;
            gpuPrims += b.prims.size();
        }
        lastGeom = b.prims.back();
        lastState = b.states[lastGeom.stateIndex < b.states.size() ? lastGeom.stateIndex : 0];
        ++rasterBatches;
        rasterBatchPrims += b.prims.size();
        rasterPrims += static_cast<unsigned long>(b.prims.size());
    }

    // Phase 2a shadow verify: mirror[offset..] vs the CPU's post-draw bytes for the same
    // range. Reuses the full-compare kernel through its uBase offset.
    void verifyRangeOnThread(uint32_t offset, const std::vector<uint8_t> &expected, uint64_t tag)
    {
        if (expected.empty() || (offset & 3u) != 0u || !ensureFullCompare())
            return;
        const uint32_t len = std::min<uint32_t>(static_cast<uint32_t>(expected.size()),
                                                kVramBytes - std::min(offset, kVramBytes));
        const uint32_t words = len / 4u;
        if (words == 0u)
            return;
        const uint32_t init[4] = {0u, 0xFFFFFFFFu, 0u, 0u};
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, refBuf);
        gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, words * 4u, expected.data());
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuf);
        gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, init);

        gl.UseProgram(fullProgram);
        gl.Uniform1ui(locFullWords, words);
        gl.Uniform1ui(locFullBase, offset / 4u);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mirrorBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, refBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, resultBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, histBuf);
        gl.DispatchCompute((words + 255u) / 256u, 1u, 1u);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        uint32_t res[4] = {0u, 0u, 0u, 0u};
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuf);
        gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, res);
        ++rasterVerifies;
        if (res[0] != 0u)
        {
            ++rasterMismRanges;
            rasterMismWords += res[0];
            if (rasterMismRanges <= 16u || (rasterMismRanges % 4096u) == 0u)
            {
                const GsGpuState &q = lastState;
                const GsGpuGeom &gg = lastGeom;
                std::fprintf(stderr,
                             "[gsgpu:raster] MISMATCH #%lu tag=%016llx off=0x%06x words=%u: "
                             "%u differ, first i=%u gpu=%08x cpu=%08x\n"
                             "[gsgpu:raster]   prim type=%u rect=(%u,%u)-(%u,%u) "
                             "frame{fbp=%u fbw=%u psm=0x%x msk=%08x} z{zbp=%u psm=0x%x msk=%u} "
                             "tex{tbp=%u tbw=%u psm=0x%x tw=%u th=%u lin=%u tfx=%u tcc=%u "
                             "cbp=%u cpsm=0x%x csa=%u} texa{ta0=%u ta1=%u aem=%u} "
                             "test{ate=%u atst=%u aref=%u afail=%u ztst=%u} "
                             "blend{%u%u%u%u fix=%u abe=%u} sprite{unclip=(%u,%u) wh=%.1fx%.1f "
                             "uv=(%.2f,%.2f)-(%.2f,%.2f) z=%u}\n",
                             rasterMismRanges, static_cast<unsigned long long>(tag),
                             offset, words, res[0], res[1], res[2], res[3],
                             q.primType, gg.minX, gg.minY, gg.maxX, gg.maxY,
                             q.fbp, q.fbw, q.fpsm, q.fbmsk, q.zbp, q.zpsm, q.zmsk,
                             q.tbp, q.tbw, q.tpsm, q.texW, q.texH, q.linear, q.tfx, q.tcc,
                             q.cbp, q.cpsm, q.csa, q.ta0, q.ta1, q.aem,
                             q.ate, q.atst, q.aref, q.afail, q.ztst,
                             q.blendA, q.blendB, q.blendC, q.blendD, q.blendFix, q.abe,
                             gg.unclipX0, gg.unclipY0, gg.spriteW, gg.spriteH,
                             gg.su0, gg.sv0, gg.su1, gg.sv1, gg.spriteZ);
            }
        }
        const GLenum err = gl.GetError();
        if (err != 0u && errorLogs < 8u)
        {
            ++errorLogs;
            std::fprintf(stderr, "[gsgpu] GL error 0x%x during raster verify\n", err);
        }
    }

    bool ensureFullCompare()
    {
        if (fullProgram != 0u)
            return true;
        fullProgram = buildProgram(kFullCompareShader, "fullcompare");
        if (!fullProgram)
            return false;
        locFullWords = gl.GetUniformLocation(fullProgram, "uWords");
        locFullBase = gl.GetUniformLocation(fullProgram, "uBase");
        gl.GenBuffers(1, &refBuf);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, refBuf);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, kVramBytes, nullptr, GL_DYNAMIC_DRAW);
        gl.GenBuffers(1, &histBuf);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, histBuf);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER, kVramPages * 4u, nullptr, GL_DYNAMIC_DRAW);
        return gl.GetError() == 0u;
    }

    // Phase-1b: whole-mirror vs whole-CPU-VRAM compare. It runs after every previously
    // posted mirror job (strict FIFO), so a divergence here is a REAL gap -- either a
    // VRAM writer with no mirror path yet, or a kernel that wrote the wrong thing.
    bool verifyFullOnThread(const uint8_t *vram, uint32_t vramSize,
                            const uint8_t *drawPageMask, bool quiet,
                            uint32_t *outFirstWord = nullptr, uint32_t *outStaleWords = nullptr)
    {
        if (!vram || vramSize < kVramBytes || !ensureFullCompare())
            return false;
        const uint32_t words = kVramBytes / 4u;
        const uint32_t init[4] = {0u, 0xFFFFFFFFu, 0u, 0u};
        const std::vector<uint32_t> zeroHist(kVramPages, 0u);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, refBuf);
        gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kVramBytes, vram);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuf);
        gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, init);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, histBuf);
        gl.BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kVramPages * 4u, zeroHist.data());

        gl.UseProgram(fullProgram);
        gl.Uniform1ui(locFullWords, words);
        gl.Uniform1ui(locFullBase, 0u);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mirrorBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, refBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, resultBuf);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, histBuf);
        gl.DispatchCompute((words + 255u) / 256u, 1u, 1u);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        uint32_t res[4] = {0u, 0u, 0u, 0u};
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuf);
        gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, res);
        std::vector<uint32_t> hist(kVramPages, 0u);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, histBuf);
        gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kVramPages * 4u, hist.data());

        ++mvRuns;
        if (outFirstWord)
            *outFirstWord = res[1];
        if (outStaleWords)
            *outStaleWords = res[0];
        if (quiet)
        {
            const GLenum qerr = gl.GetError();
            if (qerr != 0u && errorLogs < 8u)
            {
                ++errorLogs;
                std::fprintf(stderr, "[gsgpu] GL error 0x%x during quiet mverify\n", qerr);
            }
            return res[0] == 0u;
        }
        // Split the divergence: pages the CPU rasterizer wrote since the last verify are
        // EXPECTED to be stale until phase 2 mirrors draws. Everything else is a real gap.
        unsigned pagesBad = 0, pagesDraw = 0, pagesGap = 0;
        unsigned long long wordsDraw = 0, wordsGap = 0;
        for (uint32_t i = 0; i < kVramPages; ++i)
        {
            if (hist[i] == 0u)
                continue;
            ++pagesBad;
            if (drawPageMask && drawPageMask[i] != 0u)
            {
                ++pagesDraw;
                wordsDraw += hist[i];
                hist[i] = 0u; // drop from the range report below: explained, not a gap
            }
            else
            {
                ++pagesGap;
                wordsGap += hist[i];
            }
        }
        std::fprintf(stderr,
                     "[gsgpu:mverify] #%lu words=%u stale-words=%u stale-pages=%u/%u "
                     "| drawn{pages=%u words=%llu} UNEXPLAINED{pages=%u words=%llu} "
                     "(patches=%lu patch-bytes=%llu)\n",
                     mvRuns, words, res[0], pagesBad, kVramPages,
                     pagesDraw, wordsDraw, pagesGap, wordsGap, prPatches, prBytes);
        if (wordsGap != 0u)
        {
            // Name the stale pages as RANGES (a page maps straight back to a guest
            // fbp/tbp: page = block/32), so the report identifies the unmirrored writer.
            // Ranges rather than a top-N list because a top-N is blind to exactly what
            // matters: a handful of stale TEXTURE pages hiding behind saturated
            // framebuffer pages would mean a real gap, not just un-mirrored draws.
            // "+" marks a fully stale range (every word of every page differs).
            char line[1024];
            int n = std::snprintf(line, sizeof(line),
                                  "[gsgpu:mverify]   first word=%u (byte 0x%06x, page %u) "
                                  "mirror=%08x cpu=%08x | UNEXPLAINED pages:",
                                  res[1], res[1] * 4u, res[1] >> 11u, res[2], res[3]);
            constexpr uint32_t kWordsPerPage = 8192u / 4u;
            unsigned printed = 0;
            for (uint32_t i = 0; i < kVramPages && n > 0 && n < static_cast<int>(sizeof(line)) - 24;)
            {
                if (hist[i] == 0u)
                {
                    ++i;
                    continue;
                }
                const uint32_t from = i;
                bool full = true;
                while (i < kVramPages && hist[i] != 0u)
                {
                    full = full && hist[i] == kWordsPerPage;
                    ++i;
                }
                if (++printed > 16u)
                {
                    n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n), " ...");
                    break;
                }
                if (from == i - 1u)
                    n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n),
                                       " %u%s", from, full ? "+" : "");
                else
                    n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n),
                                       " %u-%u%s", from, i - 1u, full ? "+" : "");
            }
            std::fprintf(stderr, "%s\n", line);
        }
        const GLenum err = gl.GetError();
        if (err != 0u && errorLogs < 8u)
        {
            ++errorLogs;
            std::fprintf(stderr, "[gsgpu] GL error 0x%x during mverify\n", err);
        }
        return res[0] == 0u;
    }
};

GsGpuPresentDevice::GsGpuPresentDevice() : m_impl(std::make_unique<Impl>()) {}

// Stop and join the device thread; skip EGL teardown (the device is a process-lifetime
// static in the present path -- the context dies with the process, and tearing it down
// during exit races the window-system shutdown for nothing).
GsGpuPresentDevice::~GsGpuPresentDevice()
{
    Impl &d = *m_impl;
    {
        std::lock_guard<std::mutex> lk(d.mx);
        d.stop = true;
    }
    d.cv.notify_all();
    d.cvDone.notify_all(); // release producers parked on queue space / decode waiters
    if (d.worker.joinable())
        d.worker.join();
}

bool GsGpuPresentDevice::EnsureInitialized()
{
    Impl &d = *m_impl;
    std::unique_lock<std::mutex> lk(d.mx);
    if (d.state == Impl::State::Untried && !d.worker.joinable())
        d.worker = std::thread([&d] { d.threadLoop(); });
    d.cvDone.wait(lk, [&] { return d.state != Impl::State::Untried; });
    return d.state == Impl::State::Ready;
}

bool GsGpuPresentDevice::DecodeFrameRect(const uint8_t *vram,
                                         uint32_t vramSize,
                                         uint64_t sourceId,
                                         const GSFrameReg &frame,
                                         uint32_t width,
                                         uint32_t height,
                                         bool frameBaseIsPages,
                                         uint32_t originX,
                                         uint32_t originY,
                                         bool preserveAlpha,
                                         std::vector<uint8_t> &outPixels)
{
    if (!EnsureInitialized())
        return false;
    Impl &d = *m_impl;
    bool done = false, result = false;
    Impl::Job j;
    j.kind = Impl::Job::Kind::Decode;
    j.decode = {vram, vramSize, sourceId, frame, width, height,
                frameBaseIsPages, originX, originY, preserveAlpha, &outPixels};
    j.done = &done;
    j.result = &result;
    d.enqueue(std::move(j));
    std::unique_lock<std::mutex> lk(d.mx);
    d.cvDone.wait(lk, [&] { return done || d.stop; });
    return result;
}

// ★ cont.329 phase 1 (the GL renderer arc). Render one frame on the device thread and return it in
// the presenter's own buffer shape: 640x512 RGBA8, 640-pixel row stride, image top-left, rows
// top-down -- exactly what CopyFrameToHostRgbaCpu produces, so the whole present path downstream is
// untouched. Synchronous: the presenter needs the pixels now.
void GsGpuPresentDevice::SnapshotFrameGl(uint64_t targetKey, uint64_t seq)
{
    if (!EnsureInitialized())
        return;
    Impl::Job j;
    j.kind = Impl::Job::Kind::GlSnap;
    j.tag = targetKey;
    j.seq = seq;
    m_impl->enqueue(std::move(j));
}

bool GsGpuPresentDevice::RenderFrameGl(uint64_t targetKey, uint32_t displayWidth,
                                       uint32_t displayHeight, std::vector<uint8_t> &outPixels,
                                       uint64_t snapshotSeq)
{
    if (!EnsureInitialized())
        return false;
    Impl &d = *m_impl;
    bool done = false, result = false;
    Impl::Job j;
    j.kind = Impl::Job::Kind::GlFrame;
    j.tag = targetKey; // reuses the job's spare tag field rather than growing Job
    j.seq = snapshotSeq;
    j.decode.width = displayWidth;
    j.decode.height = displayHeight;
    j.decode.out = &outPixels;
    j.done = &done;
    j.result = &result;
    d.enqueue(std::move(j));
    std::unique_lock<std::mutex> lk(d.mx);
    d.cvDone.wait(lk, [&] { return done || d.stop; });
    return result;
}

// ★★★★ cont.332c: hand the presenter the scene-resolution frame, if PS2X_GS_PRESENT_HIRES armed
// one since the last call. Returns false when the feature is off or no new frame has landed -- the
// caller then keeps showing what it already has (a present-cache hit produces no new frame either).
bool GsGpuPresentDevice::TakeHiresPresentFrame(std::vector<uint8_t> &out, uint32_t &w, uint32_t &h)
{
    if (!m_impl)
        return false;
    Impl &d = *m_impl;
    std::lock_guard<std::mutex> lk(d.hiresMx);
    if (!d.hiresFresh || d.hiresBuf.empty())
        return false;
    out = d.hiresBuf;
    w = d.hiresW;
    h = d.hiresH;
    d.hiresFresh = false;
    return true;
}

void GsGpuPresentDevice::DrawBatchGl(GsGlBatch &&batch)
{
    // ★★★★ cont.331 AUTHORITY MODEL phase A: a SEED-ONLY batch (no vertices, only targetSeeds) is
    // legal now. Dirty VRAM rects must reach a target when the target is USED -- including at the
    // present resolve -- not only when the guest happens to draw into it. During an FMV the guest
    // draws almost nothing into the display buffer while macroblock uploads rewrite it every
    // frame, so a draw-triggered seed never fires and the movie never reaches the GL target.
    if ((batch.verts.empty() && batch.targetSeeds.empty()) || !EnsureInitialized())
        return;
    Impl::Job j;
    j.kind = Impl::Job::Kind::GlDraw;
    j.glBatch = std::move(batch);
    m_impl->enqueue(std::move(j));
}

void GsGpuPresentDevice::MirrorUpload(GsGpuUploadChunk &&chunk)
{
    if (!EnsureInitialized())
        return;
    Impl::Job j;
    j.kind = Impl::Job::Kind::Upload;
    j.chunk = std::move(chunk);
    m_impl->enqueue(std::move(j));
}

void GsGpuPresentDevice::MirrorVerify(GsGpuUploadChunk &&chunk)
{
    if (!EnsureInitialized())
        return;
    Impl::Job j;
    j.kind = Impl::Job::Kind::Verify;
    j.chunk = std::move(chunk);
    m_impl->enqueue(std::move(j));
}

void GsGpuPresentDevice::MirrorPatchRaw(uint32_t byteOffset, std::vector<uint8_t> &&bytes)
{
    if (bytes.empty() || !EnsureInitialized())
        return;
    Impl::Job j;
    j.kind = Impl::Job::Kind::PatchRaw;
    j.rawOffset = byteOffset;
    j.raw = std::move(bytes);
    m_impl->enqueue(std::move(j));
}

void GsGpuPresentDevice::MirrorReadback(uint32_t byteOffset, uint32_t byteLen,
                                        std::vector<uint8_t> &out)
{
    if (!EnsureInitialized())
        return;
    Impl &d = *m_impl;
    bool done = false;
    Impl::Job j;
    j.kind = Impl::Job::Kind::Readback;
    j.rawOffset = byteOffset;
    j.decode.vramSize = byteLen;
    j.readOut = &out;
    j.done = &done;
    d.enqueue(std::move(j));
    std::unique_lock<std::mutex> lk(d.mx);
    d.cvDone.wait(lk, [&] { return done || d.stop; });
}

void GsGpuPresentDevice::RasterBatch(GsGpuBatch &&batch)
{
    if (batch.prims.empty() || !EnsureInitialized())
        return;
    Impl::Job j;
    j.kind = Impl::Job::Kind::RasterBatch;
    j.batch = std::move(batch);
    m_impl->enqueue(std::move(j));
}

void GsGpuPresentDevice::MirrorVerifyRange(uint32_t byteOffset, std::vector<uint8_t> &&expected,
                                           uint64_t tag)
{
    if (expected.empty() || !EnsureInitialized())
        return;
    Impl::Job j;
    j.kind = Impl::Job::Kind::VerifyRange;
    j.rawOffset = byteOffset;
    j.raw = std::move(expected);
    j.tag = tag;
    m_impl->enqueue(std::move(j));
}

void GsGpuPresentDevice::EnablePerfTiming()
{
    m_impl->perfTiming = true; // set before the device thread takes any raster job
}

void GsGpuPresentDevice::PerfStats(unsigned long long &gpuNs, unsigned long long &prims,
                                   unsigned long &dispatches) const
{
    gpuNs = m_impl->gpuNs;
    prims = m_impl->gpuPrims;
    dispatches = m_impl->gpuDispatches;
}

bool GsGpuPresentDevice::HardwareRasterRequested() const
{
    // hwFailed is written on the device thread; a stale read here only means one more batch
    // arrives unbinned, and the device bins it itself.
    return m_impl->hwRaster && !m_impl->hwFailed;
}

void GsGpuPresentDevice::RasterStats(unsigned long &prims, unsigned long &verifies,
                                     unsigned long &mismatchRanges,
                                     unsigned long long &mismatchWords) const
{
    // Read without the lock: these are device-thread counters sampled for a log line, and
    // a torn read only misreports a diagnostic tally.
    prims = m_impl->rasterPrims;
    verifies = m_impl->rasterVerifies;
    mismatchRanges = m_impl->rasterMismRanges;
    mismatchWords = m_impl->rasterMismWords;
}

bool GsGpuPresentDevice::MirrorVerifyFull(const uint8_t *vram, uint32_t vramSize,
                                          const uint8_t *drawPageMask, bool quiet,
                                          uint32_t *outFirstWord, uint32_t *outStaleWords)
{
    if (!vram || !EnsureInitialized())
        return false;
    Impl &d = *m_impl;
    bool done = false, result = false;
    Impl::Job j;
    j.kind = Impl::Job::Kind::VerifyFull;
    j.decode.vram = vram;
    j.decode.vramSize = vramSize;
    j.drawPageMask = drawPageMask;
    j.quiet = quiet;
    j.outFirstWord = outFirstWord;
    j.outStaleWords = outStaleWords;
    j.done = &done;
    j.result = &result;
    d.enqueue(std::move(j));
    std::unique_lock<std::mutex> lk(d.mx);
    d.cvDone.wait(lk, [&] { return done || d.stop; });
    return result;
}
