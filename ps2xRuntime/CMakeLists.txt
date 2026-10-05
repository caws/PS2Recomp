cmake_minimum_required(VERSION 3.20)

project(PS2Runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PS2X_ENABLE_RUNNER_UNITY_BUILD "Build ps2EntryRunner with CMake unity build" ON)
set(PS2X_RUNNER_UNITY_BUILD_BATCH_SIZE 32 CACHE STRING "Unity build batch size for ps2EntryRunner")
option(PS2X_ENABLE_RUNNER_PCH "Precompile the heavy runtime headers for ps2EntryRunner" ON)
option(PS2X_ENABLE_SCCACHE "Use sccache as compiler launcher when available" ON)

option(PS2X_ENABLE_RUNTIME_LOGS "Enable PS2 runtime logs" OFF)
option(PS2X_ENABLE_AGRESSIVE_LOGS "Enable very verbose/agressive PS2 runtime logs" OFF)
option(PS2X_ENABLE_IOP_RPC_TRACE "Log unhandled IOP/SIF RPC trace suggestions" ON)
option(PS2X_STRICT_RETURN_DIAGNOSTICS "Route generated JR $ra returns through runtime branch diagnostics" OFF)
option(PS2X_SHOW_WINDOWS_CONSOLE "Show a console window for ps2EntryRunner on Windows release builds" ON)
option(PS2X_ENABLE_DEBUG_UI "Build the desktop runtime debug UI" ON)

if(PS2X_ENABLE_SCCACHE)
    find_program(PS2X_SCCACHE_PROGRAM sccache)

    if(PS2X_SCCACHE_PROGRAM)
        if(CMAKE_C_COMPILER)
            set(CMAKE_C_COMPILER_LAUNCHER "${PS2X_SCCACHE_PROGRAM}")
        endif()

        set(CMAKE_CXX_COMPILER_LAUNCHER "${PS2X_SCCACHE_PROGRAM}")
        message(STATUS "Using sccache: ${PS2X_SCCACHE_PROGRAM}")
    else()
        message(STATUS "sccache not found; continuing without compiler launcher")
    endif()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReleaseMode.cmake")

set(PS2X_IS_VITA OFF)

if((CMAKE_C_COMPILER MATCHES "arm-vita-eabi") OR
    (CMAKE_CXX_COMPILER MATCHES "arm-vita-eabi"))
    set(PS2X_IS_VITA ON)
endif()

set(PS2X_IS_ANDROID OFF)

if(ANDROID)
    set(PS2X_IS_ANDROID ON)
endif()

option(PS2X_VITA_CREATE_PACKAGE "Create Vita eboot/vpk targets" ON)
set(PS2X_VITA_APP_NAME "PS2 Retro X" CACHE STRING "Display name for the Vita bubble")
set(PS2X_VITA_TITLEID "RANJ00001" CACHE STRING "9-character Vita title id")
set(PS2X_VITA_VERSION "01.00" CACHE STRING "Vita app version")
set(PS2X_DEFAULT_BOOT_ELF "" CACHE STRING "Guest ELF path passed directly to main() when argv is unavailable")

if(PS2X_IS_VITA)
    if(NOT DEFINED ENV{VITASDK})
        message(FATAL_ERROR "VITASDK is not defined. Configure inside the VitaSDK environment or pass the Vita toolchain file.")
    endif()

    set(PS2X_VITASDK "$ENV{VITASDK}")

    if(NOT EXISTS "${PS2X_VITASDK}/share/vita.cmake")
        message(FATAL_ERROR "Could not find vita.cmake under ${PS2X_VITASDK}/share.")
    endif()

    include("${PS2X_VITASDK}/share/vita.cmake" REQUIRED)
else()
    include(FetchContent)
    set(FETCHCONTENT_QUIET FALSE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)

    if(PS2X_IS_ANDROID)
        set(PLATFORM "Android" CACHE STRING "" FORCE)
    endif()

    # rumble: raylib's desktop GLFW backend has no vibration output (SetGamepadVibration logs
    # "not available on target platform" and returns; GLFW's joystick API is read-only), while its
    # SDL backend drives SDL_GameControllerRumble -- the Vita and Android builds already run on SDL.
    # SDL2 is fetched and linked statically rather than taken from the system: raylib 5.5's SDL
    # backend needs SDL >= 2.0.16 (SDL_SetWindowAlwaysOnTop, SDL_OpenURL), which distro packages
    # can lag, and a player then needs no SDL package installed. `-DPS2X_RAYLIB_SDL=OFF` = GLFW.
    option(PS2X_RAYLIB_SDL "Build raylib's desktop platform on SDL2 (gamepad rumble)" ON)
    if(PS2X_RAYLIB_SDL AND NOT PS2X_IS_ANDROID)
        set(SDL_SHARED OFF CACHE BOOL "" FORCE)
        set(SDL_STATIC ON CACHE BOOL "" FORCE)
        set(SDL_TEST OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            SDL2
            GIT_REPOSITORY "https://github.com/libsdl-org/SDL.git"
            GIT_TAG "release-2.30.12"
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
        )
        FetchContent_MakeAvailable(SDL2)
        if(NOT TARGET SDL2::SDL2)
            add_library(SDL2::SDL2 ALIAS SDL2-static)
        endif()
        # raylib's LibraryConfigurations.cmake does find_package(SDL2 REQUIRED): satisfy it with the
        # target above instead of a system install.
        file(WRITE "${CMAKE_BINARY_DIR}/ps2x_sdl2_config/SDL2Config.cmake" "set(SDL2_FOUND TRUE)\n")
        set(SDL2_DIR "${CMAKE_BINARY_DIR}/ps2x_sdl2_config" CACHE PATH "" FORCE)
        set(PLATFORM "SDL" CACHE STRING "" FORCE)
        message(STATUS "ps2x: raylib platform = SDL (static SDL2 2.30.12)")
    elseif(NOT PS2X_IS_ANDROID)
        set(PLATFORM "Desktop" CACHE STRING "" FORCE)
    endif()

    FetchContent_Declare(
        raylib
        GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
        GIT_TAG "5.5" # we will migrate to 4.2.0 later, trust me it will be better
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(raylib)

    # ★★★★ cont.303: miniaudio's PulseAudio backend cannot open a playback device on this box.
    # Proven with a 10-line raylib program (no window, none of our code): the PulseAudio CONTEXT
    # connects to /run/user/1000/pulse/native and authenticates, then ma_device_init fails -- and
    # because the context SUCCEEDED, miniaudio never falls back (libasound is loaded ZERO times).
    # ALSA itself is healthy here (aplay works, and it routes through PipeWire's ALSA plugin so
    # mixing is unaffected). Blocking PulseAudio flips the device open:
    #     WARNING: AUDIO: Failed to initialize playback device   ->   IsAudioDeviceReady() = 1
    # Symptom when this bites: a run is COMPLETELY silent while the audio HLE logs perfect work,
    # which reads exactly like an audio regression (it cost most of cont.302-303 to rule out).
    # `-DPS2X_AUDIO_NO_PULSE=OFF` restores the stock backend order.
    option(PS2X_AUDIO_NO_PULSE "Skip miniaudio's PulseAudio backend and use ALSA" ON)
    if(PS2X_AUDIO_NO_PULSE AND UNIX AND NOT APPLE AND NOT PS2X_IS_ANDROID)
        target_compile_definitions(raylib PRIVATE MA_NO_PULSEAUDIO)
        message(STATUS "ps2x: miniaudio PulseAudio backend DISABLED (using ALSA)")
    endif()

    if(PS2X_ENABLE_DEBUG_UI AND NOT PS2X_IS_ANDROID)
        FetchContent_Declare(
            imgui
            GIT_REPOSITORY https://github.com/ocornut/imgui.git
            GIT_TAG "v1.92.7-docking"
            GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(imgui)

        if(NOT imgui_POPULATED)
            FetchContent_Populate(imgui)
        endif()

        set(PS2X_IMGUI_SOURCE_DIR "${imgui_SOURCE_DIR}")

        add_library(imgui STATIC
            "${PS2X_IMGUI_SOURCE_DIR}/imgui.cpp"
            "${PS2X_IMGUI_SOURCE_DIR}/imgui_draw.cpp"
            "${PS2X_IMGUI_SOURCE_DIR}/imgui_tables.cpp"
            "${PS2X_IMGUI_SOURCE_DIR}/imgui_widgets.cpp"
            "${PS2X_IMGUI_SOURCE_DIR}/imgui_demo.cpp"
        )
        target_include_directories(imgui PUBLIC "${PS2X_IMGUI_SOURCE_DIR}")

        FetchContent_Declare(
            rlImGui
            GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
            GIT_TAG "Raylib_5_5"
            GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(rlImGui)

        if(NOT rlimgui_POPULATED)
            FetchContent_Populate(rlImGui)
        endif()

        set(PS2X_RLIMGUI_SOURCE_DIR "${rlimgui_SOURCE_DIR}")

        add_library(rlImGui STATIC
            "${PS2X_RLIMGUI_SOURCE_DIR}/rlImGui.cpp"
        )
        target_include_directories(rlImGui PUBLIC "${PS2X_RLIMGUI_SOURCE_DIR}")
        target_link_libraries(rlImGui PUBLIC raylib imgui)
    endif()
endif()

add_library(ps2_host_backend INTERFACE)

set(PS2X_DEFAULT_BOOT_ELF_DEFINE "")

if(PS2X_DEFAULT_BOOT_ELF)
    set(PS2X_DEFAULT_BOOT_ELF_DEFINE "${PS2X_DEFAULT_BOOT_ELF}")
    string(REPLACE "\\" "\\\\" PS2X_DEFAULT_BOOT_ELF_DEFINE "${PS2X_DEFAULT_BOOT_ELF_DEFINE}")
    string(REPLACE "\"" "\\\"" PS2X_DEFAULT_BOOT_ELF_DEFINE "${PS2X_DEFAULT_BOOT_ELF_DEFINE}")
endif()

if(PS2X_IS_VITA)
    set(PS2X_VITA_PREFIX "${PS2X_VITASDK}/arm-vita-eabi")
    set(PS2X_VITA_EXTRA_INCLUDE_DIRS "" CACHE STRING "Extra include directories for Vita host dependencies")
    set(PS2X_VITA_LIBRARY_DIR "${PS2X_VITA_PREFIX}/lib")
    set(PS2X_VITA_REQUIRED_LIBRARY_NAMES
        raylib
        SDL2
        OpenSLES
        taihen_stub
        SceAppMgr_stub
        SceCtrl_stub
        SceGxm_stub
        SceCommonDialog_stub
        SceLibKernel_stub
        SceAudio_stub
        SceTouch_stub
        SceHid_stub
        SceMotion_stub
        SceSysmodule_stub
        SceIofilemgr_stub
        SceNetCtl_stub
        SceNet_stub
        SceDisplay_stub
        SceAppUtil_stub
        SceAudioIn_stub
        ScePower_stub
        SceProcessmgr_stub
        SceIme_stub
        libIMGEGL_stub_weak
        libgpu_es4_ext_stub_weak
        libGLESv2_stub_weak
        libGL_stub_weak
        libGLESv1_CM_stub_weak
    )

    set(PS2X_VITA_INCLUDE_DIRS
        "${PS2X_VITA_PREFIX}/include"
    )

    foreach(PS2X_VITA_OPTIONAL_INCLUDE_DIR IN ITEMS
        "${PS2X_VITA_PREFIX}/include/raylib"
        "${PS2X_VITA_PREFIX}/include/SDL2")
        if(EXISTS "${PS2X_VITA_OPTIONAL_INCLUDE_DIR}")
            list(APPEND PS2X_VITA_INCLUDE_DIRS "${PS2X_VITA_OPTIONAL_INCLUDE_DIR}")
        endif()
    endforeach()

    if(PS2X_VITA_EXTRA_INCLUDE_DIRS)
        list(APPEND PS2X_VITA_INCLUDE_DIRS ${PS2X_VITA_EXTRA_INCLUDE_DIRS})
    endif()

    if(NOT EXISTS "${PS2X_VITA_PREFIX}/include/SDL2/SDL.h")
        message(FATAL_ERROR
            "Missing SDL2 headers under ${PS2X_VITA_PREFIX}/include/SDL2. "
            "Quenom/raylib-5.5-vita requires SDL2 with PVR support installed into VitaSDK.")
    endif()

    if(NOT EXISTS "${PS2X_VITA_LIBRARY_DIR}")
        message(FATAL_ERROR "Missing Vita library directory: ${PS2X_VITA_LIBRARY_DIR}")
    endif()

    function(ps2x_find_vita_library out_var library_name)
        string(MAKE_C_IDENTIFIER "${library_name}" library_id)
        set(cache_var "PS2X_VITA_LIB_${library_id}")

        find_library(${cache_var}
            NAMES "${library_name}" "${library_name}.a" "lib${library_name}.a"
            PATHS "${PS2X_VITA_LIBRARY_DIR}"
            NO_DEFAULT_PATH
        )

        if(NOT ${cache_var})
            message(FATAL_ERROR
                "Could not find Vita library '${library_name}' under ${PS2X_VITA_LIBRARY_DIR}. "
                "Install Quenom/raylib-5.5-vita and its SDL2/PVR dependencies into VitaSDK first.")
        endif()

        set(${out_var} "${${cache_var}}" PARENT_SCOPE)
    endfunction()

    set(PS2X_VITA_RESOLVED_LIBRARIES "")

    foreach(PS2X_VITA_LIBRARY_NAME IN LISTS PS2X_VITA_REQUIRED_LIBRARY_NAMES)
        ps2x_find_vita_library(PS2X_VITA_LIBRARY_PATH "${PS2X_VITA_LIBRARY_NAME}")
        list(APPEND PS2X_VITA_RESOLVED_LIBRARIES "${PS2X_VITA_LIBRARY_PATH}")
    endforeach()

    target_compile_definitions(ps2_host_backend INTERFACE PLATFORM_VITA)
    target_include_directories(ps2_host_backend INTERFACE ${PS2X_VITA_INCLUDE_DIRS})
    target_link_libraries(ps2_host_backend INTERFACE
        ${PS2X_VITA_RESOLVED_LIBRARIES}
        m
        c
        pthread
    )
else()
    target_link_libraries(ps2_host_backend INTERFACE raylib)

    if(PS2X_IS_ANDROID)
        target_link_libraries(ps2_host_backend INTERFACE log android)
    endif()
endif()

# rotk (2026-10-02, row 242): OFF on every platform. This game's movies decode through its own libmpeg
# on the PCSX2 IPU port (cont.247), so nothing calls the FFmpeg-backed Kernel/Stubs/MPEG.cpp HLE; it
# builds its no-FFmpeg path (as on Android) and the runner no longer links libavcodec/avformat/...
# A game that routes sceMpeg* to that HLE configures with -DPS2X_ENABLE_FFMPEG=ON.
set(PS2X_ENABLE_FFMPEG_DEFAULT OFF)

option(PS2X_ENABLE_FFMPEG "Enable FFmpeg-backed MPEG video decoding" ${PS2X_ENABLE_FFMPEG_DEFAULT})

if(NOT PS2X_ENABLE_FFMPEG)
    message(STATUS "FFmpeg disabled; MPEG video decode falls back to stub frames")
    add_library(ffmpeg INTERFACE)
elseif(WIN32)
    include(ExternalProject)

    set(FFMPEG_PREBUILT_TAG "n7.1-241205")
    set(FFMPEG_PREBUILT_URL
        "https://github.com/System233/ffmpeg-msvc-prebuilt/releases/download/${FFMPEG_PREBUILT_TAG}/ffmpeg-${FFMPEG_PREBUILT_TAG}-lgpl-amd64-shared.zip")

    set(FFMPEG_PREFIX_DIR "${CMAKE_BINARY_DIR}/ThirdParty/ffmpeg-prefix")
    set(FFMPEG_SOURCE_DIR "${FFMPEG_PREFIX_DIR}/src/ffmpeg_external")
    set(FFMPEG_INCLUDE_DIR "${FFMPEG_SOURCE_DIR}/include")
    set(FFMPEG_LIB_DIR "${FFMPEG_SOURCE_DIR}/bin")
    set(FFMPEG_BIN_DIR "${FFMPEG_SOURCE_DIR}/bin")

    ExternalProject_Add(ffmpeg_external
        URL "${FFMPEG_PREBUILT_URL}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX "${FFMPEG_PREFIX_DIR}"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS
        "${FFMPEG_LIB_DIR}/avcodec.lib"
        "${FFMPEG_LIB_DIR}/avformat.lib"
        "${FFMPEG_LIB_DIR}/avutil.lib"
        "${FFMPEG_LIB_DIR}/swresample.lib"
        "${FFMPEG_LIB_DIR}/swscale.lib"
    )

    # Link against the import libraries shipped with the Windows DLLs.
    add_library(ffmpeg_avcodec STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_avcodec PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/avcodec.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_avcodec ffmpeg_external)

    add_library(ffmpeg_avformat STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_avformat PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/avformat.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_avformat ffmpeg_external)

    add_library(ffmpeg_avutil STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_avutil PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/avutil.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_avutil ffmpeg_external)

    add_library(ffmpeg_swresample STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_swresample PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/swresample.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_swresample ffmpeg_external)

    add_library(ffmpeg_swscale STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_swscale PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/swscale.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_swscale ffmpeg_external)

    add_library(ffmpeg INTERFACE)
    target_link_libraries(ffmpeg INTERFACE
        ffmpeg_avcodec
        ffmpeg_avformat
        ffmpeg_avutil
        ffmpeg_swresample
        ffmpeg_swscale
        bcrypt
        secur32
        ws2_32
        user32
        advapi32
        ole32
        shell32
    )

    set(PS2X_FFMPEG_BIN_DIR "${FFMPEG_BIN_DIR}" CACHE INTERNAL "Directory containing FFmpeg runtime DLLs")
    set(PS2X_COPY_FFMPEG_DLLS_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyFfmpegDlls.cmake"
        CACHE INTERNAL
        "Script used to stage FFmpeg runtime DLLs beside executables")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET GLOBAL
        libavcodec
        libavformat
        libavutil
        libswresample
        libswscale
    )

    add_library(ffmpeg INTERFACE)
    target_link_libraries(ffmpeg INTERFACE PkgConfig::FFMPEG)
endif()

function(ps2x_stage_ffmpeg_runtime_dlls target_name)
    if(NOT WIN32)
        return()
    endif()

    if(NOT PS2X_FFMPEG_BIN_DIR OR NOT PS2X_COPY_FFMPEG_DLLS_SCRIPT)
        return()
    endif()

    if(TARGET ffmpeg_external)
        add_dependencies(${target_name} ffmpeg_external)
    endif()

    add_custom_command(
        TARGET ${target_name} POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
        "-Dsource_dir=${PS2X_FFMPEG_BIN_DIR}"
        "-Ddest_dir=$<TARGET_FILE_DIR:${target_name}>"
        -P "${PS2X_COPY_FFMPEG_DLLS_SCRIPT}"
        VERBATIM
    )
endfunction()

# ---- cont.230: PCSX2 microVU port (GPL-3.0+, see src/lib/vu/microvu/README.md) -----------------
# Built as its own target so PCSX2's sources keep their include layout ("common/emitter/..."),
# their warnings and their own flags, and never enter the runner's unity build.
add_library(ps2x_microvu STATIC
    src/lib/vu/microvu/ps2x_microvu.cpp
    src/lib/vu/microvu/shim/common/Assertions.cpp
    src/lib/vu/microvu/pcsx2/common/AlignedMalloc.cpp
    src/lib/vu/microvu/pcsx2/common/emitter/bmi.cpp
    src/lib/vu/microvu/pcsx2/common/emitter/fpu.cpp
    src/lib/vu/microvu/pcsx2/common/emitter/groups.cpp
    src/lib/vu/microvu/pcsx2/common/emitter/jmp.cpp
    src/lib/vu/microvu/pcsx2/common/emitter/legacy.cpp
    src/lib/vu/microvu/pcsx2/common/emitter/legacy_sse.cpp
    src/lib/vu/microvu/pcsx2/common/emitter/movs.cpp
    src/lib/vu/microvu/pcsx2/common/emitter/simd.cpp
    src/lib/vu/microvu/pcsx2/common/emitter/x86emitter.cpp
    src/lib/vu/microvu/pcsx2/x86/microVU.cpp
)
target_include_directories(ps2x_microvu
    PRIVATE
        src/lib/vu/microvu/shim
        src/lib/vu/microvu/pcsx2
        src/lib/vu/microvu/pcsx2/common
        src/lib/vu/microvu/pcsx2/x86
        include
)
target_compile_options(ps2x_microvu PRIVATE -w -fno-strict-aliasing -msse4.1)
target_compile_definitions(ps2x_microvu PRIVATE PS2X_MICROVU_PORT=1)

# ---- cont.245: PCSX2 IPU port (GPL-3.0+, see src/lib/ipu/README.md) ------------------------------
# Own target, like the microVU port: PCSX2's sources keep their include layout ("IPU/IPU.h") and their
# own warnings, and never enter the runner's unity build.
add_library(ps2x_ipu STATIC
    src/lib/ipu/ps2x_ipu.cpp
    src/lib/ipu/pcsx2/IPU/IPU.cpp
    src/lib/ipu/pcsx2/IPU/IPU_Fifo.cpp
    src/lib/ipu/pcsx2/IPU/IPUdma.cpp
    src/lib/ipu/pcsx2/IPU/IPU_MultiISA.cpp
    src/lib/ipu/pcsx2/IPU/IPUdither.cpp
    src/lib/ipu/pcsx2/IPU/yuv2rgb.cpp
)
target_include_directories(ps2x_ipu
    PRIVATE
        src/lib/ipu/shim
        src/lib/ipu/pcsx2
        src/lib/vu/microvu/pcsx2
        include
)
target_compile_options(ps2x_ipu PRIVATE -w -fno-strict-aliasing -msse4.1)
target_compile_definitions(ps2x_ipu PRIVATE PS2X_IPU_PORT=1)

add_library(ps2_runtime STATIC
    src/lib/game_overrides.cpp
    src/lib/gs/ps2_gif_arbiter.cpp
    src/lib/ps2_audio.cpp
    src/lib/ps2_audio_vag.cpp
    src/lib/gs/ps2_gs_memory.cpp
    src/lib/gs/gs_frontend.cpp
    src/lib/gs/gs_cpu_backend.cpp
    src/lib/gs/gs_texpack.cpp
    src/lib/ps2_iop_host.cpp
    src/lib/ps2_dbcman_hle.cpp
    src/lib/ps2_memory.cpp
    src/lib/ps2_pad.cpp
    src/lib/ps2_runtime.cpp
    src/lib/ps2_settings.cpp
    src/lib/ps2_settings_menu.cpp
    src/lib/ps2_launcher.cpp
    src/lib/ps2_mods.cpp
    src/lib/ps2_user_dir.cpp
    src/lib/ps2_vif1_interpreter.cpp
    src/lib/ps2_pipe_capture.cpp
    src/lib/ps2_gs_pipeline.cpp
    src/lib/vu/ps2_vu1_core.cpp
    src/lib/vu/ps2_vu1_upper.cpp
    src/lib/vu/ps2_vu1_lower.cpp
    src/lib/games_database.cpp
)

# ★ cont.326 PS2X_RASTER_AVX2 (default OFF -- MEASURED A LOSS): the CPU rasterizer unit ALONE compiled with VEX
# encoding (-mavx2; NO -mfma, and -ffp-contract=off so no product+sum is ever fused). Three-operand forms remove
# the register copies the two-operand SSE encoding needs before every destructive op, and the build is bit-exact
# (hashes unchanged): Helm's bench at 1 thread retired 6.3% FEWER instructions and 4.6% fewer fused-domain uops
# (idq.dsb_uops + idq.mite_uops), with the decoded-icache coverage unchanged (92%) and the hot loops' byte size
# unchanged -- and took 2.0% MORE cycles (+2.2% wall), on every arm of an alternating A/B on a quiet box; 7 threads
# flat. The register copies it removed were free (move elimination at rename); what it changed in scheduling and
# memory-operand folding is a diffuse back-end loss with no hot instruction (progress.md cont.326). Kept as an
# option for a future 256-bit (8-lane) rasterizer, which needs the encoding; ps2_runtime.cpp refuses to start on a
# host without AVX2 when this is on.
option(PS2X_RASTER_AVX2 "Compile the CPU rasterizer unit (gs_cpu_backend.cpp) with AVX2 VEX encoding (measured slower, cont.326)" OFF)
if(PS2X_RASTER_AVX2 AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64"))
    set_source_files_properties(src/lib/gs/gs_cpu_backend.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
    target_compile_definitions(ps2_runtime PRIVATE PS2X_RASTER_AVX2=1)
    message(STATUS "Rasterizer unit: AVX2 VEX encoding (PS2X_RASTER_AVX2=ON)")
endif()

if(PS2X_ENABLE_RUNTIME_LOGS)
    target_compile_definitions(ps2_runtime PUBLIC
        PS2_RUNTIME_LOGS=1
    )
endif()

if(PS2X_ENABLE_AGRESSIVE_LOGS)
    target_compile_definitions(ps2_runtime PUBLIC
        AGRESSIVE_LOGS=1
        PS2_FUNCTION_LOG_TRACKER=1
    )
endif()

if(PS2X_ENABLE_IOP_RPC_TRACE)
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_ENABLE_IOP_RPC_TRACE=1
    )
else()
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_ENABLE_IOP_RPC_TRACE=0
    )
endif()

if(PS2X_STRICT_RETURN_DIAGNOSTICS)
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_STRICT_RETURN_DIAGNOSTICS=1
    )
endif()

target_compile_definitions(ps2_runtime PRIVATE
    PS2X_HAS_FFMPEG=$<BOOL:${PS2X_ENABLE_FFMPEG}>
)

file(GLOB_RECURSE KERNEL_SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/lib/Kernel/*.cpp"
)

target_sources(ps2_runtime PRIVATE
    ${KERNEL_SRC_FILES}
)

file(GLOB RUNNER_SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runner/*.cpp"
)

# Keep the GAME's hand-written override sources OUT of the unity build.
#
# The runner mixes two very different kinds of source: machine-generated recompiled functions
# (thousands of files, which unity batching exists to speed up) and a handful of hand-written
# override modules copied in from the game repo. Unity-batching the latter is actively harmful:
#
#   1. Batches are formed from the source list, so ADDING or REMOVING one override module
#      re-shuffles every batch after it and forces a near-full rebuild of the runner.
#   2. Batched files are #include'd into one TU, so file-scope statics (and __LINE__-keyed
#      names, as PS2_REGISTER_GAME_OVERRIDE generates) can collide across unrelated modules.
#
# The engine's scripts/03_build_game.sh writes this manifest when it installs the overrides; each
# line is one basename in src/runner/. Absent manifest = no overrides to exclude (plain upstream).
set(GAME_OVERRIDE_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/src/runner/game_overrides.manifest")
if(EXISTS "${GAME_OVERRIDE_MANIFEST}")
    file(STRINGS "${GAME_OVERRIDE_MANIFEST}" GAME_OVERRIDE_BASENAMES)
    foreach(override_base IN LISTS GAME_OVERRIDE_BASENAMES)
        set(override_src "${CMAKE_CURRENT_SOURCE_DIR}/src/runner/${override_base}")
        if(EXISTS "${override_src}")
            set_source_files_properties("${override_src}" PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
        endif()
    endforeach()
    list(LENGTH GAME_OVERRIDE_BASENAMES override_count)
    message(STATUS "Game overrides excluded from unity build: ${override_count}")
endif()

set(RUNNER_MAIN_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/runner/main.cpp")
set(ROOT_MAIN_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

if(EXISTS "${RUNNER_MAIN_CPP}")
    list(APPEND RUNNER_SRC_FILES "${RUNNER_MAIN_CPP}")
elseif(EXISTS "${ROOT_MAIN_CPP}")
    list(APPEND RUNNER_SRC_FILES "${ROOT_MAIN_CPP}")
endif()

if(PS2X_IS_ANDROID)
    add_library(ps2EntryRunner SHARED
        ${RUNNER_SRC_FILES}
    )

    target_link_options(ps2EntryRunner PRIVATE "-Wl,--undefined=ANativeActivity_onCreate")
else()
    add_executable(ps2EntryRunner
        ${RUNNER_SRC_FILES}
    )
endif()

if(PS2X_ENABLE_RUNNER_UNITY_BUILD)
    set_target_properties(ps2EntryRunner PROPERTIES
        UNITY_BUILD ON
        UNITY_BUILD_BATCH_SIZE "${PS2X_RUNNER_UNITY_BUILD_BATCH_SIZE}"
    )
endif()

# ---- cont.260 PS2X_RUNNER_SIZE_OPT (default OFF): compile the runner for SIZE (-Os).
# The runner's .text is 37.9 MB, of which 33.9 MB is generated guest code -- 99.7 bytes of host
# code per guest instruction, a 24.9x expansion over 4-byte MIPS and ~27x the L2 cache. cont.259
# measured 23.9 host cycles per guest instruction against 2.6 for the same emitted macro bodies in
# a cache-warm microbenchmark, so instruction FETCH, not instruction semantics, is the suspected
# cost. This is the test: if the guest half is i-cache-bound, building it SMALLER should make it
# FASTER even though -Os emits worse per-instruction code. A speedup confirms the mechanism (and is
# itself the win); no change is ambiguous, because -Os also degrades the code it shrinks.
option(PS2X_RUNNER_SIZE_OPT "Compile ps2EntryRunner with -Os (i-cache experiment, cont.260)" OFF)
if(PS2X_RUNNER_SIZE_OPT)
    target_compile_options(ps2EntryRunner PRIVATE -Os)
    message(STATUS "ps2EntryRunner: -Os (PS2X_RUNNER_SIZE_OPT=ON, cont.260 i-cache experiment)")
endif()

if(PS2X_ENABLE_RUNNER_PCH)
    set(PS2X_RUNNER_PCH_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_runtime_macros.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_runtime.h"
    )

    foreach(PS2X_RUNNER_GENERATED_HEADER IN ITEMS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_recompiled_functions.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_recompiled_stubs.h")
        if(EXISTS "${PS2X_RUNNER_GENERATED_HEADER}")
            list(APPEND PS2X_RUNNER_PCH_HEADERS "${PS2X_RUNNER_GENERATED_HEADER}")
        endif()
    endforeach()

    list(APPEND PS2X_RUNNER_PCH_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_syscalls.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_stubs.h"
    )

    target_precompile_headers(ps2EntryRunner PRIVATE ${PS2X_RUNNER_PCH_HEADERS})
endif()

if(PS2X_DEFAULT_BOOT_ELF_DEFINE)
    target_compile_definitions(ps2EntryRunner PRIVATE
        PS2X_DEFAULT_BOOT_ELF="${PS2X_DEFAULT_BOOT_ELF_DEFINE}"
    )
endif()

if(MSVC)
    target_compile_options(ps2EntryRunner PRIVATE /FS)
endif()

target_include_directories(ps2_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/Kernel
)

target_link_libraries(ps2_runtime PUBLIC ps2_host_backend ffmpeg ps2_iop ps2x_microvu ps2x_ipu)

# The launcher's font (ps2_launcher.cpp): Lato, SIL Open Font License 1.1 (embedding permitted), as
# Zelda 64: Recompiled ships it. Downloaded once at configure time like raylib/SDL and embedded as a
# byte array, so a player needs no font file. On any failure the launcher falls back to raylib's
# built-in font (PS2X_LAUNCHER_FONT=0).
set(PS2X_FONT_DIR "${CMAKE_BINARY_DIR}/ps2x_fonts")
set(PS2X_FONT_HEADER "${PS2X_FONT_DIR}/ps2x_launcher_font.h")
set(PS2X_LAUNCHER_FONT_OK 1)
foreach(face Lato-Regular Lato-Bold)
    set(ttf "${PS2X_FONT_DIR}/${face}.ttf")
    if(NOT EXISTS "${ttf}")
        file(DOWNLOAD "https://github.com/google/fonts/raw/main/ofl/lato/${face}.ttf" "${ttf}"
             STATUS dlStatus TLS_VERIFY ON)
        list(GET dlStatus 0 dlCode)
        if(NOT dlCode EQUAL 0)
            file(REMOVE "${ttf}")
            set(PS2X_LAUNCHER_FONT_OK 0)
            message(WARNING "ps2x: could not download ${face}.ttf; the launcher uses raylib's built-in font")
        endif()
    endif()
endforeach()
if(PS2X_LAUNCHER_FONT_OK AND NOT EXISTS "${PS2X_FONT_HEADER}")
    set(fontHeader "// Generated by ps2xRuntime/CMakeLists.txt: Lato (SIL OFL 1.1), embedded for the launcher.\n#pragma once\n")
    foreach(face Lato-Regular Lato-Bold)
        file(READ "${PS2X_FONT_DIR}/${face}.ttf" hex HEX)
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex "${hex}")
        string(REPLACE "-" "_" sym "${face}")
        string(APPEND fontHeader "static const unsigned char k${sym}[] = {${hex}};\n")
    endforeach()
    file(WRITE "${PS2X_FONT_HEADER}" "${fontHeader}")
endif()
if(PS2X_LAUNCHER_FONT_OK)
    target_include_directories(ps2_runtime PRIVATE "${PS2X_FONT_DIR}")
    target_compile_definitions(ps2_runtime PRIVATE PS2X_LAUNCHER_FONT=1)
endif()
# GPU presentation decode (gs_gpu_device.cpp): desktop-only, links EGL only -- every GL
# entry point is resolved at runtime through eglGetProcAddress, so no GL link dependency.
# PS2X_HAS_GS_GPU_DEVICE gates the gs_cpu_backend.cpp integration (0 = CPU decode only).
if(NOT PS2X_IS_VITA AND NOT PS2X_IS_ANDROID)
    target_sources(ps2_runtime PRIVATE src/lib/gs/gs_gpu_device.cpp)
    target_link_libraries(ps2_runtime PUBLIC EGL)
    target_compile_definitions(ps2_runtime PRIVATE PS2X_HAS_GS_GPU_DEVICE=1)
else()
    target_compile_definitions(ps2_runtime PRIVATE PS2X_HAS_GS_GPU_DEVICE=0)
endif()
target_link_libraries(ps2EntryRunner
    PRIVATE
    ps2_runtime
)

if(PS2X_ENABLE_DEBUG_UI AND NOT PS2X_IS_VITA AND NOT PS2X_IS_ANDROID)
    target_sources(ps2EntryRunner PRIVATE
        src/lib/ps2_debug_panel.cpp
    )
    target_link_libraries(ps2EntryRunner PRIVATE rlImGui imgui)
    target_compile_definitions(ps2EntryRunner PRIVATE PS2X_ENABLE_DEBUG_UI=1)
endif()

ps2x_stage_ffmpeg_runtime_dlls(ps2EntryRunner)

if(PS2X_IS_VITA)
    set(VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS} -d PARENTAL_LEVEL=1") 
    set(VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS} -d ATTRIBUTE2=12")

    if(PS2X_VITA_CREATE_PACKAGE)
        set(PS2X_VITA_VPK_FILES "")

        foreach(PS2X_VITA_MODULE IN ITEMS
            libgpu_es4_ext.suprx
            libIMGEGL.suprx
            libGLESv2.suprx
            libpvrPSP2_WSEGL.suprx
            libGL.suprx)
            set(PS2X_VITA_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vita/module/${PS2X_VITA_MODULE}")

            if(EXISTS "${PS2X_VITA_MODULE_PATH}")
                list(APPEND PS2X_VITA_VPK_FILES FILE "${PS2X_VITA_MODULE_PATH}" "module/${PS2X_VITA_MODULE}")
            else()
                message(WARNING
                    "Missing ${PS2X_VITA_MODULE_PATH}; the vpk will rely on ur0:data/external/ for it")
            endif()
        endforeach()

        vita_create_self(eboot.bin ps2EntryRunner UNSAFE)
        vita_create_vpk(ps2EntryRunner.vpk ${PS2X_VITA_TITLEID} eboot.bin
            VERSION ${PS2X_VITA_VERSION}
            NAME ${PS2X_VITA_APP_NAME}
            ${PS2X_VITA_VPK_FILES}
        )
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    EnableFastReleaseMode(ps2_runtime)
    EnableFastReleaseMode(ps2EntryRunner)

    if(WIN32 AND NOT PS2X_SHOW_WINDOWS_CONSOLE)
        if(MSVC)
            target_link_options(ps2EntryRunner PRIVATE
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:/SUBSYSTEM:WINDOWS>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:/ENTRY:mainCRTStartup>
            )
        elseif(MINGW)
            target_link_options(ps2EntryRunner PRIVATE
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-mwindows>
            )
        endif()
    endif()
endif()

# Work around WinAPI vs raylib symbol clash for CloseWindow on x64
if(MSVC)
    target_link_options(ps2EntryRunner PRIVATE "/FORCE:MULTIPLE")
endif()

# Vita: Work around for GCC 10's LTO plugin
if(PS2X_IS_VITA)
    target_link_options(ps2EntryRunner PRIVATE "-Wl,--allow-multiple-definition")
    target_sources(ps2EntryRunner PRIVATE src/lib/ps2_vita_runtime.cpp)
    set_source_files_properties(src/lib/ps2_vita_runtime.cpp PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION TRUE
        COMPILE_OPTIONS "-fno-lto"
    )
endif()

if(PS2X_IS_ANDROID)
    target_sources(ps2EntryRunner PRIVATE src/lib/ps2_android_runtime.cpp)
endif()

install(TARGETS ps2_runtime ps2EntryRunner
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(WIN32 AND PS2X_ENABLE_FFMPEG)
    install(DIRECTORY "${FFMPEG_BIN_DIR}/"
        DESTINATION bin
        FILES_MATCHING
        PATTERN "*.dll"
    )
endif()

install(DIRECTORY include/
    DESTINATION include
)
