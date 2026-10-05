#ifndef PS2_DEBUG_PANEL_H
#define PS2_DEBUG_PANEL_H

#include <atomic>

class PS2Runtime;

// cont.232: mirrors PS2DebugPanel::m_visible for the presenter loop (ps2_runtime.cpp), which draws at
// 60 Hz only while the overlay is open (PS2X_GS_PRESENT_LAZY). Written by the panel, read by the loop.
extern std::atomic<bool> g_ps2xDebugUiVisible;

class PS2DebugPanel
{
public:
    void initialize();
    void shutdown();
    void draw(PS2Runtime &runtime);

    bool isVisible() const { return m_visible; }
    void setVisible(bool visible) { m_visible = visible; g_ps2xDebugUiVisible.store(visible); }
    void toggleVisible() { setVisible(!m_visible); }

private:
    bool m_initialized = false;
    bool m_visible = true;
    bool m_showRegisters = true;
    unsigned int m_memoryAddress = 0x00100000u;
    unsigned int m_memoryBytes = 0x100u;
};

#endif // PS2_DEBUG_PANEL_H
