#ifndef PS2_SETTINGS_MENU_H
#define PS2_SETTINGS_MENU_H

// A controller-first menu over the settings schema (ps2_settings.h). Drawn with raylib into any
// rectangle, so the same widget serves the pre-game launcher now and can be drawn over the game
// later. It holds no window and reads no input device itself: the caller turns its devices into a
// MenuInput per frame.

#include "ps2_settings.h"
#include "raylib.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace ps2x::settings
{
    struct MenuInput
    {
        bool up = false, down = false, left = false, right = false;
        bool accept = false;   // on a value: step it forward; on an action row: trigger it
        bool reset = false;    // everything back to the preset / defaults
    };

    class SettingsMenu
    {
    public:
        // `fileValues` = what settings.ini holds now (readValues()); `actions` = extra rows at the end
        // ("Play", "Quit" in the launcher). activatedAction() reports which one was accepted.
        SettingsMenu(const std::map<std::string, std::string> &fileValues, std::vector<std::string> actions);

        void setFont(const Font *font) { m_font = font; }   // nullptr = raylib's built-in font
        void handle(const MenuInput &in);
        void draw(Rectangle area, int fontSize) const;

        // -1 = none this frame; else the index into `actions`.
        int activatedAction() const { return m_activated; }

        // Options another screen owns (the Mods screen's switches) are left out of this list.
        void hide(const std::set<std::string> &keys);

        // Direct access by "section.key", for other screens editing the same values (Mods).
        std::string value(const std::string &key) const;        // the effective value, "" if unknown
        std::string source(const std::string &key) const;       // "set" | "preset" | "default"
        std::string lockedBy(const std::string &key) const;     // "NAME=value" if the env pins it
        bool setValue(const std::string &key, const std::string &raw);
        bool stepValue(const std::string &key, int dir);        // as Left/Right on its row; false if pinned

        // What to write back: the preset (unless original) + every option set explicitly.
        std::map<std::string, std::string> values() const;
        bool dirty() const { return m_dirty; }

    private:
        struct Row
        {
            enum class Kind { Header, Preset, Option, Action } kind;
            std::string label;
            int index = -1;   // Option: into m_schema; Action: into m_actions
        };

        void buildRows();
        const Setting *find(const std::string &key) const;
        std::string effective(const Setting &s, std::string *source) const;
        void step(const Row &row, int dir);
        bool selectable(int row) const;
        void moveSelection(int dir);

        std::vector<Setting> m_schema;
        std::vector<std::string> m_actions;
        std::vector<Row> m_rows;
        std::string m_preset = "original";
        std::map<std::string, std::string> m_explicit;   // "section.key" -> normalised value
        std::map<std::string, std::string> m_unknown;    // file entries we do not know: kept verbatim
        std::set<std::string> m_hidden;
        int m_selected = 0;
        mutable int m_scrollFirst = 0;   // list scroll position (draw keeps the selection in view)
        int m_activated = -1;
        bool m_dirty = false;
        const Font *m_font = nullptr;
    };
}

#endif
