#include "ps2_settings_menu.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace ps2x::settings
{
    namespace
    {
        const Color kText{230, 230, 230, 255};
        const Color kDim{140, 140, 150, 255};
        const Color kAccent{214, 176, 92, 255};    // a warm gold, readable on the dark background
        const Color kSelectBg{60, 52, 36, 255};
        const Color kLocked{150, 110, 110, 255};

        // Text through an optional TTF font (the launcher's Lato); nullptr = raylib's built-in font.
        void drawText(const Font *font, const char *text, int x, int y, int size, Color c)
        {
            if (font) DrawTextEx(*font, text, Vector2{static_cast<float>(x), static_cast<float>(y)}, static_cast<float>(size), 0.0f, c);
            else DrawText(text, x, y, size, c);
        }

        int measureText(const Font *font, const char *text, int size)
        {
            return font ? static_cast<int>(MeasureTextEx(*font, text, static_cast<float>(size), 0.0f).x) : MeasureText(text, size);
        }

        // "render_scale" -> "Render scale"
        std::string prettify(const std::string &key)
        {
            std::string out = key;
            for (char &c : out)
                if (c == '_') c = ' ';
            if (!out.empty()) out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
            return out;
        }

        std::vector<std::string> splitChoices(const char *choices)
        {
            std::vector<std::string> v;
            std::stringstream ss(choices ? choices : "");
            std::string c;
            while (std::getline(ss, c, '|')) v.push_back(c);
            return v;
        }

        // Greedy word wrap for the description line.
        std::vector<std::string> wrap(const Font *font, const std::string &text, int fontSize, float width)
        {
            std::vector<std::string> lines;
            std::stringstream ss(text);
            std::string word, line;
            while (ss >> word)
            {
                const std::string candidate = line.empty() ? word : line + " " + word;
                if (!line.empty() && measureText(font, candidate.c_str(), fontSize) > width)
                {
                    lines.push_back(line);
                    line = word;
                }
                else
                    line = candidate;
            }
            if (!line.empty()) lines.push_back(line);
            return lines;
        }
    }

    SettingsMenu::SettingsMenu(const std::map<std::string, std::string> &fileValues, std::vector<std::string> actions)
        : m_schema(schema()), m_actions(std::move(actions))
    {
        for (const auto &[k, raw] : fileValues)
        {
            if (k == "preset")
            {
                m_preset = raw;
                continue;
            }
            bool known = false;
            for (const Setting &s : m_schema)
                if (key(s) == k)
                {
                    std::string v;
                    if (validate(s, raw, v)) m_explicit[k] = v;
                    known = true;
                }
            if (!known) m_unknown[k] = raw;
        }

        buildRows();
    }

    // Rows = the preset, then each section's options (minus the hidden ones), then the actions.
    void SettingsMenu::buildRows()
    {
        m_rows.clear();
        m_rows.push_back({Row::Kind::Preset, "Preset", -1});
        std::string section;
        for (size_t i = 0; i < m_schema.size(); ++i)
        {
            if (m_hidden.count(key(m_schema[i]))) continue;
            if (section != m_schema[i].section)
            {
                section = m_schema[i].section;
                std::string title = section;
                for (char &c : title) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                m_rows.push_back({Row::Kind::Header, title, -1});
            }
            m_rows.push_back({Row::Kind::Option, prettify(m_schema[i].key), static_cast<int>(i)});
        }
        m_rows.push_back({Row::Kind::Header, "", -1});
        for (size_t i = 0; i < m_actions.size(); ++i)
            m_rows.push_back({Row::Kind::Action, m_actions[i], static_cast<int>(i)});

        // Start on the first action ("Back"/"Play"): a player who changes nothing just presses accept.
        m_selected = 0;
        for (size_t i = 0; i < m_rows.size(); ++i)
            if (m_rows[i].kind == Row::Kind::Action) { m_selected = static_cast<int>(i); break; }
        m_scrollFirst = 0;
    }

    void SettingsMenu::hide(const std::set<std::string> &keys)
    {
        m_hidden = keys;
        buildRows();
    }

    const Setting *SettingsMenu::find(const std::string &k) const
    {
        for (const Setting &s : m_schema)
            if (key(s) == k) return &s;
        return nullptr;
    }

    std::string SettingsMenu::value(const std::string &k) const
    {
        const Setting *s = find(k);
        return s ? effective(*s, nullptr) : std::string{};
    }

    std::string SettingsMenu::source(const std::string &k) const
    {
        const Setting *s = find(k);
        std::string src;
        if (s) effective(*s, &src);
        return src;
    }

    bool SettingsMenu::stepValue(const std::string &k, int dir)
    {
        for (size_t i = 0; i < m_schema.size(); ++i)
            if (key(m_schema[i]) == k)
            {
                if (!environmentOverride(m_schema[i]).empty()) return false;
                step(Row{Row::Kind::Option, {}, static_cast<int>(i)}, dir);
                return true;
            }
        return false;
    }

    std::string SettingsMenu::lockedBy(const std::string &k) const
    {
        const Setting *s = find(k);
        return s ? environmentOverride(*s) : std::string{};
    }

    bool SettingsMenu::setValue(const std::string &k, const std::string &raw)
    {
        const Setting *s = find(k);
        std::string v;
        if (!s || !lockedBy(k).empty() || !validate(*s, raw, v)) return false;
        if (effective(*s, nullptr) != v)
        {
            m_explicit[k] = v;
            m_dirty = true;
        }
        return true;
    }

    std::string SettingsMenu::effective(const Setting &s, std::string *source) const
    {
        const std::string k = key(s);
        if (auto it = m_explicit.find(k); it != m_explicit.end())
        {
            if (source) *source = "set";
            return it->second;
        }
        for (const Preset &p : presets())
            if (m_preset == p.name)
                for (const auto &kv : p.values)
                    if (kv.first == k)
                    {
                        std::string v;
                        if (validate(s, kv.second, v))
                        {
                            if (source) *source = "preset";
                            return v;
                        }
                    }
        if (source) *source = "default";
        std::string v;
        return validate(s, s.defaultValue, v) ? v : std::string(s.defaultValue);
    }

    bool SettingsMenu::selectable(int row) const
    {
        if (row < 0 || row >= static_cast<int>(m_rows.size())) return false;
        const Row &r = m_rows[row];
        if (r.kind == Row::Kind::Header) return false;
        if (r.kind == Row::Kind::Option && !environmentOverride(m_schema[r.index]).empty()) return true;   // shown, not editable
        return true;
    }

    void SettingsMenu::moveSelection(int dir)
    {
        int r = m_selected;
        for (size_t n = 0; n < m_rows.size(); ++n)
        {
            r = (r + dir + static_cast<int>(m_rows.size())) % static_cast<int>(m_rows.size());
            if (selectable(r)) { m_selected = r; return; }
        }
    }

    void SettingsMenu::step(const Row &row, int dir)
    {
        if (row.kind == Row::Kind::Preset)
        {
            const std::vector<Preset> &ps = presets();
            if (ps.empty()) return;
            int cur = 0;
            for (size_t i = 0; i < ps.size(); ++i)
                if (m_preset == ps[i].name) cur = static_cast<int>(i);
            cur = (cur + dir + static_cast<int>(ps.size())) % static_cast<int>(ps.size());
            m_preset = ps[cur].name;
            m_dirty = true;
            return;
        }
        if (row.kind != Row::Kind::Option) return;
        const Setting &s = m_schema[row.index];
        if (!environmentOverride(s).empty()) return;   // the environment pins it: not ours to change
        const std::string cur = effective(s, nullptr);
        std::string next = cur;
        switch (s.kind)
        {
        case Kind::Bool:
            next = (cur == "true") ? "false" : "true";
            break;
        case Kind::Int:
        {
            const int v = std::clamp(std::atoi(cur.c_str()) + dir, s.minInt, s.maxInt);
            next = std::to_string(v);
            break;
        }
        case Kind::Choice:
        {
            const std::vector<std::string> cs = splitChoices(s.choices);
            if (cs.empty()) return;
            auto it = std::find(cs.begin(), cs.end(), cur);
            int i = (it == cs.end()) ? 0 : static_cast<int>(it - cs.begin());
            i = (i + dir + static_cast<int>(cs.size())) % static_cast<int>(cs.size());
            next = cs[i];
            break;
        }
        }
        if (next != cur)
        {
            m_explicit[key(s)] = next;
            m_dirty = true;
        }
    }

    void SettingsMenu::handle(const MenuInput &in)
    {
        m_activated = -1;
        if (in.up) moveSelection(-1);
        if (in.down) moveSelection(+1);
        const Row &row = m_rows[m_selected];
        if (row.kind == Row::Kind::Action)
        {
            // The buttons sit side by side: left/right moves between them.
            if (in.left && row.index > 0) moveSelection(-1);
            if (in.right && row.index + 1 < static_cast<int>(m_actions.size())) moveSelection(+1);
        }
        else
        {
            if (in.left) step(row, -1);
            if (in.right) step(row, +1);
        }
        if (in.accept)
        {
            if (row.kind == Row::Kind::Action)
                m_activated = row.index;
            else
                step(row, +1);
        }
        if (in.reset)
        {
            m_explicit.clear();
            m_preset = "original";
            m_dirty = true;
        }
    }

    std::map<std::string, std::string> SettingsMenu::values() const
    {
        std::map<std::string, std::string> out = m_unknown;   // never silently drop a hand-written line
        if (m_preset != "original") out["preset"] = m_preset;
        for (const auto &kv : m_explicit) out[kv.first] = kv.second;
        return out;
    }

    void SettingsMenu::draw(Rectangle area, int fontSize) const
    {
        // Layout: a scrolling list of options, then the description of the selected one, then the
        // action rows as a fixed bar of buttons -- always visible, however long the list gets.
        const float lineH = fontSize * 1.55f;
        const float valueX = area.x + area.width * 0.52f;
        const int small = static_cast<int>(fontSize * 0.8f);
        const int descLines = 2;
        const float buttonsH = lineH * 1.4f;
        const float descH = descLines * small * 1.3f + lineH * 0.4f;
        const float listH = area.height - buttonsH - descH;

        std::vector<int> listRows;
        for (int i = 0; i < static_cast<int>(m_rows.size()); ++i)
            if (m_rows[i].kind != Row::Kind::Action && !(m_rows[i].kind == Row::Kind::Header && m_rows[i].label.empty()))
                listRows.push_back(i);
        const int visible = std::max(1, static_cast<int>(listH / lineH));

        // Scroll so the selected option is in view; with an action selected, keep the last scroll.
        int selPos = -1;
        for (int p = 0; p < static_cast<int>(listRows.size()); ++p)
            if (listRows[p] == m_selected) selPos = p;
        int first = m_scrollFirst;
        if (selPos >= 0)
        {
            if (selPos < first) first = std::max(0, selPos - 1);   // keep a section header above it
            if (selPos >= first + visible) first = selPos - visible + 1;
        }
        first = std::clamp(first, 0, std::max(0, static_cast<int>(listRows.size()) - visible));
        m_scrollFirst = first;

        std::string selectedDescription;
        for (int p = first; p < static_cast<int>(listRows.size()) && p < first + visible; ++p)
        {
            const int i = listRows[p];
            const Row &r = m_rows[i];
            const float y = area.y + (p - first) * lineH;
            const bool sel = (i == m_selected);
            if (sel)
                DrawRectangle(static_cast<int>(area.x), static_cast<int>(y - fontSize * 0.2f),
                              static_cast<int>(area.width), static_cast<int>(lineH), kSelectBg);

            switch (r.kind)
            {
            case Row::Kind::Header:
                drawText(m_font, r.label.c_str(), static_cast<int>(area.x), static_cast<int>(y + lineH * 0.25f), small, kAccent);
                break;
            case Row::Kind::Preset:
            {
                drawText(m_font, r.label.c_str(), static_cast<int>(area.x + fontSize), static_cast<int>(y), fontSize, kText);
                const std::string v = sel ? "< " + m_preset + " >" : m_preset;
                drawText(m_font, v.c_str(), static_cast<int>(valueX), static_cast<int>(y), fontSize, kText);
                if (sel)
                    for (const Preset &pr : presets())
                        if (m_preset == pr.name) selectedDescription = pr.description;
                break;
            }
            case Row::Kind::Option:
            {
                const Setting &s = m_schema[r.index];
                const std::string env = environmentOverride(s);
                std::string source;
                const std::string cur = effective(s, &source);
                drawText(m_font, r.label.c_str(), static_cast<int>(area.x + fontSize), static_cast<int>(y), fontSize, kText);
                if (!env.empty())
                    drawText(m_font, ("locked: " + env).c_str(), static_cast<int>(valueX), static_cast<int>(y), fontSize, kLocked);
                else
                {
                    const std::string v = sel ? "< " + cur + " >" : cur;
                    drawText(m_font, v.c_str(), static_cast<int>(valueX), static_cast<int>(y), fontSize,
                             source == "default" ? kDim : kText);
                }
                if (sel)
                    selectedDescription = std::string(s.description) +
                                          (env.empty() ? "" : "  (set by the environment, so it cannot be changed here)");
                break;
            }
            case Row::Kind::Action:
                break;
            }
        }
        // A scroll hint when the list does not fit.
        if (first > 0)
            drawText(m_font, "^ more", static_cast<int>(area.x + area.width - measureText(m_font, "^ more", small)), static_cast<int>(area.y - small * 1.2f), small, kDim);
        if (first + visible < static_cast<int>(listRows.size()))
            drawText(m_font, "v more", static_cast<int>(area.x + area.width - measureText(m_font, "v more", small)),
                     static_cast<int>(area.y + visible * lineH - small), small, kDim);

        const float descY = area.y + listH + lineH * 0.2f;
        const std::vector<std::string> lines = wrap(m_font, selectedDescription, small, area.width);
        for (size_t i = 0; i < lines.size() && static_cast<int>(i) < descLines; ++i)
            drawText(m_font, lines[i].c_str(), static_cast<int>(area.x), static_cast<int>(descY + i * small * 1.3f), small, kDim);

        // The action buttons.
        float x = area.x;
        const float by = area.y + area.height - buttonsH + lineH * 0.2f;
        for (int i = 0; i < static_cast<int>(m_rows.size()); ++i)
        {
            const Row &r = m_rows[i];
            if (r.kind != Row::Kind::Action) continue;
            const int tw = measureText(m_font, r.label.c_str(), fontSize);
            const Rectangle box{x, by, static_cast<float>(tw + fontSize * 3), lineH};
            const bool sel = (i == m_selected);
            DrawRectangleRec(box, sel ? kSelectBg : Color{36, 34, 40, 255});
            DrawRectangleLinesEx(box, 2.0f, sel ? kAccent : kDim);
            drawText(m_font, r.label.c_str(), static_cast<int>(box.x + fontSize * 1.5f), static_cast<int>(box.y + (lineH - fontSize) / 2),
                     fontSize, sel ? kAccent : kText);
            x += box.width + fontSize;
        }
    }
}
