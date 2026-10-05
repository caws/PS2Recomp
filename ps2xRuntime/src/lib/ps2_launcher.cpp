#include "ps2_launcher.h"
#include "ps2_settings.h"
#include "ps2_settings_menu.h"
#include "ps2_mods.h"

#include "raylib.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <vector>

#if defined(PS2X_LAUNCHER_FONT)
#include "ps2x_launcher_font.h"   // generated: kLato_Regular[], kLato_Bold[] (SIL OFL 1.1)
#endif

namespace ps2x::launcher
{
    namespace
    {
        const Color kBg{18, 17, 21, 255};
        const Color kPanel{28, 26, 32, 255};
        const Color kText{232, 230, 226, 255};
        const Color kDim{140, 138, 150, 255};
        const Color kAccent{214, 176, 92, 255};
        const Color kSelectBg{60, 52, 36, 255};
        const Color kOk{120, 190, 120, 255};
        const Color kWarn{220, 180, 90, 255};
        const Color kBad{220, 100, 100, 255};

        GameInfo g_info{nullptr, nullptr};
        bool g_hasInfo = false;

        std::vector<Page> &pages() { static std::vector<Page> s_pages; return s_pages; }

        void setEnvVar(const std::string &name, const std::string &value)
        {
#if defined(_WIN32)
            _putenv_s(name.c_str(), value.c_str());
#else
            if (value.empty()) unsetenv(name.c_str()); else setenv(name.c_str(), value.c_str(), 1);
#endif
        }

        struct Fonts
        {
            Font regular{}, bold{};
            bool loaded = false;
        };

        void loadFonts(Fonts &f)
        {
#if defined(PS2X_LAUNCHER_FONT)
            // Rasterised once at a large size and scaled down with bilinear filtering: crisp at every
            // window size the launcher is likely to be.
            f.regular = LoadFontFromMemory(".ttf", kLato_Regular, sizeof kLato_Regular, 64, nullptr, 0);
            f.bold = LoadFontFromMemory(".ttf", kLato_Bold, sizeof kLato_Bold, 96, nullptr, 0);
            SetTextureFilter(f.regular.texture, TEXTURE_FILTER_BILINEAR);
            SetTextureFilter(f.bold.texture, TEXTURE_FILTER_BILINEAR);
            f.loaded = f.regular.texture.id != 0 && f.bold.texture.id != 0;
#else
            (void)f;
#endif
        }

        void text(const Fonts &f, bool bold, const std::string &s, float x, float y, float size, Color c)
        {
            if (f.loaded) DrawTextEx(bold ? f.bold : f.regular, s.c_str(), Vector2{x, y}, size, 0.0f, c);
            else DrawText(s.c_str(), static_cast<int>(x), static_cast<int>(y), static_cast<int>(size), c);
        }

        float measure(const Fonts &f, bool bold, const std::string &s, float size)
        {
            return f.loaded ? MeasureTextEx(bold ? f.bold : f.regular, s.c_str(), size, 0.0f).x
                            : static_cast<float>(MeasureText(s.c_str(), static_cast<int>(size)));
        }

        // Word-wrap into lines no wider than `width`.
        std::vector<std::string> wrap(const Fonts &f, bool bold, const std::string &s, float size, float width)
        {
            std::vector<std::string> lines;
            std::string line, word;
            for (size_t i = 0; i <= s.size(); ++i)
            {
                if (i == s.size() || s[i] == ' ')
                {
                    const std::string cand = line.empty() ? word : line + " " + word;
                    if (!line.empty() && measure(f, bold, cand, size) > width) { lines.push_back(line); line = word; }
                    else line = cand;
                    word.clear();
                }
                else
                    word += s[i];
            }
            if (!line.empty()) lines.push_back(line);
            return lines;
        }

        struct DiscStatus
        {
            enum class Level { Ok, Warn, Missing } level;
            std::string line;
            std::string help;   // what to do about it (shown for Missing)
        };

        // "camera_pullback" -> "Camera pullback"
        std::string prettify(std::string s)
        {
            for (char &c : s)
                if (c == '_') c = ' ';
            if (!s.empty()) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
            return s;
        }

        const settings::Setting *findSetting(const std::vector<settings::Setting> &schema, const std::string &key)
        {
            for (const settings::Setting &st : schema)
                if (settings::key(st) == key) return &st;
            return nullptr;
        }

        std::string upper(std::string s)
        {
            for (char &c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            return s;
        }

        // Game-agnostic: the executable exists, and the disc's own SYSTEM.CNF (BOOT2 = cdrom0:\NAME;1)
        // names the same executable. The full per-file verification stays in scripts/verify_disc.sh.
        DiscStatus checkDisc(const std::filesystem::path &elf)
        {
            std::error_code ec;
            const std::filesystem::path dir = elf.parent_path();
            if (!std::filesystem::exists(elf, ec))
                return {DiscStatus::Level::Missing, "Disc not found",
                        "Copy every file from your game disc into " + dir.string() + " -- the game needs " +
                            elf.filename().string() + " and the rest of the disc beside it."};

            std::string serial;
            std::ifstream cnf(dir / "SYSTEM.CNF");
            std::string line;
            while (cnf && std::getline(cnf, line))
            {
                if (upper(line).find("BOOT2") == std::string::npos) continue;
                const size_t a = line.find_last_of("\\:/");
                const size_t b = line.find(';', a == std::string::npos ? 0 : a);
                if (a != std::string::npos)
                    serial = line.substr(a + 1, (b == std::string::npos ? line.size() : b) - a - 1);
                while (!serial.empty() && std::isspace(static_cast<unsigned char>(serial.back()))) serial.pop_back();
            }
            const std::string exe = elf.filename().string();
            if (serial.empty())
                return {DiscStatus::Level::Warn, "Disc " + exe + " found, but no SYSTEM.CNF beside it", {}};
            if (upper(serial) != upper(exe))
                return {DiscStatus::Level::Warn, "The disc boots " + serial + ", but the executable is " + exe, {}};
            return {DiscStatus::Level::Ok, "Disc " + serial + " ready", {}};
        }

        // Held-direction auto-repeat for keys, d-pad and the left stick.
        struct Repeat
        {
            double heldSince = -1.0, lastFire = 0.0;
            bool fire(bool down, double now)
            {
                if (!down) { heldSince = -1.0; return false; }
                if (heldSince < 0.0) { heldSince = lastFire = now; return true; }
                if (now - heldSince > 0.4 && now - lastFire > 0.08) { lastFire = now; return true; }
                return false;
            }
        };

        struct Input
        {
            bool up = false, down = false, left = false, right = false;
            bool accept = false, back = false, reset = false, start = false;
        };

        Input readInput(Repeat &rUp, Repeat &rDown, Repeat &rLeft, Repeat &rRight)
        {
            const double now = GetTime();
            bool gu = false, gd = false, gl = false, gr = false;
            Input in;
            for (int g = 0; g < 4; ++g)   // any connected gamepad drives the launcher
            {
                if (!IsGamepadAvailable(g)) continue;
                gu |= IsGamepadButtonDown(g, GAMEPAD_BUTTON_LEFT_FACE_UP) || GetGamepadAxisMovement(g, GAMEPAD_AXIS_LEFT_Y) < -0.6f;
                gd |= IsGamepadButtonDown(g, GAMEPAD_BUTTON_LEFT_FACE_DOWN) || GetGamepadAxisMovement(g, GAMEPAD_AXIS_LEFT_Y) > 0.6f;
                gl |= IsGamepadButtonDown(g, GAMEPAD_BUTTON_LEFT_FACE_LEFT) || GetGamepadAxisMovement(g, GAMEPAD_AXIS_LEFT_X) < -0.6f;
                gr |= IsGamepadButtonDown(g, GAMEPAD_BUTTON_LEFT_FACE_RIGHT) || GetGamepadAxisMovement(g, GAMEPAD_AXIS_LEFT_X) > 0.6f;
                in.accept |= IsGamepadButtonPressed(g, GAMEPAD_BUTTON_RIGHT_FACE_DOWN);
                in.back |= IsGamepadButtonPressed(g, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT);
                in.reset |= IsGamepadButtonPressed(g, GAMEPAD_BUTTON_RIGHT_FACE_UP);
                in.start |= IsGamepadButtonPressed(g, GAMEPAD_BUTTON_MIDDLE_RIGHT);
            }
            in.up = rUp.fire(IsKeyDown(KEY_UP) || IsKeyDown(KEY_W) || gu, now);
            in.down = rDown.fire(IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S) || gd, now);
            in.left = rLeft.fire(IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A) || gl, now);
            in.right = rRight.fire(IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D) || gr, now);
            in.accept |= IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_X);
            in.back |= IsKeyPressed(KEY_ESCAPE) || IsKeyPressed(KEY_BACKSPACE);
            in.reset |= IsKeyPressed(KEY_R);
            return in;
        }

        // The bindings, as the runtime's pad backend implements them (ps2_pad.cpp).
        const char *const kControls[][3] = {
            // PS2 button,   keyboard,                 gamepad
            {"D-pad",        "Arrow keys",             "D-pad"},
            {"Left stick",   "W A S D",                "Left stick"},
            {"Right stick",  "I J K L",                "Right stick"},
            {"Cross",        "X or Space",             "A (bottom)"},
            {"Circle",       "C",                      "B (right)"},
            {"Square",       "Z",                      "X (left)"},
            {"Triangle",     "V",                      "Y (top)"},
            {"L1 / R1",      "Q / E",                  "LB / RB"},
            {"L2 / R2",      "Left Shift / 2",         "LT / RT"},
            {"L3 / R3",      "-",                      "Stick clicks"},
            {"Start",        "Enter",                  "Menu / Start"},
            {"Select",       "Tab",                    "View / Back"},
        };
    }

    bool registerGameInfo(const GameInfo &info)
    {
        g_info = info;
        g_hasInfo = true;
        return true;
    }

    const GameInfo *gameInfo() { return g_hasInfo ? &g_info : nullptr; }

    bool registerPage(const Page &page)
    {
        pages().push_back(page);
        return true;
    }

    bool shouldShow(char **argv)
    {
        if (!settings::enabled()) return false;
        if (const char *done = std::getenv("PS2X_SETTINGS_APPLIED"); done && done[0] == '1') return false;
        if (const char *e = std::getenv("PS2X_LAUNCHER"); e && e[0]) return e[0] != '0';
        const std::filesystem::path file = settings::settingsFile(argv);
        if (file.empty()) return false;
        const auto values = settings::readValues(file);
        if (auto it = values.find("launcher.show"); it != values.end())
            for (const settings::Setting &s : settings::schema())
                if (settings::key(s) == "launcher.show")
                {
                    std::string v;
                    if (settings::validate(s, it->second, v)) return v == "true";
                }
        return true;
    }

    Result run(char **argv, const std::string &windowTitle, const std::filesystem::path &elfPath)
    {
        const std::filesystem::path file = settings::settingsFile(argv);
        const DiscStatus disc = checkDisc(elfPath);
        const std::string title = g_hasInfo && g_info.title ? g_info.title : elfPath.filename().string();
        const std::string version = g_hasInfo && g_info.version ? g_info.version : "";

        SetTraceLogLevel(LOG_WARNING);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(1100, 660, windowTitle.c_str());
        SetExitKey(KEY_NULL);   // Esc means "back" here, not "close the window"
        SetTargetFPS(60);
        Fonts fonts;
        loadFonts(fonts);

        settings::SettingsMenu menu(settings::readValues(file), {"Back"});
        if (fonts.loaded) menu.setFont(&fonts.regular);

        // Mods (ps2_mods.h): each switches through its own settings key, which therefore leaves the
        // Settings list -- one place per option.
        const std::vector<mods::ModInfo> modList = mods::scan(argv);
        {
            // Every [mods] switch and every mod's own options ([mods.<id>]) belong to the Mods screen,
            // whether or not the mod's folder is present.
            std::set<std::string> owned;
            for (const settings::Setting &st : settings::schema())
            {
                const std::string sec = st.section;
                if (sec == "mods" || sec.rfind("mods.", 0) == 0) owned.insert(settings::key(st));
            }
            menu.hide(owned);
        }
        const std::vector<settings::Setting> schema = settings::schema();
        // The Mods list: each mod, and -- only while it is switched on -- its own options nested under it.
        struct ModRow { int mod; int opt; };   // opt = -1: the mod itself; else an index into its options
        auto modIsOn = [&](const mods::ModInfo &m) { return m.setting.empty() || menu.value(m.setting) == "true"; };
        auto modRows = [&]()
        {
            std::vector<ModRow> rows;
            for (size_t i = 0; i < modList.size(); ++i)
            {
                rows.push_back({static_cast<int>(i), -1});
                if (modIsOn(modList[i]))
                    for (size_t k = 0; k < modList[i].options.size(); ++k) rows.push_back({static_cast<int>(i), static_cast<int>(k)});
            }
            return rows;   // + "Back" at index rows.size()
        };
        int modSelected = 0;   // index into modRows(); == its size = "Back"
        int modFirst = 0;      // the list's scroll position (first row drawn)

        enum class Screen { Main, Settings, Mods, Controls, Page } screen = Screen::Main;
        // Game pages (row 249): the open page, its rows, the selected row, the text row being edited.
        int pageIdx = -1, pageSel = 0;
        std::vector<PageRow> pageRows;
        std::string pageStatus;
        bool editing = false;
        auto openPage = [&](int i)
        {
            pageIdx = i; pageSel = 0; editing = false; pageStatus.clear(); pageRows.clear();
            if (pages()[i].init) pages()[i].init(pageRows);
            while (pageSel < static_cast<int>(pageRows.size()) && (!pageRows[pageSel].visible || pageRows[pageSel].kind == PageRow::Kind::Label)) ++pageSel;
            screen = Screen::Page;
        };
        // PS2X_LAUNCHER_SCREEN=settings|mods|mods:<id>|controls opens on that screen (screenshots/tests;
        // unset = main). mods:<id> selects that mod's row, mods:<id>:<option> that option's row.
        if (const char *e = std::getenv("PS2X_LAUNCHER_SCREEN"); e && e[0])
        {
            screen = (e[0] == 's') ? Screen::Settings : (e[0] == 'c' ? Screen::Controls : (e[0] == 'm' ? Screen::Mods : Screen::Main));
            const std::string arg = e;
            if (screen == Screen::Mods && arg.find(':') != std::string::npos)
            {
                std::string id = arg.substr(arg.find(':') + 1), optKey;
                if (const size_t c = id.find(':'); c != std::string::npos) { optKey = "mods." + id.substr(0, c) + "." + id.substr(c + 1); id.resize(c); }
                const std::vector<ModRow> rows = modRows();
                for (size_t r = 0; r < rows.size(); ++r)
                {
                    const mods::ModInfo &mi = modList[rows[r].mod];
                    if (mi.id != id) continue;
                    if (rows[r].opt < 0 && optKey.empty()) modSelected = static_cast<int>(r);
                    if (rows[r].opt >= 0 && mi.options[rows[r].opt] == optKey) modSelected = static_cast<int>(r);
                }
            }
        }
        // The main menu: Start, each game page (row 249), then Settings, Mods, Controls, Exit.
        std::vector<std::string> items = {disc.level == DiscStatus::Level::Missing ? "Disc not found" : "Start game"};
        for (const Page &pg : pages()) items.push_back(pg.menuLabel);
        const int firstFixed = static_cast<int>(items.size());   // index of "Settings"
        for (const char *fixed : {"Settings", "Mods", "Controls", "Exit"}) items.push_back(fixed);
        if (const char *e = std::getenv("PS2X_LAUNCHER_PAGE"); e && e[0])
            for (size_t i = 0; i < pages().size(); ++i)
                if (pages()[i].id == e) openPage(static_cast<int>(i));
        int selected = 0;
        bool showHelp = false;
        Repeat rUp, rDown, rLeft, rRight;
        Result result = Result::Quit;
        bool done = false;

        // The window takes focus when it opens, so a keystroke meant for another window (typing
        // Enter elsewhere) would otherwise land here and start the game. Input is ignored for the
        // first half second and until every key held at that point has been released.
        bool armed = false;
        const double openedAt = GetTime();

        while (!done && !WindowShouldClose())
        {
            Input in = readInput(rUp, rDown, rLeft, rRight);
            if (editing) in = Input{};   // typing into a text row: letters, Space and Backspace are text, not commands
            if (!armed)
            {
                const bool anyHeld = IsKeyDown(KEY_ENTER) || IsKeyDown(KEY_SPACE) || IsKeyDown(KEY_X) ||
                                     IsKeyDown(KEY_ESCAPE) || GetKeyPressed() != 0;
                armed = (GetTime() - openedAt > 0.5) && !anyHeld;
                in = Input{};
            }
            switch (screen)
            {
            case Screen::Main:
                if (in.up) selected = (selected + static_cast<int>(items.size()) - 1) % static_cast<int>(items.size());
                if (in.down) selected = (selected + 1) % static_cast<int>(items.size());
                if (in.start && disc.level != DiscStatus::Level::Missing) { result = Result::Play; done = true; }
                if (in.accept)
                {
                    if (selected == 0)
                    {
                        if (disc.level == DiscStatus::Level::Missing) showHelp = !showHelp;
                        else { result = Result::Play; done = true; }
                    }
                    else if (selected < firstFixed) openPage(selected - 1);
                    else if (selected == firstFixed) screen = Screen::Settings;
                    else if (selected == firstFixed + 1) screen = Screen::Mods;
                    else if (selected == firstFixed + 2) screen = Screen::Controls;
                    else done = true;
                }
                if (in.back) done = true;
                break;
            case Screen::Settings:
            {
                settings::MenuInput mi;
                mi.up = in.up; mi.down = in.down; mi.left = in.left; mi.right = in.right;
                mi.accept = in.accept; mi.reset = in.reset;
                menu.handle(mi);
                if (in.back || menu.activatedAction() == 0) screen = Screen::Main;
                if (in.start && disc.level != DiscStatus::Level::Missing) { result = Result::Play; done = true; }
                break;
            }
            case Screen::Mods:
            {
                const std::vector<ModRow> rows = modRows();
                const int n = static_cast<int>(rows.size()) + 1;   // + "Back"
                modSelected = std::min(modSelected, n - 1);
                if (in.up) modSelected = (modSelected + n - 1) % n;
                if (in.down) modSelected = (modSelected + 1) % n;
                const bool onBack = modSelected == static_cast<int>(rows.size());
                if (in.back || (onBack && in.accept)) { screen = Screen::Main; break; }
                if (!onBack)
                {
                    const ModRow r = rows[modSelected];
                    const mods::ModInfo &m = modList[r.mod];
                    if (r.opt < 0)
                    {
                        // The mod's row switches it; its options appear under it (or go away) at once.
                        if ((in.accept || in.left || in.right) && !m.setting.empty())
                            menu.setValue(m.setting, menu.value(m.setting) == "true" ? "false" : "true");
                    }
                    else
                    {
                        const std::string &k = m.options[r.opt];
                        if (in.left) menu.stepValue(k, -1);
                        if (in.right || in.accept) menu.stepValue(k, +1);
                    }
                }
                if (in.start && disc.level != DiscStatus::Level::Missing) { result = Result::Play; done = true; }
                break;
            }
            case Screen::Controls:
                if (in.back || in.accept) screen = Screen::Main;
                break;
            case Screen::Page:
            {
                const int n = static_cast<int>(pageRows.size());
                auto selectable = [&](int r) { return r >= 0 && r < n && pageRows[r].visible && pageRows[r].kind != PageRow::Kind::Label; };
                PageEvent ev;
                if (editing)
                {
                    PageRow &row = pageRows[pageSel];
                    for (int c = GetCharPressed(); c > 0; c = GetCharPressed())
                        if (c >= 32 && c < 127 && row.text.size() < row.maxLen) row.text.push_back(static_cast<char>(c));
                    if ((IsKeyPressed(KEY_BACKSPACE) || IsKeyPressedRepeat(KEY_BACKSPACE)) && !row.text.empty()) row.text.pop_back();
                    if ((IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL)) && IsKeyPressed(KEY_V))
                        if (const char *clip = GetClipboardText())
                            for (const char *q = clip; *q && row.text.size() < row.maxLen; ++q)
                                if (*q >= 32 && *q < 127) row.text.push_back(*q);
                    if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER) || IsKeyPressed(KEY_ESCAPE) || IsKeyPressed(KEY_TAB)) editing = false;
                }
                else
                {
                    if (in.up || in.down)
                    {
                        const int step = in.up ? -1 : 1;
                        for (int k = 1; k <= n; ++k)
                        {
                            const int r = ((pageSel + step * k) % n + n) % n;
                            if (selectable(r)) { pageSel = r; break; }
                        }
                    }
                    if (selectable(pageSel) && pageRows[pageSel].enabled)
                    {
                        PageRow &row = pageRows[pageSel];
                        const int cn = static_cast<int>(row.choices.size());
                        if (row.kind == PageRow::Kind::Choice && cn > 0)
                        {
                            if (in.left) row.choice = (row.choice + cn - 1) % cn;
                            if (in.right || in.accept) row.choice = (row.choice + 1) % cn;
                        }
                        else if (row.kind == PageRow::Kind::Text && in.accept) { editing = true; while (GetCharPressed() > 0) {} }
                        else if (row.kind == PageRow::Kind::Button && in.accept) ev.activated = pageSel;
                    }
                    ev.back = in.back;
                }
                PageResult pr = pages()[pageIdx].update ? pages()[pageIdx].update(pageRows, ev, pageStatus) : PageResult{};
                if (!selectable(pageSel))
                    for (int r = 0; r < n; ++r) if (selectable(r)) { pageSel = r; break; }
                if (pr.kind == PageResult::Kind::Back) { screen = Screen::Main; editing = false; }
                else if (pr.kind == PageResult::Kind::Play)
                {
                    for (const auto &kv : pr.env) setEnvVar(kv.first, kv.second);
                    result = Result::Play;
                    done = true;
                }
                break;
            }
            }
            if (done) break;

            const float w = static_cast<float>(GetScreenWidth()), h = static_cast<float>(GetScreenHeight());
            const float m = std::max(28.0f, w / 22.0f);
            const float base = std::clamp(h / 28.0f, 16.0f, 34.0f);
            BeginDrawing();
            ClearBackground(kBg);

            if (screen == Screen::Main)
            {
                // Left: the game's title, "Recompiled", the disc. Right: the menu. Corner: version.
                DrawRectangle(0, 0, static_cast<int>(w * 0.56f), static_cast<int>(h), kPanel);
                const float titleSize = base * 2.1f;
                float y = h * 0.22f;
                for (const std::string &l : wrap(fonts, true, title, titleSize, w * 0.56f - 2 * m))
                {
                    text(fonts, true, l, m, y, titleSize, kText);
                    y += titleSize * 1.08f;
                }
                text(fonts, false, "Recompiled", m, y + base * 0.2f, base * 1.2f, kAccent);
                y += base * 3.0f;
                const Color dc = disc.level == DiscStatus::Level::Ok ? kOk : (disc.level == DiscStatus::Level::Warn ? kWarn : kBad);
                DrawCircle(static_cast<int>(m + base * 0.3f), static_cast<int>(y + base * 0.45f), base * 0.25f, dc);
                text(fonts, false, disc.line, m + base, y, base * 0.9f, dc);
                if (showHelp && !disc.help.empty())
                {
                    y += base * 1.6f;
                    for (const std::string &l : wrap(fonts, false, disc.help, base * 0.8f, w * 0.56f - 2 * m))
                    {
                        text(fonts, false, l, m, y, base * 0.8f, kDim);
                        y += base * 1.05f;
                    }
                }

                const float mx = w * 0.56f + m;
                float my = h * 0.5f - items.size() * base * 1.9f / 2.0f;
                for (size_t i = 0; i < items.size(); ++i)
                {
                    const bool sel = static_cast<int>(i) == selected;
                    if (sel)
                        DrawRectangleRounded(Rectangle{mx - base * 0.6f, my - base * 0.35f, w - mx - m + base * 0.6f, base * 1.75f}, 0.3f, 6, kSelectBg);
                    // The bullet is drawn, not typed: the embedded font is rasterised for ASCII only.
                    if (sel) DrawCircle(static_cast<int>(mx + base * 0.25f), static_cast<int>(my + base * 0.62f), base * 0.14f, kAccent);
                    text(fonts, sel, items[i], mx + base * 0.8f, my, base * 1.15f, sel ? kAccent : kText);
                    my += base * 1.9f;
                }
                if (!version.empty()) text(fonts, false, version, m, h - m * 0.8f, base * 0.65f, kDim);
                text(fonts, false, "Up/Down  choose     Enter / A  select     Esc / B  exit", mx, h - m * 0.8f, base * 0.65f, kDim);
            }
            else if (screen == Screen::Settings)
            {
                text(fonts, true, "Settings", m, m * 0.8f, base * 1.6f, kText);
                const Rectangle area{m, m * 0.8f + base * 3.0f, w - 2 * m, h - m * 1.6f - base * 4.2f};
                menu.draw(area, static_cast<int>(base));
                text(fonts, false, "Up/Down  choose     Left/Right  change     R / Y  reset     Esc / B  back     Start  play",
                     m, h - m * 0.8f, base * 0.65f, kDim);
            }
            else if (screen == Screen::Mods)
            {
                text(fonts, true, "Mods", m, m * 0.8f, base * 1.6f, kText);
                const float listW = w * 0.40f;
                const float top = m * 0.8f + base * 3.0f;
                const float modH = base * 1.55f, optH = base * 1.3f, opt = base * 0.85f;
                const float backH = base * 2.3f;
                const float bottom = h - m * 0.8f - base * 1.2f - backH;   // the list ends above "Back" and the hints
                const std::vector<ModRow> rows = modRows();
                float y = top;
                if (modList.empty())
                {
                    const std::filesystem::path dir = mods::modsDirectory(argv);
                    text(fonts, false, "No mods found in " + (dir.empty() ? std::string("<exe dir>/mods") : dir.string()),
                         m, y, base * 0.85f, kDim);
                    y += base * 1.6f;
                }
                // Scroll so the selected row is in view (a mod with many options can outgrow the window).
                auto rowH = [&](int r) { return rows[r].opt < 0 ? modH : optH; };
                if (modSelected < static_cast<int>(rows.size()))
                {
                    if (modSelected < modFirst) modFirst = modSelected;
                    for (;;)
                    {
                        float used = 0.f;
                        for (int r = modFirst; r <= modSelected; ++r) used += rowH(r);
                        if (used <= bottom - top || modFirst >= modSelected) break;
                        ++modFirst;
                    }
                }
                modFirst = std::clamp(modFirst, 0, std::max(0, static_cast<int>(rows.size()) - 1));
                if (modFirst > 0) text(fonts, false, "^ more", m + listW - measure(fonts, false, "^ more", base * 0.65f), top - base * 1.0f, base * 0.65f, kDim);
                int last = modFirst - 1;
                for (int r = modFirst; r < static_cast<int>(rows.size()); ++r)
                {
                    if (y + rowH(r) > bottom + 0.5f) break;
                    last = r;
                    const bool sel = r == modSelected;
                    const mods::ModInfo &mod = modList[rows[r].mod];
                    if (rows[r].opt < 0)
                    {
                        if (sel) DrawRectangle(static_cast<int>(m), static_cast<int>(y - base * 0.2f), static_cast<int>(listW), static_cast<int>(modH), kSelectBg);
                        text(fonts, sel, mod.name, m + base * 0.5f, y, base, sel ? kAccent : kText);
                        std::string state;
                        Color sc = kDim;
                        if (mod.setting.empty()) state = "-";
                        else if (!menu.lockedBy(mod.setting).empty()) { state = "locked"; sc = kBad; }
                        else if (menu.value(mod.setting) == "true") { state = "ON"; sc = kOk; }
                        else state = "OFF";
                        text(fonts, true, state, m + listW - measure(fonts, true, state, base) - base * 0.5f, y, base, sc);
                        y += modH;
                    }
                    else
                    {
                        // An option, indented under its mod, with a thin guide line down the left.
                        const std::string &k = mod.options[rows[r].opt];
                        const float ix = m + base * 1.6f;
                        if (sel) DrawRectangle(static_cast<int>(ix - base * 0.4f), static_cast<int>(y - base * 0.15f), static_cast<int>(m + listW - ix + base * 0.4f), static_cast<int>(optH), kSelectBg);
                        DrawRectangle(static_cast<int>(m + base * 0.9f), static_cast<int>(y - base * 0.15f), 2, static_cast<int>(optH) + 2, kDim);
                        text(fonts, false, prettify(k.substr(k.rfind('.') + 1)), ix, y, opt, sel ? kAccent : kText);
                        const std::string env = menu.lockedBy(k);
                        std::string v = env.empty() ? menu.value(k) : "locked";
                        if (sel && env.empty()) v = "< " + v + " >";
                        const Color vc = !env.empty() ? kBad : (menu.source(k) == "default" ? kDim : kText);
                        text(fonts, false, v, m + listW - measure(fonts, false, v, opt) - base * 0.5f, y, opt, vc);
                        y += optH;
                    }
                }
                if (last + 1 < static_cast<int>(rows.size()))
                    text(fonts, false, "v more", m + listW - measure(fonts, false, "v more", base * 0.65f), y - base * 0.1f, base * 0.65f, kDim);
                {
                    const bool sel = modSelected == static_cast<int>(rows.size());
                    const Rectangle box{m, bottom + base * 0.6f, measure(fonts, false, "Back", base) + base * 3.0f, modH};
                    DrawRectangleRec(box, sel ? kSelectBg : Color{36, 34, 40, 255});
                    DrawRectangleLinesEx(box, 2.0f, sel ? kAccent : kDim);
                    text(fonts, false, "Back", box.x + base * 1.5f, box.y + base * 0.25f, base, sel ? kAccent : kText);
                }

                // The right panel: the selected row's mod, and the option's own description when one is selected.
                const bool onRow = modSelected < static_cast<int>(rows.size());
                if (onRow)
                {
                    const ModRow sr = rows[modSelected];
                    const mods::ModInfo &mod = modList[sr.mod];
                    const float cx = m + listW + m, cw = w - cx - m;
                    float cy = top;
                    text(fonts, true, mod.name, cx, cy, base * 1.2f, kText);
                    cy += base * 1.7f;
                    std::string meta = mod.version.empty() ? "" : "version " + mod.version;
                    if (!mod.author.empty()) meta += (meta.empty() ? "by " : "  -  by ") + mod.author;
                    if (!meta.empty()) { text(fonts, false, meta, cx, cy, base * 0.75f, kDim); cy += base * 1.3f; }
                    const std::string desc = !mod.hasIni ? "No description (this folder has no mod.ini)."
                                                         : (mod.description.empty() ? "No description." : mod.description);
                    for (const std::string &l : wrap(fonts, false, desc, base * 0.85f, cw))
                    {
                        text(fonts, false, l, cx, cy, base * 0.85f, kText);
                        cy += base * 1.15f;
                    }
                    cy += base * 0.6f;
                    std::string how;
                    if (mod.setting.empty()) how = "Always active: the game registers no [mods] " + mod.id + " switch for it.";
                    else if (!menu.lockedBy(mod.setting).empty()) how = "Locked by the environment: " + menu.lockedBy(mod.setting);
                    else how = "Saved in settings.ini as [mods] " + mod.id;
                    if (sr.opt < 0 && !mod.options.empty() && !modIsOn(mod))
                        how += ". Switch it on to see its " + std::to_string(mod.options.size()) + " options.";
                    for (const std::string &l : wrap(fonts, false, how, base * 0.7f, cw))
                    {
                        text(fonts, false, l, cx, cy, base * 0.7f, kDim);
                        cy += base;
                    }
                    if (sr.opt >= 0)
                    {
                        const std::string &k = mod.options[sr.opt];
                        const settings::Setting *st = findSetting(schema, k);
                        cy += base * 0.8f;
                        text(fonts, true, prettify(k.substr(k.rfind('.') + 1)), cx, cy, base * 0.95f, kAccent);
                        cy += base * 1.4f;
                        std::string od = st ? st->description : "";
                        if (!od.empty()) od[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(od[0])));   // a sentence here
                        if (!menu.lockedBy(k).empty()) od += "  (set by the environment: " + menu.lockedBy(k) + ", so it cannot be changed here)";
                        for (const std::string &l : wrap(fonts, false, od, base * 0.8f, cw))
                        {
                            text(fonts, false, l, cx, cy, base * 0.8f, kText);
                            cy += base * 1.1f;
                        }
                    }
                }
                const bool onOption = onRow && rows[modSelected].opt >= 0;
                text(fonts, false, onOption ? "Up/Down  choose     Left/Right  change     Esc / B  back     Start  play"
                                            : "Up/Down  choose     Enter / A  switch on/off     Esc / B  back     Start  play",
                     m, h - m * 0.8f, base * 0.65f, kDim);
            }
            else if (screen == Screen::Page)
            {
                const Page &pg = pages()[pageIdx];
                text(fonts, true, pg.title, m, m * 0.8f, base * 1.6f, kText);
                const float listW = w * 0.46f;
                float y = m * 0.8f + base * 3.0f;
                const float rowH = base * 1.6f;
                for (int r = 0; r < static_cast<int>(pageRows.size()); ++r)
                {
                    const PageRow &row = pageRows[r];
                    if (!row.visible) continue;
                    const bool sel = r == pageSel && row.kind != PageRow::Kind::Label;
                    const Color tc = !row.enabled ? kDim : (sel ? kAccent : kText);
                    if (row.kind == PageRow::Kind::Label)
                    {
                        for (const std::string &l : wrap(fonts, false, row.label, base * 0.8f, listW))
                        {
                            text(fonts, false, l, m, y, base * 0.8f, kDim);
                            y += base * 1.1f;
                        }
                        y += base * 0.4f;
                        continue;
                    }
                    if (row.kind == PageRow::Kind::Button)
                    {
                        const Rectangle box{m, y, measure(fonts, false, row.label, base) + base * 3.0f, rowH - base * 0.2f};
                        DrawRectangleRec(box, sel ? kSelectBg : Color{36, 34, 40, 255});
                        DrawRectangleLinesEx(box, 2.0f, sel ? kAccent : kDim);
                        text(fonts, sel, row.label, box.x + base * 1.5f, box.y + base * 0.2f, base, tc);
                        y += rowH + base * 0.3f;
                        continue;
                    }
                    if (sel) DrawRectangle(static_cast<int>(m), static_cast<int>(y - base * 0.2f), static_cast<int>(listW), static_cast<int>(rowH), kSelectBg);
                    text(fonts, sel, row.label, m + base * 0.5f, y, base, tc);
                    std::string v;
                    if (row.kind == PageRow::Kind::Choice)
                    {
                        v = row.choices.empty() ? "" : row.choices[std::clamp(row.choice, 0, static_cast<int>(row.choices.size()) - 1)];
                        if (sel && row.enabled) v = "< " + v + " >";
                    }
                    else
                    {
                        v = row.text;
                        if (sel && editing && std::fmod(GetTime(), 1.0) < 0.6) v += "_";
                        else if (v.empty()) v = sel ? "(Enter to type)" : "-";
                    }
                    text(fonts, false, v, m + listW - measure(fonts, false, v, base) - base * 0.5f, y, base, row.enabled ? (sel ? kAccent : kText) : kDim);
                    y += rowH;
                }
                if (!pageStatus.empty())
                {
                    y += base * 0.8f;
                    for (const std::string &l : wrap(fonts, false, pageStatus, base * 0.85f, listW))
                    {
                        text(fonts, false, l, m, y, base * 0.85f, kWarn);
                        y += base * 1.15f;
                    }
                }
                if (pageSel >= 0 && pageSel < static_cast<int>(pageRows.size()) && !pageRows[pageSel].help.empty())
                {
                    const float cx = m + listW + m, cw = w - cx - m;
                    float cy = m * 0.8f + base * 3.0f;
                    for (const std::string &l : wrap(fonts, false, pageRows[pageSel].help, base * 0.8f, cw))
                    {
                        text(fonts, false, l, cx, cy, base * 0.8f, kText);
                        cy += base * 1.1f;
                    }
                }
                text(fonts, false, editing ? "Type     Backspace  delete     Ctrl+V  paste     Enter  done"
                                           : "Up/Down  choose     Left/Right  change     Enter / A  select     Esc / B  back",
                     m, h - m * 0.8f, base * 0.65f, kDim);
            }
            else
            {
                text(fonts, true, "Controls", m, m * 0.8f, base * 1.6f, kText);
                const float c0 = m, c1 = m + w * 0.26f, c2 = m + w * 0.56f;
                float y = m * 0.8f + base * 3.0f;
                text(fonts, true, "PS2", c0, y, base * 0.85f, kAccent);
                text(fonts, true, "Keyboard", c1, y, base * 0.85f, kAccent);
                text(fonts, true, "Gamepad", c2, y, base * 0.85f, kAccent);
                y += base * 1.5f;
                for (const auto &row : kControls)
                {
                    text(fonts, false, row[0], c0, y, base * 0.85f, kText);
                    text(fonts, false, row[1], c1, y, base * 0.85f, kDim);
                    text(fonts, false, row[2], c2, y, base * 0.85f, kDim);
                    y += base * 1.25f;
                }
                y += base * 0.5f;
                for (const std::string &l : wrap(fonts, false,
                                                 "Player 2: a second gamepad; or, when a gamepad has player 1, the keyboard "
                                                 "(same keys). Change it under Settings > Input > Keyboard.",
                                                 base * 0.75f, w - 2 * m))
                {
                    text(fonts, false, l, c0, y, base * 0.75f, kDim);
                    y += base;
                }
                text(fonts, false, "Esc / B  back", m, h - m * 0.8f, base * 0.65f, kDim);
            }
            EndDrawing();
        }

        if (result == Result::Play && menu.dirty())
        {
            if (settings::writeValues(file, menu.values()))
                std::fprintf(stderr, "[launcher] saved %s\n", file.string().c_str());
            else
                std::fprintf(stderr, "[launcher] WARNING could not write %s -- starting with the previous settings\n",
                             file.string().c_str());
        }
        if (fonts.loaded)
        {
            UnloadFont(fonts.regular);
            UnloadFont(fonts.bold);
        }
        CloseWindow();
        std::fprintf(stderr, "[launcher] %s\n", result == Result::Play ? "play" : "quit");
        return result;
    }
}
