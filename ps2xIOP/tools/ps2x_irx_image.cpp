// ps2x_irx_image (rotk row 275): turn an IOP module (IRX) into a plain MIPS ELF relocated to a fixed IOP load address,
// for ps2xRecomp's R3000 mode.
//
//   ps2x_irx_image <in.irx> <base-hex> <out.elf>
//
// The image is placed and relocated by ps2xIOP's own module loader (IopModuleLoader::load), exactly as the IOP
// emulator does at runtime, so the recompiled code and the bytes the emulator executes are the same: everything below
// <base> in the IOP pool is reserved first, so the loader's first-fit allocation lands on <base>.
//
// Output: ET_EXEC, one PT_LOAD at <base>, the module's allocatable sections at their relocated addresses, and a symbol
// table holding
//   * the module's own symbols, shifted to <base> (function sizes are what the recompiler uses as boundaries);
//   * one 8-byte FUNC symbol per import stub, named __iopimport_<library>_<ordinal> (ps2sdk irx.h import tables, magic
//     0x41E00000: `jr ra; addiu zero, zero, <ordinal>` pairs after a 20-byte header naming the library).
// Prints the base, size, entry and gp on stdout.
#include "../src/emulator/core/iop_memory.h"
#include "../src/emulator/services/iop_module_loader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using ps2x::iop::detail::IopMemory;
using ps2x::iop::detail::IopModuleLoader;

namespace
{
#pragma pack(push, 1)
    struct Ehdr
    {
        uint8_t ident[16];
        uint16_t type, machine;
        uint32_t version, entry, phoff, shoff, flags;
        uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
    };
    struct Phdr
    {
        uint32_t type, offset, vaddr, paddr, filesz, memsz, flags, align;
    };
    struct Shdr
    {
        uint32_t name, type, flags, addr, offset, size, link, info, addralign, entsize;
    };
    struct Sym
    {
        uint32_t name, value, size;
        uint8_t info, other;
        uint16_t shndx;
    };
#pragma pack(pop)
    static_assert(sizeof(Ehdr) == 52 && sizeof(Phdr) == 32 && sizeof(Shdr) == 40 && sizeof(Sym) == 16);

    constexpr uint32_t SHT_PROGBITS = 1, SHT_SYMTAB = 2, SHT_STRTAB = 3, SHT_NOBITS = 8;
    constexpr uint32_t SHF_ALLOC = 2, SHF_EXECINSTR = 4;
    constexpr uint32_t kImportMagic = 0x41E00000u;

    template <typename T>
    T readAt(const std::vector<uint8_t> &bytes, size_t offset)
    {
        T value{};
        if (offset + sizeof(T) <= bytes.size())
            std::memcpy(&value, bytes.data() + offset, sizeof(T));
        return value;
    }

    struct OutSection
    {
        std::string name;
        Shdr header{};
        std::vector<uint8_t> data;
    };

    uint32_t addString(std::string &table, const std::string &s)
    {
        const uint32_t offset = static_cast<uint32_t>(table.size());
        table += s;
        table.push_back('\0');
        return offset;
    }
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::fprintf(stderr, "usage: %s <in.irx> <base-hex> <out.elf>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    const std::vector<uint8_t> image((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const uint32_t wantBase = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 16));
    if (image.size() < sizeof(Ehdr) || std::memcmp(image.data(), "\x7f" "ELF", 4) != 0)
    {
        std::fprintf(stderr, "not an ELF: %s\n", argv[1]);
        return 1;
    }

    // Place it where the emulator will: reserve the pool below the wanted base, then load with the real loader.
    IopMemory memory;
    if (wantBase > IopMemory::HeapBase && memory.allocate(wantBase - IopMemory::HeapBase, 16u, IopMemory::HeapBase) == 0u)
    {
        std::fprintf(stderr, "cannot reserve 0x%x..0x%x\n", IopMemory::HeapBase, wantBase);
        return 1;
    }
    const auto loaded = IopModuleLoader::load(image, memory, wantBase);
    if (!loaded || loaded.base != wantBase || !loaded.relocationsComplete)
    {
        std::fprintf(stderr, "load failed or landed at 0x%x (wanted 0x%x), relocations complete=%d\n", loaded.base,
                     wantBase, loaded.relocationsComplete ? 1 : 0);
        return 1;
    }
    const uint32_t base = loaded.base;
    const auto ram = memory.ram();

    // The input's sections, symbols and string table.
    const Ehdr eh = readAt<Ehdr>(image, 0);
    std::vector<Shdr> shdrs(eh.shnum);
    for (uint32_t i = 0; i < eh.shnum; ++i)
        shdrs[i] = readAt<Shdr>(image, eh.shoff + i * sizeof(Shdr));
    auto sectionName = [&](uint32_t index) -> std::string {
        if (eh.shstrndx >= shdrs.size())
            return {};
        const size_t at = shdrs[eh.shstrndx].offset + shdrs[index].name;
        return at < image.size() ? std::string(reinterpret_cast<const char *>(image.data() + at)) : std::string{};
    };

    // Output sections: [0] null, then every allocatable input section at base + addr, from the relocated RAM.
    std::vector<OutSection> out(1);
    std::vector<int> outIndexOf(shdrs.size(), 0);
    for (uint32_t i = 0; i < shdrs.size(); ++i)
    {
        const Shdr &s = shdrs[i];
        if ((s.flags & SHF_ALLOC) == 0u || s.size == 0u)
            continue;
        OutSection o;
        o.name = sectionName(i);
        o.header = s;
        o.header.addr = s.addr + base;
        o.header.link = o.header.info = 0u;
        if (s.type != SHT_NOBITS)
        {
            o.header.type = SHT_PROGBITS;
            o.data.assign(ram.begin() + o.header.addr, ram.begin() + o.header.addr + s.size);
        }
        outIndexOf[i] = static_cast<int>(out.size());
        out.push_back(std::move(o));
    }

    // Symbols: the module's own (shifted), then the import stubs.
    std::string strtab(1, '\0');
    std::vector<Sym> syms(1);
    for (uint32_t i = 0; i < shdrs.size(); ++i)
    {
        if (shdrs[i].type != SHT_SYMTAB || shdrs[i].entsize < sizeof(Sym))
            continue;
        const Shdr &names = shdrs[shdrs[i].link];
        for (uint32_t k = 1; k < shdrs[i].size / shdrs[i].entsize; ++k)
        {
            Sym s = readAt<Sym>(image, shdrs[i].offset + k * shdrs[i].entsize);
            if (s.shndx == 0u || s.shndx >= shdrs.size() || outIndexOf[s.shndx] == 0)
                continue;   // undefined / absolute / non-allocated: nothing to place
            const size_t at = names.offset + s.name;
            const std::string name = at < image.size() ? reinterpret_cast<const char *>(image.data() + at) : "";
            if (name.empty())
                continue;
            s.name = addString(strtab, name);
            s.value += base;
            s.shndx = static_cast<uint16_t>(outIndexOf[s.shndx]);
            syms.push_back(s);
        }
    }
    uint32_t importCount = 0u;
    for (size_t o = 1; o < out.size(); ++o)
    {
        if ((out[o].header.flags & SHF_EXECINSTR) == 0u || out[o].data.empty())
            continue;
        const std::vector<uint8_t> &d = out[o].data;
        for (size_t off = 0; off + 20u <= d.size(); off += 4u)
        {
            if (readAt<uint32_t>(d, off) != kImportMagic)
                continue;
            char library[9] = {};
            std::memcpy(library, d.data() + off + 12u, 8u);
            for (size_t stub = off + 20u; stub + 8u <= d.size(); stub += 8u)
            {
                const uint32_t jump = readAt<uint32_t>(d, stub), slot = readAt<uint32_t>(d, stub + 4u);
                if (jump == 0u && slot == 0u)
                    break;
                if (jump != 0x03E00008u || (slot & 0xFFFF0000u) != 0x24000000u)
                    break;
                Sym s{};
                s.name = addString(strtab, "__iopimport_" + std::string(library) + "_" + std::to_string(slot & 0xFFFFu));
                s.value = out[o].header.addr + static_cast<uint32_t>(stub);
                s.size = 8u;
                s.info = 0x12;   // STB_GLOBAL | STT_FUNC
                s.shndx = static_cast<uint16_t>(o);
                syms.push_back(s);
                ++importCount;
            }
        }
    }

    // A module without function symbols of its own (rotk LIBSD): seed function starts the way z3xox/BT3-Recomp's
    // tools/iop/irx_functions.py does -- the entry, the export table's function pointers, JAL targets, and lui+addiu/ori
    // pairs pointing into the code -- and let each function run to the next start or table.
    size_t ownFunctions = 0u;
    for (const Sym &s : syms)
        if ((s.info & 0xFu) == 2u && s.size != 0u)
            ++ownFunctions;
    if (ownFunctions == importCount)
    {
        std::vector<std::pair<uint32_t, uint32_t>> tables;   // [start, end) of import/export tables
        std::vector<uint32_t> starts;
        uint32_t codeLo = 0u, codeHi = 0u;
        size_t textIndex = 0u;
        for (size_t o = 1; o < out.size(); ++o)
        {
            if ((out[o].header.flags & SHF_EXECINSTR) == 0u || out[o].data.empty())
                continue;
            textIndex = o;
            const std::vector<uint8_t> &d = out[o].data;
            const uint32_t at = out[o].header.addr;
            codeLo = at;
            codeHi = at + static_cast<uint32_t>(d.size());
            for (size_t off = 0; off + 20u <= d.size(); off += 4u)
            {
                const uint32_t magic = readAt<uint32_t>(d, off);
                if (magic == kImportMagic)
                {
                    size_t end = off + 20u;
                    while (end + 8u <= d.size() && readAt<uint32_t>(d, end) != 0u)
                        end += 8u;
                    tables.emplace_back(at + static_cast<uint32_t>(off), at + static_cast<uint32_t>(end + 8u));
                }
                else if (magic == 0x41C00000u)   // export table: header, then function pointers up to a 0
                {
                    size_t end = off + 20u;
                    while (end + 4u <= d.size() && readAt<uint32_t>(d, end) != 0u)
                    {
                        starts.push_back(readAt<uint32_t>(d, end));
                        end += 4u;
                    }
                    tables.emplace_back(at + static_cast<uint32_t>(off), at + static_cast<uint32_t>(end + 4u));
                }
            }
            for (size_t off = 0; off + 4u <= d.size(); off += 4u)
            {
                const uint32_t word = readAt<uint32_t>(d, off);
                if ((word >> 26) == 3u)   // jal
                    starts.push_back(((at + static_cast<uint32_t>(off)) & 0xF0000000u) | ((word & 0x03FFFFFFu) << 2));
                if ((word >> 26) == 0x0Fu && off + 8u <= d.size())   // lui rt, hi ; addiu/ori rt, rt, lo (within 2 slots)
                {
                    const uint32_t rt = (word >> 16) & 31u;
                    for (size_t k = 4u; k <= 8u && off + k + 4u <= d.size(); k += 4u)
                    {
                        const uint32_t next = readAt<uint32_t>(d, off + k);
                        const uint32_t op = next >> 26;
                        if ((op == 0x09u || op == 0x0Du) && ((next >> 21) & 31u) == rt)
                        {
                            const uint32_t lo = next & 0xFFFFu;
                            const uint32_t value = (word << 16) + (op == 0x09u ? static_cast<uint32_t>(static_cast<int16_t>(lo)) : lo);
                            if ((value & 3u) == 0u)
                                starts.push_back(value);
                            break;
                        }
                    }
                }
            }
        }
        starts.push_back(loaded.entry);
        for (const Sym &s : syms)
            if ((s.info & 0xFu) == 2u && s.size == 8u)
                starts.push_back(s.value);   // the import stubs bound the functions before them
        auto inTable = [&](uint32_t a) {
            for (const auto &t : tables)
                if (a >= t.first && a < t.second)
                    return true;
            return false;
        };
        std::vector<uint32_t> bounds;
        for (uint32_t a : starts)
            if (a >= codeLo && a < codeHi && !inTable(a))
                bounds.push_back(a);
        for (const auto &t : tables)
            bounds.push_back(t.first);
        bounds.push_back(codeHi);
        std::sort(bounds.begin(), bounds.end());
        bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
        uint32_t seeded = 0u;
        for (size_t i = 0; i + 1u < bounds.size(); ++i)
        {
            const uint32_t a = bounds[i];
            bool isStub = inTable(a);
            for (const Sym &s : syms)
                if (s.value == a && s.size == 8u)
                    isStub = true;
            if (isStub)
                continue;
            Sym s{};
            char name[32];
            std::snprintf(name, sizeof(name), "FUN_%08x", a);
            s.name = addString(strtab, name);
            s.value = a;
            s.size = bounds[i + 1u] - a;
            s.info = 0x12;
            s.shndx = static_cast<uint16_t>(textIndex);
            syms.push_back(s);
            ++seeded;
        }
        std::printf("no function symbols: seeded %u functions\n", seeded);
    }

    // Layout: ehdr, phdr, section data, symtab, strtab, shstrtab, section headers.
    std::string shstrtab(1, '\0');
    std::vector<uint8_t> file(sizeof(Ehdr) + sizeof(Phdr), 0u);
    auto append = [&](const void *p, size_t n) -> uint32_t {
        while (file.size() % 16u)
            file.push_back(0u);
        const uint32_t at = static_cast<uint32_t>(file.size());
        const uint8_t *b = static_cast<const uint8_t *>(p);
        file.insert(file.end(), b, b + n);
        return at;
    };
    for (size_t o = 1; o < out.size(); ++o)
    {
        out[o].header.name = addString(shstrtab, out[o].name);
        out[o].header.offset = out[o].data.empty() ? static_cast<uint32_t>(file.size()) : append(out[o].data.data(), out[o].data.size());
    }
    OutSection symtab, strsec, shstr;
    symtab.header.type = SHT_SYMTAB;
    symtab.header.entsize = sizeof(Sym);
    symtab.header.addralign = 4u;
    symtab.header.info = 1u;
    symtab.header.offset = append(syms.data(), syms.size() * sizeof(Sym));
    symtab.header.size = static_cast<uint32_t>(syms.size() * sizeof(Sym));
    symtab.header.link = static_cast<uint32_t>(out.size() + 1u);
    symtab.header.name = addString(shstrtab, ".symtab");
    strsec.header.type = SHT_STRTAB;
    strsec.header.offset = append(strtab.data(), strtab.size());
    strsec.header.size = static_cast<uint32_t>(strtab.size());
    strsec.header.name = addString(shstrtab, ".strtab");
    shstr.header.type = SHT_STRTAB;
    shstr.header.name = addString(shstrtab, ".shstrtab");
    shstr.header.offset = append(shstrtab.data(), shstrtab.size());
    shstr.header.size = static_cast<uint32_t>(shstrtab.size());
    out.push_back(symtab);
    out.push_back(strsec);
    out.push_back(shstr);

    std::vector<Shdr> headers;
    for (const auto &o : out)
        headers.push_back(o.header);
    const uint32_t shoff = append(headers.data(), headers.size() * sizeof(Shdr));

    Ehdr oh{};
    std::memcpy(oh.ident, "\x7f" "ELF\x01\x01\x01", 7);
    oh.type = 2;      // ET_EXEC
    oh.machine = 8;   // EM_MIPS
    oh.version = 1;
    oh.entry = loaded.entry;
    oh.phoff = sizeof(Ehdr);
    oh.shoff = shoff;
    oh.ehsize = sizeof(Ehdr);
    oh.phentsize = sizeof(Phdr);
    oh.phnum = 1;
    oh.shentsize = sizeof(Shdr);
    oh.shnum = static_cast<uint16_t>(headers.size());
    oh.shstrndx = static_cast<uint16_t>(headers.size() - 1u);
    std::memcpy(file.data(), &oh, sizeof(oh));

    // One PT_LOAD covering the image (file bytes from the relocated RAM, so its contents match the sections).
    const uint32_t imageAt = append(ram.data() + base, loaded.size);
    Phdr ph{};
    ph.type = 1;
    ph.offset = imageAt;
    ph.vaddr = ph.paddr = base;
    ph.filesz = ph.memsz = loaded.size;
    ph.flags = 7;
    ph.align = 0x10;
    std::memcpy(file.data() + sizeof(Ehdr), &ph, sizeof(ph));

    std::ofstream o(argv[3], std::ios::binary);
    o.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamsize>(file.size()));
    uint64_t hash = 1469598103934665603ull;   // FNV-1a 64 of the IRX file: what ps2x::iop::native matches a loaded module by
    for (uint8_t b : image)
        hash = (hash ^ b) * 1099511628211ull;
    std::printf("base=0x%x size=0x%x entry=0x%x gp=0x%x symbols=%zu imports=%u hash=0x%016llx\n", base, loaded.size,
                loaded.entry, loaded.gp, syms.size() - 1u, importCount, static_cast<unsigned long long>(hash));
    return o ? 0 : 1;
}
