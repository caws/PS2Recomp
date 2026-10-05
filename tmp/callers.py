#!/usr/bin/env python3
"""callers.py <hexaddr> [depth] : walk UP the static call graph in tmp/generated/*.cpp."""
import sys, os, re, subprocess
GEN='/home/caws/Documents/projects/decompilations/rotk_decomp/tmp/generated'
def callers(addr):
    out=subprocess.run(['grep','-rli',f'0x{addr}u',GEN],capture_output=True,text=True).stdout.split()
    res=set()
    for f in out:
        b=os.path.basename(f)
        if 'register_functions' in b: continue
        m=re.search(r'_0x([0-9a-f]+)\.cpp$',b)
        if m and m.group(1)!=addr: res.add(m.group(1))
    return sorted(res)
start=sys.argv[1].lower(); depth=int(sys.argv[2]) if len(sys.argv)>2 else 4
seen={start}; level=[start]
for d in range(depth):
    nxt=[]
    for a in level:
        cs=callers(a)
        print(f"{'  '*d}0x{a} <- " + (", ".join('0x'+c for c in cs) if cs else "(no static caller)"))
        for c in cs:
            if c not in seen: seen.add(c); nxt.append(c)
    if not nxt: break
    level=nxt
