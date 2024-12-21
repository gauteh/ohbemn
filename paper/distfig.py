#!/usr/bin/env python

import sys
import shutil
from pathlib import Path

texf = Path(sys.argv[1])
outd = Path(sys.argv[2])

assert outd.is_dir()

print(f'Distributing figures and TeX for {texf}.. to {outd}..')

flsf = texf.parent / (texf.stem + '.fls')

figexts = ['.jpg', '.png', '.pdf']
figi = 1
fmap = {}

for line in open(flsf).readlines():
    if 'INPUT' in line:
        p = Path(line.split(' ', maxsplit=1)[1].strip())
        if p.suffix in figexts:
            if not fmap.get(p):
                fmap[p] = f'Fig{figi:02d}{p.suffix}'
                print(f'{p} -> {fmap[p]}')
                shutil.copy(p, (outd / fmap[p]))
                figi += 1

# Re-write TeX
tex = open(texf).read()

for old, new in fmap.items():
    tex = tex.replace(str(old), new)

newtexf = outd / texf.name
print(f'Writing new TeX to: {newtexf}..')
open(newtexf, 'wt').write(tex)

