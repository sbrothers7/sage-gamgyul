r"""Find DLLs on PATH that may crash Ollama's GPU discovery (0xc0000005). -> logs\gpu_fix.log"""
import os, fnmatch, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / 'logs' / 'gpu_fix.log'; LOG.parent.mkdir(exist_ok=True)
PAT = ['msvcp140*.dll', 'vcruntime140*.dll', 'cudart*.dll', 'cublas*.dll', 'ggml*.dll',
       'llama*.dll', 'libomp*.dll', 'nvcuda*.dll', 'vulkan-1*.dll', 'amdhip*.dll',
       'concrt140*.dll', 'libiomp*.dll', 'mkl*.dll']
with open(LOG, 'a', encoding='utf-8') as f:
    def w(s): print(s); f.write(s + '\n')
    w(f'\n===== PATH scan {time.strftime("%H:%M:%S")} =====')
    for d in os.environ.get('PATH', '').split(';'):
        d = d.strip()
        if not d: continue
        p = Path(d)
        if not p.is_dir(): w(f'  (missing) {d}'); continue
        try: names = os.listdir(p)
        except Exception: w(f'  (no access) {d}'); continue
        hits = [n for n in names if any(fnmatch.fnmatch(n.lower(), x) for x in PAT)]
        w(f'  {d}' + (f'   <-- {", ".join(sorted(hits)[:8])}' if hits else ''))
    sysdir = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32'
    for n in ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'nvcuda.dll'):
        f2 = sysdir / n
        w(f'  system32 {n}: ' + (f'{f2.stat().st_size} B, {time.strftime("%Y-%m-%d", time.localtime(f2.stat().st_mtime))}' if f2.exists() else 'MISSING'))
