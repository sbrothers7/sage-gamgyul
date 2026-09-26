r"""
Diagnose: why are local models slow, which models follow the answer format,
and how are Zenodo CitrusUAT labels stored.   Output: logs\diagnose.log
"""
import base64, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
LOG = ROOT / 'logs' / 'diagnose.log'; LOG.parent.mkdir(exist_ok=True)
_f = open(LOG, 'w', encoding='utf-8')
HOST = 'http://localhost:11434'
CANDIDATES = ['moondream', 'gemma3:4b', 'qwen2.5vl:7b',
              'qwen2.5vl:3b', 'llava:7b', 'granite3.2-vision']


def log(*a):
    s = ' '.join(str(x) for x in a)
    try: print(s, flush=True)
    except Exception: print(s.encode('ascii', 'replace').decode(), flush=True)
    _f.write(s + '\n'); _f.flush()


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
        out = r.stdout or r.stderr
        for enc in ('utf-8', 'cp949'):
            try: return out.decode(enc)
            except UnicodeDecodeError: pass
        return out.decode('utf-8', 'replace')
    except Exception as ex:
        return f'ERR {ex}'


def api(path, body=None, timeout=900):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode() if body else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def tags():
    return [m['name'] for m in json.loads(api('/api/tags')).get('models', [])]


def pull(name):
    if any(t in (name, name + ':latest') for t in tags()): return True
    log(f'  pulling {name} ...')
    try:
        last = ''
        req = urllib.request.Request(HOST + '/api/pull', data=json.dumps({'model': name}).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                ev = json.loads(line)
                if 'error' in ev: log('  pull error', ev['error']); return False
                last = ev.get('status', last)
        log(f'  {name}: {last}'); return last == 'success'
    except Exception as ex:
        log('  pull failed', ex); return False


def main():
    import vp_core as C
    from PIL import Image
    log('=' * 70); log('diagnose', time.strftime('%Y-%m-%d %H:%M:%S'))
    log(sh('ollama --version').strip())
    log(sh('nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv'))

    # ── images ──
    spec = ROOT / 'data' / 'citrusB_mendeley.json'
    pilot, _, tax = C.load_dataset(spec, n_images=5, seed=7)
    for s in pilot:
        im = Image.open(s['image_path'])
        log(f"img {Path(s['image_path']).name} {im.size} {os.path.getsize(s['image_path']) >> 10} KB  gt={s['gt_disease']}")
    P = C.build_prompts(tax)

    # ── models ──
    for m in CANDIDATES:
        if not pull(m): continue
    raw = {}
    log('\n' + '=' * 70)
    log(f"{'model':<22}{'load s':>8}{'prompt tok':>11}{'gen tok':>9}{'tok/s':>8}{'total s':>9}  parse  processor")
    for m in [t for t in tags()]:
        raw[m] = []
        for i, s in enumerate(pilot[:3]):
            b64 = base64.b64encode(open(s['image_path'], 'rb').read()).decode()
            t0 = time.time()
            try:
                r = json.loads(api('/api/chat', {'model': m, 'stream': False,
                        'messages': [{'role': 'user', 'content': P['r1'], 'images': [b64]}],
                        'options': {'temperature': 0.2, 'num_predict': 700}}))
            except Exception as ex:
                log(f'{m}: ERROR {ex}'); break
            dt = time.time() - t0
            txt = r.get('message', {}).get('content', '')
            p = C.parse_response(txt, m, tax)
            ok = bool(p['crop_category'] and p['disease_category'])
            ev, evd = r.get('eval_count', 0), r.get('eval_duration', 1) / 1e9
            ps = ''
            if i == 0:
                ps = ' '.join(sh('ollama ps').strip().splitlines()[1:2])
            log(f"{m:<22}{r.get('load_duration', 0) / 1e9:>8.1f}{r.get('prompt_eval_count', 0):>11}"
                f"{ev:>9}{ev / max(evd, 1e-6):>8.1f}{dt:>9.1f}  {'OK ' if ok else 'BAD'}  "
                f"pred={p['disease_category']} gt={s['gt_disease']}  {ps}")
            raw[m].append({'gt': s['gt_disease'], 'text': txt})
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'diag_raw.json').write_text(json.dumps(raw, ensure_ascii=False, indent=1),
                                                     encoding='utf-8')
    for m, rs in raw.items():
        if rs:
            log(f'\n--- {m} sample answer ---\n' + rs[0]['text'][:500])
    log(sh('nvidia-smi --query-gpu=memory.used,memory.total --format=csv'))

    # ── Ollama server log (GPU offload lines) ──
    sl = Path(os.environ.get('LOCALAPPDATA', '')) / 'Ollama' / 'server.log'
    if sl.exists():
        lines = sl.read_text(encoding='utf-8', errors='replace').splitlines()
        keys = ('offload', 'library=', 'inference compute', 'layers', 'vram', 'cuda', 'CPU')
        hits = [l for l in lines if any(k.lower() in l.lower() for k in keys)]
        log('\n--- server.log (gpu lines, last 40) ---'); [log(l[:300]) for l in hits[-40:]]

    # ── Zenodo labels ──
    z = Path('C:/citrus-data/zenodo_8294078')
    log('\n--- zenodo non-image files ---')
    for f in sorted(z.rglob('*')):
        if f.is_file() and f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.zip', '.part'):
            log(f'{f}  {f.stat().st_size >> 10} KB')
            if f.suffix.lower() in ('.csv', '.txt', '.json', '.md', '.tsv'):
                try: log('   ' + f.read_text(encoding='utf-8', errors='replace')[:1500].replace('\n', '\n   '))
                except Exception as ex: log('   read err', ex)
            if f.suffix.lower() in ('.xlsx', '.xls'):
                log('   (excel)')
    imgs = z / 'CitrusUAT_dataset' / 'CitrusUAT_dataset' / 'Images'
    if imgs.exists():
        names = sorted(p.name for p in imgs.iterdir())
        log(f'\nImages: {len(names)} files; first 40 / every 25th:')
        for n in names[:40]: log('  ' + n)
        for n in names[::25]: log('  ' + n)
        subd = [p for p in imgs.iterdir() if p.is_dir()]
        log('subdirs:', [p.name for p in subd])
        if names:
            im = Image.open(next(p for p in imgs.iterdir() if p.is_file()))
            log('size', im.size)
    log('\nDONE')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback; log(traceback.format_exc())
