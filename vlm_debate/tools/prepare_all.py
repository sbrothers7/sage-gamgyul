r"""
One-shot setup: everything needed before the real experiment.

    py tools\prepare_all.py            (4_prepare_all.bat runs this)

Steps (each is skipped if already done, so re-running is safe):
  1) pip: pillow, openai
  2) Ollama: install (winget -> official installer), start server
  3) Models: moondream, gemma3:4b, qwen2.5vl:7b  (RTX 3070 8GB plan)
  4) Data (in parallel with 3):
       A  Mendeley 3f83gxmv57 v2  (leaves)
       B  Zenodo 8294078 CitrusUAT
     stored OUTSIDE OneDrive: D:\citrus-data  (or C:\citrus-data if no D:)
  5) data/*.json specs + .env
  6) dedup A vs B
  7) selftest, bench, 10-image pilot run
Everything is logged to logs\prepare_all.log
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys, threading, time, traceback, zipfile
import urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGD = ROOT / 'logs'; LOGD.mkdir(exist_ok=True)
LOG = LOGD / 'prepare_all.log'
STATE = LOGD / 'prepare_state.json'
PY = sys.executable
MODELS = ['moondream', 'gemma3:4b', 'qwen2.5vl:7b']
HOST = 'http://localhost:11434'
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) citrus-vlm-setup'}
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

DATA = Path('D:/citrus-data') if Path('D:/').exists() else Path('C:/citrus-data')

_lk = threading.Lock()
_logf = open(LOG, 'a', encoding='utf-8')
state = {}


def log(msg, tag='main'):
    line = f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}"
    with _lk:
        try:
            print(line, flush=True)
        except Exception:
            print(line.encode('ascii', 'replace').decode(), flush=True)
        _logf.write(line + '\n'); _logf.flush()


def save_state(**kw):
    with _lk:
        state.update(kw)
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding='utf-8')


def sh(cmd, tag, timeout=None, cwd=None):
    """run a command, stream output into the log, return exit code"""
    log('$ ' + (cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))), tag)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    try:
        p = subprocess.Popen(cmd, cwd=cwd or ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             shell=isinstance(cmd, str))
    except Exception as ex:
        log(f'cannot start: {ex}', tag); return -1
    t0 = time.time()
    for raw in iter(p.stdout.readline, b''):
        for enc in ('utf-8', 'cp949'):
            try:
                s = raw.decode(enc); break
            except UnicodeDecodeError:
                s = raw.decode('utf-8', 'replace')
        s = s.rstrip()
        if s:
            log('  ' + s[-400:], tag)
        if timeout and time.time() - t0 > timeout:
            p.kill(); log('TIMEOUT', tag); return -9
    return p.wait()


# ── 1. pip ───────────────────────────────────────────────────────────────
def step_pip():
    rc = sh([PY, '-m', 'pip', 'install', '--disable-pip-version-check', '-q',
             'pillow', 'openai'], 'pip')
    save_state(pip=rc == 0)


# ── 2. Ollama ────────────────────────────────────────────────────────────
def ollama_exe():
    w = shutil.which('ollama')
    if w: return w
    for c in (Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Ollama/ollama.exe',
              Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Ollama/ollama.exe'):
        if c.exists(): return str(c)
    return None


def serving():
    try:
        with urllib.request.urlopen(HOST + '/api/tags', timeout=4) as r:
            return [m['name'] for m in json.loads(r.read()).get('models', [])]
    except Exception:
        return None


def download(url, dst: Path, tag, expect_size=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and expect_size and dst.stat().st_size == expect_size:
        log(f'already downloaded: {dst.name}', tag); return True
    tmp = dst.with_suffix(dst.suffix + '.part')
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, 'wb') as f:
                total = int(r.headers.get('Content-Length') or 0) or expect_size or 0
                got, last = 0, -10
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); got += len(b)
                    pct = got * 100 // total if total else -1
                    if total and pct >= last + 10:
                        last = pct; log(f'{dst.name}: {pct}% ({got >> 20} MB)', tag)
            if expect_size and tmp.stat().st_size != expect_size:
                raise IOError(f'size mismatch {tmp.stat().st_size} != {expect_size}')
            os.replace(tmp, dst)
            log(f'downloaded {dst.name} ({dst.stat().st_size >> 20} MB)', tag)
            return True
        except Exception as ex:
            log(f'download failed ({attempt + 1}/3) {url}: {ex}', tag)
            time.sleep(5)
    return False


def step_ollama():
    exe = ollama_exe()
    if not exe:
        log('Ollama not installed -> winget', 'ollama')
        sh('winget install -e --id Ollama.Ollama --silent --accept-package-agreements '
           '--accept-source-agreements', 'ollama', timeout=1800)
        exe = ollama_exe()
    if not exe:
        log('winget did not work -> official installer', 'ollama')
        inst = Path(os.environ.get('TEMP', str(ROOT))) / 'OllamaSetup.exe'
        if download('https://ollama.com/download/OllamaSetup.exe', inst, 'ollama'):
            sh([str(inst), '/VERYSILENT', '/NORESTART', '/SUPPRESSMSGBOXES'], 'ollama', timeout=1800)
        exe = ollama_exe()
    if not exe:
        save_state(ollama='install_failed'); raise RuntimeError('Ollama install failed')
    log(f'ollama at {exe}', 'ollama')
    if serving() is None:
        log('starting ollama serve', 'ollama')
        flags = 0x00000008 | 0x00000200   # DETACHED_PROCESS | NEW_PROCESS_GROUP
        subprocess.Popen([exe, 'serve'], creationflags=flags, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        for _ in range(60):
            if serving() is not None: break
            time.sleep(1)
    tags = serving()
    if tags is None:
        save_state(ollama='not_serving'); raise RuntimeError('Ollama server not responding')
    save_state(ollama='ok', ollama_exe=exe)
    log(f'server ok, models: {tags}', 'ollama')


# ── 3. models ────────────────────────────────────────────────────────────
def pull(name):
    have = serving() or []
    if any(h == name or h == name + ':latest' for h in have):
        log(f'{name}: already present', 'model'); return True
    body = json.dumps({'model': name, 'stream': True}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(HOST + '/api/pull', data=body,
                                         headers={'Content-Type': 'application/json'})
            last_status, last_pct = None, -10
            with urllib.request.urlopen(req, timeout=600) as r:
                for line in r:
                    try: ev = json.loads(line)
                    except Exception: continue
                    if 'error' in ev: raise RuntimeError(ev['error'])
                    st = ev.get('status', '')
                    if ev.get('total'):
                        pct = ev.get('completed', 0) * 100 // ev['total']
                        if pct >= last_pct + 20 or st != last_status:
                            log(f"{name}: {st[:30]} {pct}% of {ev['total'] >> 20} MB", 'model')
                            last_pct = pct
                    elif st != last_status:
                        log(f'{name}: {st}', 'model')
                    last_status = st
            if last_status == 'success': return True
        except Exception as ex:
            log(f'{name}: pull error ({attempt + 1}/3): {ex}', 'model'); time.sleep(10)
    return False


def step_models():
    ok = [m for m in MODELS if pull(m)]
    save_state(models_ok=ok, models_failed=[m for m in MODELS if m not in ok])
    return ok


# ── 4. data ──────────────────────────────────────────────────────────────
def extract(zp: Path, dst: Path, tag):
    mark = dst / '.extracted'
    if mark.exists(): return
    log(f'extracting {zp.name} -> {dst}', tag)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(dst)
    # nested zips (Mendeley often zips folders inside the zip)
    for inner in list(dst.rglob('*.zip')):
        sub = inner.with_suffix('')
        if not (sub / '.extracted').exists():
            log(f'  inner zip {inner.name}', tag)
            try:
                with zipfile.ZipFile(inner) as z: z.extractall(sub)
                (sub / '.extracted').write_text('ok')
            except zipfile.BadZipFile:
                log(f'  bad inner zip {inner}', tag)
    mark.write_text('ok')


def get_zenodo():
    tag, dst = 'zenodo', DATA / 'zenodo_8294078'
    try:
        req = urllib.request.Request('https://zenodo.org/api/records/8294078', headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            rec = json.loads(r.read())
    except Exception as ex:
        log(f'record lookup failed: {ex}', tag); return None
    meta = rec.get('metadata', {})
    log(f"title: {meta.get('title')} | license: {meta.get('license')}", tag)
    files = rec.get('files') or []
    if isinstance(files, dict): files = files.get('entries', [])
    for f in files:
        key = f.get('key') or f.get('filename')
        url = (f.get('links') or {}).get('self') or (f.get('links') or {}).get('content')
        size = f.get('size') or f.get('filesize')
        log(f'file {key} {int(size or 0) >> 20} MB', tag)
        target = dst / '_download' / key
        if not download(url, target, tag, size): return None
        if key.lower().endswith('.zip'):
            extract(target, dst / Path(key).stem, tag)
    return dst


def find_local_mendeley():
    """already downloaded by hand? look in Downloads / D:"""
    home = Path.home()
    cands = []
    for base in (home / 'Downloads', Path('D:/'), home / 'OneDrive/Desktop', home / 'Desktop'):
        if base.exists():
            try:
                cands += list(base.glob('*3f83gxmv57*.zip'))
                cands += list(base.glob('*[Cc]itrus*[Ll]eaves*.zip'))
            except Exception:
                pass
    return cands[0] if cands else None


def get_mendeley():
    tag, dst = 'mendeley', DATA / 'mendeley_3f83gxmv57_v2'
    zp = dst / '_download' / '3f83gxmv57-2.zip'
    if (dst / 'x' / '.extracted').exists():
        return dst / 'x'
    ok = zp.exists() and zipfile.is_zipfile(zp)
    urls = ['https://data.mendeley.com/public-api/zip/3f83gxmv57/download/2',
            'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/3f83gxmv57-2.zip']
    for u in urls:
        if ok: break
        if download(u, zp, tag) and zipfile.is_zipfile(zp):
            ok = True
        else:
            log(f'not a usable zip from {u}', tag)
            if zp.exists(): zp.unlink()
    if not ok:
        local = find_local_mendeley()
        if local:
            log(f'using local file {local}', tag); zp = local; ok = True
    if not ok:
        log('MENDELEY DOWNLOAD FAILED - manual download needed', tag); return None
    extract(zp, dst / 'x', tag)
    return dst / 'x'


def tree(root: Path, tag, depth=4):
    """log folder structure with image counts (so Claude can check it)"""
    def walk(d, lv):
        if lv > depth: return
        try: subs = sorted([s for s in d.iterdir() if s.is_dir()])
        except Exception: return
        for s in subs:
            n = sum(1 for f in s.rglob('*') if f.suffix.lower() in IMG_EXT)
            log(f"{'  ' * lv}{s.name}/  {n} img", tag)
            walk(s, lv + 1)
    log(f'tree of {root}', tag); walk(root, 0)


def leaf_class_dirs(root: Path):
    """directories that directly contain images"""
    out = []
    for d in [root] + [p for p in root.rglob('*') if p.is_dir()]:
        try:
            if any(f.suffix.lower() in IMG_EXT for f in d.iterdir() if f.is_file()):
                out.append(d)
        except Exception:
            pass
    return out


def build_spec(root: Path, out_json: Path, source: str, tag, only_under=None):
    sys.path.insert(0, str(ROOT / 'src'))
    from prepare_dataset import map_disease
    classes, counts = {}, {}
    for d in leaf_class_dirs(root):
        if only_under and not any(p.name.lower() in only_under for p in [d] + list(d.parents)):
            continue
        name = map_disease(d.name)
        classes.setdefault(name, []).append(str(d))
        counts[name] = counts.get(name, 0) + sum(1 for f in d.iterdir()
                                                 if f.suffix.lower() in IMG_EXT)
    spec = {'crop': 'Citrus', 'source': source, 'counts': counts, 'classes': classes}
    out_json.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    log(f'spec {out_json.name}: ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())), tag)
    return spec


def step_data():
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(DATA).free >> 30
        log(f'data folder {DATA} (free {free} GB)', 'data')
        a = get_mendeley()
        if a:
            tree(a, 'mendeley')
            build_spec(a, ROOT / 'data' / 'citrusB_mendeley.json',
                       'Mendeley 3f83gxmv57 v2 (Rauf et al., Sargodha) - leaves only',
                       'mendeley', only_under=('leaves', 'leaf'))
        b = get_zenodo()
        if b:
            tree(b, 'zenodo')
            build_spec(b, ROOT / 'data' / 'citrus_zenodo.json',
                       'Zenodo 8294078 CitrusUAT (C. sinensis)', 'zenodo')
        save_state(data_root=str(DATA), mendeley=str(a) if a else None,
                   zenodo=str(b) if b else None)
    except Exception:
        log(traceback.format_exc(), 'data'); save_state(data_error=True)


# ── 5. .env ──────────────────────────────────────────────────────────────
def set_env(pairs):
    env = ROOT / '.env'
    if not env.exists():
        shutil.copy(ROOT / '.env.example', env)
    t = None
    for enc in ('utf-8-sig', 'utf-16', 'cp949', 'utf-8'):
        try:
            t = env.read_text(encoding=enc)
            if '\x00' not in t: break
        except (UnicodeDecodeError, UnicodeError):
            pass
    t = t or ''
    for k, v in pairs.items():
        line = f'{k}={v}'
        if re.search(rf'(?m)^\s*{k}\s*=.*$', t):
            t = re.sub(rf'(?m)^\s*{k}\s*=.*$', lambda m: line, t)
        else:
            t = t.rstrip('\n') + '\n' + line + '\n'
    env.write_text(t, encoding='utf-8')
    log('.env: ' + ', '.join(f'{k}={v}' for k, v in pairs.items()))


# ══════════════════════════════════════════════════════════════════════════
def main():
    log('=' * 60)
    log(f'prepare_all start  python={sys.version.split()[0]}  root={ROOT}')
    sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader', 'gpu')
    step_pip()

    data_t = threading.Thread(target=step_data, daemon=True)
    data_t.start()
    models = []
    try:
        step_ollama()
        models = step_models()
    except Exception as ex:
        log(f'OLLAMA/MODEL STEP FAILED: {ex}')
    data_t.join()

    if models:
        set_env({'OLLAMA_MODELS': ','.join(models), 'CDDM_ROOT': 'data/citrusB_mendeley.json'})

    if state.get('mendeley') and state.get('zenodo'):
        leaves = [p for p in Path(state['mendeley']).rglob('*')
                  if p.is_dir() and p.name.lower() in ('leaves', 'leaf')]
        a = leaves[0] if leaves else Path(state['mendeley'])
        rc = sh([PY, 'src/prepare_dataset.py', 'dedup', str(a), state['zenodo'],
                 '--out', 'results/overlap_mendeley_zenodo.json'], 'dedup')
        save_state(dedup_rc=rc)

    rc = sh([PY, 'tools/selftest/make_fake_data.py'], 'selftest')
    rc = rc or sh([PY, 'tools/selftest/run_selftest.py'], 'selftest')
    save_state(selftest_rc=rc)

    if models and state.get('mendeley'):
        rc = sh([PY, 'tools/bench_models.py', '--images', '3'], 'bench', timeout=3600)
        save_state(bench_rc=rc)
        rc = sh([PY, 'tools/run_experiment.py', '--n', '10', '--tag', 'pilot10_ev-on'],
                'pilot', timeout=4 * 3600)
        save_state(pilot_rc=rc)

    save_state(finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    log('ALL DONE. Claude will read logs\\prepare_all.log')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        log(traceback.format_exc())
