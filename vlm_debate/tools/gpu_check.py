r"""Check whether Ollama actually puts the model on the GPU.  exit 0 = GPU, 1 = CPU/none"""
import json, os, sys, time, subprocess, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / 'logs' / 'gpu_fix.log'; LOG.parent.mkdir(exist_ok=True)
tag = sys.argv[1] if len(sys.argv) > 1 else '?'
H = 'http://localhost:11434'


def log(s):
    print(s, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(s + '\n')


def api(p, body=None, t=300):
    r = urllib.request.Request(H + p, data=json.dumps(body).encode() if body else None,
                               headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=t) as x: return json.loads(x.read())


log(f'\n===== [{tag}] {time.strftime("%H:%M:%S")} =====')
for _ in range(40):
    try: api('/api/tags', t=3); break
    except Exception: time.sleep(1)
else:
    log('server not responding'); sys.exit(2)
try:
    log('version ' + str(api('/api/version')))
    r = api('/api/generate', {'model': 'gemma3:4b', 'prompt': 'Say hi.', 'stream': False,
                              'options': {'num_predict': 40}, 'keep_alive': '60s'})
    tps = r.get('eval_count', 0) / max(r.get('eval_duration', 1) / 1e9, 1e-6)
    ps = api('/api/ps').get('models', [])
    for m in ps:
        log(f"ps: {m['name']} size={m.get('size', 0) >> 20}MB size_vram={m.get('size_vram', 0) >> 20}MB")
    smi = subprocess.run('nvidia-smi --query-gpu=memory.used --format=csv,noheader',
                         shell=True, capture_output=True, text=True).stdout.strip()
    log(f'gen speed {tps:.1f} tok/s | nvidia-smi used {smi}')
    vram = sum(m.get('size_vram', 0) for m in ps)
    ok = vram > 0
    log('RESULT: ' + ('GPU OK' if ok else 'CPU ONLY'))
except Exception as ex:
    log(f'error {ex}'); ok = False

# discovery lines from both log sources
for lf in [Path(os.environ.get('LOCALAPPDATA', '')) / 'Ollama' / 'server.log' if os.name == 'nt'
           else Path.home() / '.ollama' / 'logs' / 'server.log',
           ROOT / 'logs' / f'serve_{tag}.log']:
    if lf.exists():
        L = lf.read_text(encoding='utf-8', errors='replace').splitlines()
        keys = ('inference compute', 'discover', 'gpu', 'cuda', 'nvml', 'nvidia', 'library',
                'no compatible', 'error', 'fail', 'vram', 'offload')
        hits = [l for l in L if any(k in l.lower() for k in keys)]
        log(f'--- {lf.name}: {len(hits)} matching lines, last 25 ---')
        for l in hits[-25:]: log('  ' + l[:260])
sys.exit(0 if ok else 1)
