r"""
Headless experiment run (same pipeline as the web console, no browser).

    py tools\run_experiment.py --n 10 --tag pilot
    py tools\run_experiment.py --n 30 --no-evidence --tag ablation_off

Results go to results\<date>_<tag>\ (CSV, transcripts, summary.json, run_config.json)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from bench_models import load_env          # also loads .env on import
import vp_core as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.getenv('CDDM_ROOT', 'data/citrusB_mendeley.json'))
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--temp', type=float, default=2.0)
    ap.add_argument('--debate', type=int, default=3)
    ap.add_argument('--agents', default=os.getenv('OLLAMA_MODELS', ''))
    ap.add_argument('--no-evidence', action='store_true', help='anti-sycophancy OFF')
    ap.add_argument('--enforce', action='store_true',
                    help='anti-sycophancy HARD: changes without new evidence are reverted in code')
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--tag', default='run')
    ap.add_argument('--reuse-r1', default='', help='vida_raw_bank.json of an earlier run (same seed/n)')
    a = ap.parse_args()

    ds = Path(a.dataset)
    if not ds.is_absolute(): ds = ROOT / ds
    out = ROOT / 'results' / f"{time.strftime('%Y%m%d_%H%M')}_{a.tag}"
    agents = [x.strip() for x in a.agents.split(',') if x.strip()] or None
    if agents and not a.mock:
        # Ollama lists 'moondream' as 'moondream:latest' -> match to real names
        avail = list(C.Providers().list_agents())
        fixed = []
        for x in agents:
            m = [v for v in avail if v == x or v == x + ':latest']
            if not m:
                sys.exit(f'agent not available: {x}  (available: {avail})')
            fixed.append(m[0])
        agents = fixed
    cfg = dict(C.DEFAULT_CFG, dataset_root=str(ds), output_dir=str(out), n_images=a.n,
               n_debate=a.debate, seed=a.seed, softmax_temp=a.temp, mock=a.mock,
               use_judge=False, require_new_evidence=not a.no_evidence, agents=agents,
               vida_cache=a.reuse_r1 or None, enforce_evidence=a.enforce)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(cfg, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'run_config.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
    seen = {'pct': -10}

    class R(C.Runner):
        def log(self, msg):
            print(msg, flush=True)
            super().log(msg)

    def ev(**st):
        if st.get('total'):
            pct = st['done'] * 100 // st['total']
            if pct < seen['pct']: seen['pct'] = -10        # new phase
            if pct >= seen['pct'] + 10:
                seen['pct'] = pct
                print(f"  .. {st.get('phase')} {st['done']}/{st['total']}", flush=True)

    # keep the computer awake while running (no permanent setting change)
    try:
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        elif sys.platform == 'darwin':
            import subprocess
            subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    except Exception:
        pass
    r = R(cfg, on_event=ev)
    r.run()
    s = r.state.get('summary') or {}
    print(json.dumps(s, ensure_ascii=False)[:2000])
    print(f'phase={r.state.get("phase")}  out={out}')
    sys.exit(0 if r.state.get('phase') == 'done' else 1)


if __name__ == '__main__':
    main()
