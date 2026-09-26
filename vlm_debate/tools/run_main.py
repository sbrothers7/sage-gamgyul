r"""
Main experiment (anti-sycophancy ablation), meant to run overnight.

  1) ev-ON  : n images, anti-sycophancy ON
  2) ev-OFF : same images, R1 answers REUSED from run 1, anti-sycophancy OFF
     -> the only difference between the two runs is the debate rule.
Log: logs\main.log     Results: results\<date>_main<n>_ev-on / _ev-off
"""
import argparse, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / 'logs' / 'main.log'


def log(s):
    try: print(s, flush=True)
    except Exception: print(s.encode('ascii', 'replace').decode(), flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(s + '\n')


def run(args):
    log(f"\n$ run_experiment {' '.join(args)}   [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    p = subprocess.Popen([sys.executable, '-u', 'tools/run_experiment.py'] + args, cwd=ROOT,
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_dir = None
    for line in p.stdout:
        s = line.decode('utf-8', 'replace').rstrip()
        log('  ' + s)
        if 'out=' in s and 'phase=' in s:
            out_dir = s.split('out=', 1)[1].strip()
    rc = p.wait()
    log(f'rc={rc}  [{time.strftime("%Y-%m-%d %H:%M:%S")}]')
    return rc, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/citrusA_mendeley_4cls.json')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    base = ['--dataset', a.dataset, '--n', str(a.n), '--seed', str(a.seed)]
    name = Path(a.dataset).stem.split('_')[0]
    log('=' * 70)
    log(f'MAIN START {time.strftime("%Y-%m-%d %H:%M:%S")}  dataset={a.dataset} n={a.n} seed={a.seed}')
    rc, on_dir = run(base + ['--tag', f'main{a.n}_{name}_ev-on'])
    if rc != 0 or not on_dir:
        log('ev-ON failed - stopping'); return
    cache = Path(on_dir) / 'vida_raw_bank.json'
    rc, off_dir = run(base + ['--tag', f'main{a.n}_{name}_ev-off', '--no-evidence',
                              '--reuse-r1', str(cache)])
    log(f'MAIN END {time.strftime("%Y-%m-%d %H:%M:%S")}  on={on_dir}  off={off_dir}')


if __name__ == '__main__':
    main()
