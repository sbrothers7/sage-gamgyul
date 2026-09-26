r"""
Night 2
  1) Mendeley HARD : same 190 images, R1 reused from the ev-on run,
                     changes without new evidence are reverted in code
  2) Zenodo  ON    : cross-dataset check (CitrusUAT, 12 classes, 180 images)
  3) analyze.py    -> results\ANALYSIS.md
Log: logs\main.log
"""
import subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from run_main import run, log


def main():
    log('=' * 70); log(f'NIGHT2 START {time.strftime("%Y-%m-%d %H:%M:%S")}')
    on = sorted((ROOT / 'results').glob('*main200_citrusA_ev-on'))
    if on:
        cache = on[-1] / 'vida_raw_bank.json'
        run(['--dataset', 'data/citrusA_mendeley_4cls.json', '--n', '200', '--seed', '42',
             '--enforce', '--reuse-r1', str(cache), '--tag', 'main200_citrusA_ev-hard'])
    else:
        log('no ev-on run found - skipping HARD')
    run(['--dataset', 'data/citrus_zenodo.json', '--n', '180', '--seed', '42',
         '--tag', 'main180_zenodo_ev-on'])
    subprocess.run([sys.executable, 'tools/analyze.py'], cwd=ROOT)
    log(f'NIGHT2 END {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
