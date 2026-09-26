r"""
GPU 확인 → 모델 추천 → 내려받기 → .env 반영 까지 한 번에.

    py tools\gpu_setup.py            # 확인만 (아무것도 바꾸지 않음)
    py tools\gpu_setup.py --apply    # 모델 내려받고 .env 까지 갱신

하는 일
  1) nvidia-smi 로 GPU 와 VRAM 확인
  2) Ollama 설치·실행 여부 확인
  3) VRAM 에 맞는 비전 모델 조합 추천
  4) --apply 면 ollama pull 실행 후 .env 의 OLLAMA_MODELS 갱신
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 프로젝트 루트
ENV  = ROOT / '.env'
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# 비전(이미지 입력) 가능한 Ollama 모델. (이름, 대략 VRAM GB, 등급 설명)
CANDIDATES = [
    ('qwen2.5vl:3b',  3.2, '가벼움 · 형식 준수 양호'),
    ('gemma3:4b',     3.6, '가벼움 · 서술 풍부'),
    ('moondream',     1.8, '매우 가벼움 · 정확도 낮음'),
    ('llava:7b',      5.0, '중간'),
    ('qwen2.5vl:7b',  6.5, '중간 · 3b 보다 정확'),
    ('llava:13b',     9.0, '무거움'),
    ('gemma3:12b',    9.5, '무거움'),
    ('qwen2.5vl:32b', 22.0, '매우 무거움'),
]


def run(cmd, timeout=20):
    """명령을 실행하고 (성공여부, 출력) 을 돌려준다."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, shell=False)
        return r.returncode == 0, (r.stdout or r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as ex:
        return False, str(ex)


# ── 1. GPU ─────────────────────────────────────────────────────────────
def detect_gpu():
    if not shutil.which('nvidia-smi'):
        return None, 'nvidia-smi 를 찾을 수 없습니다 (드라이버 미설치 또는 PATH 문제)'
    ok, out = run(['nvidia-smi',
                   '--query-gpu=name,memory.total,driver_version',
                   '--format=csv,noheader,nounits'])
    if not ok or not out:
        return None, f'nvidia-smi 실행 실패: {out[:200]}'
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 2:
        return None, f'출력 해석 실패: {line}'
    try:
        vram_gb = round(int(parts[1]) / 1024, 1)
    except ValueError:
        return None, f'VRAM 값 해석 실패: {parts[1]}'
    return {'name': parts[0], 'vram_gb': vram_gb,
            'driver': parts[2] if len(parts) > 2 else '?'}, None


# ── 2. Ollama ──────────────────────────────────────────────────────────
def detect_ollama():
    info = {'installed': bool(shutil.which('ollama')), 'serving': False,
            'models': [], 'version': None}
    if info['installed']:
        ok, out = run(['ollama', '--version'])
        if ok:
            info['version'] = out.splitlines()[0]
    try:
        with urllib.request.urlopen(f'{OLLAMA_HOST}/api/tags', timeout=4) as r:
            tags = json.loads(r.read())
        info['serving'] = True
        info['models'] = [m['name'] for m in tags.get('models', [])]
    except Exception:
        pass
    return info


# ── 3. 추천 ────────────────────────────────────────────────────────────
def recommend(vram_gb, n_agents=3):
    """VRAM 안에 '하나씩' 올라갈 수 있는 모델 중에서 실력 차이가 나게 고른다.

    실행 코드가 에이전트 단위로 묶어 호출하므로 모델이 동시에 상주할 필요는
    없다. 따라서 '각 모델이 개별적으로 들어가는가'만 따지고, 그 안에서
    가벼운 것 ~ 무거운 것을 고르게 뽑아 능력 차이를 만든다.
    (전부 비슷한 실력이면 토론 효과 Δ 가 움직이지 않는다.)
    """
    if vram_gb is None:                       # CPU 전용
        budget, note = 4.0, 'GPU 를 못 찾아 CPU 기준으로 골랐습니다 — 매우 느립니다'
    else:
        budget = max(vram_gb - 1.2, 1.0)      # 1.2GB 는 OS·디스플레이 몫
        note = None

    elig = sorted([c for c in CANDIDATES if c[1] <= budget], key=lambda c: c[1])
    if not elig:                              # 제일 작은 것도 안 들어가면
        return [min(CANDIDATES, key=lambda c: c[1])], budget, \
               'VRAM 이 부족합니다. CPU 로 돌게 되어 매우 느립니다'

    n = min(n_agents, len(elig))
    if n == 1:
        picked = [elig[-1]]
    else:                                     # 가장 작은 것 ~ 가장 큰 것을 고르게
        idx = [round(i * (len(elig) - 1) / (n - 1)) for i in range(n)]
        picked = [elig[i] for i in sorted(set(idx))]

    concurrent = sum(p[1] for p in picked)
    return picked, concurrent, note


# ── 4. .env 갱신 ───────────────────────────────────────────────────────
def read_env_text():
    if not ENV.exists():
        return '', None
    for enc in ('utf-8-sig', 'utf-16', 'cp949', 'utf-8'):
        try:
            t = ENV.read_text(encoding=enc)
            if '\x00' in t:
                continue
            return t, enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return '', None


def update_env(models):
    text, enc = read_env_text()
    enc = enc or 'utf-8'
    line = 'OLLAMA_MODELS=' + ','.join(models)
    if re.search(r'(?m)^\s*OLLAMA_MODELS\s*=.*$', text):
        text = re.sub(r'(?m)^\s*OLLAMA_MODELS\s*=.*$', line, text)
    else:
        text = text.rstrip('\n') + ('\n\n' if text else '') + line + '\n'
    ENV.write_text(text, encoding=enc)
    return line


# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true',
                    help='모델을 내려받고 .env 를 갱신한다')
    ap.add_argument('--agents', type=int, default=3, help='에이전트 수 (기본 3)')
    args = ap.parse_args()

    print('=' * 66)
    print('  실행 환경 점검')
    print('=' * 66)

    gpu, gpu_err = detect_gpu()
    if gpu:
        print(f"  GPU        : {gpu['name']}")
        print(f"  VRAM       : {gpu['vram_gb']} GB")
        print(f"  드라이버   : {gpu['driver']}")
    else:
        print(f"  GPU        : 확인 불가 — {gpu_err}")

    oll = detect_ollama()
    print(f"  Ollama     : {'설치됨 ' + (oll['version'] or '') if oll['installed'] else '설치 안 됨'}")
    print(f"  Ollama 서버: {'실행 중' if oll['serving'] else '응답 없음'}")
    if oll['models']:
        print(f"  보유 모델  : {', '.join(oll['models'])}")

    print()
    print('=' * 66)
    print('  추천 구성')
    print('=' * 66)
    picked, used, note = recommend(gpu['vram_gb'] if gpu else None, args.agents)
    if note:
        print(f"  ⚠ {note}")
    for name, gb, desc in picked:
        have = ' (이미 있음)' if any(m.split(':')[0] == name.split(':')[0]
                                     for m in oll['models']) else ''
        print(f"    {name:<16} 약 {gb:>4.1f} GB   {desc}{have}")
    biggest = max(p[1] for p in picked)
    print()
    if gpu:
        print(f"  가장 큰 모델 {biggest:.1f} GB / VRAM {gpu['vram_gb']} GB — 하나씩 올려 실행합니다.")
        if used <= max(gpu['vram_gb'] - 1.2, 1.0):
            print(f"  세 모델 합계도 {used:.1f} GB 라 동시에 상주 가능 — 모델 교체 없이 더 빠릅니다.")
        else:
            print(f"  세 모델 합계는 {used:.1f} GB 로 동시 상주는 안 되지만,")
            print(f"  실행 코드가 에이전트 단위로 묶어 호출하므로 교체는 라운드당 1회뿐입니다.")
    print()
    print('  실력 차이가 나는 모델을 섞는 것이 핵심입니다.')
    print('  전부 비슷하면 토론 전후 차이(Δ)가 거의 움직이지 않습니다.')

    names = [p[0] for p in picked]
    if not args.apply:
        print()
        print('-' * 66)
        print('  지금은 확인만 했습니다. 실제로 적용하려면:')
        print('      py tools\\gpu_setup.py --apply')
        print('-' * 66)
        return

    if not oll['installed']:
        print('\n⛔ Ollama 가 설치되어 있지 않습니다. https://ollama.com 에서 설치 후 다시 실행하세요.')
        sys.exit(1)

    print()
    print('=' * 66)
    print('  모델 내려받기')
    print('=' * 66)
    failed = []
    for n in names:
        print(f'  → ollama pull {n}')
        ok, out = run(['ollama', 'pull', n], timeout=3600)
        if not ok:
            failed.append(n)
            print(f'    실패: {out[:200]}')
    if failed:
        print(f'\n  실패한 모델: {", ".join(failed)}')
        names = [n for n in names if n not in failed]
    if not names:
        print('  내려받은 모델이 없어 .env 를 바꾸지 않았습니다.')
        sys.exit(1)

    line = update_env(names)
    print()
    print(f'  .env 갱신: {line}')
    print(f'  위치: {ENV}')
    print()
    print('  이제 run_app.bat 을 실행하면 에이전트 목록에 이 모델들이 뜹니다.')


if __name__ == '__main__':
    main()
