r"""
모델별 실제 속도를 재고, 본 실험이 얼마나 걸릴지 추정한다.

    py tools\bench_models.py                      # .env 의 모델로 3장씩
    py tools\bench_models.py --images 5 --plan 490

실제 이미지로 R1 프롬프트를 던져 응답 시간과 파싱 성공 여부를 잰다.
밤새 돌릴지 며칠에 나눌지를 이 결과로 판단한다.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

# .env 읽기 (vp_app 과 같은 방식)
def load_env(p: Path):
    if not p.exists():
        return
    for enc in ('utf-8-sig', 'utf-16', 'cp949', 'utf-8'):
        try:
            t = p.read_text(encoding=enc)
            if '\x00' in t:
                continue
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return
    for line in t.splitlines():
        line = line.strip().lstrip('﻿')
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

load_env(ROOT / '.env')
import vp_core as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.getenv('CDDM_ROOT', 'data/citrusB_mendeley.json'))
    ap.add_argument('--images', type=int, default=3, help='모델당 측정 장수')
    ap.add_argument('--plan', type=int, default=490, help='본 실험 예정 장수')
    ap.add_argument('--agents', type=int, default=3, help='본 실험 토론 참가자 수')
    args = ap.parse_args()

    ds = Path(args.dataset)
    if not ds.is_absolute():
        ds = ROOT / ds
    pilot, missing, tax = C.load_dataset(ds, n_images=args.images, seed=7)
    if not pilot:
        sys.exit(f'이미지를 못 찾았습니다: {ds}')
    prompts = C.build_prompts(tax)
    print(f'데이터셋 {ds}')
    print(f'분류체계 {len(tax)}개 클래스 · 측정용 {len(pilot)}장\n')

    prov = C.Providers()
    agents = prov.list_agents()
    if not agents:
        sys.exit('사용 가능한 에이전트가 없습니다. gpu_setup.py --apply 를 먼저 실행하세요.')

    print(f'{"모델":<22}{"평균(초)":>10}{"최소":>8}{"최대":>8}{"파싱성공":>8}')
    print('-' * 58)

    rows = []
    for name in agents:
        fn = prov.make_fn(name)
        times, ok_parse = [], 0
        for s in pilot:
            t0 = time.time()
            raw = fn(s['image_path'], prompts['r1'])
            dt = time.time() - t0
            times.append(dt)
            p = C.parse_response(raw, name, tax)
            if p['crop_category'] and p['disease_category']:
                ok_parse += 1
        avg = statistics.mean(times)
        rows.append({'model': name, 'avg': avg, 'min': min(times),
                     'max': max(times), 'parse_ok': ok_parse, 'n': len(times)})
        print(f'{name:<22}{avg:>10.1f}{min(times):>8.1f}{max(times):>8.1f}'
              f'{f"{ok_parse}/{len(times)}":>8}')

    # ── 본 실험 예상 시간 ───────────────────────────────────────────────
    print()
    print('=' * 58)
    n_deb = min(args.agents, len(rows))
    top = sorted(rows, key=lambda r: r['avg'])[:n_deb]
    per_img_r1 = sum(r['avg'] for r in rows)                 # R1: 전 에이전트
    per_img_deb = sum(r['avg'] for r in top) * 2             # R2+R3: 참가자만
    total_sec = args.plan * (per_img_r1 + per_img_deb)
    h, m = divmod(int(total_sec / 60), 60)
    print(f'  이미지 {args.plan}장 · 에이전트 {len(rows)}개 · 토론 {n_deb}개 기준')
    print(f'  예상 소요 시간 : 약 {h}시간 {m}분')
    print(f'  (R1 {args.plan * len(rows)}회 + 토론 {args.plan * n_deb * 2}회)')
    print('=' * 58)
    if total_sec > 6 * 3600:
        print('  6시간을 넘습니다. 장수를 줄이거나 밤에 돌리는 편이 낫습니다.')

    out = ROOT / 'results' / 'bench.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'rows': rows, 'plan': args.plan,
                               'est_seconds': total_sec}, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print(f'\n  결과 저장: {out}')


if __name__ == '__main__':
    main()
