"""
무비용 자체 점검 — API 키도 데이터셋도 필요 없다.

합성 이미지 + 가짜 에이전트로 VIDA→PANDA 전 과정을 돌려
코드가 끝까지 도는지만 확인한다. 숫자는 무작위이므로 해석하지 않는다.

코드를 고친 뒤 유료 API 나 긴 GPU 실행을 태우기 전에 이걸 먼저 돌린다.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import vp_core as C

DATA = HERE / 'fake_data' / 'images'
OUT  = HERE / 'mock_out'

FAILED = []


def check(label, cond, detail=''):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(label)


def main():
    if not DATA.exists():
        sys.exit(f'합성 데이터가 없습니다. 먼저 make_fake_data.py 를 실행하세요.\n  {DATA}')

    print('=' * 60)
    print('  자체 점검 (가짜 에이전트)')
    print('=' * 60)

    # 1. 분류체계 자동 인식
    tax = C.Taxonomy.from_folder(DATA)
    check('분류체계 자동 인식', tax is not None and len(tax) > 0,
          f'{len(tax) if tax else 0}개 클래스')

    # 2. 데이터 로딩 + 고유 id
    pilot, missing, tax2 = C.load_dataset(DATA, n_images=8, seed=42)
    check('데이터셋 로딩', len(pilot) == 8, f'{len(pilot)}장')
    check('이미지 고유 id 부여', len({p['uid'] for p in pilot}) == len(pilot))

    # 3. 파서
    sample = ("Crop Category: Tomato\nDisease Category: Late Blight\n"
              "Diseased: Yes\nReasoning: dark lesions.\nConfidence: 0.8")
    p = C.parse_response(sample, 'test', tax2)
    check('응답 파서', p['crop_category'] == 'Tomato'
          and p['disease_category'] == 'Late Blight' and p['confidence'] == 0.8)

    # 4. 전체 파이프라인
    cfg = dict(C.DEFAULT_CFG, dataset_root=str(DATA), output_dir=str(OUT),
               n_images=8, n_debate=3, mock=True)
    r = C.Runner(cfg)
    r.run()
    s = r.state
    check('파이프라인 완주', s['phase'] == 'done', s.get('error') or '')
    check('이미지 수 일치', len(s['transcripts']) == len(s['panda_rows']) == 8,
          f"기록 {len(s['transcripts'])} / 행 {len(s['panda_rows'])}")
    check('에이전트별 지표 생성', len(s['metrics']) == 5)
    check('요약 계산', 'delta' in s.get('summary', {}))

    # 5. 결과 파일
    for f in ('vida_r1_results.csv', 'panda_results.csv', 'vida_metrics.csv',
              'panda_transcripts.json', 'summary.json'):
        check(f'결과 파일 {f}', (OUT / f).exists())

    # 6. 토론 기록 구조
    t = next(iter(s['transcripts'].values()))
    check('토론 기록 3라운드', all(k in list(t['rounds'].values())[0]
                                   for k in ('r1', 'r2', 'r3')))

    print('=' * 60)
    if FAILED:
        print(f'  실패 {len(FAILED)}건: {", ".join(FAILED)}')
        sys.exit(1)
    print('  전부 통과. 결과물은 아래에 있습니다 (숫자는 무작위, 해석 금지):')
    print(f'  {OUT}')
    print('=' * 60)


if __name__ == '__main__':
    main()
