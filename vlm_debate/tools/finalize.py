r"""
After the GPU works:
  1) Zenodo CitrusUAT -> 12 class folders, resized to 768px (originals 4128x3096)
  2) Mendeley spec without Melanose (13 images, too few)
  3) pick 3 models with different skill (weak / mid / strong) on 16 real images
  4) .env update, 10-image pilot
Log: logs\finalize.log
"""
import json, os, re, subprocess, sys, time, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'tools'))
LOG = ROOT / 'logs' / 'finalize.log'
_f = open(LOG, 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(map(str, a))
    try: print(s, flush=True)
    except Exception: print(s.encode('ascii', 'replace').decode(), flush=True)
    _f.write(s + '\n'); _f.flush()


ZMAP = {'citrusleafminer': 'Leaf Miner', 'fe': 'Iron Deficiency', 'greasyspot': 'Greasy Spot',
        'hlb': 'Greening', 'healthy': 'Healthy', 'mg': 'Magnesium Deficiency',
        'mn': 'Manganese Deficiency', 'n': 'Nitrogen Deficiency', 'redscale': 'Red Scale',
        'redscalesequelae': 'Red Scale Sequelae', 'texasmite': 'Texas Mite',
        'zn': 'Zinc Deficiency'}


def zenodo():
    from PIL import Image
    src = Path('C:/citrus-data/zenodo_8294078/CitrusUAT_dataset/CitrusUAT_dataset/Images')
    dst = Path('C:/citrus-data/zenodo_768')
    counts, unknown = {}, set()
    for f in sorted(src.iterdir()):
        if not f.is_file(): continue
        key = re.sub(r'[^a-z]', '', f.stem.rsplit('_', 1)[0].lower())
        cls = ZMAP.get(key)
        if not cls: unknown.add(f.stem.rsplit('_', 1)[0]); continue
        out = dst / cls / (f.stem + '.jpg')
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(f).convert('RGB'); im.thumbnail((768, 768))
            im.save(out, quality=90)
        counts[cls] = counts.get(cls, 0) + 1
    spec = {'crop': 'Citrus', 'source': 'Zenodo 8294078 CitrusUAT (C. sinensis), Images resized to 768px',
            'counts': counts, 'classes': {c: [str(dst / c)] for c in sorted(counts)}}
    (ROOT / 'data' / 'citrus_zenodo.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    log('zenodo classes:', ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))
    if unknown: log('UNKNOWN prefixes:', unknown)


def mendeley4():
    s = json.loads((ROOT / 'data' / 'citrusB_mendeley.json').read_text(encoding='utf-8'))
    mel = s['classes'].pop('Melanose', None); s['counts'].pop('Melanose', None)
    s['source'] += ' | Melanose excluded (13 images)'
    (ROOT / 'data' / 'citrusA_mendeley_4cls.json').write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding='utf-8')
    log('mendeley 4cls:', s['counts'])
    if mel:
        names = sorted(p.name for p in Path(mel[0]).iterdir())
        log('melanose files (for the note):', names)


def select_models():
    import vp_core as C
    ds = ROOT / 'data' / 'citrusA_mendeley_4cls.json'
    pilot, _, tax = C.load_dataset(ds, n_images=16, seed=11)
    P = C.build_prompts(tax)
    prov = C.Providers()
    cands = [a for a in prov.list_agents() if not a.startswith('moondream')]
    log(f'\nmodel selection on {len(pilot)} images: {cands}')
    log(f"{'model':<26}{'acc':>6}{'parse':>7}{'sec/img':>9}")
    rows = []
    for a in cands:
        fn = prov.make_fn(a); ok = parsed = 0; ts = []
        for s in pilot:
            t0 = time.time(); raw = fn(s['image_path'], P['r1']); ts.append(time.time() - t0)
            p = C.parse_response(raw, a, tax)
            if p['disease_category']: parsed += 1
            ok += int(p['disease_category'] == s['gt_disease'] and p['crop_category'] == s['gt_crop'])
        r = {'model': a, 'acc': ok / len(pilot), 'parse': parsed / len(pilot),
             'sec': statistics.mean(ts)}
        rows.append(r)
        log(f"{a:<26}{r['acc']:>6.2f}{r['parse']:>7.2f}{r['sec']:>9.1f}")
    good = sorted([r for r in rows if r['parse'] >= 0.8], key=lambda r: (r['acc'], -r['sec']))
    if len(good) < 3: good = sorted(rows, key=lambda r: r['acc'])
    pick = [good[0], good[len(good) // 2], good[-1]]
    names = []
    for r in pick:
        if r['model'] not in names: names.append(r['model'])
    for r in good:
        if len(names) >= 3: break
        if r['model'] not in names: names.append(r['model'])
    log('PICKED (weak -> strong):', names)
    (ROOT / 'results' / 'model_select.json').write_text(json.dumps({'rows': rows, 'picked': names,
        'n_images': len(pilot), 'date': time.strftime('%Y-%m-%d %H:%M')}, indent=1), encoding='utf-8')
    return names


def set_env(pairs):
    from prepare_all import set_env as se
    se(pairs)


def main():
    zenodo()
    mendeley4()
    names = select_models()
    set_env({'OLLAMA_MODELS': ','.join(names), 'CDDM_ROOT': 'data/citrusA_mendeley_4cls.json'})
    log('\npilot run (10 images, anti-sycophancy ON)')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    p = subprocess.Popen([sys.executable, '-u', 'tools/run_experiment.py', '--n', '10',
                          '--dataset', 'data/citrusA_mendeley_4cls.json',
                          '--agents', ','.join(names), '--tag', 'pilot10_ev-on'],
                         cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        log('  ' + line.decode('utf-8', 'replace').rstrip())
    log('pilot rc', p.wait())
    log('DONE')


if __name__ == '__main__':
    try: main()
    except Exception:
        import traceback; log(traceback.format_exc())
