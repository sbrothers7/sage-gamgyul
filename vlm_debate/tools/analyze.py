r"""
Analyse all results\*main* runs -> results\ANALYSIS.md  (standard library only)

  - per run: accuracy before/after debate, delta, exact McNemar p
  - runs that share the same R1 (same images): paired comparison of final answers
  - per agent: R1->R3 accuracy, wrong->right / right->wrong flips
  - rule compliance: share of opinion changes with no new visual evidence
"""
import csv, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results'


def mcnemar(b, c):
    """exact two-sided McNemar (binomial) p-value; b, c = discordant counts"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def load(d):
    rows = list(csv.DictReader(open(d / 'panda_results.csv', encoding='utf-8-sig')))
    summ = json.loads((d / 'summary.json').read_text(encoding='utf-8'))
    cfg = json.loads((d / 'run_config.json').read_text(encoding='utf-8'))
    tr = json.loads((d / 'panda_transcripts.json').read_text(encoding='utf-8'))
    return rows, summ, cfg, tr


def cond_name(cfg, summ):
    if summ.get('enforce_evidence') or cfg.get('enforce_evidence'): return 'HARD (code-enforced)'
    return 'ON (prompt rule)' if cfg.get('require_new_evidence') else 'OFF (no rule)'


def main():
    runs = [d for d in sorted(RES.iterdir()) if d.is_dir() and 'main' in d.name
            and (d / 'summary.json').exists()]
    out = [f'# Analysis — {time.strftime("%Y-%m-%d %H:%M")}\n']
    data = {}
    for d in runs:
        rows, summ, cfg, tr = load(d)
        data[d.name] = (rows, summ, cfg, tr)
    # ── per run ──
    out.append('## 1. Debate effect per run (R1 weighted vote -> R3 weighted vote)\n')
    out.append('| run | condition | n | before | after | Δ | wrong→right | right→wrong | McNemar p |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for name, (rows, summ, cfg, tr) in data.items():
        b = sum(1 for r in rows if r['baseline_both_correct'] == '0' and r['consensus_both_correct'] == '1')
        c = sum(1 for r in rows if r['baseline_both_correct'] == '1' and r['consensus_both_correct'] == '0')
        out.append(f"| {name} | {cond_name(cfg, summ)} | {len(rows)} | {summ['baseline_both']:.3f} | "
                   f"{summ['consensus_both']:.3f} | {summ['delta']:+.3f} | {b} | {c} | {mcnemar(b, c):.3f} |")
    # ── paired between runs with same R1 ──
    out.append('\n## 2. Conditions compared on the same images (shared R1)\n')
    names = list(data)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = data[names[i]][0], data[names[j]][0]
            ka = [(r['image_name'], r['class_label'], r['r1_consensus_disease']) for r in A]
            kb = [(r['image_name'], r['class_label'], r['r1_consensus_disease']) for r in B]
            if ka == kb: pairs.append((names[i], names[j]))
    if not pairs: out.append('(no pairs)')
    for x, y in pairs:
        A, B = data[x][0], data[y][0]
        b = sum(1 for p, q in zip(A, B) if p['consensus_both_correct'] == '0' and q['consensus_both_correct'] == '1')
        c = sum(1 for p, q in zip(A, B) if p['consensus_both_correct'] == '1' and q['consensus_both_correct'] == '0')
        out.append(f"- **{cond_name(data[x][2], data[x][1])}** vs **{cond_name(data[y][2], data[y][1])}**: "
                   f"only-second-right {b}, only-first-right {c}, McNemar p = {mcnemar(b, c):.3f}")
    # ── agents ──
    out.append('\n## 3. Individual agents (accuracy R1 → R2 → R3, flips R1→R3)\n')
    out.append('| run | agent | R1 | R2 | R3 | changed | wrong→right | right→wrong |')
    out.append('|---|---|---|---|---|---|---|---|')
    for name, (rows, summ, cfg, tr) in data.items():
        for a in summ['debate_agents']:
            acc = [sum(int(r[f'r{k}_{a}_both_correct']) for r in rows) / len(rows) for k in (1, 2, 3)]
            g = [r['gt_disease'] for r in rows]
            r1 = [r[f'r1_{a}_disease'] for r in rows]; r3 = [r[f'r3_{a}_disease'] for r in rows]
            ch = sum(1 for p, q in zip(r1, r3) if p != q)
            w2r = sum(1 for p, q, t in zip(r1, r3, g) if p != t and q == t)
            r2w = sum(1 for p, q, t in zip(r1, r3, g) if p == t and q != t)
            out.append(f'| {name} | {a} | {acc[0]:.3f} | {acc[1]:.3f} | {acc[2]:.3f} | {ch} | {w2r} | {r2w} |')
    # ── compliance / convergence ──
    out.append('\n## 4. Rule compliance and convergence\n')
    out.append('| run | opinion changes (R1→R2, R2→R3) | changes with NO new evidence | reverted by code | unanimous R1 | unanimous R3 | self-named influence |')
    out.append('|---|---|---|---|---|---|---|')
    for name, (rows, summ, cfg, tr) in data.items():
        ch = noev = selfn = 0
        for v in tr.values():
            for a, r in v['rounds'].items():
                for rn, pv in (('r2', 'r1'), ('r3', 'r2')):
                    x, y = r.get(rn, {}), r.get(pv, {})
                    ev = str(x.get('new_visual_evidence') or '').strip().strip('.').upper()
                    if x.get('influenced_by') == a: selfn += 1
                    if x.get('disease_category') != y.get('disease_category'):
                        ch += 1; noev += ev in ('', 'NONE', 'N/A', 'NA', 'NO')
        u1 = sum(1 for r in rows if float(r['r1_disease_agreement']) >= 0.999) / len(rows)
        u3 = sum(1 for r in rows if float(r['r3_disease_agreement']) >= 0.999) / len(rows)
        out.append(f"| {name} | {ch} | {noev} ({noev / max(ch, 1):.0%}) | {summ.get('reverted_changes', '-')} | "
                   f"{u1:.0%} | {u3:.0%} | {selfn} |")
    # ── per class / confusion ──
    out.append('\n## 5. Per class accuracy (before → after) and what the group answered\n')
    for name, (rows, summ, cfg, tr) in data.items():
        out.append(f'**{name}**\n')
        by = defaultdict(list)
        for r in rows: by[r['gt_disease']].append(r)
        for cls, rs in sorted(by.items()):
            b = sum(int(r['baseline_both_correct']) for r in rs) / len(rs)
            c = sum(int(r['consensus_both_correct']) for r in rs) / len(rs)
            pred = Counter(r['r3_consensus_disease'] for r in rs).most_common(4)
            out.append(f"- {cls} (n={len(rs)}): {b:.2f} → {c:.2f}  | final answers: " +
                       ', '.join(f'{k} {v}' for k, v in pred))
        out.append('')
    (RES / 'ANALYSIS.md').write_text('\n'.join(out), encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
