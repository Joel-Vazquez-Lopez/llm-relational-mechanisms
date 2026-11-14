#!/usr/bin/env python3
"""EXP-103 development-only final paired analysis. Read-only by default.

Reads EXISTING reports/exp103/factorial_dev/*.json. Does not import training
scripts, torch, datasets, reserved examples or any protected evaluation code.
The optional --write creates one new immutable analysis file; it never modifies
or rewrites training results.

Use from repository root:
  python3 scripts/exp103_results_audit.py
  python3 scripts/exp103_results_audit.py --write
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
import json
import math

SEEDS = (10501, 10503, 10509, 10513, 10521, 10527, 10529, 10533)
EXPOSURES = ('familiar', 'target')
POLICIES = ('quartet', 'distributed')
CELLS = ('AA', 'PA', 'AP')
MILESTONES = (800, 1600, 2400, 3200)
METRICS = ('C_query', 'accuracy4', 'wrong_event_rate', 'wrong_role_rate',
           'same_answer_rate', 'per_event_both_correct', 'full_vocab_accuracy')
MAIN = ('exposure', 'policy', 'interaction')


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def mean(values):
    return sum(values)/len(values)


def ci_bootstrap(values, *, repetitions=30000, bootstrap_seed=1033200):
    """Descriptive percentile CI across PAIRED seeds, NOT individual sentences."""
    rng = Random(bootstrap_seed)
    n = len(values)
    x = sorted(mean([values[rng.randrange(n)] for _ in range(n)])
               for _ in range(repetitions))
    return (x[int(.025*repetitions)], x[int(.975*repetitions)])


def pp(x):
    return f'{x*100:+.2f} pp'


def percentage(x):
    return f'{x*100:.2f}%'


def check_close(a, b, name, tol=1e-5):
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f'Metric identity failed: {name} {a} vs {b}')


def validate_cell(m):
    assert m['sentences'] == 72, f"Unexpected dev set: {m['sentences']}"
    for k in METRICS:
        v = m[k]
        assert isinstance(v, (float, int)) and 0.0-1e-6 <= v <= 1.0+1e-6, (k, v)
    check_close(m['accuracy4'] + m['wrong_event_rate'] + m['wrong_role_rate'], 1., 'question_error_partition')
    check_close(m['same_answer_rate'], (m['role_copy_first'] + m['role_copy_second'])/2, 'copy_mean')
    check_close(m['per_event_both_correct'], (m['both_correct_first'] + m['both_correct_second'])/2, 'event_both_mean')
    assert m['C_query'] <= m['per_event_both_correct'] + 1e-6
    assert m['C_query'] <= m['accuracy4'] + 1e-6
    assert len(m['per_cell_C_query']) == 9


def read_results(root):
    folder = root/'reports/exp103/factorial_dev'
    protocol_path = folder/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['study'] == 'EXP-103-factorial-development'
    assert tuple(protocol['seeds']) == SEEDS
    assert tuple(protocol['factors']['exposure']) == EXPOSURES
    assert tuple(protocol['factors']['policy']) == POLICIES
    assert protocol['steps'] == 3200 and tuple(protocol['milestones']) == MILESTONES
    expected = canonical_hash(protocol)
    reports = {}
    for seed in SEEDS:
        for exposure in EXPOSURES:
            for policy in POLICIES:
                name = f'seed{seed}_{exposure}_{policy}.json'
                j = json.loads((folder/name).read_text())
                assert (j['study'], j['seed'], j['exposure'], j['policy'], j['step']) == (
                    'EXP-103-factorial-development', seed, exposure, policy, 3200)
                assert j['protocol_sha256'] == expected, f'Frozen protocol mismatch: {name}'
                assert j['data_sha256'] == protocol['data_sha256']
                assert j['evidence_status'] == 'EXPLORATORY_DEVELOPMENT_ONLY'
                assert j['protected_accessed'] is False
                assert j['reserved_cases_generated'] is False
                assert j['LI_cases_generated'] is False
                assert [r['step'] for r in j['records']] == list(MILESTONES)
                for rec in j['records']:
                    assert set(rec['dev']) == set(CELLS)
                    for cell in CELLS:
                        validate_cell(rec['dev'][cell])
                reports[(seed, exposure, policy)] = j
    assert len(reports) == 32
    return reports, protocol


def metric(reports, seed, exposure, policy, cell, key, *, step=3200):
    rows = reports[(seed, exposure, policy)]['records']
    rec = next(r for r in rows if r['step']==step)
    return rec['dev'][cell][key]


def paired_effects(reports, cell, key):
    result = {name:[] for name in MAIN}
    for seed in SEEDS:
        fq=metric(reports,seed,'familiar','quartet',cell,key)
        fd=metric(reports,seed,'familiar','distributed',cell,key)
        tq=metric(reports,seed,'target','quartet',cell,key)
        td=metric(reports,seed,'target','distributed',cell,key)
        result['exposure'].append(((tq+td)-(fq+fd))/2)
        result['policy'].append(((tq+fq)-(td+fd))/2)
        result['interaction'].append((tq-td)-(fq-fd))
    return result


def gate_for_model(reports, seed, exposure, policy):
    return all(metric(reports,seed,exposure,policy,cell,'accuracy4') >= .95 -1e-9 and
               metric(reports,seed,exposure,policy,cell,'per_event_both_correct') >= .90 - 1e-9
               for cell in ('AA','PA'))


def markdown(reports, protocol):
    lines=['# EXP-103: Source-matched factorial study — development-only audit',
           '', '**Status:** descriptive, exploratory, based on frozen step-3200 endpoints. No protected or reserved cases generated.',
           '', '## Final AP (first active, second passive): C_query', '',
           '| Exposure / update schedule | Mean across 8 seeds | Seeds ≥ 90% |',
           '|---|---:|---:|']
    for e in EXPOSURES:
        for p in POLICIES:
            a=[metric(reports,s,e,p,'AP','C_query') for s in SEEDS]
            lines.append(f'| {e} / {p} | {percentage(mean(a))} | {sum(x>=.9 for x in a)}/8 |')
    lines += ['', '## Paired factorial contrasts on AP C_query', '',
              '| Contrast | Mean effect | Descriptive paired-seed bootstrap 95% CI | Positive seeds |',
              '|---|---:|---:|---:|']
    effects = paired_effects(reports,'AP','C_query')
    for name in MAIN:
        values = effects[name]
        lo,hi = ci_bootstrap(values,bootstrap_seed=1033200+MAIN.index(name))
        lines.append(f'| {name} | {pp(mean(values))} | [{pp(lo)}, {pp(hi)}] | {sum(v>0 for v in values)}/8 |')
    lines += ['', '**Definitions:** exposure = mean(target) − mean(familiar) across policies; '
              'policy = mean(quartet) − mean(distributed) across exposure arms; '
              'interaction = (target quartet − target distributed) − (familiar quartet − familiar distributed).',
              '', '**Per-seed paired primary endpoint:**', '',
              '| Seed | Exposure effect | Policy effect | Interaction |', '|---|---:|---:|---:|']
    for i,seed in enumerate(SEEDS):
        lines.append(f'| {seed} | {pp(effects["exposure"][i])} | {pp(effects["policy"][i])} | {pp(effects["interaction"][i])} |')
    lines += ['', '## AP error taxonomy at step 3200', '',
              '| Condition | Question accuracy | Wrong event | Wrong role | Same-answer | Two-query both-correct |',
              '|---|---:|---:|---:|---:|---:|']
    for e in EXPOSURES:
        for p in POLICIES:
            v=lambda k: mean([metric(reports,s,e,p,'AP',k) for s in SEEDS])
            lines.append(f'| {e}/{p} | '+ ' | '.join(percentage(v(k)) for k in
                         ('accuracy4','wrong_event_rate','wrong_role_rate','same_answer_rate','per_event_both_correct'))+' |')
    lines += ['', '## Secondary paired contrast summaries', '',
              '| AP outcome | Exposure main effect | Policy main effect | 2×2 interaction |',
              '|---|---:|---:|---:|']
    for k in ('accuracy4','wrong_event_rate','wrong_role_rate','same_answer_rate','per_event_both_correct'):
        d=paired_effects(reports,'AP',k)
        lines.append(f'| {k} | '+ ' | '.join(pp(mean(d[x])) for x in MAIN) +' |')
    lines += ['', '## Familiar development competence gate: AA and PA', '',
              'Individual model passes only when **both AA and PA** have question accuracy ≥95% '
              'and per-event both-role correctness ≥90%. Predeclared requirement: ≥6 of 8 seeds in **every arm**.',
              '', '| Arm | Passing seeds | Meets ≥6/8 |', '|---|---:|---:|']
    gate={}
    for e in EXPOSURES:
        for p in POLICIES:
            n=sum(gate_for_model(reports,s,e,p) for s in SEEDS)
            gate[(e,p)] = n
            lines.append(f'| {e}/{p} | {n}/8 | {"YES" if n>=6 else "NO"} |')
    lines += ['', '**Overall gate:** '+('PASS' if all(n>=6 for n in gate.values()) else 'FAIL') +
              '. A passing gate is not permission to open reserved evaluation. This runner has no protected-evaluation phase.',
              '', '## AP learning trajectories (mean C_query)', '',
              '| Step | Familiar/Q | Familiar/D | Target/Q | Target/D |',
              '|---:|---:|---:|---:|---:|']
    for step in MILESTONES:
        vals=[mean([metric(reports,s,e,p,'AP','C_query',step=step) for s in SEEDS])
              for e in EXPOSURES for p in POLICIES]
        lines.append(f'| {step} | '+' | '.join(percentage(v) for v in vals)+' |')
    lines += ['', '## Interpretation and limitations', '',
              '- The unit of replication is the paired random seed (n=8), not the 72 examples per dev cell.',
              '- These are **development** comparisons after exploratory study design; intervals are descriptive, not preregistered confirmation.',
              '- The sources/questions are exactly matched **within each four-update window** across schedule policies, but not within individual updates or optimizer trajectories.',
              '- Target versus familiar exposure intentionally changes the training construction. This is not a direct ML-080 replication.',
              '- Do not infer that absence of mean scheduling effect proves equivalence. A zero-looking mean may hide seed-level heterogeneity.',
              '- The AP set uses familiar modifiers and held-out name sets; it is not the protected L+I composition evaluation.',
              '- The historical outcomes and EXP-102 results should remain immutable; no post hoc checkpoint selection or seed exclusion.',
              '', 'PASS: 32/32 frozen endpoint reports, 8 paired seeds, 4 arms, 3 dev cells, 4 time points, '
              'metric identities and protected-access assertions.', '']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1],
                    help='Experiment repository root, inferred from scripts/ location')
    ap.add_argument('--write',action='store_true',help='Create immutable analysis markdown; no overwrite')
    a=ap.parse_args()
    reports,prot=read_results(a.root)
    result=markdown(reports,prot)
    print(result)
    if a.write:
        dest=a.root/'reports/exp103/analysis/factorial_dev_3200_audit.md'
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if dest.read_text()!=result:
                raise FileExistsError('Existing analysis differs; refusing overwrite: '+str(dest))
            print('EXISTING AUDIT: verified identical',dest)
        else:
            dest.write_text(result)
            print('SAVED',dest)

if __name__ == '__main__':
    main()
