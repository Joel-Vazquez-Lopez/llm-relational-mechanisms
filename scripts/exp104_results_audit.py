#!/usr/bin/env python3
"""Read-only EXP-104 source-matched schedule audit; optional write-once derived outputs.

Run from the EXP-102 worktree:
    python3 scripts/exp104_results_audit.py
    python3 scripts/exp104_results_audit.py --write

Does not import training/generator modules, generate any examples, load any
model checkpoints, train anything, or access LI/reserved/protected cases.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random
from statistics import mean, median
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts/exp104_source_matched_schedule.py'
BASE = ROOT / 'scripts/exp102_v2_familiar_training.py'
OUT = ROOT / 'reports/exp104/source_matched_dev'
PROTO = OUT / 'protocol.json'
AUDIT = ROOT / 'reports/exp104/analysis'
SEEDS = (10601,10603,10609,10613,10621,10627,10629,10633)
POLICIES = ('quartet','distributed')
MILESTONES = tuple(range(400,3201,400))
CELLS = frozenset('bare/bare L/bare bare/L I/bare bare/I L/I I/L L/L I/I'.split())
METRICS = ('C_query','accuracy4','same_answer_rate','wrong_event_rate',
           'wrong_role_rate','per_event_both_correct')


def digest(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def filehash(path):
    return sha256(path.read_bytes()).hexdigest()


def close(a,b,tol=2e-5):
    return abs(a-b) <= tol


def verify_dev(m):
    assert m['sentences'] == 288
    for name in METRICS:
        assert 0.0 <= m[name] <= 1.0, (name,m[name])
    hit = (m['correct_event_identity_first']+m['correct_event_identity_second'])/2
    copy = (m['role_copy_first']+m['role_copy_second'])/2
    both = (m['both_correct_first']+m['both_correct_second'])/2
    assert close(1-hit,m['wrong_event_rate'])
    assert close(hit-m['accuracy4'],m['wrong_role_rate'])
    assert close(copy,m['same_answer_rate'])
    assert close(both,m['per_event_both_correct'])
    assert close(m['accuracy4']+m['wrong_event_rate']+m['wrong_role_rate'],1.0)
    assert m['C_query'] <= min(m['both_correct_first'],m['both_correct_second'])+2e-5
    assert m['per_event_both_correct'] <= m['accuracy4']+2e-5
    assert set(m['per_cell_C_query']) == CELLS
    for x in m['per_cell_C_query'].values(): assert 0 <= x <= 1


def load():
    protocol=json.loads(PROTO.read_text())
    assert protocol['runner_sha256'] == filehash(RUNNER), 'Runner modified since freeze'
    assert protocol['base_runner_sha256'] == filehash(BASE), 'Base runner modified since freeze'
    assert protocol['data_sha256'] and protocol['seeds'] == list(SEEDS)
    assert protocol['policies'] == list(POLICIES)
    assert protocol['steps'] == 3200 and protocol['milestones'] == list(MILESTONES)
    assert protocol['train_sentences'] == 2304 and protocol['development_sentences'] == 288
    assert protocol['status'] == 'EXPLORATORY_FAMILIAR_DEVELOPMENT_ONLY'
    phash=digest(protocol)
    data={}
    for seed in SEEDS:
        for policy in POLICIES:
            path=OUT/f'seed{seed}_{policy}_step3200.json'
            report=json.loads(path.read_text())
            assert report['study']=='EXP-104-source-matched-schedule'
            assert report['seed']==seed and report['policy']==policy and report['step']==3200
            assert report['protocol_sha256']==phash
            assert report['data_sha256']==protocol['data_sha256']
            assert report['evidence_status']=='EXPLORATORY_ONLY'
            assert report['stage']=='FAMILIAR_DEV_ONLY'
            assert report['protected_accessed'] is False
            assert report['LI_cases_generated'] is False
            assert report['reserved_set_cases_generated'] is False
            assert [r['step'] for r in report['records']] == list(MILESTONES)
            for rec in report['records']:
                verify_dev(rec['dev'])
            data[(seed,policy)] = report['records']
    return data,phash


def bootstrap_ci(vals, seed=104104, repetitions=50000):
    rng=random.Random(seed)
    n=len(vals)
    draws=sorted(mean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(repetitions))
    return draws[int(repetitions*.025)], draws[int(repetitions*.975)]


def signflip_p(vals):
    n=len(vals); obs=abs(mean(vals)); count=0
    for signs in itertools.product((-1,1),repeat=n):
        if abs(mean(s*x for s,x in zip(signs,vals))) >= obs - 1e-12: count+=1
    return count/2**n


def delta(data, metric, checkpoint_index=-1):
    return [data[(seed,'quartet')][checkpoint_index]['dev'][metric]
           - data[(seed,'distributed')][checkpoint_index]['dev'][metric] for seed in SEEDS]


def pct(v):return f'{100*v:+.2f} pp'


def render(data,phash):
    lines=['# EXP-104 — source-matched scheduling, complete development audit','',
           '**Status:** exploratory; frozen 3,200-step endpoint, eight paired seeds. No protected, LI, or reserved examples generated.','',
           f'Protocol SHA256 (canonical JSON): `{phash}`','',
           '## Primary endpoint: DEV C_query at step 3200','',
           '| Seed | Quartet | Distributed | Paired difference |','|---:|---:|---:|---:|']
    diffs=delta(data,'C_query')
    for seed,d in zip(SEEDS,diffs):
        q=data[(seed,'quartet')][-1]['dev']['C_query']; p=data[(seed,'distributed')][-1]['dev']['C_query']
        lines.append(f'| {seed} | {q:.2%} | {p:.2%} | {pct(d)} |')
    qmean=mean(data[(s,'quartet')][-1]['dev']['C_query'] for s in SEEDS)
    dmean=mean(data[(s,'distributed')][-1]['dev']['C_query'] for s in SEEDS)
    lo,hi=bootstrap_ci(diffs)
    lines.extend(['',f'Quartet mean: **{qmean:.2%}**; distributed mean: **{dmean:.2%}**.',
                  f'Mean paired difference: **{pct(mean(diffs))}** (3,200-step endpoint).',
                  f'Descriptive paired-seed bootstrap 95% interval: **[{pct(lo)}, {pct(hi)}]**.',
                  f'Wins/ties/reversals: **{sum(x>0 for x in diffs)}/{sum(x==0 for x in diffs)}/{sum(x<0 for x in diffs)}**. '
                  f'Two-sided exact sign-flip p (exploratory only): {signflip_p(diffs):.4f}.','',
                  '## Learning trajectories (secondary)','',
                  '| Step | Quartet mean C_query | Distributed mean C_query | Paired gap | Positive pairs |',
                  '|---:|---:|---:|---:|---:|'])
    for i,step in enumerate(MILESTONES):
        qs=[data[(s,'quartet')][i]['dev']['C_query'] for s in SEEDS]
        ds=[data[(s,'distributed')][i]['dev']['C_query'] for s in SEEDS]
        dd=[a-b for a,b in zip(qs,ds)]
        lines.append(f'| {step} | {mean(qs):.2%} | {mean(ds):.2%} | {pct(mean(dd))} | {sum(x>0 for x in dd)}/8 |')
    # Equal-spaced checkpoints, not true AUC. Unlike endpoint, was not registered primary.
    avgd=[mean(data[(s,'quartet')][i]['dev']['C_query']-data[(s,'distributed')][i]['dev']['C_query']
               for i in range(len(MILESTONES))) for s in SEEDS]
    alc,auc=bootstrap_ci(avgd,seed=104105)
    lines.extend(['',f'Average over eight sampled development checkpoints (secondary, not time-integrated AUC): '
                  f'{pct(mean(avgd))}; descriptive bootstrap 95% interval [{pct(alc)}, {pct(auc)}]; '
                  f'{sum(x>0 for x in avgd)}/8 seed-pairs positive.','',
                  '## Secondary errors and accuracy at step 3200','',
                  '| Metric | Quartet mean | Distributed mean | Quartet − distributed |',
                  '|---|---:|---:|---:|'])
    for k in METRICS[1:]:
        qa=[data[(s,'quartet')][-1]['dev'][k] for s in SEEDS]
        da=[data[(s,'distributed')][-1]['dev'][k] for s in SEEDS]
        lines.append(f'| {k} | {mean(qa):.2%} | {mean(da):.2%} | {pct(mean(a-b for a,b in zip(qa,da)))} |')
    gates={policy:sum(data[(s,policy)][-1]['dev']['C_query']>=.90 and
                        data[(s,policy)][-1]['dev']['accuracy4']>=.95 for s in SEEDS)
           for policy in POLICIES}
    passed=all(n>=6 for n in gates.values())
    lines.extend(['','## Predeclared familiar-development gate','',
                  f'Quartet: **{gates["quartet"]}/8**; distributed: **{gates["distributed"]}/8**. '
                  f'Overall: **{"PASS" if passed else "FAIL"}** (requires ≥6/8 in both).','',
                  '## Interpretation','',
                  '- The original primary endpoint shows no consistent across-seed quartet advantage under source–question matching.',
                  '- Earlier milestones suggest a possible learning-speed advantage; this is secondary and exploratory, not evidence of a guaranteed benefit.',
                  '- Different order, per-update source diversity, gradient covariance, and optimization trajectories remain inseparable in this intervention.',
                  '- Source–question multisets are matched within four-update windows, not within individual gradient updates.',
                  '- This study cannot establish equivalence or explain the original EXP-102 gap uniquely. The failure of the competence gate does not authorize test access.',
                  '- Neither L+I nor reserved data were used; all comparisons remain familiar-development only.','',
                  f'**PASS:** {len(data)}/16 completed reports; fixed seed pairs and eight milestones; hashes, metric identities and protected flags verified.',''])
    return '\n'.join(lines),gates


def write_once(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text()!=content: raise FileExistsError(f'Derived file changed; refusing overwrite: {path}')
        print('ALREADY VERIFIED',path)
    else:
        path.write_text(content)
        print('SAVED',path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write',action='store_true',help='Write the audited, reproducible report only; no training outputs changed.')
    args=ap.parse_args()
    data,phash=load()
    md,gate=render(data,phash)
    print(md)
    if args.write:
        write_once(AUDIT/'source_matched_dev_3200_audit.md',md)
        rows=StringIO()
        writer=csv.writer(rows,lineterminator="\n")
        writer.writerow(['seed','quartet_C_query','distributed_C_query','paired_gap'])
        for s in SEEDS:
            q=data[(s,'quartet')][-1]['dev']['C_query']
            d=data[(s,'distributed')][-1]['dev']['C_query']
            writer.writerow([s,f'{q:.9f}',f'{d:.9f}',f'{q-d:.9f}'])
        write_once(AUDIT/'source_matched_dev_3200_pairs.csv',rows.getvalue())


if __name__=='__main__':
    main()
