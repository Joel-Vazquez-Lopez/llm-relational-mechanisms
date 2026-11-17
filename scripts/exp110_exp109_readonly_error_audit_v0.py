#!/usr/bin/env python3
"""EXP-110 v0: read-only post-hoc error analysis of *all* frozen EXP-109 JSON reports.

No torch, training, checkpoint access, source generation, or protected/LI scoring.
Experimental unit: seed (two numerical runs averaged first).

Usage (from exp102-clean-design worktree):
  python3 scripts/exp110_exp109_readonly_error_audit_v0.py
  python3 scripts/exp110_exp109_readonly_error_audit_v0.py --write

--write saves an additive JSON+Markdown audit under reports/exp110/read_only_error_audit_v0.
It never mutates any EXP-109 report and refuses to overwrite differing audit files.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
import json

ROOT = Path(__file__).resolve().parents[1]
IN_DEFAULT = ROOT / 'reports/exp109/replication_v0'
OUT_DEFAULT = ROOT / 'reports/exp110/read_only_error_audit_v0'
SEEDS = (10901, 10903, 10907, 10909, 10913, 10919, 10921, 10927)
REPEATS = (1, 2)
ARMS = ('pa_only', 'indirect_pp_aa', 'direct_pa_ap')
CONDITIONS = ('AP', 'PA_literal', 'AA', 'PP', 'AP_rename', 'AP_reverse_second')
METRICS = ('C_query', 'accuracy4', 'full_vocab_accuracy', 'wrong_event_rate',
           'wrong_role_rate', 'same_answer_rate')
CLASSES = ('correct', 'reversed', 'collapsed', 'wrong_event_name')
T7 = 2.364624251
N = 144
TOL = 1.5e-5


def close(a, b):
    return abs(float(a)-float(b)) <= TOL


def summarize(xs):
    assert len(xs) == len(SEEDS)
    m = mean(xs)
    s = stdev(xs)
    margin = T7*s/sqrt(len(xs))
    return {'mean': m, 'sd': s, 'ci95': [m-margin, m+margin],
            'per_seed': dict(zip(map(str, SEEDS), xs)),
            'n_independent_seeds': len(xs)}


def load(folder):
    records, files = {}, {}
    for seed in SEEDS:
        for repeat in REPEATS:
            path = folder / f'seed{seed}_repeat{repeat}_mps_step6400.json'
            if not path.is_file():
                raise FileNotFoundError(f'Missing required run: {path}. No partial analysis.')
            raw = path.read_bytes()
            r = json.loads(raw)
            assert r['study'] == 'EXP-109' and r['seed'] == seed and r['repeat'] == repeat
            assert r['device'] == 'mps' and r['steps'] == 6400
            assert r['no_protected_or_LI'] is True
            assert set(r['results']) == set(ARMS)
            for arm in ARMS:
                d = r['results'][arm]
                assert set(d['conditions']) == set(CONDITIONS)
                for condition in CONDITIONS:
                    metrics = d['conditions'][condition]
                    assert metrics['n'] == N and all(0<=metrics[k]<=1 for k in METRICS)
                classes = d['AP_second_event_pair_classes_n']
                assert sum(classes[c] for c in CLASSES) == N
                assert all(isinstance(classes[c], int) and classes[c]>=0 for c in CLASSES)
                p = d['AP_PA_literal_paired_n']
                assert sum(p.values()) == N
                assert set(p) == {'both_correct','AP_wrong_PA_correct',
                                  'AP_correct_PA_wrong','both_wrong'}
                assert close(d['conditions']['AP']['C_query'],
                             (p['both_correct']+p['AP_correct_PA_wrong'])/N)
                assert close(d['conditions']['PA_literal']['C_query'],
                             (p['both_correct']+p['AP_wrong_PA_correct'])/N)
            locked = r['predeclared_outcomes']
            assert close(locked['primary_pa_only_gap'],
                         r['results']['pa_only']['conditions']['PA_literal']['C_query']-
                         r['results']['pa_only']['conditions']['AP']['C_query'])
            assert close(locked['secondary_indirect_AP_lift'],
                         r['results']['indirect_pp_aa']['conditions']['AP']['C_query']-
                         r['results']['pa_only']['conditions']['AP']['C_query'])
            records[(seed, repeat)] = r
            files[path.name] = sha256(raw).hexdigest()
    # All runs must have the same frozen evaluation and protocol.
    assert len({x['protocol_digest'] for x in records.values()}) == 1
    assert len({x['evaluation_sha256'] for x in records.values()}) == 1
    assert len({x['freeze_commit'] for x in records.values()}) == 1
    assert all(records[(s,1)]['schedule_sha256'] == records[(s,2)]['schedule_sha256']
               for s in SEEDS)
    manifest = folder / 'manifest.json'
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        assert all(r['protocol_digest']==m['protocol_digest'] for r in records.values())
        assert all(r['evaluation_sha256']==m['evaluation_audit']['evaluation_sha256']
                   for r in records.values())
    return records, files


def seed_values(records, fn):
    return [mean(fn(records[(s,r)]) for r in REPEATS) for s in SEEDS]


def calculate(records, files):
    arm_condition = {}
    for arm in ARMS:
        arm_condition[arm] = {}
        for condition in CONDITIONS:
            arm_condition[arm][condition] = {
                metric: summarize(seed_values(records, lambda r, a=arm,c=condition,k=metric:
                    r['results'][a]['conditions'][c][k])) for metric in METRICS
            }
    roles = {arm: {c: summarize(seed_values(records,
                  lambda r,a=arm,k=c: r['results'][a]['AP_second_event_pair_classes_n'][k]/N))
                  for c in CLASSES} for arm in ARMS}
    comparisons = {}
    for metric in METRICS:
        comparisons[metric] = {
            'AP_minus_literal_PA_in_baseline': summarize(seed_values(records,lambda r,k=metric:
                r['results']['pa_only']['conditions']['AP'][k]-
                r['results']['pa_only']['conditions']['PA_literal'][k])),
            'indirect_minus_baseline_on_AP': summarize(seed_values(records,lambda r,k=metric:
                r['results']['indirect_pp_aa']['conditions']['AP'][k]-
                r['results']['pa_only']['conditions']['AP'][k]))
        }
    predeclared = {
        k: summarize(seed_values(records,lambda r, kk=k:r['predeclared_outcomes'][kk]))
        for k in ('primary_pa_only_gap', 'secondary_indirect_AP_lift')
    }
    # Sanity-check the locked primary contrast against the new descriptive aggregator.
    assert close(predeclared['primary_pa_only_gap']['mean'],
                 -comparisons['C_query']['AP_minus_literal_PA_in_baseline']['mean'])
    assert close(predeclared['secondary_indirect_AP_lift']['mean'],
                 comparisons['C_query']['indirect_minus_baseline_on_AP']['mean'])
    competence = {a:sum(bool(records[(s,r)]['predeclared_outcomes']
                              ['arm_competence_descriptive'][a])
                        for s in SEEDS for r in REPEATS) for a in ARMS}
    within_repeats = {}
    for metric in ('primary_pa_only_gap','secondary_indirect_AP_lift'):
        signed = [records[(s,2)]['predeclared_outcomes'][metric]-
                  records[(s,1)]['predeclared_outcomes'][metric] for s in SEEDS]
        within_repeats[metric] = {
            'signed_rep2_minus_rep1': dict(zip(map(str,SEEDS),signed)),
            'mean_absolute_difference':mean(map(abs,signed)),
            'maximum_absolute_difference':max(map(abs,signed))
        }
    paired = {a: {k:summarize(seed_values(records,lambda r,aa=a,kk=k:
                        r['results'][aa]['AP_PA_literal_paired_n'][kk]/N))
                   for k in ('both_correct','AP_wrong_PA_correct',
                             'AP_correct_PA_wrong','both_wrong')}
              for a in ARMS}
    return {
        'status':'POST_HOC_DESCRIPTIVE_ANALYSIS_NOT_A_NEW_CONFIRMATORY_TEST',
        'source_study':'EXP-109', 'frozen_primary_preserved':True,
        'experimental_unit':'8 seeds, 2 MPS numerical repeats averaged within seed',
        'source_files_sha256':files,
        'freeze_commit':next(iter(records.values()))['freeze_commit'],
        'protocol_digest':next(iter(records.values()))['protocol_digest'],
        'evaluation_sha256':next(iter(records.values()))['evaluation_sha256'],
        'locked_outcomes':predeclared,'descriptive_arm_conditions':arm_condition,
        'descriptive_AP_second_event_pair_classes':roles,
        'descriptive_comparisons':comparisons,
        'descriptive_AP_PA_paired_outcomes':paired,
        'competence_counts_out_of_16_not_filtered':competence,
        'within_seed_numerical_variation':within_repeats,
        'limitations':[
            'All reported new comparisons/error classes are post-hoc descriptive; no multiplicity correction.',
            'The per-question wrong-event and wrong-role categories partition errors; same-answer can overlap them.',
            'Second-event pair classes apply to native four-name candidate predictions, not oracle-restricted labels.',
            'Original AP vs literal PA changes event positions and question references despite preserving clause text.',
            'Indirect vs baseline differs in second-slot passive training and is not isolated grammatical abstraction.',
            'Same synthetic grammar and development name partition; not independent external generalization.',
            'No model internals assessed; internal circuit hypotheses remain untested.'
        ],
        'no_training_or_protected_access':True
    }


def pct(x): return f'{x*100:.1f}%'
def pp(x): return f'{x*100:+.1f} pp'


def markdown(out):
    arms=out['descriptive_arm_conditions'];cls=out['descriptive_AP_second_event_pair_classes']
    comparisons=out['descriptive_comparisons']; locked=out['locked_outcomes']
    lines=['# EXP-110 — Read-only error audit of EXP-109', '',
           '**Status:** post-hoc descriptive analysis of all 16 frozen EXP-109 reports. '
           'Not a fresh confirmatory test. No training, checkpoint reading, LI, reserved or protected evaluation.', '',
           '## Frozen endpoints (unchanged)', '',
           '| Effect | Seed-level mean | 95% t interval (8 seeds) |',
           '|---|---:|---:|']
    for key,label in [('primary_pa_only_gap','PA-only: literal PA − AP'),
                      ('secondary_indirect_AP_lift','Indirect AP − baseline AP')]:
        d=locked[key];lo,hi=d['ci95']
        lines.append(f'| {label} | {pp(d["mean"])} | [{pp(lo)}, {pp(hi)}] |')
    lines.extend(['', '## Native scores by construction (averages of seed means)', '',
                  '| Arm / construction | All four correct | Question accuracy | Wrong event | Wrong role | Same answer |',
                  '|---|---:|---:|---:|---:|---:|'])
    for arm in ARMS:
        for cond in ('AP','PA_literal'):
            d=arms[arm][cond]
            lines.append('| '+arm+' / '+cond+' | '+' | '.join(pct(d[k]['mean']) for k in
                ('C_query','accuracy4','wrong_event_rate','wrong_role_rate','same_answer_rate'))+' |')
    lines.extend(['', '## Baseline AP − literal PA: paired descriptive error changes', '',
                  '| Metric | Seed-mean difference | 95% t interval |',
                  '|---|---:|---:|'])
    for key in ('accuracy4','full_vocab_accuracy','wrong_event_rate',
                'wrong_role_rate','same_answer_rate'):
        d=comparisons[key]['AP_minus_literal_PA_in_baseline'];lo,hi=d['ci95']
        lines.append(f'| {key} | {pp(d["mean"])} | [{pp(lo)}, {pp(hi)}] |')
    lines.extend(['', '## AP second-event agent/patient pair taxonomy', '',
                  'Mutually exclusive outcomes under native four-name candidate predictions, '
                  'averaged across seed-level proportions. These are **not** oracle-restricted outcomes.', '',
                  '| Arm | Both correct | Reversed | Collapsed | Wrong event name |',
                  '|---|---:|---:|---:|---:|'])
    for arm in ARMS:
        lines.append('| '+arm+' | '+' | '.join(pct(cls[arm][k]['mean']) for k in CLASSES)+' |')
    lines.extend(['', '## AP joint outcomes after literal clause swap', '',
                  '| Arm | AP wrong, PA correct | AP correct, PA wrong | Both correct | Both wrong |',
                  '|---|---:|---:|---:|---:|'])
    outcomes=out['descriptive_AP_PA_paired_outcomes']
    for arm in ARMS:
        lines.append('| '+arm+' | '+' | '.join(pct(outcomes[arm][k]['mean']) for k in
            ('AP_wrong_PA_correct','AP_correct_PA_wrong','both_correct','both_wrong'))+' |')
    v=out['within_seed_numerical_variation']
    lines.extend(['', '## MPS numerical variability (repeat 2 − repeat 1)', '',
                  'These are absolute differences in outcome proportions, not independent seed effects.', '',
                  '| Outcome | Mean absolute difference | Maximum difference |',
                  '|---|---:|---:|'])
    for key in v:
        d=v[key]
        lines.append(f'| {key} | {pct(d["mean_absolute_difference"])} | '
                     f'{pct(d["maximum_absolute_difference"])} |')
    lines.extend(['', '## Interpretation and boundaries', '',
                  '- The original PA-only AP/PA gap is reproduced, and the AP error excess is dominated '
                  'numerically by wrong-role answers rather than wrong-event answers.',
                  '- AP second-event pair reversals and collapses are distinguishable behavioral outcomes; '
                  'this does not identify an internal role-binding circuit.',
                  '- Comparisons in this document beyond the two frozen endpoints are **exploratory**. '
                  'Their confidence intervals are descriptive and not adjusted for multiple comparisons.',
                  '- No successful checkpoints were selected, and no competence filtering was applied.',
                  '- Second-position voice, sentence-position heuristics and optimization remain alternative explanations.',
                  '- The original reserved and L+I tests remain closed.', '',
                  '## Falsifiable mechanistic *hypothesis* (not a result)', '',
                  'A model that has learned a first-event-passive role heuristic may fail to '
                  'invert agent/patient mapping when passive syntax moves into the second event. '
                  'Before circuit claims, compare role-specific logit margins and controlled '
                  'within-model interventions on matched clause-swap counterfactuals, with '
                  'position/event/perturbation controls and independently frozen replication.', '',
                  '## Provenance', '',
                  f'- Frozen EXP-109 commit: `{out["freeze_commit"]}`',
                  f'- Protocol digest: `{out["protocol_digest"]}`',
                  f'- Evaluation digest: `{out["evaluation_sha256"]}`',
                  '- Sources: all 16 JSON reports in `reports/exp109/replication_v0/`',
                  '- This analyzer reads source reports only; it does not generate or score examples.', ''])
    return '\n'.join(lines)


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:
            raise FileExistsError(f'Refusing to overwrite different existing audit: {path}')
        print('VERIFIED', path)
    else:
        with path.open('xb') as fp: fp.write(data)
        print('SAVED', path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=IN_DEFAULT)
    parser.add_argument('--out-dir',type=Path,default=OUT_DEFAULT)
    parser.add_argument('--write',action='store_true',help='Write derived JSON and Markdown audit (never overwrite different contents).')
    args=parser.parse_args()
    records,files=load(args.input_dir)
    out=calculate(records,files)
    print('PASS: validated 16/16 frozen EXP-109 reports; 8 independent seeds × 2 repeats × 3 arms')
    for key in ('primary_pa_only_gap','secondary_indirect_AP_lift'):
        a=out['locked_outcomes'][key]
        print(f'{key}: {pp(a["mean"])} 95% CI [{pp(a["ci95"][0])}, {pp(a["ci95"][1])}]')
    arms=out['descriptive_arm_conditions']
    for arm in ARMS:
        print(arm+' AP C_query='+pct(arms[arm]['AP']['C_query']['mean'])+
              ' literal PA='+pct(arms[arm]['PA_literal']['C_query']['mean']))
    for metric in ('wrong_event_rate','wrong_role_rate','same_answer_rate'):
        d=out['descriptive_comparisons'][metric]['AP_minus_literal_PA_in_baseline']
        print('PA-only AP minus PA_literal '+metric+': '+pp(d['mean']))
    print('PA-only AP second-event pair classes: '+', '.join(
        k+'='+pct(out['descriptive_AP_second_event_pair_classes']['pa_only'][k]['mean']) for k in CLASSES))
    print('POST-HOC DESCRIPTIVE AUDIT ONLY; frozen study unchanged; no LI/reserved/protected access')
    if args.write:
        write_once(args.out_dir/'analysis.json',
                   (json.dumps(out,indent=2,sort_keys=True)+'\n').encode())
        write_once(args.out_dir/'analysis.md', markdown(out).encode())

if __name__=='__main__':main()
