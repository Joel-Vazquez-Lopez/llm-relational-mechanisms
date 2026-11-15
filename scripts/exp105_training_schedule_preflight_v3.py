#!/usr/bin/env python3
"""EXP-105 v3: UNFROZEN development-only candidate training schedule audit.

No training, model/checkpoint reads, reserved examples, protected evaluation,
or writing unless --write is explicitly specified. Requires the EXP-105 v0,
v1 and v2 scripts in the same scripts/ directory. Designed as an auditable
proposal for a *later* disposable development-only pilot, not a preregistered
confirmatory protocol.

Run from repo root:
    python3 scripts/exp105_training_schedule_preflight_v3.py
    python3 scripts/exp105_training_schedule_preflight_v3.py --write
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import exp105_generator_preflight as v0
import exp105_position_control_preflight_v1 as v1
import exp105_exposure_matching_preflight_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/exp105/preflight'
SEEDS = (10701, 10703, 10709, 10713, 10721, 10727, 10729, 10733)
STEPS = 3200  # CANDIDATE pilot budget, not a confirmatory freeze
BLOCK = 45    # 9 modifier cells x 5 neutral frames
BASE_PER_VOICE = 7  # 7 AA + 7 PA = 14 shared base sources/update
DOSE_PER_UPDATE = 2
QUERIES_PER_SOURCE = 4
BASE_LABELS = 56
DOSE_LABELS = 8
LABELS_PER_UPDATE = 64
SAME_MODS = (('bare','bare'), ('bare','L'), ('L','bare'), ('L','L'), ('I','I'))
SWAP_MODS = ((('bare','I'), ('I','bare')), (('L','I'), ('I','L')))
STRATA = tuple((mods, fi) for mods in v0.TRAIN_MODS for fi in range(len(v1.FRAMES)))
assert len(STRATA) == BLOCK


def digest(o):
    return sha256(json.dumps(o, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def pools():
    train_sets, dev_sets, part = v0.partitions()
    base, aa, ap, dev_fam, dev_li = v0.generate(train_sets, dev_sets)
    pa = [v0.make_source(events, 'PA', mods, 'dose', cid)
          for j, cid, events in v0.cores(train_sets, 'train') if j == 2
          for mods in v0.TRAIN_MODS]
    assert len(pa) == len(ap) == len(aa) == 2700
    assert [(p.core_id,p.mods,p.gold) for p in pa] == [(a.core_id,a.mods,a.gold) for a in ap]
    common = []
    paired = []
    bpool, dpool = defaultdict(list), defaultdict(list)
    for source in base:
        for fi, frame in enumerate(v1.FRAMES):
            i = len(common)
            s = v1.expand(source, frame)
            common.append(s)
            bpool[(s.voice, s.mods, fi)].append(i)
    for a, b in zip(pa, ap):
        for fi, frame in enumerate(v1.FRAMES):
            i = len(paired)
            p, t = v1.expand(a, frame), v1.expand(b, frame)
            paired.append((p,t,fi))
            layout_parity = int(p.core_id.split('-')[1]) % 2
            dpool[(p.mods,fi,layout_parity)].append(i)
            assert p.gold == t.gold and p.core_id == t.core_id
            assert v0.parse_independent(v1.unframe(p.text)) == p.gold
            assert v0.parse_independent(v1.unframe(t.text)) == t.gold
    assert len(common) == 54000 and len(paired) == 13500
    assert set(bpool) == {(voice, mods, fi) for voice in ('AA','PA') for mods,fi in STRATA}
    assert set(dpool) == {(m,fi,parity) for m,fi in STRATA for parity in (0,1)}
    assert all(len(v) == 600 for v in bpool.values())
    assert all(len(v) == 150 for v in dpool.values())
    assert not ({s.text for s in common} & {p.text for p,t,fi in paired})
    assert not ({s.text for s in common} & {t.text for p,t,fi in paired})
    assert not any('LI' in x.mods for x in common)
    assert not any('LI' in p.mods or 'LI' in t.mods for p,t,_ in paired)
    assert all(s.voice in ('AA','PA') for s in common)
    assert all(p.voice == 'PA' and t.voice == 'AP' for p,t,_ in paired)
    assert {s.text for s in common}.isdisjoint(s.text for s in dev_fam + dev_li)
    assert {p.text for p,t,_ in paired}.isdisjoint(s.text for s in dev_fam + dev_li)
    assert {t.text for p,t,_ in paired}.isdisjoint(s.text for s in dev_fam + dev_li)
    return common, paired, bpool, dpool, part['partition_digest']


def templates():
    """45 paired-dose templates: each modifier/frame cell appears exactly twice.

    Matching I-modifier placement in opposite event positions cancels the
    otherwise 2-token per-source punctuation-length difference. A template
    contains two *different source cores* carrying the paired modifier cells.
    """
    result=[]
    for fi in range(len(v1.FRAMES)):
        for mod in SAME_MODS:
            result.append((mod, mod, fi))
        for m1, m2 in SWAP_MODS:
            result.extend([(m1,m2,fi), (m1,m2,fi)])
    assert len(result) == BLOCK
    hist=Counter((mod,fi) for m1,m2,fi in result for mod in (m1,m2))
    assert set(hist) == set(STRATA) and set(hist.values()) == {2}
    return result


def pick(rng, population, excluded, objects, core=False):
    """Sample with replacement across updates, without duplicates within an update."""
    for _ in range(100):
        i = population[rng.randrange(len(population))]
        if i not in excluded and (not core or objects[i].core_id not in excluded):
            return i
    raise AssertionError('Unable to select distinct source for this update')


def iter_schedule(seed, common, paired, bpool, dpool):
    """Yield the *actual audited* deterministic update indices.

    A future development pilot should IMPORT and consume this function, not
    copy its randomization logic. Yields (step, base_ix, dose_ix, base_strata,
    dose_strata). The two arms share indices; dose_ix is mapped to PA/AP.
    """
    rng = Random(seed + 1053000)
    for block_start in range(0, STEPS, BLOCK):
        base_slots={}
        for voice in ('AA','PA'):
            for slot in range(BASE_PER_VOICE):
                seq=list(STRATA)
                rng.shuffle(seq)
                base_slots[(voice,slot)] = seq
        tasks=templates()
        rng.shuffle(tasks)
        steps_in_block = min(BLOCK, STEPS - block_start)
        dose_stratum_occurrence=Counter()
        for off in range(steps_in_block):
            base_ix=[]
            base_strata=[]
            used=set()
            for voice in ('AA','PA'):
                for slot in range(BASE_PER_VOICE):
                    mod,fi=base_slots[(voice,slot)][off]
                    ix=pick(rng,bpool[(voice,mod,fi)],used,common)
                    used.add(ix)
                    base_ix.append(ix)
                    base_strata.append((voice,mod,fi))
            m1,m2,fi=tasks[off]
            dose_ix=[]
            dose_strata=[]
            used_dose_cores=set()
            for mod in (m1,m2):
                parity=dose_stratum_occurrence[(mod,fi)]%2
                dose_stratum_occurrence[(mod,fi)]+=1
                pool=dpool[(mod,fi,parity)]
                for _ in range(100):
                    ix=pool[rng.randrange(len(pool))]
                    if paired[ix][0].core_id not in used_dose_cores:
                        break
                else:
                    raise AssertionError('Could not choose two distinct paired source cores')
                used_dose_cores.add(paired[ix][0].core_id)
                dose_ix.append(ix)
                dose_strata.append((mod,fi,parity))
            assert len(base_ix)==14 and len(set(base_ix))==14
            assert len(dose_ix)==2 and len(set(dose_ix))==2
            yield block_start+off+1,base_ix,dose_ix,base_strata,dose_strata
        if steps_in_block==BLOCK:
            assert len(dose_stratum_occurrence)==BLOCK
            assert set(dose_stratum_occurrence.values())=={2}


def run_seed(seed, common, paired, bpool, dpool):
    schedule=sha256()
    totals=Counter()
    dose_hist=Counter()
    base_hist=Counter()
    complete_window_matches=0
    window_F=Counter()
    window_T=Counter()
    for step,base_ix,dose_ix,base_strata,dose_strata in iter_schedule(seed,common,paired,bpool,dpool):
        base_hist.update(base_strata)
        dose_hist.update((mod,fi) for mod,fi,parity in dose_strata)
        fam=[common[i] for i in base_ix]+[paired[i][0] for i in dose_ix]
        targ=[common[i] for i in base_ix]+[paired[i][1] for i in dose_ix]
        assert len(fam)==len(targ)==16
        assert sum(s.voice.count('P') for s in fam)==9
        assert sum(s.voice.count('P') for s in targ)==9
        assert sum(s.voice=='AA' for s in fam)==7
        assert sum(s.voice=='PA' for s in fam)==9
        assert sum(s.voice=='AP' for s in targ)==2
        assert sum(s.voice=='PA' for s in targ)==7
        assert [s.gold for s in fam]==[s.gold for s in targ]
        assert len(fam)*QUERIES_PER_SOURCE==LABELS_PER_UPDATE
        tokens_f=Counter()
        tokens_t=Counter()
        lengths_f=Counter()
        lengths_t=Counter()
        for s in fam:
            ts=v0.tokens(s.text)
            tokens_f.update(ts)
            lengths_f[len(ts)]+=1
        for s in targ:
            ts=v0.tokens(s.text)
            tokens_t.update(ts)
            lengths_t[len(ts)]+=1
        assert tokens_f==tokens_t,f'token multiset mismatch seed={seed} step={step}'
        window_F.update(lengths_f)
        window_T.update(lengths_t)
        item={'step':step,'base':base_ix,'dose':dose_ix}
        schedule.update((json.dumps(item,separators=(',',':'))+'\n').encode())
        totals['updates']+=1
        totals['source_slots']+=16
        totals['label_slots']+=64
        totals['base_labels']+=56
        totals['dose_labels']+=8
        totals['paired_passive_event_occurrences_per_arm']+=9
        totals['per_update_token_multiset_matches']+=1
        totals['per_update_source_length_hist_matches']+=(lengths_f==lengths_t)
        if step % BLOCK == 0:
            assert window_F==window_T,f'45-update window length histogram mismatch seed={seed}, step={step}'
            complete_window_matches+=1
            window_F.clear()
            window_T.clear()
    complete_blocks=STEPS//BLOCK
    for voice in ('AA','PA'):
        for mod,fi in STRATA:
            assert complete_blocks*BASE_PER_VOICE <= base_hist[(voice,mod,fi)] <= (complete_blocks+1)*BASE_PER_VOICE
    for mod,fi in STRATA:
        assert complete_blocks*2 <= dose_hist[(mod,fi)] <= (complete_blocks+1)*2
    assert complete_window_matches==complete_blocks
    assert totals['updates']==STEPS
    assert totals['label_slots']==STEPS*LABELS_PER_UPDATE
    return {
        'seed':seed,'schedule_sha256':schedule.hexdigest(),
        'updates':totals['updates'],'label_slots':totals['label_slots'],
        'base_labels':totals['base_labels'],'dose_labels':totals['dose_labels'],
        'matched_token_multiset_updates':totals['per_update_token_multiset_matches'],
        'matched_source_length_hist_updates':totals['per_update_source_length_hist_matches'],
        'different_source_length_hist_updates':STEPS-totals['per_update_source_length_hist_matches'],
        'complete_45step_windows_with_matched_source_length_hist':complete_window_matches,
        'passive_events_per_arm':totals['paired_passive_event_occurrences_per_arm'],
        'base_stratum_min':min(base_hist.values()),'base_stratum_max':max(base_hist.values()),
        'dose_stratum_min':min(dose_hist.values()),'dose_stratum_max':max(dose_hist.values()),
    }


def audit():
    common,paired,bpool,dpool,partition_hash=pools()
    # v2 audit validates aggregate exposure matching and no held-out syntax.
    inherited=v2.report()
    assert inherited['partition_digest']==partition_hash
    assert inherited['selected_primary_exposure_contrast']=='PA vs AP'
    assert inherited['per_arm']['PA']['missing_dev_LI_token_position_pairs']==0
    assert inherited['per_arm']['AP']['missing_dev_LI_token_position_pairs']==0
    rows=[run_seed(s,common,paired,bpool,dpool) for s in SEEDS]
    assert len({r['schedule_sha256'] for r in rows})==len(SEEDS)
    assert all(r['matched_token_multiset_updates']==STEPS for r in rows)
    assert all(r['complete_45step_windows_with_matched_source_length_hist']==STEPS//BLOCK for r in rows)
    return {
        'study':'EXP-105', 'phase':'UNFROZEN_SCHEDULE_PREFLIGHT_V3',
        'design_only':True, 'training_performed':False,'protected_examples_generated':False,
        'note':'Development-only *candidate* pilot schedule, not a frozen confirmatory protocol',
        'dependency_sha256':{
            p.name:file_hash(p) for p in (Path(v0.__file__),Path(v1.__file__),Path(v2.__file__),Path(__file__))},
        'partition_digest':partition_hash,
        'paired_seeds':list(SEEDS), 'candidate_optimizer_updates':STEPS,
        'labels_per_update':LABELS_PER_UPDATE,
        'base_source_count_per_update':2*BASE_PER_VOICE,
        'dose_source_count_per_update':DOSE_PER_UPDATE,
        'base_labels_per_update':BASE_LABELS,'dose_labels_per_update':DOSE_LABELS,
        'query_types_per_update':[16]*4,
        'passive_events_per_update_per_arm':9,
        'source_sampling':'without duplicate source per update; with replacement between updates',
        'stratification':'each 45-update full block: 7 base draws per voice/modifier/frame stratum and 2 dose draws per modifier/frame stratum, balanced 1/1 by source-core layout parity',
        'per_update_controls':[
            'identical base source IDs and all four questions',
            'matched semantic core ID, modifier cell, neutral frame and gold four labels in each dose pair',
            'same 16 source slots, 64 labels, passive count, question frequencies, token multiset',
            'source length histogram matched per full 45-update window, not necessarily per update',
        ],
        'dose_templates':45,
        'schedule_hashes_by_seed':rows,
        'remaining_limits':[
            'PA and AP differ in the event position of the passive construction; this is the intervention.',
            'Individual-update source length histograms can differ for L/I and I/L modifier doses; exact length matching is only over full 45-update windows.',
            'Absolute token positions and syntactic contexts are not identical; same token/length marginals do not equate positional frequencies.',
            'This is a proposed quartet-only scheduling regime; no quartet-vs-distributed contrast is included.',
            'Train/development grammar belongs to the existing synthetic family; independent replication remains needed.',
            'The complete candidate training protocol (model, loss, optimizer, gate, analysis) is not yet frozen.',
            'No learning or transfer result is claimed by a schedule audit.',
        ],
    }


def markdown(a):
    lines=[
        '# EXP-105 — prospective PA/AP update-schedule audit v3', '',
        '**Status: UNFROZEN design-only candidate. No training, checkpoints, reserved examples or protected evaluation.**','',
        f"Fixed candidate schedule: {a['candidate_optimizer_updates']} optimizer updates, {len(SEEDS)} fresh paired seeds, 64 labels/update.",
        f"Each update: 7 shared AA + 7 shared PA + 2 matched-dose PA/AP sources; all four questions per source.",
        f"Each arm receives 9 passive-event occurrences per update; the dose changes passive *event position*.",
        '',
        '| Per-update invariant | Status |', '|---|---|',
        '| Matched shared-base source IDs | PASS |',
        '| Matched paired dose semantic cores and four gold labels | PASS |',
        '| 16 sources / 64 labels / 16 of each query | PASS |',
        '| 9 passive-event occurrences per arm | PASS |',
        '| Aggregate token multiset in each individual update | PASS |',
        '| Source input-length histogram in each individual update | NOT GUARANTEED (documented) |',
        '| Source input-length histogram per complete 45-update window | PASS |',
        '| Nine modifier cells × five frame types and layout parity balanced over each full 45-update block | PASS |',
        '',
        '| Seed | SHA256 prefix | Updates audited | Token matches | Length matches (per update) |', '|---:|---|---:|---:|---:|',
    ]
    for r in a['schedule_hashes_by_seed']:
        lines.append(f"| {r['seed']} | `{r['schedule_sha256'][:16]}` | {r['updates']} | {r['matched_token_multiset_updates']}/{r['updates']} | {r['matched_source_length_hist_updates']}/{r['updates']} |")
    lines += [ '',
        f"**PASS:** {len(SEEDS)*STEPS:,} proposed paired updates audited. No model trained.", '',
        '## Limits before training',
        *['- '+s for s in a['remaining_limits']], '',
        '**Next:** peer-review the full pilot protocol and code for label leakage, source ordering, and optimizer setup before any training.', ''
    ]
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--write', action='store_true',help='Write new v3 report files once; never overwrite')
    args=p.parse_args()
    a=audit()
    s=markdown(a)
    print(s,flush=True)
    if args.write:
        OUT.mkdir(parents=True,exist_ok=True)
        paths=(OUT/'training_schedule_preflight_v3.md',OUT/'training_schedule_preflight_v3.json')
        if any(x.exists() for x in paths):
            raise FileExistsError('Refusing to overwrite an existing EXP-105 v3 audit')
        paths[0].write_text(s,encoding='utf-8')
        paths[1].write_text(json.dumps(a,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        for path in paths:print('SAVED',path)

if __name__=='__main__': main()
