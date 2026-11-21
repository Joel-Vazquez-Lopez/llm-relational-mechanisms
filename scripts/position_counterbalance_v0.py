#!/usr/bin/env python3
"""POSITION-CB v0: same-voice AA+PP versus mixed-voice AP+PA.

Run prepare, smoke, commit the manifest and all frozen files, then run on MPS.
No protected/reserved or within-event L+I examples are generated or scored.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from random import Random
from statistics import mean, stdev

import torch

import exp105_generator_preflight as gen
import exp105_position_control_preflight_v1 as frames
import exp105_development_pilot_v0 as pilot
import exp105_training_schedule_preflight_v3 as sched
import exp109_replication_v0 as exp109
import exp112_clause_state_causal_patch_v0 as exp112
import exp113_position_update_factorial_v0 as exp113
import exp114_independent_replication_v0 as exp114
import phase2_01_coverage_curve_v0 as phase201
import phase2_02_structural_transfer_v0 as phase202
import phase2_03_distractor_event_binding_v0 as phase203

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PROTOCOL = ROOT / 'paper/POSITION_CB_v0_protocol.md'
LAUNCHER = ROOT / 'scripts/run_position_counterbalance_v0.sh'
OUT = ROOT / 'reports/position_cb_v0'
CKPT = ROOT / 'results/position_cb_v0'
MANIFEST = OUT / 'manifest.json'
SEEDS = (12701, 12703, 12707, 12709, 12713, 12719, 12721, 12727)
REPEATS = (1, 2)
ARMS = ('same_voice', 'mixed_voice')
DIRECTIONS = ('AA', 'PP', 'AP', 'PA')
STEPS, PAIRS_PER_STEP, EVAL_SEED, PER_MOD = 6400, 8, 1272026, 16
TC95 = 2.364624251
assert not (set(SEEDS) & (set(exp109.SEEDS) | set(exp114.SEEDS) | set(phase201.SEEDS)
                          | set(phase202.SEEDS) | set(phase203.SEEDS)))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_once(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise FileExistsError('Frozen record differs: ' + str(path))
        print('VERIFIED', path, flush=True)
        return
    with path.open('x', encoding='utf8') as f:
        json.dump(obj, f, sort_keys=True, indent=2)
        f.write('\n')
    print('SAVED', path, flush=True)


def resources():
    common, dose, _bp, dp, partition, dev, vocab = exp109.resources()
    assert len(dose) == 13500 and len(dev) == 288
    assert set(dp) == {(mod, fi, parity) for mod in gen.TRAIN_MODS
                       for fi in range(len(frames.FRAMES)) for parity in (0, 1)}
    return dose, dp, partition, dev, vocab


def matched_pair(item):
    aa, pp, ap, pa = (item[k] for k in ('AA', 'PP', 'AP', 'PA'))
    assert aa.gold == pp.gold == ap.gold == pa.gold
    assert aa.core_id == pp.core_id == ap.core_id == pa.core_id
    assert aa.mods == pp.mods == ap.mods == pa.mods
    assert Counter(gen.tokens(aa.text) + gen.tokens(pp.text)) == Counter(
        gen.tokens(ap.text) + gen.tokens(pa.text))
    return (aa, pp), (ap, pa)


def schedule(seed, dp, limit=STEPS):
    assert seed in SEEDS and 0 < limit <= STEPS
    rng = Random(seed + 1272701)
    cells = list(sched.STRATA)
    assert len(cells) == 45
    block_idx = -1
    order = []
    for step in range(1, limit + 1):
        b = (step - 1) // len(cells)
        if b != block_idx:
            order = list(cells)
            rng.shuffle(order)
            block_idx = b
        mods, fi = order[(step - 1) % len(cells)]
        indices = rng.sample(dp[(mods, fi, 0)], 4) + rng.sample(dp[(mods, fi, 1)], 4)
        assert len(set(indices)) == PAIRS_PER_STEP
        rng.shuffle(indices)
        yield step, tuple(indices), (mods, fi)


def batch_sources(dose, indices):
    same, mixed = [], []
    for idx in indices:
        pair_same, pair_mixed = matched_pair(dose[idx])
        same.extend(pair_same)
        mixed.extend(pair_mixed)
    assert len(same) == len(mixed) == 16
    assert all(a.gold == b.gold for a, b in zip(same, mixed))
    return {'same_voice': same, 'mixed_voice': mixed}


def audit_schedule(seed, dp):
    h, freq = sha256(), Counter()
    for step, indices, (mods, fi) in schedule(seed, dp):
        freq[(mods, fi)] += 1
        payload = {'step': step, 'indices': list(indices), 'cell': [list(mods), fi]}
        h.update((json.dumps(payload, separators=(',', ':')) + '\n').encode())
    assert set(freq) == set(sched.STRATA) and max(freq.values()) - min(freq.values()) <= 1
    return {'seed': seed, 'steps': STEPS, 'schedule_sha256': h.hexdigest(),
            'strata_min': min(freq.values()), 'strata_max': max(freq.values())}


def previous_exclusions(dev):
    """Recreate *already evaluated* development surfaces; never score them."""
    old_text, old_orders, _a109, _b114 = phase201.previous_dev_exclusions(dev)
    _train, dev_sets, _part = gen.partitions()
    lookup = {frozenset(x): i for i, x in enumerate(dev_sets)}
    old114, _audit114 = exp114.novel_evaluation(dev)
    old117, a117 = phase201.evaluation(dev)
    old118, a118 = phase202.evaluation(dev)
    old119, a119 = phase203.evaluation(dev)
    for audit, old_manifest in ((a117, phase201.MANIFEST),
                                (a118, phase202.MANIFEST),
                                (a119, phase203.MANIFEST)):
        assert audit['evaluation_sha256'] == json.loads(
            old_manifest.read_text())['evaluation_audit']['evaluation_sha256']
    for bundle in (old114, old117):
        for group in bundle:
            for src in group.values():
                old_text.add(src.text)
            i = lookup[frozenset(group['AP'].gold)]
            old_orders.setdefault(i, set()).update((
                tuple(group['AP'].gold), tuple(group['PA_literal'].gold)))
    for group in old118:
        for version in phase202.VERSIONS:
            for src in group['source'][version].values():
                old_text.add(src.text)
        example = group['source']['canonical']['AP']
        i = lookup[frozenset(example.gold)]
        old_orders.setdefault(i, set()).update((tuple(example.gold),
                                                 tuple(example.gold[2:] + example.gold[:2])))
    for group in old119:
        for ref in phase203.REFERENTS:
            for d in phase203.DIRECTIONS:
                for pos in phase203.POSITIONS:
                    old_text.add(group['sources'][ref][d][pos].text)
        e = group['sources'][phase203.REFERENTS[0]]['AP']['canonical']
        i = lookup[frozenset(e.gold)]
        old_orders.setdefault(i, set()).update((tuple(e.gold),
                                                 tuple(e.gold[2:] + e.gold[:2])))
    return old_text, old_orders, {'phase2_01': a117['evaluation_sha256'],
                                  'phase2_02': a118['evaluation_sha256'],
                                  'phase2_03': a119['evaluation_sha256']}


def evaluation(dev):
    """New 144 source cores; no model inference and no reserved partition access."""
    old_text, old_orders, predecessor = previous_exclusions(dev)
    _train, dev_sets, partition = gen.partitions()
    rng = Random(EVAL_SEED)
    seen, used = set(old_text), defaultdict(set)
    groups, per_mod = [], Counter()
    for mods in gen.TRAIN_MODS:
        ids = list(range(len(dev_sets)))
        rng.shuffle(ids)
        for i in ids:
            perms = [p for p in permutations(dev_sets[i], 4)
                     if p not in old_orders.get(i, set()) and p not in used[i]
                     and (p[2], p[3], p[0], p[1]) not in old_orders.get(i, set())
                     and (p[2], p[3], p[0], p[1]) not in used[i]]
            rng.shuffle(perms)
            for a, b, c, d in perms:
                v0, v1 = gen.VERBS[(i+11) % len(gen.VERBS)], gen.VERBS[(i+13) % len(gen.VERBS)]
                assert v0 != v1
                events = (gen.Event(a, b, v0, 'A', c, d, i % 2),
                          gen.Event(c, d, v1, 'A', a, b, (i+1) % 2))
                pack = {voice: gen.make_source(events, voice, mods, 'position_cb_dev',
                                                f'newdev127-{i}-{per_mod[mods]}')
                        for voice in DIRECTIONS}
                texts = {s.text for s in pack.values()}
                if len(texts) != 4 or texts & seen:
                    continue
                if any(len(['<bos>'] + gen.tokens(s.text) + ['<sep>']
                           + gen.tokens(q) + ['<answer>']) > pilot.MAX_TOKENS
                       for s in pack.values() for q in gen.Q):
                    continue
                assert all(s.gold == (a,b,c,d) for s in pack.values())
                assert all(gen.parse_independent(s.text) == s.gold for s in pack.values())
                assert all(not any('LI' in m for m in s.mods) for s in pack.values())
                assert Counter(gen.tokens(pack['AA'].text) + gen.tokens(pack['PP'].text)) == Counter(
                    gen.tokens(pack['AP'].text) + gen.tokens(pack['PA'].text))
                seen.update(texts)
                used[i].update(((a,b,c,d), (c,d,a,b)))
                groups.append(pack)
                per_mod[mods] += 1
                break
            if per_mod[mods] == PER_MOD:
                break
        assert per_mod[mods] == PER_MOD, ('Insufficient unseen dev permutations', mods, per_mod[mods])
    assert len(groups) == 144 and len(seen - old_text) == 576
    material = [[(voice, g[voice].text, list(g[voice].gold)) for voice in DIRECTIONS]
                for g in groups]
    audit = {'evaluation_sha256': digest(material), 'anchors': 144,
             'presentations': 576, 'no_prior_surface_or_core_order_overlap': True,
             'no_protected_reserved_or_LI': True, 'same_grammar_dev_names': True,
             'partition_digest': partition['partition_digest'],
             'per_modifier': {str(k): v for k,v in sorted(per_mod.items())},
             'prior_public_eval_hashes': predecessor}
    return groups, audit


def hashes():
    names = ('exp105_generator_preflight.py', 'exp105_position_control_preflight_v1.py',
             'exp105_development_pilot_v0.py', 'exp105_training_schedule_preflight_v3.py',
             'exp106_voice_recombination_pilot_v0.py', 'exp109_replication_v0.py',
             'exp112_clause_state_causal_patch_v0.py', 'exp113_position_update_factorial_v0.py',
             'exp114_independent_replication_v0.py', 'phase2_01_coverage_curve_v0.py',
             'phase2_02_structural_transfer_v0.py', 'phase2_03_distractor_event_binding_v0.py')
    return {n: sha(ROOT/'scripts'/n) for n in names}


def protocol(vocab, part, dev, audit):
    return {'study': 'POSITION-CB-v0', 'stage': 'FROZEN_BEFORE_FIRST_TRAINING',
            'seeds': list(SEEDS), 'repeats': list(REPEATS), 'arms': list(ARMS),
            'directions': list(DIRECTIONS), 'steps': STEPS,
            'pairs_per_update': PAIRS_PER_STEP, 'labels_per_update': 64,
            'training': 'AA+PP versus AP+PA on the same 8 source cores/update',
            'matching': 'same source cores, labels and aggregate token bags; per-event voice balance',
            'nonmatched': 'within-source co-occurrence and all absolute token positions',
            'primary': 'across-arm familiar minus unseen C_query, paired seed t95 CI >0',
            'secondary': 'unseen minus familiar complete pair reversals, gated on primary',
            'eval_seed': EVAL_SEED, 'evaluation_sha256': audit['evaluation_sha256'],
            'vocab_digest': digest(sorted(vocab.items())),
            'dev_digest': digest([(s.text, s.gold) for s in dev]),
            'partition_digest': part, 'no_protected_or_LI': True,
            'script_sha256': sha(SCRIPT), 'protocol_sha256': sha(PROTOCOL),
            'launcher_sha256': sha(LAUNCHER), 'dependency_hashes': hashes(),
            'no_checkpoint_selection': True, 'no_exclusion_or_optional_stopping': True}


def prepare():
    if MANIFEST.exists():
        load_and_verify()
        print('PREPARE VERIFIED existing manifest, without scoring models')
        return
    dose, dp, part, dev, vocab = resources()
    for item in dose:
        matched_pair(item)  # audit every source-core AA+PP / AP+PA token bag
    groups, audit = evaluation(dev)
    assert all(set(gen.tokens(s.text)) <= set(vocab) for g in groups for s in g.values())
    study = protocol(vocab, part, dev, audit)
    schedule_hashes = {str(seed): audit_schedule(seed, dp) for seed in SEEDS}
    assert len({v['schedule_sha256'] for v in schedule_hashes.values()}) == 8
    write_once(MANIFEST, {'protocol': study, 'protocol_digest': digest(study),
                          'evaluation_audit': audit, 'schedule_audit': schedule_hashes,
                          'pair_audit': {'all_13500_variants_checked': True,
                                         'balanced_A_P_per_event_position': True,
                                         'exact_aggregate_token_bags': True}})
    print('PREPARE PASS: 144 fresh same-grammar dev anchors; commit manifest before RUN')


def load_and_verify():
    if not MANIFEST.exists():
        raise FileNotFoundError('Run PREPARE and commit the new manifest first')
    dose, dp, part, dev, vocab = resources()
    groups, audit = evaluation(dev)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['evaluation_audit'] == audit
    p = protocol(vocab, part, dev, audit)
    assert manifest['protocol'] == p and manifest['protocol_digest'] == digest(p)
    return manifest, (dose, dp, dev, vocab), groups


def freeze_verified():
    files = ('scripts/position_counterbalance_v0.py',
             'scripts/run_position_counterbalance_v0.sh',
             'paper/POSITION_CB_v0_protocol.md',
             'reports/position_cb_v0/manifest.json')
    try:
        head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,
                                       text=True).strip()
        for rel in files:
            committed = subprocess.check_output(['git','show',head+':'+rel],
                                                cwd=ROOT, stderr=subprocess.DEVNULL)
            if committed != (ROOT/rel).read_bytes():
                raise RuntimeError('Frozen bytes do not match git HEAD: '+rel)
        return head
    except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError) as err:
        raise RuntimeError('COMMIT runner, protocol, launcher and manifest before training') from err


def models(seed, device, n_vocab):
    torch.manual_seed(seed)
    initial = pilot.Model(n_vocab).to(device)
    result = {}
    for arm in ARMS:
        model = deepcopy(initial).to(device)
        assert all(torch.equal(x,y) for x,y in zip(model.parameters(), initial.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=pilot.LR,
                                      weight_decay=pilot.WEIGHT_DECAY)
        result[arm] = (model, optimizer)
    return result


def encode_groups(groups, vocab):
    return {v: pilot.encode_sources([g[v] for g in groups], vocab) for v in DIRECTIONS}


def score(model, enc, device):
    result = {}
    for voice, data in enc.items():
        logits = exp113.no_touch_identity(model, data, device)
        row = exp112.metrics(logits, data)
        _ids, _readout, gold, candidates = data
        g = gold.view(-1, 4)
        pred = candidates.gather(1, logits.gather(1, candidates).argmax(
            1, keepdim=True)).flatten().view(-1, 4)
        for event in range(2):
            k = event * 2
            p0,p1,g0,g1 = pred[:,k],pred[:,k+1],g[:,k],g[:,k+1]
            wrong = ~(((p0==g0)|(p0==g1)) & ((p1==g0)|(p1==g1)))
            classes = {'correct': (p0==g0)&(p1==g1)&~wrong,
                       'reversed': (p0==g1)&(p1==g0)&~wrong,
                       'collapsed': (p0==p1)&~wrong,
                       'wrong_event_name': wrong}
            assert torch.all(torch.stack(list(classes.values())).sum(0) == 1)
            row[f'event{event+1}_pair_classes'] = {
                name: float(mask.float().mean()) for name,mask in classes.items()}
        result[voice] = row
    return result


def smoke(device):
    dose, dp, _part, dev, vocab = resources()
    nets = models(SEEDS[0], device, len(vocab))
    for _step, indices, _cell in schedule(SEEDS[0], dp, limit=2):
        arms = batch_sources(dose, indices)
        for arm in ARMS:
            loss = pilot.one_update(*nets[arm],
                    pilot.encode_sources(arms[arm], vocab, check=False), device)
            assert math.isfinite(loss)
    old, _audit = exp109.evaluation(dev)
    legacy = [{'AA':g['AA'], 'PP':g['PP'], 'AP':g['AP'],
               'PA':g['PA_literal']} for g in old[:2]]
    for arm in ARMS:
        outcome = score(nets[arm][0].eval(), encode_groups(legacy, vocab), device)
        assert all(0 <= m['C_query'] <= 1 for m in outcome.values())
    print('SMOKE PASS: 2 optimizer steps and OLD explored dev scoring only')


def run(seed, repeat, device):
    if device.type != 'mps':
        raise ValueError('Scientific training requires Apple MPS; CPU for smoke only')
    assert seed in SEEDS and repeat in REPEATS
    man, (dose, dp, _dev, vocab), groups = load_and_verify()
    freeze_commit = freeze_verified()
    report = OUT / f'seed{seed}_repeat{repeat}_mps_step{STEPS}.json'
    ckpt = CKPT / f'seed{seed}_repeat{repeat}_mps_step{STEPS}.pt'
    if report.exists() or ckpt.exists():
        raise FileExistsError('Refusing to overwrite immutable report or checkpoint')
    enc = encode_groups(groups, vocab)
    nets = models(seed, device, len(vocab))
    h, history = sha256(), []
    for step, idxs, (mods, fi) in schedule(seed, dp):
        src = batch_sources(dose, idxs)
        losses = {arm: pilot.one_update(*nets[arm],
                  pilot.encode_sources(src[arm], vocab, check=False), device)
                  for arm in ARMS}
        payload = {'step':step, 'indices':list(idxs), 'cell':[list(mods),fi]}
        h.update((json.dumps(payload, separators=(',',':'))+'\n').encode())
        if step in exp109.MILESTONES:
            history.append({'step':step, 'losses':losses})
            print('STEP',step,'losses',
                  ' '.join(f'{arm}:{losses[arm]:.4f}' for arm in ARMS),flush=True)
    assert h.hexdigest() == man['schedule_audit'][str(seed)]['schedule_sha256']
    outcomes = {arm: score(nets[arm][0].eval(), enc, device) for arm in ARMS}
    record = {'study':'POSITION-CB-v0', 'status':'SEALED_NEW_SEED',
              'seed':seed, 'repeat':repeat, 'steps':STEPS, 'device':'mps',
              'freeze_commit':freeze_commit, 'script_sha256':sha(SCRIPT),
              'protocol_digest':man['protocol_digest'],
              'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
              'schedule_sha256':h.hexdigest(), 'loss_history':history,
              'outcomes':outcomes, 'no_protected_reserved_or_LI':True}
    CKPT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save({'study':'POSITION-CB-v0','seed':seed,'repeat':repeat,
                'step':STEPS,'protocol_digest':man['protocol_digest'],
                'models':{a:nets[a][0].state_dict() for a in ARMS}}, ckpt)
    write_once(report,record)
    print('SEALED, no per-run outcome disclosed. Finish all 16 runs first.')


def label_sets(arm):
    return (('AA','PP'),('AP','PA')) if arm=='same_voice' \
        else (('AP','PA'),('AA','PP'))


def measure(record, arm, familiar, metric):
    seen, unseen = label_sets(arm)
    target = seen if familiar else unseen
    if metric in ('C_query','accuracy4','full_vocab_accuracy'):
        return mean(record['outcomes'][arm][d][metric] for d in target)
    if metric in ('wrong_role','wrong_event'):
        return mean(record['outcomes'][arm][d]['by_query'][q][metric]
                    for d in target for q in ('first_agent','first_patient',
                                              'second_agent','second_patient'))
    if metric in ('correct','reversed','collapsed','wrong_event_name'):
        return mean(record['outcomes'][arm][d][f'event{e}_pair_classes'][metric]
                    for d in target for e in (1,2))
    raise KeyError(metric)


def ci(values):
    assert len(values)==8
    m=mean(values)
    radius=TC95*stdev(values)/math.sqrt(8)
    return {'mean':m,'ci95':[m-radius,m+radius],
            'per_seed':dict(zip(map(str,SEEDS),values)),
            'positive_seeds':sum(x>0 for x in values),
            'negative_seeds':sum(x<0 for x in values)}


def summary():
    man, _resources, _groups = load_and_verify()
    records, missing = {}, []
    for seed in SEEDS:
        for repeat in REPEATS:
            path = OUT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.json'
            checkpoint = CKPT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.pt'
            if not path.exists() or not checkpoint.exists():
                missing.append(f'{seed}/{repeat}')
                continue
            rec=json.loads(path.read_text())
            assert (rec['study'],rec['seed'],rec['repeat'],rec['steps']) == (
                'POSITION-CB-v0',seed,repeat,STEPS)
            assert rec['script_sha256']==sha(SCRIPT)
            assert rec['protocol_digest']==man['protocol_digest']
            assert rec['evaluation_sha256']==man['evaluation_audit']['evaluation_sha256']
            assert rec['schedule_sha256']==man['schedule_audit'][str(seed)]['schedule_sha256']
            assert rec['no_protected_reserved_or_LI'] and set(rec['outcomes'])==set(ARMS)
            for arm in ARMS:
                assert set(rec['outcomes'][arm])==set(DIRECTIONS)
                for d in DIRECTIONS:
                    for e in (1,2):
                        classes=rec['outcomes'][arm][d][f'event{e}_pair_classes']
                        assert abs(sum(classes.values())-1)<1e-5
            records[seed,repeat]=rec
    if missing:
        print('INCOMPLETE: '+str(len(missing))+'/16 missing; no summary written')
        print('Missing:',', '.join(missing))
        return
    def across(fn):
        return ci([mean(fn(records[s,r]) for r in REPEATS) for s in SEEDS])
    def pooled(rec,metric,familiar):
        return mean(measure(rec,arm,familiar,metric) for arm in ARMS)
    primary=across(lambda r:pooled(r,'C_query',True)-pooled(r,'C_query',False))
    secondary=across(lambda r:pooled(r,'reversed',False)-pooled(r,'reversed',True))
    primary_pass=primary['ci95'][0]>0
    secondary_pass=secondary['ci95'][0]>0 if primary_pass else None
    statistics={arm:{exposure:{metric:across(
                lambda r,a=arm,m=metric,f=(exposure=='familiar'):measure(r,a,f,m))
                for metric in ('C_query','accuracy4','full_vocab_accuracy',
                               'correct','reversed','collapsed','wrong_event_name',
                               'wrong_role','wrong_event')}
               for exposure in ('familiar','unseen')} for arm in ARMS}
    result={'study':'POSITION-CB-v0','status':'COMPLETE_MATCHED_VOICE_COOCCURRENCE',
            'source_records':16,'model_states':32,'n_independent_seeds':8,
            'no_protected_reserved_or_LI':True,
            'protocol_digest':man['protocol_digest'],
            'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
            'tests':{'primary_familiar_minus_unseen_C_query':
                     {'estimate':primary,'passes':primary_pass},
                     'gated_secondary_reversal_increase':
                     {'estimate':secondary,'passes':secondary_pass}},
            'arms':statistics,
            'limitations':'Same old grammar; different within-source voice cooccurrence; exact aggregate token bags but not all absolute position marginals'}
    write_once(OUT/'analysis_summary.json',result)
    print('SUMMARY PASS: 16 processes, 32 trained models, 8 independent seeds')
    for key,obj in result['tests'].items():
        est=obj['estimate']
        print(key,f"{100*est['mean']:+.2f} pp CI "
              f"[{100*est['ci95'][0]:+.2f},{100*est['ci95'][1]:+.2f}] "
              f"pass={obj['passes']}")
    for arm in ARMS:
        print(arm,'C_query familiar',statistics[arm]['familiar']['C_query']['mean'],
              'unseen',statistics[arm]['unseen']['C_query']['mean'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('prepare','smoke','run','summary'),required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--repeat',type=int)
    p.add_argument('--device',choices=('cpu','mps'),default='cpu')
    args=p.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='smoke':smoke(torch.device(args.device))
    elif args.phase=='run':
        if args.seed is None or args.repeat is None:
            p.error('RUN requires --seed and --repeat')
        run(args.seed,args.repeat,torch.device(args.device))
    else:summary()


if __name__=='__main__':
    main()
