#!/usr/bin/env python3
"""EXP-106: voice-combination completion, UNFROZEN DEVELOPMENT-ONLY pilot.

Question: does seeing PP and AA, but *never AP*, enable generalization to AP?
Two matched primary dose conditions:
   INDIRECT: 1 PP + 1 AA on distinct matched source cores, no AP.
   DIRECT:   1 PA + 1 AP on those SAME source cores.
Each dose has one active/one passive first event, and one active/one passive
second event, four names per source, 4 questions/source, and exactly equal
per-update aggregate token bags. Matched direct/indirect are the key contrast.
Third descriptive control: 2 PA (never AP/PP), with same total passive count
but NOT the same event-position marginals; NOT a fully matched causal control.

Training uses only sources from pre-existing TRAIN name sets, bare/L/I mods,
neutral whole-sentence frames and never LI within an event. Development scoring
uses only AA, PA, AP and NEW PP *familiar-modifier* dev sources; no LI generated.
No protected/reserved examples are rendered, inspected or scored. Never use
107xx candidate confirmatory seeds. This is NOT a registered experiment.

From repository root:
  python3 scripts/exp106_voice_recombination_pilot_v0.py --phase prepare
  python3 scripts/exp106_voice_recombination_pilot_v0.py --phase smoke --device cpu
  python3 -u scripts/exp106_voice_recombination_pilot_v0.py --phase run --seed 10811 --device mps
  python3 scripts/exp106_voice_recombination_pilot_v0.py --phase summary
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import torch

import exp105_generator_preflight as gen
import exp105_position_control_preflight_v1 as frames
import exp105_training_schedule_preflight_v3 as sched
import exp105_development_pilot_v0 as pilot

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / 'reports/exp106/voice_recombination_pilot_v0'
CKPT = ROOT / 'results/exp106/voice_recombination_pilot_v0'
SEEDS = (10811, 10817)  # Fresh EXPLORATORY seeds, exclude 107xx and earlier 10801/10803.
STEPS = 6400
HALF = sched.STEPS
MILESTONES = (800, 1600, 2400, 3200, 4000, 4800, 5600, 6400)
ARMS = ('pa_only', 'indirect_pp_aa', 'direct_pa_ap')
LABELS = 64


def digest(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def code_hashes():
    return {p.name: sha256(p.read_bytes()).hexdigest() for p in
            (SCRIPT, Path(gen.__file__), Path(frames.__file__),
             Path(sched.__file__), Path(pilot.__file__))}


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'Existing file differs; will not overwrite: {path}')
        print('VERIFIED', path, flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('SAVED', path, flush=True)


def pools():
    """Rebuild only TRAIN grammar; deliberately avoid v0.generate's dev-LI design pool."""
    train_sets, dev_sets, part = gen.partitions()
    common = []
    dose = []
    bp = defaultdict(list)
    dp = defaultdict(list)
    for j, core_id, events in gen.cores(train_sets, 'train'):
        for mod in gen.TRAIN_MODS:
            if j in (0, 1):
                for voice in ('AA', 'PA'):
                    src = gen.make_source(events, voice, mod, 'base', core_id)
                    for fi, frame in enumerate(frames.FRAMES):
                        idx = len(common)
                        expanded = frames.expand(src, frame)
                        common.append(expanded)
                        bp[(voice, mod, fi)].append(idx)
            elif j == 2:
                voices = {v: gen.make_source(events, v, mod, 'dose', core_id)
                          for v in ('AA', 'PA', 'AP', 'PP')}
                for fi, frame in enumerate(frames.FRAMES):
                    idx = len(dose)
                    d = {v: frames.expand(s, frame) for v, s in voices.items()}
                    dose.append(d)
                    parity = int(core_id.split('-')[1]) % 2
                    dp[(mod, fi, parity)].append(idx)
    assert len(common) == 54000 and len(dose) == 13500
    assert set(bp) == {(v,m,fi) for v in ('AA','PA') for m,fi in sched.STRATA}
    assert set(dp) == {(m,fi,p) for m,fi in sched.STRATA for p in (0,1)}
    assert all(len(z)==600 for z in bp.values())
    assert all(len(z)==150 for z in dp.values())
    assert all(not any('LI' in mod for mod in s.mods) for s in common)
    assert all(not any('LI' in mod for mod in d['PP'].mods) for d in dose)
    return common, dose, bp, dp, part['partition_digest']


def dev_sources():
    """Retain the exact existing 216 familiar dev examples; append 72 PP examples."""
    legacy = pilot.familiar_dev_sources()  # only AA/PA/AP, no LI
    _train, dev_sets, _part = gen.partitions()
    pool = defaultdict(list)
    for _j, core_id, events in gen.cores(dev_sets, 'dev'):
        for mod in gen.TRAIN_MODS:
            pool[mod].append(gen.make_source(events, 'PP', mod, 'dev_familiar', core_id))
    rng = Random(106450)
    extra = []
    for mod in sorted(pool):
        extra.extend(rng.sample(pool[mod], pilot.DEV_PER_CELL))
    assert len(legacy)==216 and len(extra)==72
    assert all(s.voice=='PP' for s in extra)
    result = legacy+extra
    assert len({s.text for s in result})==len(result)==288
    assert all('LI' not in s.mods for s in result)
    return result


def train_schedule(seed, common, dose, bp, dp, limit=STEPS):
    """Shared base schedule from exact EXP-105 v3; new two-source same-cell doses.

    Each 45-step block visits every one of 9 modifier x 5 frame cells once.
    Source-core parity is paired 0/1; each chosen core has 4 variants AA PA AP PP.
    Core-to-condition assignment is independently randomized per update.
    """
    assert seed in SEEDS
    count = 0
    for half, s in enumerate((seed, seed+100000)):
        rng = Random(s + 1060000)
        cells = []
        block = -1
        for inner_step, base, _old_dose, _bs, _ds in sched.iter_schedule(s,common,
                                                 [(d['PA'],d['AP'],0) for d in dose],bp,dp):
            if (inner_step-1)//sched.BLOCK != block:
                block = (inner_step-1)//sched.BLOCK
                cells = list(sched.STRATA)
                rng.shuffle(cells)
            mod, fi = cells[(inner_step-1)%sched.BLOCK]
            a = rng.choice(dp[(mod, fi, 0)])
            b = rng.choice(dp[(mod, fi, 1)])
            assert dose[a]['PA'].core_id != dose[b]['PA'].core_id
            if rng.randrange(2):
                a,b=b,a
            step = half*HALF+inner_step
            count += 1
            yield step, base, (a,b), (mod,fi)
            if count == limit:
                return
    assert count==limit


def arm_sources(common, dose, base, selected):
    a,b = selected
    shared = [common[i] for i in base]
    return {
        'pa_only': shared+[dose[a]['PA'], dose[b]['PA']],
        'indirect_pp_aa': shared+[dose[a]['PP'], dose[b]['AA']],
        'direct_pa_ap': shared+[dose[a]['PA'], dose[b]['AP']],
    }


def verify_step(arms, mod=None, fi=None):
    assert set(arms)==set(ARMS)
    srcs=list(arms.values())
    assert all(len(row)==16 for row in srcs)
    assert [s.gold for s in srcs[0]]==[s.gold for s in srcs[1]]==[s.gold for s in srcs[2]]
    assert [s.core_id for s in srcs[0]]==[s.core_id for s in srcs[1]]==[s.core_id for s in srcs[2]]
    assert all(sum(s.voice.count('P') for s in row)==9 for row in srcs)
    assert all(sum(s.voice=='AP' for s in row)==0 for row in srcs[:2])
    assert sum(s.voice=='AP' for s in srcs[2])==1
    assert sum(s.voice=='PP' for s in srcs[1])==1
    # The primary arms have identical *event-slot* voice frequencies.
    assert all(sum(s.voice[0]=='P' for s in row)==8 and
               sum(s.voice[1]=='P' for s in row)==1 for row in srcs[1:])
    assert sum(s.voice[0]=='P' for s in srcs[0])==9
    assert sum(s.voice[1]=='P' for s in srcs[0])==0
    assert sum(s.voice=='AA' for s in srcs[1])==8  # seven common AA + one AA dose
    assert all(not any('LI' in x for x in s.mods) for row in srcs for s in row)
    if mod is not None:
        assert all(s.mods == mod for row in srcs for s in row[-2:])
        assert all(s.text.startswith(frames.FRAMES[fi]) for row in srcs for s in row[-2:])
    # Primary matched contrast: exact per-update aggregate token bags.
    indirect=Counter(w for s in arms['indirect_pp_aa'] for w in gen.tokens(s.text))
    direct=Counter(w for s in arms['direct_pa_ap'] for w in gen.tokens(s.text))
    assert indirect==direct, 'Indirect/direct token multiset mismatch'
    # The intervention necessarily changes passive clustering / source length.
    length_i=Counter(len(gen.tokens(s.text)) for s in arms['indirect_pp_aa'])
    length_d=Counter(len(gen.tokens(s.text)) for s in arms['direct_pa_ap'])
    return length_i==length_d


def audit(seed, common, dose, bp, dp):
    h = sha256()
    tok_matches = 0
    length_matches = 0
    strata = Counter()
    core_order = Counter()
    for step, b, pair, stratum in train_schedule(seed,common,dose,bp,dp):
        arms = arm_sources(common,dose,b,pair)
        length_matches += verify_step(arms,*stratum)
        tok_matches += 1
        strata[stratum]+=1
        core_order[tuple(pair)]+=1
        h.update((json.dumps({'step':step,'base':b,'dose':list(pair),
                              'stratum':[list(stratum[0]),stratum[1]]},
                             separators=(',',':'))+'\n').encode())
    assert tok_matches==STEPS
    assert set(strata)==set(sched.STRATA)
    assert min(strata.values())>=STEPS//sched.BLOCK
    assert max(strata.values()) <= (2*(HALF//sched.BLOCK)+2)
    return {'seed':seed,'steps':STEPS,'schedule_sha256':h.hexdigest(),
            'exact_indirect_direct_token_multiset_updates':tok_matches,
            'matched_indirect_direct_source_length_hist_updates':length_matches,
            'different_indirect_direct_source_length_hist_updates':STEPS-length_matches,
            'min_modifier_frame_uses':min(strata.values()),
            'max_modifier_frame_uses':max(strata.values()),
            'all_primary_arms_9_passive_events_per_update':True,
            'both_indirect_and_pa_only_never_include_AP':True,
            'no_LI_or_reserved_examples':True}


def setup():
    parent = pilot.read_and_verify_audit()  # enforce unchanged preflight sources
    common,dose,bp,dp,partition = pools()
    assert partition==parent['partition_digest']
    dev=dev_sources()
    paired=[(d['PA'],d['AP'],0) for d in dose]
    vocab=pilot.vocab_for(common,paired,dev)
    assert all(set(gen.tokens(s.text)) <= set(vocab) for s in dev)
    # Four-voice development data remains drawn only from existing dev sets.
    return common,dose,bp,dp,dev,vocab,partition


def protocol(common,dose,dev,vocab,partition):
    return {
        'study':'EXP-106', 'stage':'UNFROZEN_EXPLORATORY_VOICE_RECOMBINATION_PILOT_V0',
        'purpose':'Does PP+AA evidence without AP transfer to unseen AP?',
        'arms':list(ARMS), 'seeds':list(SEEDS), 'steps':STEPS,
        'milestones':list(MILESTONES), 'device_not_frozen':True,
        'control':'2 PA; never AP or PP; matched passive dose count but not passive position',
        'primary_matched_contrast':'1 PP + 1 AA vs 1 PA + 1 AP on same two cores',
        'dose_count':'2 sources, 8 questions, 2 passive events per update in each arm',
        'shared_base':'7 AA + 7 PA sources; 56 questions per update',
        'matching':'identical core IDs, same gold, modifiers, frame, total passives, event-slot A/P marginals and exact per-update bag of source tokens for indirect/direct',
        'unmatched':'source-length histograms and within-source cooccurrence (the intended composition); PA-only position distribution',
        'training':'AdamW lr=.002 wd=.01; original EXP105 model; cross entropy over full vocabulary',
        'evaluation':'288 development-only familiar-modifier AA/PA/AP/PP sources; candidate-restricted four name C_query, accuracy4, error types; no LI',
        'selection':'fixed step 6400; no best checkpoint selection',
        'development_digest':digest([(s.text,s.gold) for s in dev]),
        'vocab_digest':digest(sorted(vocab.items())),
        'train_partition_digest':partition,
        'source_count_common':len(common),'source_count_dose':len(dose),
        'vocab_size':len(vocab),'code_sha256':code_hashes(),
        'confirmatory_seeds_untouched':list(sched.SEEDS),
        'not_confirmatory':True,'protected_access':False,'LI_scoring':False,
        'caveats':['Direct exposure to AP predictably helps AP; this is a competence/upper anchor, not transfer evidence.',
                   'If indirect beats PA-only, difference includes presence of second-position passive: does not isolate abstract binding.',
                   'Matching marginal tokens does not match sequence lengths, within-source voice clustering, or positional token contexts.',
                   'MPS has proven non-bitwise reproducible for EXP105; independent seeds and repeated runs are needed for inference.',
                   'The familiar development AP/PP conditions use new name combinations but the same synthetic grammar family.']}


def load_audit(proto):
    p=OUT/'protocol.json'; a=OUT/'schedule_audit.json'
    if not (p.is_file() and a.is_file()):
        raise FileNotFoundError('Run --phase prepare first; no existing files will be overwritten')
    if json.loads(p.read_text())!=proto:
        raise RuntimeError('Prepared protocol does not match source code/data; refusing run')
    manifest=json.loads(a.read_text())
    assert manifest['protocol_digest']==digest(proto)
    return manifest


def models(seed,device,n_vocab):
    torch.manual_seed(seed)
    ref=pilot.Model(n_vocab).to(device)
    result={}
    for arm in ARMS:
        m=deepcopy(ref).to(device)
        assert all(torch.equal(x,y) for x,y in zip(ref.parameters(),m.parameters()))
        o=torch.optim.AdamW(m.parameters(),lr=pilot.LR,weight_decay=pilot.WEIGHT_DECAY)
        result[arm]=(m,o)
    return result


def run(seed,device,common,dose,bp,dp,dev,vocab,proto,manifest,smoke=False):
    steps=2 if smoke else STEPS
    out=OUT/f'seed{seed}_{device.type}_step{steps}.json'
    ckp=CKPT/f'seed{seed}_{device.type}_step{steps}.pt'
    if not smoke and (out.exists() or ckp.exists()):
        raise FileExistsError(f'Run artifacts already exist; refusing overwrite: {out} / {ckp}')
    network=models(seed,device,len(vocab))
    encoded_dev=pilot.encode_sources(dev,vocab)
    history=[]
    h=sha256()
    for step,base,pair,stratum in train_schedule(seed,common,dose,bp,dp,limit=steps):
        sources=arm_sources(common,dose,base,pair)
        verify_step(sources,*stratum)
        enc={arm:pilot.encode_sources(sources[arm],vocab,check=False) for arm in ARMS}
        assert all(torch.equal(enc['pa_only'][2],enc[a][2]) and
                   torch.equal(enc['pa_only'][3],enc[a][3]) for a in ARMS[1:])
        h.update((json.dumps({'step':step,'base':base,'dose':list(pair),
                              'stratum':[list(stratum[0]),stratum[1]]},
                             separators=(',',':'))+'\n').encode())
        losses={arm:pilot.one_update(*network[arm],enc[arm],device) for arm in ARMS}
        if step in MILESTONES or step==steps:
            metrics={arm:pilot.score(network[arm][0],encoded_dev,dev,device,len(vocab))
                     for arm in ARMS}
            history.append({'step':step,'metrics':metrics,'losses':losses})
            print(f'SEED {seed} STEP {step} '+ ' '.join(
                f'{arm} AA={metrics[arm]["per_voice_C_query"]["AA"]:.3f}'
                f' PA={metrics[arm]["per_voice_C_query"]["PA"]:.3f}'
                f' AP={metrics[arm]["per_voice_C_query"]["AP"]:.3f}'
                f' PP={metrics[arm]["per_voice_C_query"]["PP"]:.3f}'
                for arm in ARMS),flush=True)
    if smoke:
        assert len(history)==1 and history[-1]['step']==2
        print('SMOKE PASS: three matched initialized models, two updates, AA/PA/AP/PP dev scoring; no files written',flush=True)
        return
    assert h.hexdigest()==manifest['seeds'][str(seed)]['schedule_sha256']
    report={'study':'EXP-106','phase':'DEVELOPMENT_ONLY','seed':seed,'device':str(device),
            'step':STEPS,'protocol_digest':digest(proto),'schedule_sha256':h.hexdigest(),
            'history':history,'no_LI_scoring':True,'no_protected_evaluation':True}
    OUT.mkdir(parents=True,exist_ok=True)
    CKPT.mkdir(parents=True,exist_ok=True)
    torch.save({'models':{a:network[a][0].state_dict() for a in ARMS},
                'optimizers':{a:network[a][1].state_dict() for a in ARMS},
                'seed':seed,'step':STEPS,'protocol_digest':digest(proto)},ckp)
    with out.open('x') as f:
        json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print('SAVED',out,flush=True)
    print('SAVED',ckp,flush=True)
    print('DEVELOPMENT ONLY: protected and L+I remain closed',flush=True)


def summary(proto):
    rows=[]
    for seed in SEEDS:
        files=list(OUT.glob(f'seed{seed}_*_step{STEPS}.json'))
        if not files:
            print('MISSING seed',seed)
            continue
        for path in files:
            rec=json.loads(path.read_text())
            assert rec['protocol_digest']==digest(proto)
            m=rec['history'][-1]['metrics']
            print(f'SEED {seed} device={rec["device"]} '+ ' '.join(
                f'{arm}: AP={m[arm]["per_voice_C_query"]["AP"]:.4f} PP={m[arm]["per_voice_C_query"]["PP"]:.4f}'
                for arm in ARMS))
            rows.append({'seed':seed,'device':rec['device'], 'metrics':m})
    print('REPORT: exploratory only, no L+I, no confirmation; do not infer population-level effects from two seeds')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','smoke','run','summary'),required=True)
    p.add_argument('--seed',type=int,default=10811,choices=SEEDS)
    p.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    a=p.parse_args()
    common,dose,bp,dp,dev,vocab,part=setup()
    proto=protocol(common,dose,dev,vocab,part)
    if a.phase=='prepare':
        checks={str(seed):audit(seed,common,dose,bp,dp) for seed in SEEDS}
        assert all(x['exact_indirect_direct_token_multiset_updates']==STEPS for x in checks.values())
        write_once(OUT/'protocol.json',proto)
        write_once(OUT/'schedule_audit.json',{'protocol_digest':digest(proto),'seeds':checks})
        for seed in SEEDS:
            s=checks[str(seed)]
            print(f'PASS seed={seed}: {STEPS}/{STEPS} matched indirect/direct token-multiset updates; '
                  f'{s["different_indirect_direct_source_length_hist_updates"]} unmatched per-update length histograms')
        print('PREPARE PASS: no training, no protected/LI evaluations, confirmatory 107xx seeds untouched')
        return
    manifest=load_audit(proto)
    if a.phase=='summary':
        summary(proto)
        return
    devc=pilot.get_device(a.device)
    run(a.seed,devc,common,dose,bp,dp,dev,vocab,proto,manifest,smoke=a.phase=='smoke')

if __name__=='__main__':main()
