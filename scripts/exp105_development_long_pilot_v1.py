#!/usr/bin/env python3
"""EXP-105 v1 development-only long-horizon learnability pilot.

Uses the existing EXP-105 v3 audited scheduler without changing its logic.
First 3200 updates use exactly the original v3 sequence; second 3200 use
an independently derived deterministic v3 sequence. This is a NEW
exploratory protocol, not a confirmation or protected evaluation.

Run (repository root):
 python3 scripts/exp105_development_long_pilot_v1.py --phase prepare
 python3 scripts/exp105_development_long_pilot_v1.py --phase smoke --device mps
 python3 -u scripts/exp105_development_long_pilot_v1.py --phase run --seeds 10801 --device mps
 python3 -u scripts/exp105_development_long_pilot_v1.py --phase run --seeds 10803 --device mps
 python3 scripts/exp105_development_long_pilot_v1.py --phase summary
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import torch

import exp105_development_pilot_v0 as pilot
import exp105_training_schedule_preflight_v3 as v3
import exp105_generator_preflight as generator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/exp105/development_long_pilot_v1'
CKPT = ROOT / 'results/exp105/development_long_pilot_v1'
SCRIPT = Path(__file__).resolve()
SEEDS = (10801, 10803)  # Exploratory ONLY; never use the 107xx proposed confirmatory panel.
STEPS = 6400
MILESTONES = (800, 1200, 1600, 2400, 3200, 4000, 4800, 5600, 6400)
OFFSET_SEED = 100_000


def digest(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def artifact_write_once(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise FileExistsError(f'Existing artifact differs; refusing to overwrite: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, sort_keys=True, indent=2)+'\n')
        print('SAVED', path, flush=True)


def setup():
    existing = pilot.read_and_verify_audit()
    common, paired, bpool, dpool, part_digest = v3.pools()
    assert existing['partition_digest'] == part_digest
    dev = pilot.familiar_dev_sources()
    assert len(dev) == 216 and set(x.voice for x in dev)=={'AA','PA','AP'}
    assert all('LI' not in x.mods for x in dev)
    vocab = pilot.vocab_for(common, paired, dev)
    parent = pilot.make_protocol(existing, common, paired, dev, vocab)
    pilot.ensure_protocol(parent, create=False)
    return existing, parent, common, paired, bpool, dpool, dev, vocab


def schedule(seed, common, paired, bpool, dpool):
    """Preserve original 3200-step prefix, then add another 3200-step schedule.

    Each half is the same v3 generator with a different deterministic RNG seed.
    Sampling may repeat sources between windows, just as in v3.
    """
    assert seed in SEEDS
    for half, sched_seed in enumerate((seed, seed+OFFSET_SEED)):
        yielded=0
        for inner_step, base, dose, base_strata, dose_strata in v3.iter_schedule(
                sched_seed, common, paired, bpool, dpool):
            assert inner_step == yielded+1
            yielded+=1
            yield half*v3.STEPS+inner_step, base, dose, base_strata, dose_strata
        assert yielded==v3.STEPS==3200


def audit_schedule(seed, common, paired, bpool, dpool):
    h=sha256()
    length_discrepancy=0
    matching=0
    completed_blocks=0
    window_f=Counter()
    window_t=Counter()
    for step, base, dose, _, _ in schedule(seed, common, paired, bpool, dpool):
        fam=[common[i] for i in base]+[paired[i][0] for i in dose]
        targ=[common[i] for i in base]+[paired[i][1] for i in dose]
        assert len(fam)==len(targ)==16 and [s.gold for s in fam]==[s.gold for s in targ]
        assert sum(s.voice.count('P') for s in fam)==sum(s.voice.count('P') for s in targ)==9
        tf=Counter(word for s in fam for word in generator.tokens(s.text))
        tt=Counter(word for s in targ for word in generator.tokens(s.text))
        assert tf==tt, f'Per-update token mismatch seed={seed} step={step}'
        matching+=1
        lf=Counter(len(generator.tokens(s.text)) for s in fam)
        lt=Counter(len(generator.tokens(s.text)) for s in targ)
        length_discrepancy += lf != lt
        window_f.update(lf)
        window_t.update(lt)
        item={'step':step,'base':base,'dose':dose}
        h.update((json.dumps(item,separators=(',',':'))+'\n').encode())
        # Two independent 3200-step halves. Each has 71 full 45-step blocks plus 5 steps.
        inner=step-v3.STEPS*((step-1)//v3.STEPS)
        if inner % v3.BLOCK == 0:
            assert window_f==window_t, f'45-update length mismatch seed={seed}, step={step}'
            completed_blocks+=1
            window_f.clear()
            window_t.clear()
        if inner==v3.STEPS:
            # The trailing 5 steps are not a complete stratification block.
            window_f.clear();window_t.clear()
    assert matching==STEPS and completed_blocks==2*(v3.STEPS//v3.BLOCK)
    return {'seed':seed,'total_updates':matching,'schedule_sha256':h.hexdigest(),
            'token_matching_updates':matching,
            'different_source_length_histogram_updates':length_discrepancy,
            'complete_balanced_45_step_blocks':completed_blocks,
            'remaining_partial_blocks_per_half':5,
            'no_LI_scoring':True,'no_reserved_evaluation':True}


def make_protocol(existing, parent, dev, vocab):
    files=(SCRIPT, *pilot.source_files())
    return {
        'study':'EXP-105','stage':'UNFROZEN_EXPLORATORY_DEVELOPMENT_LONG_PILOT_V1',
        'seeds':list(SEEDS),'steps':STEPS,'milestones':list(MILESTONES),
        'initialization':'paired identical model weights within each seed',
        'model':'exactly imports exp105_development_pilot_v0 Model',
        'optimizer':'exactly imports exp105_development_pilot_v0 AdamW setup',
        'training':'7 common AA + 7 common PA + 2 PA/AP dose, four queries per source',
        'schedule':'first 3200 original v3 seed; second 3200 original v3 RNG seeded with seed+100000',
        'schedule_first_half_compatible_with_pilot_v0':True,
        'note':'Starts from scratch. Old .pt checkpoints contain no optimizer states, so continuation is not mathematically equivalent.',
        'full_vocabulary_loss':'cross_entropy','decoding':'four-name candidate restricted, plus full vocab',
        'data_split_digest':existing['partition_digest'],
        'parent_protocol_hash':digest(parent),
        'pilot_seed_prohibition':'Only 10801/10803, not prospective 107xx',
        'development_samples':len(dev),'development_digest':digest([(s.text,s.gold) for s in dev]),
        'vocab_digest':digest(sorted(vocab.items())),'vocab_size':len(vocab),
        'code_hashes':{f.name:pilot.hash_file(f) for f in files},
        'primary_pilot_goal':'Does both-arm AA and PA learnability improve at 6400?',
        'secondary_pilot_goals':['AP development C_query','per-voice C_query',
                                 'wrong-event, wrong-role and same-answer rates','learning trajectories'],
        'no_confirmatory_claim':True,'protected_access':False,'LI_scoring':False,
        'checkpoint_selection':'fixed step 6400; no best checkpoint selection',
    }


def read_frozen(proto):
    path=OUT/'protocol.json'
    if not path.is_file():
        raise FileNotFoundError('Run --phase prepare first')
    if json.loads(path.read_text())!=proto:
        raise PermissionError('Long-pilot protocol/source changed since prepare; refusing')
    manifest_path=OUT/'schedule_audit.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Run --phase prepare first to audit both 6400-step schedules')
    manifest=json.loads(manifest_path.read_text())
    assert manifest['protocol_digest']==digest(proto)
    return manifest


def run_pair(seed, device, common, paired, bpool, dpool, dev, vocab, proto, manifest):
    path=OUT/f'pilot_seed{seed}_step{STEPS}.json'
    ckpath=CKPT/f'pilot_seed{seed}_step{STEPS}.pt'
    if path.exists() or ckpath.exists():
        if path.exists() and ckpath.exists():
            print(f'ALREADY COMPLETE seed {seed}: no overwrite', flush=True)
            return
        raise FileExistsError(f'Incomplete pair for seed {seed}; inspect manually')
    (f,of),(t,ot)=pilot.make_models(seed,device,len(vocab))
    de=pilot.encode_sources(dev,vocab)
    hist=[]
    h=sha256()
    for step, b, dose, _, _ in schedule(seed,common,paired,bpool,dpool):
        fam=[common[i] for i in b]+[paired[i][0] for i in dose]
        tar=[common[i] for i in b]+[paired[i][1] for i in dose]
        assert [s.gold for s in fam]==[s.gold for s in tar]
        ef=pilot.encode_sources(fam,vocab,check=False)
        et=pilot.encode_sources(tar,vocab,check=False)
        assert torch.equal(ef[2],et[2]) and torch.equal(ef[3],et[3])
        h.update((json.dumps({'step':step,'base':b,'dose':dose},separators=(',',':'))+'\n').encode())
        lf=pilot.one_update(f,of,ef,device)
        lt=pilot.one_update(t,ot,et,device)
        if step in MILESTONES:
            fm=pilot.score(f,de,dev,device,len(vocab))
            tm=pilot.score(t,de,dev,device,len(vocab))
            hist.append({'step':step,'familiar':fm,'target':tm})
            print(f'SEED {seed} STEP {step:4d} lossF={lf:.4f} lossT={lt:.4f} '
                  f'F AA={fm["per_voice_C_query"]["AA"]:.3f} F PA={fm["per_voice_C_query"]["PA"]:.3f} F AP={fm["per_voice_C_query"]["AP"]:.3f} '
                  f'T AA={tm["per_voice_C_query"]["AA"]:.3f} T PA={tm["per_voice_C_query"]["PA"]:.3f} T AP={tm["per_voice_C_query"]["AP"]:.3f}',flush=True)
    assert hist[-1]['step']==STEPS and h.hexdigest()==manifest['seeds'][str(seed)]['schedule_sha256']
    rec={'study':'EXP-105','phase':'UNFROZEN_LONG_DEV_PILOT','seed':seed,'step':STEPS,
         'development_only':True,'no_LI_scoring':True,'protected_evaluated':False,
         'protocol_digest':digest(proto),'schedule_sha256':h.hexdigest(),
         'device':str(device),'history':hist}
    OUT.mkdir(parents=True,exist_ok=True)
    CKPT.mkdir(parents=True,exist_ok=True)
    # Atomic-ish: checkpoint written first; if interrupted, inspect partial pair manually.
    torch.save({'seed':seed,'steps':STEPS,'familiar':f.state_dict(),
                'target':t.state_dict(),'protocol_digest':digest(proto)},ckpath)
    path.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print('SAVED',path,flush=True)
    print('SAVED',ckpath,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('prepare','smoke','run','summary'),required=True)
    ap.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    ap.add_argument('--seeds',type=int,nargs='+',default=[10801])
    args=ap.parse_args()
    assert set(args.seeds)<=set(SEEDS) and len(set(args.seeds))==len(args.seeds), 'Only exploratory 10801,10803 allowed'
    existing,parent,common,paired,bpool,dpool,dev,vocab=setup()
    proto=make_protocol(existing,parent,dev,vocab)
    if args.phase=='prepare':
        audits={str(seed):audit_schedule(seed,common,paired,bpool,dpool) for seed in SEEDS}
        artifact_write_once(OUT/'protocol.json',proto)
        manifest={'protocol_digest':digest(proto),'seeds':audits,'status':'DEVELOPMENT ONLY'}
        artifact_write_once(OUT/'schedule_audit.json',manifest)
        for seed in SEEDS:
            a=audits[str(seed)]
            print(f'PASS seed={seed}: {a["token_matching_updates"]}/6400 matched token updates; '
                  f'{a["complete_balanced_45_step_blocks"]} complete balanced 45-step blocks',flush=True)
        print('PREPARE PASS: no models trained, no protected/LI scores, no original files changed',flush=True)
        return
    manifest=read_frozen(proto)
    if args.phase=='smoke':
        device=pilot.get_device(args.device)
        (f,of),(t,ot)=pilot.make_models(SEEDS[0],device,len(vocab))
        for step,b,dose,_,_ in schedule(SEEDS[0],common,paired,bpool,dpool):
            fam=[common[i] for i in b]+[paired[i][0] for i in dose]
            tar=[common[i] for i in b]+[paired[i][1] for i in dose]
            ef=pilot.encode_sources(fam,vocab)
            et=pilot.encode_sources(tar,vocab)
            assert torch.equal(ef[2],et[2]) and torch.equal(ef[3],et[3])
            lf=pilot.one_update(f,of,ef,device)
            lt=pilot.one_update(t,ot,et,device)
            print(f'SMOKE step={step} lossF={lf:.4f} lossT={lt:.4f}',flush=True)
            if step==2: break
        de=pilot.encode_sources(dev[:9],vocab)
        for model in (f,t):
            s=pilot.score(model,de,dev[:9],device,len(vocab))
            assert all(0<=s[k]<=1 for k in ('accuracy4','C_query','wrong_event_rate','wrong_role_rate'))
        print('SMOKE PASS: paired updates, gradient checks and familiar development scoring; no checkpoints saved')
        return
    if args.phase=='run':
        device=pilot.get_device(args.device)
        for seed in args.seeds:
            assert manifest['seeds'][str(seed)]['total_updates']==STEPS
            print(f'RUN: exploratory development seed={seed} full restart at step 1; no LI scores',flush=True)
            run_pair(seed,device,common,paired,bpool,dpool,dev,vocab,proto,manifest)
        print('COMPLETE development-only pilot; no confirmatory seeds or tests accessed',flush=True)
        return
    assert args.phase=='summary'
    rows=[]
    for seed in SEEDS:
        path=OUT/f'pilot_seed{seed}_step{STEPS}.json'
        if not path.exists():
            print(f'MISSING pilot seed={seed}')
            continue
        rec=json.loads(path.read_text())
        assert rec['protocol_digest']==digest(proto) and rec['step']==STEPS
        r=rec['history'][-1]
        af=r['familiar']['per_voice_C_query']
        at=r['target']['per_voice_C_query']
        gap=at['AP']-af['AP']
        print(f'{seed} familiar AA={af["AA"]:.3f} PA={af["PA"]:.3f} AP={af["AP"]:.3f} '
              f'target AA={at["AA"]:.3f} PA={at["PA"]:.3f} AP={at["AP"]:.3f} AP_gap={gap:+.3f}')
        rows.append((seed, gap))
    print(f'{len(rows)}/{len(SEEDS)} pilot pairs complete; all results exploratory; no protected/L+I scoring')


if __name__=='__main__':
    main()
