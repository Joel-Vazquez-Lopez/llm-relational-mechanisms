#!/usr/bin/env python3
"""EXP-103: source-matched 2x2 developmental factorial experiment.

Exposure: first-passive familiar vs second-passive targeted dose.
Update composition: source-quartets vs distributed complementary queries.

Within each 4-update window, policies see the SAME 64 source sentences and
EXACTLY the same 256 (source, question) labels, once each. Every update has
64 labels (16 per question), with 56 common-base and 8 dose labels.

New development study. NO reserved-set text, NO L+I composition, NO old
protected test regeneration. No changes to old checkpoints or old reports.

At repository root:
    python3 scripts/exp103_factorial_bridge.py --phase prepare
    python3 scripts/exp103_factorial_bridge.py --phase smoke --device mps
    python3 -u scripts/exp103_factorial_bridge.py --phase run --seeds 10501 --device mps
    python3 scripts/exp103_factorial_bridge.py --phase summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

import torch
from torch.nn import functional as F

import exp102_v2_familiar_training as v2
from exp102_syntax_preflight import Event, NAMES, VERBS, TRAIN, gold, parse_text, render

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / 'reports/exp103/factorial_dev'
CKPT = ROOT / 'results/exp103/factorial_dev'
PROTOCOL = OUT / 'protocol.json'
SEEDS = (10501, 10503, 10509, 10513, 10521, 10527, 10529, 10533)
EXPOSURES = ('familiar', 'target')
POLICIES = ('quartet', 'distributed')
STEPS = 3200
MILESTONES = (800, 1600, 2400, 3200)
SOURCE_SEED = 103101
BASE_PER_CELL = 64  # 9 modifier pairs x 2 voice pairs = 1152 source sentences
DOSE_PER_CELL = 8  # 9 modifier pairs x 1 voice pair = 72 matched dose sources
DEV_PER_CELL = 8  # 9 modifier pairs x 3 voice pairs = 216 development sentences
BASE_PER_WINDOW = 56
DOSE_PER_WINDOW = 8
LABELS_PER_UPDATE = 64


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def filehash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick(grouped, wanted, per_cell, seed, exclude=None):
    rng = Random(seed)
    output = []
    exclude = set() if exclude is None else set(exclude)
    for key in sorted(wanted):
        eligible = [r for r in grouped[key] if r[0] not in exclude]
        assert len(eligible) >= per_cell, (key, len(eligible))
        output.extend(rng.sample(eligible, per_cell))
    rng.shuffle(output)
    assert len({r[0] for r in output}) == len(output)
    return output


def swap_dose(row):
    """Reorder two semantic events, keeping verbs/agents/patients/modifiers.

    The familiar source has first passive / second active. The target
    source has first active / second passive. Gold order follows events.
    """
    source, (a,b,c,d), mods, voice = row
    assert voice == ('passive', 'active')
    verb0 = VERBS[(NAMES.index(a) + NAMES.index(c)) % len(VERBS)]
    verb1 = VERBS[(NAMES.index(b) + NAMES.index(d) + 1) % len(VERBS)]
    if verb1 == verb0:
        verb1 = VERBS[(VERBS.index(verb0) + 1) % len(VERBS)]
    e0 = Event(a,b,verb0,'passive',c,d,0)
    e1 = Event(c,d,verb1,'active',a,b,1)
    assert render((e0,e1), mods) == source, 'Source reconstruction mismatch'
    swapped_mods = tuple(reversed(mods))
    swapped_events = (e1,e0)
    target = render(swapped_events, swapped_mods)
    answer = gold(swapped_events)
    assert tuple(x for pair in parse_text(target) for x in pair) == answer
    assert target != source
    return (target, answer, swapped_mods, ('active','passive'))


def datasets():
    m = v2.split_manifest()  # Reserved four-name-set identifiers only.
    tr_sets = m['partitions']['train_name_sets']
    dv_sets = m['partitions']['development_name_sets']
    alltrain = v2.make_pool(tr_sets)
    alldev = v2.make_pool(dv_sets)
    base_keys = [k for k in alltrain if k[1] in (('active','active'), ('passive','active'))]
    dose_keys = [k for k in alltrain if k[1] == ('passive','active')]
    dev_keys = [k for k in alldev if k[1] in (('active','active'), ('passive','active'), ('active','passive'))]
    assert len(base_keys) == 18 and len(dose_keys) == 9 and len(dev_keys) == 27
    base = pick(alltrain, base_keys, BASE_PER_CELL, SOURCE_SEED)
    familiar_dose = pick(alltrain, dose_keys, DOSE_PER_CELL, SOURCE_SEED+1,
                         exclude={r[0] for r in base})
    target_dose = [swap_dose(r) for r in familiar_dose]
    dev = pick(alldev, dev_keys, DEV_PER_CELL, SOURCE_SEED+2)
    assert len(base) == 1152 and len(familiar_dose) == len(target_dose) == 72 and len(dev) == 216
    assert not {r[0] for r in base} & {r[0] for r in familiar_dose}
    assert not {r[0] for r in base} & {r[0] for r in target_dose}
    assert not {r[0] for r in familiar_dose} & {r[0] for r in target_dose}
    assert not {r[0] for r in dev} & ({r[0] for r in base} | {r[0] for r in familiar_dose} | {r[0] for r in target_dose})
    train_sets = {tuple(sorted(r[1])) for r in base + familiar_dose + target_dose}
    dev_sets = {tuple(sorted(r[1])) for r in dev}
    assert train_sets.isdisjoint(dev_sets)
    # Base supervision never queries the second passive event, because there
    # are no second-passive base sources. Only target dose does that.
    assert all(r[3][1] == 'active' for r in base)
    assert all(r[3] == ('passive','active') for r in familiar_dose)
    assert all(r[3] == ('active','passive') for r in target_dose)
    assert all(len(set(r[1])) == 4 and tuple(x for p in parse_text(r[0]) for x in p) == r[1]
               for r in base+familiar_dose+target_dose+dev)
    assert all(set(r[2]) <= {'bare','L','I'} for r in base+familiar_dose+target_dose+dev)
    assert len({r[2] for r in base}) == len(TRAIN)
    out = {
        'base':base,
        'dose':{'familiar':familiar_dose,'target':target_dose},
        'dev':{v:[r for r in dev if r[3] == v] for v in
                (('active','active'),('passive','active'),('active','passive'))},
    }
    assert all(len(v)==72 for v in out['dev'].values())
    identity = digest({
        'base':[(r[0],r[1]) for r in base],
        'dose':{k:[(r[0],r[1]) for r in rows] for k,rows in out['dose'].items()},
        'dev':{str(k):[(r[0],r[1]) for r in rows] for k,rows in out['dev'].items()},
    })
    return out, identity


def build_window(rng, nbase, ndose):
    """Build 64 DISTINCT source identifiers, 56 base + 8 dose.

    Every quartet step: 14 base + 2 dose sources x four queries.
    Every distributed step: 56 base + 8 dose sources x one query,
    rotated over all four questions in a Latin square. Every step has
    16 examples/question, including 14 base + 2 dose examples/question.
    """
    bs = rng.sample(range(nbase), BASE_PER_WINDOW)
    ds = [nbase+x for x in rng.sample(range(ndose), DOSE_PER_WINDOW)]
    rng.shuffle(bs);rng.shuffle(ds)
    groups = []
    for u in range(4):
        group = bs[u*14:(u+1)*14] + ds[u*2:(u+1)*2]
        rng.shuffle(group)
        groups.append(group)
    flat = [x for group in groups for x in group]
    offsets_base = [i for i in range(4) for _ in range(14)]
    offsets_dose = [i for i in range(4) for _ in range(2)]
    rng.shuffle(offsets_base);rng.shuffle(offsets_dose)
    offsets = {}
    for x in flat:
        offsets[x] = offsets_base.pop() if x < nbase else offsets_dose.pop()
    assert not offsets_base and not offsets_dose
    assert len(set(flat)) == len(flat) == 64
    return groups, flat, offsets


def update_labels(groups, flat, offsets, policy, u):
    if policy == 'quartet':
        items = [(x,q) for x in groups[u] for q in range(4)]
    elif policy == 'distributed':
        items = [(x,(offsets[x]+u)%4) for x in flat]
    else:
        raise ValueError(policy)
    assert len(items) == LABELS_PER_UPDATE
    assert Counter(q for _,q in items) == Counter({0:16,1:16,2:16,3:16})
    assert sum(x >= 1152 for x,_ in items) == 8  # nbase fixed by protocol
    return items


def schedule_integrity(seed, nbase, ndose, windows=12):
    rng = Random(seed + 103000)
    counts = Counter()
    for w in range(windows):
        groups, flat, offsets = build_window(rng,nbase,ndose)
        schedules = {}
        for p in POLICIES:
            steps = [update_labels(groups,flat,offsets,p,u) for u in range(4)]
            observed = Counter(item for step in steps for item in step)
            expected = Counter({(x,q):1 for x in flat for q in range(4)})
            assert observed == expected, f'{seed} window {w} {p}: mismatched source-question multiset'
            schedules[p] = observed
        assert schedules['quartet'] == schedules['distributed']
        counts['windows'] += 1
        counts['labels'] += 256
    return dict(counts)


def protocol(data_digest):
    return {
        'study':'EXP-103-factorial-development',
        'evidence_status':'NEW_EXPLORATORY_DEVELOPMENT_ONLY',
        'script_sha256':filehash(SCRIPT),
        'dependency_sha256':v2.source_hashes() | {
            'exp102_v2_familiar_training.py':filehash(Path(v2.__file__)),
        },
        'split_hash':v2.EXPECTED_SPLIT_HASH,
        'data_sha256':data_digest,
        'seeds':list(SEEDS),
        'factors':{'exposure':list(EXPOSURES),'policy':list(POLICIES)},
        'steps':STEPS,'milestones':list(MILESTONES),
        'sources':{'base':1152,'familiar_dose':72,'target_dose':72,
                   'per_window':64,'base_per_window':56,'dose_per_window':8},
        'batch':{'labels_per_update':64,'query_counts':[16,16,16,16],
                 'four_updates_exact_multiset_matched':True},
        'training':'3-layer 4-head width96, AdamW lr0.002 wd0.01, full-vocabulary CE, MPS/CPU',
        'primary_development': 'AP (first active / second passive) C_query at final step 3200: 2x2 exposure × batching interaction',
        'secondary_development':['AP per-question accuracy','AP wrong-event','AP wrong-role',
           'AP role-copy','AP per-event both-correct','AA/PA familiar competence','trajectory'],
        'competence_gate':'For generalization claims, >=6/8 seeds in EACH of four arms must reach AA and PA development per-question accuracy >=0.95 and average per-event both-correct >=0.90; this study NEVER opens reserved data regardless.',
        'existing_reserved_split':'Identifiers read only; zero reserved examples generated',
        'syntax':'Only familiar bare/L/I cells; L+I composition NEVER generated',
        'known_limits':'Matched source-question multiset per four updates, not matching per-update source diversity or Adam parameter trajectory; historical studies have different distributions.',
        'checkpoint_selection':'Fixed final step, no seed exclusion, no early stopping',
    }


def load_lock(now):
    if not PROTOCOL.is_file():
        raise FileNotFoundError(f'Run --phase prepare first ({PROTOCOL})')
    frozen = json.loads(PROTOCOL.read_text())
    if frozen != now:
        raise PermissionError('EXP-103 runner, dependencies, schedule, or data changed after freezing')
    return frozen


def write_once(path, object_value, torch_file=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Artifact already exists; refusing overwrite: {path}')
    temp = path.with_name(path.name+'.tmp')
    if temp.exists():
        raise FileExistsError(f'Leftover temporary artifact: {temp}')
    try:
        if torch_file:
            torch.save(object_value,temp)
        else:
            temp.write_text(json.dumps(object_value,indent=2,sort_keys=True)+'\n')
        os.replace(temp,path)
    finally:
        if temp.exists():
            temp.unlink()


def get_encoded(data,exposure):
    rows = data['base'] + data['dose'][exposure]
    return v2.encode(rows)


def dev_scores(model, decoded_dev, data, device):
    out = {}
    for voices, name in ((('active','active'),'AA'),(('passive','active'),'PA'),(('active','passive'),'AP')):
        m = v2.score(model,decoded_dev[name],data['dev'][voices],device)
        hit=(m['correct_event_identity_first']+m['correct_event_identity_second'])/2
        m['wrong_event_rate']=1-hit
        m['wrong_role_rate']=hit-m['accuracy4']
        m['same_answer_rate']=(m['role_copy_first']+m['role_copy_second'])/2
        m['per_event_both_correct']=(m['both_correct_first']+m['both_correct_second'])/2
        assert abs(m['accuracy4']+m['wrong_event_rate']+m['wrong_role_rate']-1)<1e-5
        out[name]=m
    return out


def run_one(seed, exposure, policy, data, data_digest, device, decoded_dev):
    dest = OUT / f'seed{seed}_{exposure}_{policy}.json'
    ckpt = CKPT / f'seed{seed}_{exposure}_{policy}.pt'
    if dest.exists() and ckpt.exists():
        j=json.loads(dest.read_text())
        assert j['seed']==seed and j['exposure']==exposure and j['policy']==policy and j['step']==STEPS
        assert j['protocol_sha256']==digest(protocol(data_digest))
        print('ALREADY COMPLETE:',seed,exposure,policy,flush=True)
        return
    if dest.exists() or ckpt.exists():
        raise FileExistsError(f'Partial artifact pair: {dest}, {ckpt}')
    torch.manual_seed(seed)
    model = v2.Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    x, readout, y, _ = get_encoded(data,exposure)
    nbase=len(data['base']); ndose=len(data['dose'][exposure])
    sampler=Random(seed+103000)
    records=[]
    step=0
    print(f'RUN {seed} {exposure} {policy}',flush=True)
    for w in range(STEPS//4):
        groups,flat,offsets=build_window(sampler,nbase,ndose)
        for u in range(4):
            ix=[4*s+q for s,q in update_labels(groups,flat,offsets,policy,u)]
            model.train()
            logits=model(x[ix].to(device),readout[ix].to(device))
            loss=F.cross_entropy(logits,y[ix].to(device))
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            step+=1
            if step in MILESTONES:
                dev=dev_scores(model,decoded_dev,data,device)
                records.append({'step':step,'dev':dev,'last_batch_loss':float(loss.detach())})
                print(f'STEP {step:4d} loss={loss.item():.4f} '
                      f'AP C_query={dev["AP"]["C_query"]:.3f} '
                      f'PA C_query={dev["PA"]["C_query"]:.3f} '
                      f'AA C_query={dev["AA"]["C_query"]:.3f} '
                      f'AP copy={dev["AP"]["same_answer_rate"]:.3f} '
                      f'AP wrong_event={dev["AP"]["wrong_event_rate"]:.3f}',flush=True)
    assert step == STEPS
    report={'study':'EXP-103-factorial-development','seed':seed,'exposure':exposure,
            'policy':policy,'step':STEPS,'device':device,'protocol_sha256':digest(protocol(data_digest)),
            'data_sha256':data_digest,'records':records,'protected_accessed':False,
            'reserved_cases_generated':False,'LI_cases_generated':False,
            'evidence_status':'EXPLORATORY_DEVELOPMENT_ONLY'}
    write_once(ckpt,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'metadata':report},torch_file=True)
    write_once(dest,report)
    print('SAVED',dest,flush=True)
    print('SAVED',ckpt,flush=True)


def show_summary():
    completed=[]
    for seed in SEEDS:
        names=[OUT / f'seed{seed}_{e}_{p}.json' for e in EXPOSURES for p in POLICIES]
        if not all(n.exists() for n in names):
            continue
        m={(e,p):json.loads((OUT/f'seed{seed}_{e}_{p}.json').read_text())['records'][-1]['dev']['AP']['C_query']
           for e in EXPOSURES for p in POLICIES}
        interaction=(m['target','quartet']-m['target','distributed']) - (m['familiar','quartet']-m['familiar','distributed'])
        completed.append(interaction)
        print(f'{seed}: '+ ' '.join(f'{e}/{p}={m[e,p]:.3f}' for e in EXPOSURES for p in POLICIES)
             +f' interaction={interaction:+.3f}')
    print(f'{len(completed)}/{len(SEEDS)} complete 2x2 seed quartets')
    if completed:
        print(f'EXPLORATORY mean 2x2 interaction = {sum(completed)/len(completed):+.4f}; '
              f'positive={sum(x>0 for x in completed)}/{len(completed)}')
    print('DEV ONLY: no L+I or reserved evaluation; no retroactive seed selection.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('prepare','smoke','run','summary'),required=True)
    ap.add_argument('--seeds',type=int,nargs='*')
    ap.add_argument('--device',choices=('cpu','mps'),default='mps')
    args=ap.parse_args()
    if args.seeds is not None and (len(set(args.seeds)) != len(args.seeds) or not args.seeds
                                    or any(s not in SEEDS for s in args.seeds)):
        ap.error(f'Choose distinct seeds from {SEEDS}')
    if args.phase == 'summary':
        show_summary();return
    if args.device=='mps' and args.phase in ('run','smoke') and not torch.backends.mps.is_available():
        ap.error('MPS unavailable')
    data, data_digest=datasets()
    now=protocol(data_digest)
    if args.phase=='prepare':
        for s in SEEDS:
            audit=schedule_integrity(s,len(data['base']),len(data['dose']['target']),windows=STEPS//4)
            assert audit['labels']==STEPS*LABELS_PER_UPDATE
        if PROTOCOL.exists():
            load_lock(now)
            print('EXISTING PROTOCOL VERIFIED:',PROTOCOL)
        else:
            write_once(PROTOCOL,now)
            print('PREPARED IMMUTABLE PROTOCOL:',PROTOCOL)
        print(f'BASE={len(data["base"])} DOSE={len(data["dose"]["familiar"])} '
              f'DEV={sum(map(len,data["dev"].values()))}')
        print(f'MATCHED MULTISETS: PASS ({STEPS//4} windows x {len(SEEDS)} seeds; both batching policies).')
        print('LEAK CHECK: PASS. No reserved or L+I cases generated.')
        return
    load_lock(now)
    decoded_dev={'AA':v2.encode(data['dev'][('active','active')]),
                 'PA':v2.encode(data['dev'][('passive','active')]),
                 'AP':v2.encode(data['dev'][('active','passive')])}
    if args.phase=='smoke':
        for exposure in EXPOSURES:
            for policy in POLICIES:
                torch.manual_seed(SEEDS[0]);model=v2.Model().to(args.device)
                opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
                x,readout,y,_=get_encoded(data,exposure)
                rng=Random(SEEDS[0]+103000)
                groups,flat,offsets=build_window(rng,len(data['base']),len(data['dose'][exposure]))
                ix=[4*s+q for s,q in update_labels(groups,flat,offsets,policy,0)]
                pred=model(x[ix].to(args.device),readout[ix].to(args.device))
                loss=F.cross_entropy(pred,y[ix].to(args.device))
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                sc=dev_scores(model,decoded_dev,data,args.device)
                assert torch.isfinite(loss).item()
                print(f'SMOKE {exposure}/{policy} loss={loss.item():.4f} AP C_query={sc["AP"]["C_query"]:.4f}')
        print('PASS: no checkpoints or experimental reports written')
        return
    for seed in (args.seeds if args.seeds is not None else SEEDS):
        for exposure in EXPOSURES:
            for policy in POLICIES:
                run_one(seed,exposure,policy,data,data_digest,args.device,decoded_dev)
    show_summary()


if __name__=='__main__':
    main()
