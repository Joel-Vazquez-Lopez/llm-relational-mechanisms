#!/usr/bin/env python3
"""EXP-101 R3: production DATA/TRAINING candidate for code inspection ONLY.

RUNNABLE: --phase preflight (stdlib only; no torch, model, checkpoint, or
protected item access).  Training, checkpoint evaluation, and protected phases
are NOT exposed in this CLI and MUST NOT be enabled until the full protocol,
source hashes and split manifests have been reviewed and frozen.  This module
contains a callable, but deliberately unexposed, training implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from exp101_generator_preflight import (Event, Packet, NAMES, CONDS, VOICES,
                                        render_packet, validate_case, other_event_query)
from exp101_render_validator import VERBS

SEEDS = (907, 911, 919, 929, 937, 941, 947, 953)
ARMS = ("familiar", "target")
FAMILIAR = ((True,"active"),(True,"passive"),(False,"active"))
DEV_CONDS = ("bare", "L", "I")
STEPS = 12800
SHARED_BATCH = 56
DOSE_BATCH = 8
SPLIT_RNG = 101103
WIDTH, HEADS, LAYERS, MAX_TOKENS = 96, 4, 3, 64
SPLIT_REPORT = Path("reports/exp101/production_preflight_candidate.json")
TOKEN_RE = re.compile(r"'s|[A-Za-z]+|[;,.?]")
VOCAB_WORDS = tuple(dict.fromkeys(("<pad>", "<bos>", "<sep>", "<answer>") +
    NAMES + VERBS + ("By", "by", "near", "gate", "was", "'s",
    "In", "the", "first", "second", "event", "who", "agent", "patient",
    ",", ";", ".", "?")))
VOCAB = {token:i for i,token in enumerate(VOCAB_WORDS)}


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def all_matchings():
    """15 pair-disjoint matchings of 16 names; only split *IDs* are generated."""
    ordered=list(NAMES)
    random.Random(SPLIT_RNG).shuffle(ordered)
    circle=ordered[:]
    out=[]
    for _ in range(15):
        m=tuple(tuple(sorted((circle[i],circle[-i-1]))) for i in range(8))
        out.append(m)
        circle=[circle[0],circle[-1],*circle[1:-1]]
    assert len(out)==15
    ps=[{p for matching in out[a:b] for p in matching}
        for a,b in ((0,9),(9,12),(12,15))]
    assert tuple(map(len,ps))==(72,24,24)
    assert not ps[0]&ps[1] and not ps[0]&ps[2] and not ps[1]&ps[2]
    assert len(set.union(*ps))==120
    return tuple(out[:9]),tuple(out[9:12]),tuple(out[12:15])


def examples(rounds):
    """All three familiar grammar cells, 3 reference constructions, 2 queries.

    Construct from only train/development matchings; no protected case call.
    """
    for matching in rounds:
        for i,pair in enumerate(matching):
            for vi,verb in enumerate(VERBS):
                for cond in DEV_CONDS:
                    for target_first,voice in FAMILIAR:
                        for reverse in (0,1):
                            for other_voice in VOICES:
                                for layout in (0,1):
                                    shift=1+(vi+2*reverse+layout)%7
                                    opair=matching[(i+shift)%8]
                                    a,p=pair if not reverse else pair[::-1]
                                    c,d=opair
                                    # The distractor direction is balanced by
                                    # layout, independent of target reversal.
                                    if (i+vi+layout)%2:c,d=d,c
                                    target=Event(a,p,verb,c,d,voice,layout)
                                    other=Event(c,d,VERBS[(vi+2+i%3)%6],
                                                a if layout else p,
                                                p if layout else a,
                                                other_voice,1-layout)
                                    nn=tuple(sorted(set(pair+opair)))
                                    # exact same *participant* answer pool for
                                    # agent and patient queries.
                                    pkt=render_packet(target,other,nn,cond,target_first)
                                    validate_case(pkt)
                                    yield pkt


def exposure_doses(train_rounds):
    """64 paired dose question items per arm, 32 per adjunct component."""
    matching=train_rounds[0]
    dose={arm:[] for arm in ARMS}
    for i,vi,cond,direction,role in itertools.product(
            range(4),range(2),("L","I"),(0,1),("agent","patient")):
        pair=matching[i]
        c,d=matching[(i+3)%8]
        a,p=pair if not direction else pair[::-1]
        target=Event(a,p,VERBS[vi],c,d,"passive",(i+vi)%2)
        other=Event(c,d,VERBS[(vi+2+i)%6],a,p,
                    "active" if (i+direction)%2 else "passive",(i+direction)%2)
        names=tuple(sorted(set(pair+matching[(i+3)%8])))
        for arm in ARMS:
            pkt=render_packet(target,other,names,cond,target_first=(arm=="familiar"))
            validate_case(pkt)
            dose[arm].append((pkt,role))
    assert all(len(dose[arm])==64 for arm in ARMS)
    assert all(Counter(p.condition for p,_ in dose[arm])=={"L":32,"I":32}
               for arm in ARMS)
    # Paired dose items share verb, people, voice, condition and gold role;
    # only clause order and first/second question ordinal change.
    for (a,ra),(b,rb) in zip(dose["familiar"],dose["target"]):
        assert ra==rb and a.correct_agent==b.correct_agent
        assert a.correct_patient==b.correct_patient
        assert a.event==b.event and a.other_event==b.other_event
        assert a.text[:-1].split(" ; ")==b.text[:-1].split(" ; ")[::-1]
    return dose


def token_sequence(pkt: Packet, role: str):
    q=pkt.questions[0 if role=="agent" else 1]
    tokens=["<bos>"]+TOKEN_RE.findall(pkt.text)+["<sep>"]+TOKEN_RE.findall(q)+["<answer>"]
    if len(tokens)>MAX_TOKENS:raise ValueError(f"Token length {len(tokens)} > {MAX_TOKENS}")
    unknown=set(tokens)-set(VOCAB)
    if unknown:raise ValueError(f"Unknown tokens {unknown}")
    return tuple(tokens)


def dataset_manifest():
    train,dev,sealed=all_matchings()
    # Do not construct *any* protected text, source, labels or model queries.
    training=list(examples(train))
    development=list(examples(dev))
    doses=exposure_doses(train)
    assert training and development
    assert all(pkt.condition!="LI" for pkt in training+development)
    train_pairs={p for m in train for p in m}
    dev_pairs={p for m in dev for p in m}
    trtext={(pkt.text,role,pkt.questions[0 if role=="agent" else 1])
            for pkt in training for role in ("agent","patient")}
    dvtext={(pkt.text,role,pkt.questions[0 if role=="agent" else 1])
            for pkt in development for role in ("agent","patient")}
    assert not trtext&dvtext
    assert len(trtext) == 2*len(training), "Duplicate training rendered prompts"
    assert len(dvtext) == 2*len(development), "Duplicate development rendered prompts"
    for pkt in itertools.chain(training,development):
        for role in ("agent","patient"):token_sequence(pkt,role)
    for arm in ARMS:
        for pkt,role in doses[arm]:token_sequence(pkt,role)
    # Both role answers are checked independently per realization; 96+ events
    # in every familiar development cell, no second-passive development case.
    cell_counts=Counter((pkt.condition,pkt.target_first,pkt.event.voice)
                        for pkt in development)
    assert len(cell_counts)==9 and min(cell_counts.values())>=96
    assert len({pkt.event.verb for pkt in development})==6
    assert {pkt.condition for pkt in development}==set(DEV_CONDS)
    # Equality across two roles is guaranteed by identical 2-query packets.
    memo={
        "status":"PRODUCTION_PREFLIGHT_CANDIDATE_NOT_FROZEN",
        "split_rng":SPLIT_RNG,
        "train_pair_ids":train,
        "dev_pair_ids":dev,
        # SEAL: only identifiers + digest, no protected sentences generated.
        "protected_pair_ids_sha256":stable_hash(sealed),
        "protected_pair_count":sum(map(len,sealed)),
        "train_examples":len(training),"dev_examples":len(development),
        "train_prompt_hash":stable_hash(sorted(trtext)),
        "dev_prompt_hash":stable_hash(sorted(dvtext)),
        "dose_hashes":{arm:stable_hash([(p.text,role) for p,role in rows])
                       for arm,rows in doses.items()},
        "dev_cells":{f"{cond}/{('first' if pos else 'second')}/{voice}":n
                     for (cond,pos,voice),n in sorted(cell_counts.items())},
        "vocab_hash":stable_hash(VOCAB_WORDS),
        "max_observed_token_length":max(len(token_sequence(p,role))
            for p in itertools.chain(training,development)
            for role in ("agent","patient")),
        "training_budget":{"seeds":SEEDS,"arms":ARMS,"steps":STEPS,
                           "base_batch":SHARED_BATCH,"dose_batch":DOSE_BATCH},
    }
    return memo, training, development, doses


def encode(packets_and_roles):
    """Tensorize packet role questions. Lazy import makes preflight model-free."""
    import torch
    examples=list(packets_and_roles)
    seqs=[token_sequence(pkt,role) for pkt,role in examples]
    n=max(len(s) for s in seqs)
    ids=torch.full((len(examples),n),VOCAB["<pad>"],dtype=torch.long)
    readout=[];gold=[];candidates=[]
    for i,((pkt,role),seq) in enumerate(zip(examples,seqs)):
        ids[i,:len(seq)]=torch.tensor([VOCAB[x] for x in seq])
        readout.append(len(seq)-1)
        gold.append(VOCAB[pkt.correct_agent if role=="agent" else pkt.correct_patient])
        candidates.append([VOCAB[x] for x in sorted(pkt.names)])
    return ids,torch.tensor(readout),torch.tensor(gold),torch.tensor(candidates)


def make_model():
    """Exactly ML-080 Block/Model structure, resized to R3 vocabulary."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1=nn.LayerNorm(WIDTH)
            self.attn=nn.MultiheadAttention(WIDTH,HEADS,batch_first=True)
            self.n2=nn.LayerNorm(WIDTH)
            self.mlp=nn.Sequential(nn.Linear(WIDTH,4*WIDTH),nn.GELU(),
                                   nn.Linear(4*WIDTH,WIDTH))
        def forward(self,x,mask):
            z=self.n1(x)
            x=x+self.attn(z,z,z,attn_mask=mask,need_weights=False)[0]
            return x+self.mlp(self.n2(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=nn.Embedding(len(VOCAB),WIDTH)
            self.pos=nn.Embedding(MAX_TOKENS,WIDTH)
            self.blocks=nn.ModuleList([Block() for _ in range(LAYERS)])
            self.norm=nn.LayerNorm(WIDTH)
            self.out=nn.Linear(WIDTH,len(VOCAB))
        def forward(self,ids,readout):
            batch,n=ids.shape
            if n>MAX_TOKENS:raise ValueError("Sequence too long")
            position=torch.arange(n,device=ids.device)
            x=self.embed(ids)+self.pos(position)
            mask=torch.ones(n,n,dtype=torch.bool,device=ids.device).triu(1)
            for block in self.blocks:x=block(x,mask)
            return self.out(self.norm(x[torch.arange(batch,device=ids.device),readout]))
    return Model()


def train_one(seed:int,arm:str,approved_manifest_hash:str,
              device:str="cpu",root:Path=Path("results/exp101_candidate")):
    """IMPLEMENTED, NOT CLI-EXPOSED. NEVER invoke before external freeze approval.

    Reuses ML-080 AdamW, learning rate, weight decay and fixed 12,800 updates.
    Both paired arms use identical initializations and base minibatch streams.
    All seed/arm outcomes are retained; no best-checkpoint selection.
    """
    import os
    import torch
    from torch.nn import functional as F
    manifest,base,dev,dose=dataset_manifest()
    h=stable_hash(manifest)
    if approved_manifest_hash!=h:
        raise PermissionError("Manifest is not approved/frozen; no training permitted")
    if seed not in SEEDS or arm not in ARMS:raise ValueError("Unregistered arm/seed")
    torch.manual_seed(seed)
    model=make_model().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.01)
    flat_base=[(p,r) for p in base for r in ("agent","patient")]
    base_data=encode(flat_base)
    dose_data=encode(dose[arm])
    rg_base=torch.Generator().manual_seed(seed+800000)
    rg_dose=torch.Generator().manual_seed(seed+810000)
    folder=root/f"seed_{seed}"/arm
    folder.mkdir(parents=True,exist_ok=True)
    state=folder/"state.pt"
    step=0
    if state.exists():
        checkpoint=torch.load(state,map_location="cpu",weights_only=True)
        if checkpoint["manifest"]!=h or checkpoint["seed"]!=seed or checkpoint["arm"]!=arm:
            raise AssertionError("Checkpoint does not match frozen run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rg_base.set_state(checkpoint["base_rng"])
        rg_dose.set_state(checkpoint["dose_rng"])
        step=checkpoint["step"]
    def save():
        obj=dict(manifest=h,seed=seed,arm=arm,step=step,
                 model=model.state_dict(),optimizer=optimizer.state_dict(),
                 base_rng=rg_base.get_state(),dose_rng=rg_dose.get_state())
        tmp=state.with_suffix(".tmp")
        torch.save(obj,tmp)
        os.replace(tmp,state)
    while step<STEPS:
        model.train()
        ib=torch.randint(len(flat_base),(SHARED_BATCH,),generator=rg_base)
        idose=torch.randint(len(dose[arm]),(DOSE_BATCH,),generator=rg_dose)
        bx,br,bg,_=base_data; dx,dr,dg,_=dose_data
        a,b=bx[ib],dx[idose];n=max(a.shape[1],b.shape[1])
        ids=torch.cat((F.pad(a,(0,n-a.shape[1]),value=VOCAB["<pad>"]),
                       F.pad(b,(0,n-b.shape[1]),value=VOCAB["<pad>"]))).to(device)
        readout=torch.cat((br[ib],dr[idose])).to(device)
        gold=torch.cat((bg[ib],dg[idose])).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(ids,readout),gold)
        loss.backward();optimizer.step();step+=1
        if step%800==0 or step in (3200,STEPS):save()
    return {"seed":seed,"arm":arm,"step":step,"manifest":h}


def competence_gate(seed_arm_scores):
    """Pure-data gate on nine dev cells, using all eight paired seeds.

    Input: {(seed,arm): {cell: {events,individual_accuracy,both_correct}}}
    The implementation that measures these from frozen checkpoints will be
    reviewed separately; no checkpoint is loaded by this candidate CLI.
    """
    required={f"{cond}/{('first' if first else 'second')}/{voice}"
              for cond in DEV_CONDS for first,voice in FAMILIAR}
    passing={}
    for seed in SEEDS:
        for arm in ARMS:
            rows=seed_arm_scores.get((seed,arm),{})
            passing[(seed,arm)]=len(rows)==9 and all(
                cell in rows and rows[cell]["events"]>=96 and
                rows[cell]["individual_accuracy"]>=0.95 and
                rows[cell]["both_correct"]>=0.90 for cell in required)
    joint=[seed for seed in SEEDS if all(passing[(seed,arm)] for arm in ARMS)]
    return {"jointly_passing":joint,"eligible":len(joint)>=6,
            "all_seed_pairs_in_final_analysis":SEEDS,"per_model":passing}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase",choices=("preflight",),required=True,
                    help="No train/evaluate/protected phases in inspection candidate")
    args=ap.parse_args()
    report,train,dev,dose=dataset_manifest()
    print("EXP-101 PRODUCTION DATA PREFLIGHT (candidate, NOT FROZEN)")
    print(json.dumps(report,sort_keys=True,indent=2))
    print("No torch imported, model loaded/trained, protected sentence generated, or evaluation opened")

if __name__=="__main__":main()
