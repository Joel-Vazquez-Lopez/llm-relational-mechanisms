#!/usr/bin/env python3
"""TRAIN-SURFACE-FACTORIAL-v0 prospective matched four-arm experiment.

Fresh scratch training only, 12 new independent seeds. Source material is
reconstructed EXCLUSIVELY from 300 original TRAIN name-set identifiers. Never
train or score original dev, reserved, protected, or within-event L+I.
"""
import argparse,json,math,subprocess
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import permutations, product
from pathlib import Path
from random import Random
from statistics import mean,stdev
import torch
from torch import nn
from torch.nn import functional as F
import train_surface_factorial_v0_preflight as pf

# Byte-for-byte architecture design and readout conventions of
# exp105_development_pilot_v0, reimplemented locally to avoid a legacy import
# chain that is incomplete in the public repository checkout.
WIDTH,HEADS,LAYERS,MAX_TOKENS=96,4,3,48
LR,WD=0.002,0.01

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1=nn.LayerNorm(WIDTH)
        self.attn=nn.MultiheadAttention(WIDTH,HEADS,dropout=0.0,batch_first=True)
        self.n2=nn.LayerNorm(WIDTH)
        self.mlp=nn.Sequential(nn.Linear(WIDTH,WIDTH*4),
                               nn.GELU(),nn.Linear(WIDTH*4,WIDTH))
    def forward(self,x,mask):
        z=self.n1(x)
        x=x+self.attn(z,z,z,attn_mask=mask,need_weights=False)[0]
        return x+self.mlp(self.n2(x))

class Model(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,WIDTH)
        self.pos=nn.Embedding(MAX_TOKENS,WIDTH)
        self.blocks=nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm=nn.LayerNorm(WIDTH)
        self.out=nn.Linear(WIDTH,vocab_size)
    def forward(self,ids,readout):
        bs,n=ids.shape
        assert n<=MAX_TOKENS
        positions=torch.arange(n,device=ids.device)
        x=self.embed(ids)+self.pos(positions)
        mask=torch.ones(n,n,dtype=torch.bool,device=ids.device).triu(1)
        for block in self.blocks:x=block(x,mask)
        return self.out(self.norm(x[torch.arange(bs,device=ids.device),readout]))

def encode_sources(sources,vocab,check=True):
    """Faithfully reproduces EXP-105's four-query restricted-choice encoding."""
    seqs,labels,choices,lens=[],[],[],[]
    for source in sources:
        if check:
            assert pf.gen.parse_independent(pf.frame.unframe(source.text))==source.gold
            assert len(set(source.gold))==4 and source.mods in pf.MODS
        candidates=tuple(sorted(source.gold))
        for question,answer in zip(pf.gen.Q,source.gold):
            toks=["<bos>"]+pf.gen.tokens(source.text)+["<sep>"]+pf.gen.tokens(question)+["<answer>"]
            assert set(toks)<=set(vocab) and len(toks)<=MAX_TOKENS
            seqs.append([vocab[t] for t in toks])
            labels.append(vocab[answer])
            choices.append([vocab[name] for name in candidates])
            lens.append(len(toks))
    assert len(seqs)==len(sources)*4
    matrix=torch.full((len(seqs),max(lens)),vocab["<pad>"],dtype=torch.long)
    for ix,row in enumerate(seqs):matrix[ix,:len(row)]=torch.tensor(row,dtype=torch.long)
    return (matrix,torch.tensor([n-1 for n in lens],dtype=torch.long),
            torch.tensor(labels),torch.tensor(choices))

def one_update(model,optimizer,encoded,device):
    model.train()
    ids,readout,labels,_=encoded
    logits=model(ids.to(device),readout.to(device))
    loss=F.cross_entropy(logits,labels.to(device))
    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    optimizer.step()
    return float(loss.item())

@torch.inference_mode()
def score_native(model,encoded_by_voice,device,chunk_size=128):
    """Native restricted-answer C_query plus full-vocab and pair taxonomy."""
    model.eval()
    result={}
    for voice,encoded in encoded_by_voice.items():
        ids,readout,labels,candidates=encoded
        predictions=[];vocab_preds=[]
        for start in range(0,len(ids),chunk_size):
            end=start+chunk_size
            logits=model(ids[start:end].to(device),
                         readout[start:end].to(device)).float().cpu()
            c=candidates[start:end]
            pred=c.gather(1,logits.gather(1,c).argmax(
                dim=1,keepdim=True)).flatten()
            predictions.append(pred)
            vocab_preds.append(logits.argmax(1))
        pred=torch.cat(predictions).view(-1,4)
        gold=labels.view(-1,4)
        correct=(pred==gold)
        out={"C_query":float(correct.all(1).float().mean()),
             "accuracy4":float(correct.float().mean()),
             "full_vocab_accuracy":float(
                 (torch.cat(vocab_preds)==labels).float().mean())}
        for event in range(2):
            ix=event*2
            p0,p1=pred[:,ix],pred[:,ix+1]
            g0,g1=gold[:,ix],gold[:,ix+1]
            wrong=~(((p0==g0)|(p0==g1))&((p1==g0)|(p1==g1)))
            classes={"correct":(p0==g0)&(p1==g1)&~wrong,
                     "reversed":(p0==g1)&(p1==g0)&~wrong,
                     "collapsed":(p0==p1)&~wrong,
                     "wrong_event_name":wrong}
            assert torch.all(torch.stack(list(classes.values())).sum(0)==1)
            out["event%d_pair_classes"%(event+1)]={
                key:float(value.float().mean()) for key,value in classes.items()}
        result[voice]=out
    return result


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
LAUNCH=ROOT/"scripts/run_train_surface_factorial_v0.sh"
PROTOCOL=ROOT/"paper/TRAIN_SURFACE_FACTORIAL_V0_PROTOCOL.md"
OUT=pf.OUT
SOURCE=pf.SOURCE
STUDY=OUT/"study_manifest.json"
WEIGHTS=ROOT/"results/train_surface_factorial_v0"
SEEDS=(14901,14903,14907,14909,14913,14919,
       14921,14927,14931,14933,14939,14943)
STEPS=6400
T95=2.200985160082949
ARMS=pf.ARMS
VOICES=pf.VOICE_PAIR
POPS=("train_original","unseen_roles","unseen_quartets")
MILESTONES=(800,1600,2400,3200,4000,4800,5600,6400)

def sha(x):return sha256(Path(x).read_bytes()).hexdigest()
def digest(x):return pf.digest(x)
def save_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:
        json.dump(obj,f,sort_keys=True,indent=2);f.write("\n")

def fixtures():
    if not SOURCE.exists():raise FileNotFoundError("Prepare SOURCE manifest first")
    src=json.loads(SOURCE.read_text())
    if digest(src)!=digest(pf.build()):raise RuntimeError("Frozen SOURCE manifest drift")
    train,held,original,_=pf.partition_cores()
    pool={"train":{},"unseen_roles":{},"unseen_quartets":{}}
    sig={"train_variants":[],"test_unseen_roles":[],"test_unseen_quartets":[]}
    for ix in train:
        events=original[ix]
        gold=tuple(n for e in events for n in (e.agent,e.patient))
        rows={}
        for fi in range(5):
            for mi in range(9):
                r={}
                for v in VOICES:
                    s=pf.source(events,gold,v,fi,mi)
                    r[v]=s
                    sig["train_variants"].append((ix,fi,mi,v,s.text,s.gold))
                rows[fi,mi]=r
        pool["train"][ix]=rows
    for target,ids in (("unseen_roles",train),("unseen_quartets",held)):
        for ix in ids:
            events=original[ix]
            gold=tuple(n for e in events for n in (e.agent,e.patient))
            if target=="unseen_roles":
                _,assignments=pf.previous.all_roles(gold)
            else:
                assignments=tuple(permutations(gold))
            rows=[]
            for j,roles in enumerate(assignments):
                r={}
                for v in VOICES:
                    s=pf.source(events,roles,v,0,0,group="test_"+target)
                    r[v]=s
                    sig["test_"+target].append((ix,j,v,s.text,s.gold))
                rows.append(r)
            pool[target][ix]=tuple(rows)
    assert all(digest(sig[k])==src["source_hashes"][k] for k in sig)
    words=set()
    for rows in pool["train"].values():
        for d in rows.values():
            for s in d.values():words.update(pf.gen.tokens(s.text))
    for q in pf.gen.Q:words.update(pf.gen.tokens(q))
    special=("<pad>","<bos>","<sep>","<answer>")
    vocab={w:i for i,w in enumerate(special+tuple(sorted(words-set(special))))}
    for t in ("unseen_roles","unseen_quartets"):
        for rows in pool[t].values():
            for row in rows:
                for s in row.values():assert set(pf.gen.tokens(s.text))<=set(vocab)
    return train,held,pool,vocab,src

def schedule(seed,train):
    assert seed in SEEDS
    rng=Random(seed+14931013)
    cells=list(product(range(5),range(9)))
    counter=0;order=[]
    for step in range(1,STEPS+1):
        chosen=tuple(rng.sample(train,8))
        assigned=[]
        for ix in chosen:
            if counter%45==0:
                order=list(cells);rng.shuffle(order)
            assigned.append(order[counter%45]);counter+=1
        yield step,chosen,tuple(assigned)

def sched_sha(seed,train):
    h=sha256();freq=Counter()
    for step,chosen,cells in schedule(seed,train):
        assert len(set(chosen))==8
        freq.update(cells)
        h.update((json.dumps((step,chosen,cells),separators=(",",":"))+"\n").encode())
    assert len(freq)==45 and max(freq.values())-min(freq.values())<=1
    return h.hexdigest()

def expected(train,held,vocab,src):
    dependencies=("exp105_generator_preflight.py",
                  "exp105_position_control_preflight_v1.py",
                  "core_diversity_v1_preflight.py",
                  "train_surface_factorial_v0_preflight.py")
    p={
      "study":"TRAIN-SURFACE-FACTORIAL-v0","status":"FROZEN_AFTER_COMMIT_NOT_RUN",
      "train_ids_sha256":digest(train),"held_ids_sha256":digest(held),
      "reuses_old_240_60_train_quartet_split":True,
      "source_manifest_sha256":sha(SOURCE),"sources":src["source_hashes"],
      "seeds":list(SEEDS),"steps":STEPS,"arms":list(ARMS),
      "voice_pair":list(VOICES),"vocab_sha256":digest(sorted(vocab.items())),
      "training":"240 original fixed role cores; 8 name quartets per update, "
                 "2 matched AP/PA voice sources per quartet; 64 label targets",
      "factorial":"bare / five neutral frames / nine bare-L-I adjunct pairs / both",
      "shared_quartet_and_45_cell_schedule":True,
      "identical_initial_weights_within_seed":True,
      "optimizer":"AdamW lr=0.002 wd=0.01 6400 updates, final checkpoint only",
      "architecture":"3-layer width96 4-head causal attention 48-position",
      "common_primary_test":"same bare/no-frame 16 never-trained role arrangements "
                            "on each of the 240 training quartets, 2 familiar voices",
      "common_secondary_test":"same bare/no-frame 24 arrangements on each of "
                              "60 withheld name quartets, 2 familiar voices",
      "primary_contrast":"0.5*((bare-adjunct_only)+(frame_only-frame_adjunct))",
      "success_all":"two-sided seed t CI lower>0, mean >= +0.10, "
                    "positive in >=9/12 independent new seeds",
      "gated_secondary":"same contrast on 60 withheld quartets, "
                         "CI lower>0 only if primary passes",
      "prefix_effect_and_factorial_interaction":"exploratory",
      "training_competence":"original train core bare/no-frame accuracy",
      "no_seed_filtering_or_optional_stopping":True,
      "no_old_dev_reserved_protected_LI":True,
      "known_unmatched_features":src["explicitly_unmatched_factors"],
      "script_sha256":sha(SCRIPT),"launcher_sha256":sha(LAUNCH),
      "protocol_sha256":sha(PROTOCOL),
      "dependency_sha256":{d:sha(ROOT/"scripts"/d) for d in dependencies},
      "schedule_sha256":{str(s):sched_sha(s,train) for s in SEEDS}}
    return {"protocol":p,"protocol_digest":digest(p)}

def prepare():
    train,held,pool,vocab,src=fixtures()
    m=expected(train,held,vocab,src)
    if STUDY.exists():
        if json.loads(STUDY.read_text())!=m:raise RuntimeError("Study manifest changed")
        print("PREPARE VERIFIED")
    else:
        save_once(STUDY,m)
        print("PREPARE PASS")
    print("12 independent fresh seeds, 48 new models; no inference",m["protocol_digest"])

def verified():
    train,held,pool,vocab,src=fixtures()
    m=expected(train,held,vocab,src)
    if not STUDY.exists() or json.loads(STUDY.read_text())!=m:
        raise RuntimeError("Prepare and commit frozen study manifest")
    return train,held,pool,vocab,m

def frozen():
    paths=("scripts/train_surface_factorial_v0_preflight.py",
           "scripts/train_surface_factorial_v0.py",
           "scripts/run_train_surface_factorial_v0.sh",
           "paper/TRAIN_SURFACE_FACTORIAL_V0_PROTOCOL.md",
           "reports/train_surface_factorial_v0/source_manifest.json",
           "reports/train_surface_factorial_v0/study_manifest.json")
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],
                                     cwd=ROOT,text=True).strip()
        for path in paths:
            blob=subprocess.check_output(["git","show",head+":"+path],
                    cwd=ROOT,stderr=subprocess.DEVNULL)
            if blob!=(ROOT/path).read_bytes():
                raise RuntimeError("Unfrozen changed source file: "+path)
        return head
    except (subprocess.CalledProcessError,FileNotFoundError) as e:
        raise RuntimeError("Commit complete protocol before training") from e

def make_batch(chosen,cells,pool):
    rows={a:[] for a in ARMS}
    for ix,(fi,mi) in zip(chosen,cells):
        lookup={"bare":(0,0),"frame_only":(fi,0),
                "adjunct_only":(0,mi),"frame_adjunct":(fi,mi)}
        for arm,(f,m) in lookup.items():
            rows[arm].extend(pool["train"][ix][f,m][v] for v in VOICES)
        gold=[tuple(s.gold for s in rows[a][-2:]) for a in ARMS]
        assert all(x==gold[0] for x in gold)
    assert all(len(rows[a])==16 for a in ARMS)
    return rows

def eval_data(train,held,pool,vocab):
    groups={
       "train_original":[pool["train"][ix][0,0] for ix in train],
       "unseen_roles":[x for ix in train for x in pool["unseen_roles"][ix]],
       "unseen_quartets":[x for ix in held for x in pool["unseen_quartets"][ix]]
    }
    return {p:{v:encode_sources([row[v] for row in rows],vocab,check=False)
               for v in VOICES} for p,rows in groups.items()}

def smoke(device):
    train,held,pool,vocab,m=verified()
    cells=tuple(product(range(2),range(4)))
    batch=make_batch(train[:8],cells,pool)
    torch.manual_seed(1222026)
    for arm in ARMS:
        model=Model(len(vocab)).to(device)
        data=encode_sources(batch[arm],vocab,check=True)
        with torch.inference_mode():
            assert model(data[0].to(device),data[1].to(device)).shape==(64,len(vocab))
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        loss=one_update(model,opt,data,device)
        assert math.isfinite(loss)
        enc={v:encode_sources([batch[arm][i]],vocab) for i,v in enumerate(VOICES)}
        result=score_native(model,enc,device,chunk_size=4)
        assert all(0<=result[v]["C_query"]<=1 for v in VOICES)
        print("TRAIN-ONLY SMOKE",arm,"loss",loss)
    print("SMOKE PASS; no test-model inference performed")

def run(seed,device):
    if seed not in SEEDS:raise ValueError("Unknown prospective seed")
    if device.type!="mps":raise ValueError("Scientific training requires MPS")
    report=OUT/("seed%d_mps_step6400.json"%seed)
    weights=WEIGHTS/("seed%d_mps_step6400.pt"%seed)
    if report.exists() or weights.exists():
        raise FileExistsError("Refusing to overwrite existing sealed run")
    train,held,pool,vocab,m=verified()
    commit=frozen()
    tests=eval_data(train,held,pool,vocab)
    torch.manual_seed(seed)
    original=Model(len(vocab)).to(device)
    nets={}
    for arm in ARMS:
        net=deepcopy(original).to(device)
        assert all(torch.equal(a,b) for a,b in zip(net.parameters(),original.parameters()))
        nets[arm]=(net,torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD))
    trace=sha256();history=[]
    for step,chosen,cells in schedule(seed,train):
        batch=make_batch(chosen,cells,pool)
        losses={}
        for arm in ARMS:
            losses[arm]=one_update(*nets[arm],encode_sources(batch[arm],vocab,check=False),device)
        trace.update((json.dumps((step,chosen,cells),separators=(",",":"))+"\n").encode())
        if step in MILESTONES:
            history.append({"step":step,"losses":losses})
            print("STEP",step,losses,flush=True)
    assert trace.hexdigest()==m["protocol"]["schedule_sha256"][str(seed)]
    results={arm:{pop:score_native(nets[arm][0].eval(),tests[pop],device)
                  for pop in POPS} for arm in ARMS}
    OUT.mkdir(parents=True,exist_ok=True)
    WEIGHTS.mkdir(parents=True,exist_ok=True)
    torch.save({"seed":seed,"steps":STEPS,"study":"TRAIN-SURFACE-FACTORIAL-v0",
                "protocol_digest":m["protocol_digest"],
                "models":{arm:nets[arm][0].state_dict() for arm in ARMS}},weights)
    save_once(report,{"study":"TRAIN-SURFACE-FACTORIAL-v0",
                      "seed":seed,"steps":STEPS,"status":"SEALED_NEW_SEED",
                      "protocol_digest":m["protocol_digest"],
                      "schedule_sha256":trace.hexdigest(),
                      "frozen_commit":commit,
                      "no_old_dev_reserved_protected_LI":True,
                      "loss_history":history,"outcomes":results})
    print("SEALED",report,"Do not analyze until all 12 seeds complete")

def cquery(r,arm,pop):
    return mean(r["outcomes"][arm][pop][v]["C_query"] for v in VOICES)

def effect(r,pop):
    a={arm:cquery(r,arm,pop) for arm in ARMS}
    adjunct=.5*((a["bare"]-a["adjunct_only"])+
                (a["frame_only"]-a["frame_adjunct"]))
    prefix=.5*((a["bare"]-a["frame_only"])+
               (a["adjunct_only"]-a["frame_adjunct"]))
    interaction=(a["bare"]-a["adjunct_only"])-(a["frame_only"]-a["frame_adjunct"])
    return adjunct,prefix,interaction

def ci(values):
    assert len(values)==12
    m=mean(values);radius=2.200985160082949*stdev(values)/math.sqrt(12)
    return {"mean":m,"ci95":[m-radius,m+radius],
            "positive_seeds":sum(v>0 for v in values),
            "per_seed":dict(zip(map(str,SEEDS),values))}

def summary():
    train,held,pool,vocab,m=verified()
    reports={}
    for seed in SEEDS:
        p=OUT/("seed%d_mps_step6400.json"%seed)
        if not p.exists():
            print("INCOMPLETE: missing",seed,"No partial analysis")
            return
        r=json.loads(p.read_text())
        assert r["study"]=="TRAIN-SURFACE-FACTORIAL-v0" and r["seed"]==seed
        assert r["protocol_digest"]==m["protocol_digest"]
        assert r["schedule_sha256"]==m["protocol"]["schedule_sha256"][str(seed)]
        assert r["no_old_dev_reserved_protected_LI"] and set(r["outcomes"])==set(ARMS)
        reports[seed]=r
    primary=ci([effect(reports[s],"unseen_roles")[0] for s in SEEDS])
    passed=primary["ci95"][0]>0 and primary["mean"]>=.10 and primary["positive_seeds"]>=9
    secondary=ci([effect(reports[s],"unseen_quartets")[0] for s in SEEDS])
    arms={pop:{arm:{metric:ci([mean(reports[s]["outcomes"][arm][pop][v][metric]
                                  for v in VOICES) for s in SEEDS])
                    for metric in ("C_query","accuracy4","full_vocab_accuracy")}
                   for arm in ARMS} for pop in POPS}
    pair_classes={pop:{arm:{str(event):{cls:ci([
        mean(reports[seed]["outcomes"][arm][pop][v]
             ["event%d_pair_classes"%event][cls] for v in VOICES)
        for seed in SEEDS])
        for cls in ("correct","reversed","collapsed","wrong_event_name")}
        for event in (1,2)} for arm in ARMS} for pop in POPS}
    desc={pop:{"prefix_effect":ci([effect(reports[s],pop)[1] for s in SEEDS]),
               "adjunct_effect":ci([effect(reports[s],pop)[0] for s in SEEDS]),
               "interaction":ci([effect(reports[s],pop)[2] for s in SEEDS])}
          for pop in POPS}
    output={"study":"TRAIN-SURFACE-FACTORIAL-v0",
      "status":"COMPLETE_12_INDEPENDENT_SEEDS_48_MODELS",
      "protocol_digest":m["protocol_digest"],
      "primary":{"estimate":primary,"passes":passed,
                 "success":"CI lower>0, mean>=0.10, >=9/12 positive seeds"},
      "gated_secondary":{"estimate":secondary,"confirmatory_gate_open":passed,
                         "passes":secondary["ci95"][0]>0 if passed else None},
      "arm_metrics":arms,"exploratory_factorial":desc,
      "exploratory_pair_error_classes":pair_classes,
      "no_excluded_seeds":True,"no_old_dev_reserved_protected_LI":True}
    save_once(OUT/"analysis_summary.json",output)
    print("SUMMARY PASS: 12 seeds / 48 new models")
    print("PRIMARY adjunct degradation",primary,"passes",passed)
    if passed:print("GATED SECONDARY",secondary)
    else:print("GATED SECONDARY NOT CONFIRMATORY")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("prepare","smoke","run","summary"))
    p.add_argument("--device",choices=("cpu","mps"),default="cpu")
    p.add_argument("--seed",type=int)
    a=p.parse_args()
    if a.phase=="prepare":prepare()
    elif a.phase=="smoke":smoke(torch.device(a.device))
    elif a.phase=="run":
        if a.seed is None:p.error("--phase run requires --seed")
        run(a.seed,torch.device(a.device))
    else:summary()

if __name__=="__main__":main()
