#!/usr/bin/env python3
"""EXP-101 generator-only audit, no model import, training or protected data.

The finite public-development synthetic packet generator tests *metadata to
rendered text* independently of a source-only reference parser. It does not
create held-out split cases. All tests are purely syntactic/combinatorial.
"""
from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter
from dataclasses import dataclass, replace

from exp101_render_validator import parse_sentence, VERBS

NAMES = ("Aven", "Bero", "Cira", "Dovan", "Eris", "Falen", "Goran", "Hali",
         "Iven", "Jara", "Korin", "Lessa", "Maren", "Nero", "Orin", "Pela")
CONDS = ("bare", "L", "I", "LI")
LAYOUTS = (0, 1)
VOICES = ("active", "passive")

@dataclass(frozen=True)
class Event:
    agent: str
    patient: str
    verb: str
    # for LI, the two different off-event names; L=gate owner; I=near-person
    by_name: str
    near_name: str
    voice: str
    layout: int

@dataclass(frozen=True)
class Packet:
    text: str
    query_event: int
    names: tuple[str, ...]
    correct_agent: str
    correct_patient: str
    event: Event
    condition: str
    target_first: bool


def render_event(ev: Event, cond: str) -> str:
    assert cond in CONDS and ev.layout in LAYOUTS and ev.voice in VOICES
    assert ev.agent not in (ev.by_name, ev.near_name)
    assert ev.patient not in (ev.by_name, ev.near_name)
    if ev.voice == "active":
        # Both constructs use the identical component strings on their own and
        # in the held-out LI composition; no new lexical markers at test.
        core = (f"{ev.agent}, near {ev.near_name}, {ev.verb} {ev.patient}"
                if cond in ("I", "LI")
                else f"{ev.agent} {ev.verb} {ev.patient}")
    else:
        core = (f"{ev.patient} was {ev.verb} near {ev.near_name} by {ev.agent}"
                if cond in ("I", "LI")
                else f"{ev.patient} was {ev.verb} by {ev.agent}")
    if cond in ("L", "LI"):
        core = (f"By {ev.by_name}'s gate, {core}" if ev.layout == 0
                else f"{core} by {ev.by_name}'s gate")
    return core


def case_events(anchor: int, reverse: int, target_voice: str,
                other_voice: str, by_swap: int, lay_target: int,
                lay_other: int, other_swap: int, other_reverse: int) -> tuple[Event, Event, tuple[str, ...]]:
    # Development-only deterministic groups, not frozen heldout pair IDs.
    base_anchor = anchor % 24
    start = (base_anchor // 6) * 4
    base_names = NAMES[start:start + 4]
    # Both unordered pairs appear as the queried event across the fixed
    # development universe. This breaks an A/B-vs-C/D lexical status cue.
    names = (base_names[2:] + base_names[:2]) if anchor >= 24 else base_names
    a,p,c,d = names
    if reverse: a,p = p,a
    if by_swap: c,d = d,c
    target = Event(a,p,VERBS[base_anchor%6],c,d,target_voice,lay_target)
    # Independent distractor role direction (not tied to locative assignment).
    oc, od = (names[3], names[2]) if other_reverse else (names[2], names[3])
    other = Event(oc,od,VERBS[(base_anchor+2)%6],
                  names[other_swap],names[1-other_swap],other_voice,lay_other)
    return target,other,names


def render_packet(target: Event, other: Event, names: tuple[str,...],
                  condition: str, target_first: bool) -> Packet:
    clauses=(target, other) if target_first else (other, target)
    text=" ; ".join(render_event(e, condition) for e in clauses)+"."
    return Packet(text,0 if target_first else 1,names,
                  target.agent,target.patient,target,condition,target_first)


def development_cases(cond: str):
    for anchor,rev,tv,ov,by,lt,lo,otherswap,otherrev,pos in itertools.product(
        range(48), (0,1), VOICES, VOICES, (0,1), LAYOUTS, LAYOUTS, (0,1), (0,1), (False,True)
    ):
        target,other,names=case_events(anchor,rev,tv,ov,by,lt,lo,otherswap,otherrev)
        yield render_packet(target,other,names,cond,pos)


def words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z]+",s)


def names_in(s: str, candidates: tuple[str,...]) -> list[str]:
    return [x for x in words(s) if x in candidates]


def chosen_clause(pkt: Packet) -> str:
    return pkt.text[:-1].split(" ; ")[pkt.query_event]


def shallow_core(clause: str, names: tuple[str,...]):
    # The malicious baseline strips known adjunct phrases, then uses
    # surface order + voice. This intentionally performs *no query binding*.
    s=re.sub(r"^By [A-Z][a-z]+'s gate, ","",clause)
    s=re.sub(r"^Near [A-Z][a-z]+, ","",s)
    s=re.sub(r" by [A-Z][a-z]+'s gate$","",s)
    s=re.sub(r", near [A-Z][a-z]+, "," ",s)
    s=re.sub(r" near [A-Z][a-z]+ by "," by ",s)
    ns=names_in(s,names)
    if len(ns)!=2: return (None,None)
    return ((ns[1],ns[0]) if " was " in s else (ns[0],ns[1]))


def adversarial_adjunct_singleton(pkt: Packet) -> tuple[str|None,str|None]:
    # SELECT adjunct-bearing clause, globally singleton names, infer by
    # voice-conditioned core order. Never read pkt.query_event.
    clauses=pkt.text[:-1].split(" ; ")
    counts=Counter(names_in(pkt.text,pkt.names))
    def score(x):
        adj=int("gate" in x)+int("near" in x.lower())
        singles=sum(counts[n]==1 for n in names_in(x,pkt.names))
        return (adj,singles)
    idx=max(range(2),key=lambda i:score(clauses[i])) # leftmost on ties
    return shallow_core(clauses[idx],pkt.names)


def lexrank(pkt):
    a,b=sorted(pkt.names)[:2]
    return a,b


def first_last(pkt):
    nn=names_in(chosen_clause(pkt),pkt.names)
    return nn[0],nn[-1]


def nearest(pkt):
    toks=words(chosen_clause(pkt));v=pkt.event.verb;i=toks.index(v)
    l=[x for x in toks[:i] if x in pkt.names]
    r=[x for x in toks[i+1:] if x in pkt.names]
    if not l or not r:return None,None
    return (r[0],l[-1]) if "was" in toks[:i] else (l[-1],r[0])


def byrule(pkt,which):
    cl=chosen_clause(pkt);toks=words(cl)
    if "was" not in toks:
        return nearest(pkt)
    followers=[]
    for i,w in enumerate(toks):
        if w.lower()=="by":
            x=next((z for z in toks[i+1:] if z in pkt.names),None)
            if x:followers.append(x)
    left=[z for z in toks[:toks.index(pkt.event.verb)] if z in pkt.names]
    return ((followers[0] if which=="first" else followers[-1]) if followers else None,
            left[-1] if left else None)


def eventblind(pkt):
    return shallow_core(pkt.text[:-1].split(" ; ")[0],pkt.names)


BASELINES={
    "adjunct_singleton_voice_order":adversarial_adjunct_singleton,
    "lexical_rank":lexrank,
    "queried_clause_first_last":first_last,
    "nearest_verb":nearest,
    "first_by":lambda p:byrule(p,"first"),
    "last_by":lambda p:byrule(p,"last"),
    "event_blind_first":eventblind,
}


def validate_case(p: Packet):
    a,b=parse_sentence(p.text,p.names)
    parsed=(a,b)[p.query_event]
    if (parsed.agent,parsed.patient,parsed.verb,parsed.voice)!=(
            p.correct_agent,p.correct_patient,p.event.verb,p.event.voice):
        raise AssertionError(f"Independent rendered parser mismatch: {p}")
    if a.agent==b.agent or a.patient==b.patient or len({a.agent,a.patient,b.agent,b.patient})!=4:
        raise AssertionError("Two events must have four distinct participants")
    if p.condition=="LI":
        assert sorted(names_in(p.text,p.names))==sorted(list(p.names)*2),p
        assert sorted(a.locative_names+b.locative_names)==sorted(p.names),p
        # Both clauses have both adjunct families; no unique adjunct target.
        assert p.text.count("gate")==2 and p.text.lower().count("near")==2
    else:
        # Bare/single component conditions are reference cells and are
        # balanced at source/role across the full development factorial.
        assert p.text.count("gate")== (0 if p.condition in ("bare","I") else 2)
        assert p.text.lower().count("near")== (0 if p.condition in ("bare","L") else 2)
    return (a,b)


def check_transformations():
    n=0
    for anchor,ov,by,lt,lo,otherswap,otherrev in itertools.product(range(48),VOICES,(0,1),LAYOUTS,LAYOUTS,(0,1),(0,1)):
        packets={}
        for voice,rev,pos in itertools.product(VOICES,(0,1),(False,True)):
            target,other,names=case_events(anchor,rev,voice,ov,by,lt,lo,otherswap,otherrev)
            p=render_packet(target,other,names,"LI",pos)
            packets[(voice,rev,pos)]=p
            validate_case(p)
            n+=1
        for rev,pos in itertools.product((0,1),(False,True)):
            a=packets[("active",rev,pos)]
            b=packets[("passive",rev,pos)]
            assert (a.correct_agent,a.correct_patient)==(b.correct_agent,b.correct_patient)
        for voice,pos in itertools.product(VOICES,(False,True)):
            a=packets[(voice,0,pos)];b=packets[(voice,1,pos)]
            assert (a.correct_agent,a.correct_patient)==(b.correct_patient,b.correct_agent)
        for voice,rev in itertools.product(VOICES,(0,1)):
            a=packets[(voice,rev,True)];b=packets[(voice,rev,False)]
            assert a.text[:-1].split(" ; ")==b.text[:-1].split(" ; ")[::-1]
            assert (a.correct_agent,a.correct_patient)==(b.correct_agent,b.correct_patient)
    return n


def old_packet_attack_demo():
    # User's original flaw: target the only adjunct-bearing clause.
    # Same four people/verb; other event is bare; eight transformation cells.
    n=0
    for voice,rev,pos in itertools.product(VOICES,(0,1),(False,True)):
        t,o,names=case_events(0,rev,voice,"active",0,0,0,0,0)
        clauses=[render_event(t,"LI"),render_event(o,"bare")]
        text=" ; ".join(clauses if pos else clauses[::-1])+"."
        p=Packet(text,0 if pos else 1,names,t.agent,t.patient,t,"LI",pos)
        if adversarial_adjunct_singleton(p)==(p.correct_agent,p.correct_patient):
            n+=1
    return n


def run():
    assert len(NAMES)==16 and len(set(NAMES))==16
    assert len(set(VERBS))==6
    assert old_packet_attack_demo()==8
    role_counts=Counter()
    for anchor, target_rev, other_rev in itertools.product(range(48),(0,1),(0,1)):
        target,other,nn=case_events(anchor,target_rev,"active","active",0,0,0,0,other_rev)
        for e,status in ((target,"target"),(other,"distractor")):
            role_counts[(e.agent,status,"agent")]+=1
            role_counts[(e.patient,status,"patient")]+=1
    for n in NAMES:
        if len({role_counts[(n,event,role)] for event in ("target","distractor") for role in ("agent","patient")}) != 1:
            raise AssertionError(f"Core participant {n} not balanced over event and role: {role_counts}")
    trans=check_transformations()
    results={}
    for cond in CONDS:
        cases=list(development_cases(cond))
        for p in cases:validate_case(p)
        # Counterbalance target/distractor location, target role direction,
        # order of locative components, and every event participant mention.
        for a in range(48):
            start = ((a%24)//6)*4
            quartet=tuple(NAMES[start:start+4])
            oriented = (quartet[2:]+quartet[:2]) if a >= 24 else quartet
            rows=[p for p in cases if p.names==oriented and p.event.verb==VERBS[(a%24)%6]]
            if not rows:raise AssertionError("Missing anchor balance block")
            mentions=Counter(n for p in rows for n in names_in(p.text,p.names))
            if len(set(mentions.values()))!=1:
                raise AssertionError(f"Name mention imbalance {cond}, {a}: {mentions}")
        if cond=="LI":
            assert all(sorted(Counter(names_in(p.text,p.names)).values())==[2,2,2,2] for p in cases)
        elif cond in ("L","I"):
            assert all(sorted(Counter(names_in(p.text,p.names)).values())==[1,1,2,2] for p in cases)
        else:
            assert all(sorted(Counter(names_in(p.text,p.names)).values())==[1,1,1,1] for p in cases)
        report={"n_cells":len(cases),"baseline_accuracy":{}}
        for name,fn in BASELINES.items():
            right=sum(fn(p)==(p.correct_agent,p.correct_patient) for p in cases)
            # In this static check, primary is voice-paired correctness.
            # Score on all paired active/passive identities using same other voice.
            valid=0;total=0;valid_second=0;total_second=0
            for anch,rev,ov,by,lt,lo,otherswap,otherrev,pos in itertools.product(range(48),(0,1),VOICES,(0,1),LAYOUTS,LAYOUTS,(0,1),(0,1),(False,True)):
                t,o,names=case_events(anch,rev,"active",ov,by,lt,lo,otherswap,otherrev)
                a=render_packet(t,o,names,cond,pos)
                t2=replace(t,voice="passive")
                b=render_packet(t2,o,names,cond,pos)
                ok = fn(a)==(a.correct_agent,a.correct_patient) and fn(b)==(b.correct_agent,b.correct_patient)
                if ok:valid+=1
                total+=1
                if not pos:
                    total_second += 1
                    valid_second += int(ok)
            report["baseline_accuracy"][name]={
                "joint_correct":right,"total":len(cases),
                "voice_pair_correct":valid,"voice_pairs":total,
                "voice_pair_rate":valid/total,
                "second_event_voice_pair_correct":valid_second,
                "second_event_voice_pairs":total_second,
                "second_event_voice_pair_rate":valid_second/total_second,
                "all_eight_perfect":False, # asserted below explicitly for each group
            }
            if cond=="LI":
                if valid==total:
                    raise AssertionError(f"Shallow shortcut {name} solves primary LI voice test")
                for anch,ov,by,lt,lo,otherswap,otherrev in itertools.product(range(48),VOICES,(0,1),LAYOUTS,LAYOUTS,(0,1),(0,1)):
                    full=[]
                    for tv,rv,pos2 in itertools.product(VOICES,(0,1),(False,True)):
                        tt,oo,nn=case_events(anch,rv,tv,ov,by,lt,lo,otherswap,otherrev)
                        pp=render_packet(tt,oo,nn,cond,pos2)
                        full.append(fn(pp)==(pp.correct_agent,pp.correct_patient))
                    if all(full):raise AssertionError(f"Shallow shortcut {name} passes all 8")
        results[cond]=report
    full=results["LI"]
    print("EXP-101 GENERATOR-ONLY PREFLIGHT PASS")
    print("Independent rendered-text parser: PASS")
    print("Legacy attack baseline gets 8/8 on one-adjunct-bearing-clause packet: PASS")
    print("Voice / direction / physical-event-order semantic invariants: PASS",trans)
    print("Composed LI: both clauses adjunct-bearing, all four names appear twice: PASS")
    print("No source/grammar model or protected example read; no training.")
    for name, vals in full["baseline_accuracy"].items():
        print(f"  {name}: {vals['joint_correct']}/{vals['total']} LI individual two-role; "
              f"voice-pair {vals['voice_pair_correct']}/{vals['voice_pairs']}, "
              f"target-second {vals['second_event_voice_pair_correct']}/{vals['second_event_voice_pairs']}")
    result={"study":"EXP-101", "status":"STATIC_DEV_ONLY_NOT_FROZEN", "legacy_attack_eight_of_eight":8,
            "independent_parser_validated":True,"transformation_packets_checked":trans,
            "conditions":results}
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=("preflight",),required=True)
    args=ap.parse_args()
    run()