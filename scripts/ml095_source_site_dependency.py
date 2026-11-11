"""ML-095: readout dependency of the post-L1 source-end state (EXPLORATORY).

Run in repository root:
    python3 scripts/ml095_source_site_dependency.py --phase preflight
    python3 scripts/ml095_source_site_dependency.py --phase plan
    # Review script and freeze manifest; commit BEFORE model inference.
    python3 scripts/ml095_source_site_dependency.py --phase run

Question: Is the near-zero ML-094 source-end transfer caused by low native
readout dependency on the <sep> state at that site? We test causal source-site
sensitivity using direct zeroing and an on-cohort, length-matched source-state
shuffle. Source participant and question-cue interventions are positive controls.

This reuses ML-092/093/094 calibration sentences and ML-088 trained models; it
DOES NOT open protected items, train, claim native circuit localization, or
provide independent replication. Zeroing is off-manifold and two-name zeroing
is not a token-count matched control. A null effect cannot establish that the
site encodes no relational information.
"""
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91
import ml092_source_state_transfer as m92
import ml094_source_end_role_control as m94

SEEDS = (701,709,719,727,733,739,743,751)
ARMS = ('familiar','target')
ROOT = Path('results/ml088_fresh_models')
PLAN = Path('reports/ml095/frozen_discovery_plan.json')
REPORT = Path('reports/ml095/discovery_evaluation.json')
MODES = ('native','self_sep','zero_sep_L1','shuffle_sep_L1','zero_sep_L2',
         'zero_sep_L1_L2','zero_second_agent_L1','zero_second_patient_L1',
         'zero_second_pair_L1','zero_first_pair_L1',
         'zero_role_cue_L1','zero_event_cue_L1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def second_acc(m):
    z=m['per_question_four_choice_accuracy']
    return .5*(z['second_agent']+z['second_patient'])


def bootstrap(x, n=20000, seed=95095):
    rng=random.Random(seed)
    v=sorted(mean(x[rng.randrange(len(x))] for _ in x) for _ in range(n))
    return [v[int(.025*n)],v[int(.975*n)-1]]


def prepare():
    # m94.prepare invokes the full frozen data configurator exactly once.
    old_hash, groups, m93_hash=m94.prepare()
    m94_hash=m94.freeze(m94.specification(old_hash,groups,m93_hash),False)
    if not m94.REPORT.is_file():
        raise FileNotFoundError('Preserve ML-094 discovery report first: '+str(m94.REPORT))
    previous=json.loads(m94.REPORT.read_text())
    assert previous['experiment']=='ML-094' and previous['plan_sha256']==m94_hash
    assert previous['protected_cases_generated'] is False
    assert len(previous['rows'])==16
    assert len(groups['passive'])==96 and tuple(ml.SEEDS)==SEEDS
    return old_hash,groups,m94_hash


def specification(old_hash, groups, prev_hash):
    return {
      'experiment':'ML-095',
      'status':'EXPLORATORY_POST_L1_SOURCE_SITE_DEPENDENCY',
      'script_sha256':sha(__file__),
      'dependency_hashes': {k:sha(v.__file__) for k,v in
           [('ml080',ml),('ml088',m88),('ml091',m91),('ml092',m92),('ml094',m94)]},
      'ml088_protocol_sha256':old_hash,
      'ml094_frozen_plan_sha256':prev_hash,
      'ml094_report_sha256':sha(m94.REPORT),
      'models':'Same 16 ML-088 frozen step-12800 checkpoints, 8 paired seeds',
      'examples':'Exact 96 previously examined ML-092/093/094 passive calibration sentences; 4 queries per sentence',
      'sentence_hash':digest([m91.sentence(g[0]) for g in groups['passive']]),
      'seeds':SEEDS, 'arms':ARMS, 'modes':MODES,
      'sites':'Post-L1 residual, except zero_sep_L2 after L2 and zero_sep_L1_L2 at both sites; all other computation unchanged',
      'shuffle':'At post-L1 SEP, donor is a different sentence with identical SEP index (same source length), cyclic within each index group, same donor state shared across four questions; no label-based selection',
      'primary_descriptive':'Across familiar seeds, source-end zero/shuffle impact on native four-choice prediction rate and absolute output logit shift, compared descriptively with second-participant source zeroing; no confirmatory criterion',
      'secondary':'Post-L2 SEP zeroing, both-layer SEP zeroing, first-event participant controls, question-role/event cue controls, native accuracy, four-choice error taxonomy, event-role logit gating and per-seed impact',
      'limitations':'Direct zeroing and cross-sentence shuffling can be off-manifold; token counts, source state norms, and downstream paths differ; null effects on answers do not imply an information-free residual. Previously explored data/models; neither independent confirmation nor a native binding circuit test.',
      'protected_safety':'No protected examples, no ML-094 report overwrite, no ML-088 protected evaluation, one-run report lock.'
    }


def freeze(spec, create=False):
    doc=json.loads(json.dumps({**spec,'sha256':digest(spec)}))
    if create and not PLAN.is_file():
        PLAN.parent.mkdir(parents=True,exist_ok=True)
        PLAN.write_text(json.dumps(doc,indent=2)+'\n')
    if not PLAN.is_file():
        raise FileNotFoundError('Run --phase plan first')
    assert json.loads(PLAN.read_text()) == doc, 'Frozen ML-095 script/dependencies changed'
    return doc['sha256']


def positions(groups, device):
    result=defaultdict(list)
    for g in groups:
        c=g[2]
        agent2,patient2=m92.participants(c,'second')
        agent1,patient1=m92.participants(c,'first')
        for q in g:
            toks=ml.prompt(q)
            sep=toks.index('<sep>')
            def only(token, a=0, b=None):
                b=len(toks) if b is None else b
                hits=[i for i in range(a,b) if toks[i]==token]
                assert len(hits)==1,(token,toks,hits)
                return hits[0]
            result['sep'].append(sep)
            result['second_agent'].append(only(agent2,0,sep))
            result['second_patient'].append(only(patient2,0,sep))
            result['first_agent'].append(only(agent1,0,sep))
            result['first_patient'].append(only(patient1,0,sep))
            result['role_cue'].append(only(q.role,sep+1))
            result['event_cue'].append(only('second' if q.target_pos==1 else 'first',sep+1))
    assert all(len(v)==4*len(groups) for v in result.values())
    for j in range(len(groups)):
        for key in ('sep','second_agent','second_patient','first_agent','first_patient'):
            assert len(set(result[key][4*j:4*j+4]))==1,key
        assert len(set(result['role_cue'][4*j:4*j+4]))==1
        assert len(set(result['event_cue'][4*j:4*j+4]))==1
    return {k:torch.tensor(v,device=device,dtype=torch.long) for k,v in result.items()}


def shuffle_owners(groups):
    bysep=defaultdict(list)
    for j,g in enumerate(groups):
        bysep[m94.source_sep(g[0])].append(j)
    owners=[None]*len(groups)
    for site,indices in sorted(bysep.items()):
        assert len(indices)>=2,(site,indices)
        # Source order changes but source-end position remains exactly matched.
        for i,src in enumerate(indices):
            others=indices[(i+1)%len(indices)]
            assert src!=others
            assert m91.sentence(groups[src][0])!=m91.sentence(groups[others][0])
            owners[src]=others
    assert all(isinstance(x,int) for x in owners)
    return owners


@torch.inference_mode()
def evaluate_model(model, groups, device, modes=MODES):
    qs=[q for g in groups for q in g]
    ids,ro,_,_=ml.encode(qs)
    ids,ro=ids.to(device),ro.to(device)
    n=len(qs);batch_len=ids.shape[1]
    ix=torch.arange(n,device=device)
    key=positions(groups,device)
    mask=torch.ones(batch_len,batch_len,device=device,dtype=torch.bool).triu(1)
    x0=model.embed(ids)+model.pos(torch.arange(batch_len,device=device))
    x1=model.blocks[0](x0,mask)
    shuffled=shuffle_owners(groups)
    donor_state=x1[torch.tensor([4*shuffled[j//4] for j in range(n)],device=device),
                   key['sep'][torch.tensor([4*shuffled[j//4] for j in range(n)],device=device)]]
    ref=model.out(model.norm(model.blocks[2](model.blocks[1](x1,mask),mask)[ix,ro])).float().cpu()
    direct=model(ids,ro).float().cpu()
    assert float((ref-direct).abs().max())<2e-4
    result={}
    native_sem=semantic_predictions(ref,groups)
    for mode in modes:
        x=x1.clone()
        norm_total=0.
        if mode=='self_sep':
            site=key['sep']
            values=x1[ix,site]
            x[ix,site]=values
        if mode in ('zero_sep_L1','zero_sep_L1_L2'):
            site=key['sep']
            norm_total=float(x[ix,site].norm(dim=-1).mean())
            x[ix,site]=0.
        if mode=='shuffle_sep_L1':
            site=key['sep']
            norm_total=float((donor_state-x[ix,site]).norm(dim=-1).mean())
            x[ix,site]=donor_state
        sites={
            'zero_second_agent_L1':('second_agent',),
            'zero_second_patient_L1':('second_patient',),
            'zero_second_pair_L1':('second_agent','second_patient'),
            'zero_first_pair_L1':('first_agent','first_patient'),
            'zero_role_cue_L1':('role_cue',),
            'zero_event_cue_L1':('event_cue',),
        }.get(mode,())
        if sites:
            norms=[]
            for name in sites:
                site=key[name]
                norms.append(x[ix,site].norm(dim=-1))
                x[ix,site]=0.
            norm_total=float(torch.stack(norms).mean())
        x=model.blocks[1](x,mask)
        if mode in ('zero_sep_L2','zero_sep_L1_L2'):
            site=key['sep']
            if mode=='zero_sep_L2': norm_total=float(x[ix,site].norm(dim=-1).mean())
            x[ix,site]=0.
        x=model.blocks[2](x,mask)
        z=model.out(model.norm(x[ix,ro])).float().cpu()
        score=m91.analyze_logits(z,groups)
        pred=semantic_predictions(z,groups)
        result[mode]={
           'second_accuracy':second_acc(score),
           'four_choice_accuracy':score['four_choice_accuracy'],
           'full_vocab_accuracy':score['full_vocab_accuracy'],
           'event_role_gating':score['second_event_role_gating'],
           'wrong_event_rate':score['wrong_event_rate'],
           'correct_event_wrong_role_rate':score['correct_event_wrong_role_rate'],
           'mean_output_logit_l2_change':float((z-ref).norm(dim=-1).mean()),
           'max_abs_output_logit_change':float((z-ref).abs().max()),
           'changed_four_choice_prediction_rate':float((pred!=native_sem).float().mean()),
           'mean_edited_token_norm':norm_total,
        }
    assert abs(result['native']['mean_output_logit_l2_change'])<1e-6
    assert abs(result['self_sep']['mean_output_logit_l2_change'])<1e-5
    return result


def semantic_predictions(z,groups):
    n=len(groups)
    assert z.shape[0]==4*n
    ids=torch.tensor([m91.name_ids(g) for g in groups],dtype=torch.long)
    score=z.reshape(n,4,-1).gather(2,ids[:,None,:].expand(n,4,4))
    return score.argmax(-1)


@torch.inference_mode()
def preflight(groups):
    torch.manual_seed(95095)
    model=ml.Model().eval()
    z=evaluate_model(model,groups['passive'],'cpu')
    assert z['native']['mean_output_logit_l2_change']==0.0
    assert z['self_sep']['max_abs_output_logit_change']<1e-5
    assert z['zero_sep_L1']['max_abs_output_logit_change']>1e-8
    assert z['zero_second_pair_L1']['max_abs_output_logit_change']>1e-8
    assert z['zero_role_cue_L1']['max_abs_output_logit_change']>1e-8
    assert math.isfinite(z['shuffle_sep_L1']['event_role_gating'])
    print('PREFLIGHT PASS: 96 passive calibration sentences, 4 questions each, same-index SEP donor shuffle')
    print('Native/self L1 parity and positive source/question-site patch pathways pass on an untrained model.')
    print('No trained checkpoint loaded; no protected examples accessed.')


@torch.inference_mode()
def run(old_hash,plan_hash,groups,device):
    if REPORT.exists():
        raise FileExistsError(f'ML-095 report exists; refusing rerun: {REPORT}')
    passed,counts=m88.model_gate(old_hash)
    assert passed, ('Competence gate failed',counts)
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            model=ml.load_checkpoint(ROOT/f'seed_{seed}'/arm/'model_step_12800.pt',seed,arm,old_hash,device)
            result=evaluate_model(model,groups['passive'],device)
            rows.append({'seed':seed,'arm':arm,'interventions':result})
            def fmt(name):
                x=result[name]
                return f'flip={x["changed_four_choice_prediction_rate"]:.3f}, logitL2={x["mean_output_logit_l2_change"]:.3f}, dAcc2={x["second_accuracy"]-result["native"]["second_accuracy"]:+.3f}'
            print(f'seed={seed} arm={arm:8} '
                  f'SEPzero [{fmt("zero_sep_L1")}]  '
                  f'SEPshuffle [{fmt("shuffle_sep_L1")}]  '
                  f'SECpairzero [{fmt("zero_second_pair_L1")}]',flush=True)
    def col(arm,name,k):
        return [r['interventions'][name][k] for r in rows if r['arm']==arm]
    summary={}
    for arm in ARMS:
        summary[arm]={
           mode:{k:mean(col(arm,mode,k)) for k in ('changed_four_choice_prediction_rate',
             'mean_output_logit_l2_change','max_abs_output_logit_change',
             'mean_edited_token_norm','second_accuracy','event_role_gating')}
           for mode in MODES
        }
        source=col(arm,'zero_sep_L1','changed_four_choice_prediction_rate')
        participant=col(arm,'zero_second_pair_L1','changed_four_choice_prediction_rate')
        contrast=[p-s for p,s in zip(participant,source)]
        summary[arm]['descriptive_participant_minus_sep_flip_rate_mean']=mean(contrast)
        summary[arm]['descriptive_participant_minus_sep_flip_rate_bootstrap95']=bootstrap(contrast)
    out={'experiment':'ML-095','status':'EXPLORATORY_SOURCE_SITE_DEPENDENCY',
         'plan_sha256':plan_hash,'ml088_protocol_sha256':old_hash,
         'protected_examples_generated':False,'n_checkpoints':16,
         'rows':rows,'summary':summary,'interpretation_warning':(
             'Direct zeros are off-manifold, SEP is one token vs second-pair two, '
             'norms differ, and a null impact does not show absent information. '
             'Repeated model/data exploratory analysis; not a native circuit proof.')}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(out,indent=2)+'\n')
    print('\nML-095 EXPLORATORY SUMMARY\n'+json.dumps(summary,indent=2))
    print('Saved',REPORT)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',required=True,choices=('preflight','plan','run'))
    ap.add_argument('--device',default='auto',choices=('auto','cpu','mps','cuda'))
    args=ap.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    old,groups,prior=prepare()
    spec=specification(old,groups,prior)
    if args.phase=='preflight':
        if PLAN.exists():freeze(spec)
        preflight(groups)
    else:
        h=freeze(spec,create=args.phase=='plan')
        if args.phase=='plan':
            print('ML-095 frozen plan',PLAN,'sha256',h)
            print('No trained inference performed.')
        else:
            run(old,h,groups,ml.select_device(args.device))


if __name__=='__main__':main()
