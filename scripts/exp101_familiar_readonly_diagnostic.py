#!/usr/bin/env python3
"""EXP-101 post-gate familiar-task diagnostic, read-only and exploratory.

Run ONLY from the original pinned RC7 training worktree, with its existing
frozen_release.json and results/exp101/seed_*/{familiar,target}/state.pt.
No training, no gradients, no protected-case construction or inference.
Evaluates 48 deterministically selected TRAIN and DEV packets per familiar
cell at offset 0, and the exact same DEV packets at offsets 0..8.
This script is not part of the frozen EXP-101 source manifest and does not
change the original registered development gate or its inference rule.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

import exp101_production_candidate as prod
import exp101_release as rel
import exp101_runtime as runtime
from exp101_render_validator import parse_sentence, parse_query

SOURCE_COMMIT = '3ae081129b0c480bd6ac57cb8fa7f7b952f54b09'
SOURCE_FINGERPRINT = '7ee59f50f928f0b995b2df1d942b2aa6bca0dc274840951c0995a3a73452b123'
EXPECTED_DATA_HASH = '1ebeb2327c243a95793895ea0105fab7b0d8eb53bbca53c70c8d3095c3db9b67'
N_PER_CELL = 48
BATCH_PACKETS = 24
OFFSETS = tuple(range(9))
CELLS = tuple(f"{condition}/{event}/{voice}" for condition in ('bare','L','I')
              for event,voice in (('first','active'),('first','passive'),('second','active')))


def cell_name(p):
    return f"{p.condition}/{'first' if p.target_first else 'second'}/{p.event.voice}"


def select_fixed(packets, name):
    by_cell=defaultdict(list)
    for p in packets:
        if p.condition not in ('bare','L','I'):
            raise AssertionError('Non-familiar packet in diagnostic')
        by_cell[cell_name(p)].append(p)
    if set(by_cell)!=set(CELLS):
        raise AssertionError(f'{name}: unexpected cell coverage')
    fixed={}
    for cell in CELLS:
        rows=by_cell[cell]
        if len(rows)<N_PER_CELL:raise AssertionError(f'{name}/{cell}: too few rows')
        # Sort independent of generator traversal; immutable hash selection.
        sort_key=lambda p: (hashlib.sha256((p.text+'|'+p.questions[0]+'|'+p.questions[1]).encode()).hexdigest(),p.text)
        fixed[cell]=sorted(rows,key=sort_key)[:N_PER_CELL]
    return fixed


def independent_alignment_check(p):
    assert p.event.agent != p.event.patient
    assert len(set(p.names))==4
    parsed=parse_sentence(p.text,p.names)
    correct=parsed[p.query_event]
    wrong=parsed[1-p.query_event]
    assert (correct.agent,correct.patient,correct.voice)==(p.correct_agent,p.correct_patient,p.event.voice)
    assert (wrong.agent,wrong.patient)==(p.other_event.agent,p.other_event.patient)
    for role,question in (('agent',p.questions[0]),('patient',p.questions[1])):
        ordinal,parsed_role=parse_query(question)
        assert (ordinal,parsed_role)==(p.query_event,role)
    ids,readout,labels,candidates=prod.encode([(p,'agent'),(p,'patient')])
    for i,role in enumerate(('agent','patient')):
        tokens=prod.token_sequence(p,role)
        assert tokens[-1]=='<answer>'
        assert int(readout[i])==len(tokens)-1
        assert ids[i,int(readout[i])].item()==prod.VOCAB['<answer>']
        expected_name=p.correct_agent if role=='agent' else p.correct_patient
        assert int(labels[i])==prod.VOCAB[expected_name]
        assert expected_name in [prod.VOCAB_WORDS[int(x)] for x in candidates[i]]
        assert set(prod.VOCAB_WORDS[int(x)] for x in candidates[i])==set(p.names)
        assert ids[i,:len(tokens)].tolist()==[prod.VOCAB[t] for t in tokens]
    return {'source':p.text,'agent_question':p.questions[0],
            'patient_question':p.questions[1],
            'metadata_gold':[p.correct_agent,p.correct_patient],
            'surface_parser_gold':[correct.agent,correct.patient],
            'other_event_gold':[wrong.agent,wrong.patient],
            'readout_positions':[int(x) for x in readout],
            'label_ids':[int(x) for x in labels],
            'choice_ids':[int(x) for x in candidates[0]]}


def interpret(a,p,person_names):
    gold=(p.correct_agent,p.correct_patient)
    other=(p.other_event.agent,p.other_event.patient)
    aa,pp=a
    if (aa,pp)==gold:cls='both_correct'
    elif (aa,pp)==gold[::-1]:cls='complete_role_reversal'
    elif aa==pp:cls='same_person'
    elif set((aa,pp))==set(other):cls='both_from_wrong_event'
    else:cls='other_mixed'
    return cls, {'correct_agent':int(aa==gold[0]),
                 'correct_patient':int(pp==gold[1]),
                 'both_correct':int((aa,pp)==gold),
                 'same_person':int(aa==pp),
                 'complete_reversal':int((aa,pp)==gold[::-1]),
                 'wrong_event_both':int(set((aa,pp))==set(other)),
                 'wrong_event_any':int(aa in other)+int(pp in other),
                 'classification':cls}


def evaluate_cell(model,packets,device,offset):
    import torch
    import torch.nn.functional as F
    sums=defaultdict(float); examples=[]
    for i in range(0,len(packets),BATCH_PACKETS):
        group=packets[i:i+BATCH_PACKETS]
        items=[(p,role) for p in group for role in ('agent','patient')]
        ids,readout,gold,choices=prod.encode(items)
        with torch.inference_mode():
            logits=model(ids.to(device),readout.to(device),position_offset=offset).float()
            loss=F.cross_entropy(logits,gold.to(device),reduction='none')
            argmax=logits.argmax(1)
            opts=choices.to(device)
            chosen=opts.gather(1,logits.gather(1,opts).argmax(1,keepdim=True)).flatten()
        full_pred=[prod.VOCAB_WORDS[int(j)] for j in argmax.cpu().tolist()]
        native_pred=[prod.VOCAB_WORDS[int(j)] for j in chosen.cpu().tolist()]
        batch_loss=loss.cpu().tolist()
        full_hits=(argmax.cpu()==gold).int().tolist()
        for j,p in enumerate(group):
            a=2*j; b=2*j+1
            cls,x=interpret((native_pred[a],native_pred[b]),p,p.names)
            for k,v in x.items():
                if k=='classification':sums['class_'+v]+=1
                else:sums[k]+=v
            sums['full_vocab_loss']+=batch_loss[a]+batch_loss[b]
            sums['full_vocab_correct']+=full_hits[a]+full_hits[b]
            sums['n_packets']+=1
            examples.append({'source':p.text,'questions':list(p.questions),
                'gold':[p.correct_agent,p.correct_patient],
                'other_event':[p.other_event.agent,p.other_event.patient],
                'four_choice_prediction':[native_pred[a],native_pred[b]],
                'full_vocab_prediction':[full_pred[a],full_pred[b]],
                'individual_full_vocab_loss':[round(batch_loss[a],5),round(batch_loss[b],5)],
                'category':cls})
    n=sums['n_packets'];assert n==len(packets)
    ans={'n_packets':int(n),'n_queries':2*int(n),
        'full_vocab_mean_cross_entropy':round(sums['full_vocab_loss']/(2*n),6),
        'full_vocab_accuracy':round(sums['full_vocab_correct']/(2*n),6),
        'four_choice_accuracy':round((sums['correct_agent']+sums['correct_patient'])/(2*n),6),
        'agent_accuracy':round(sums['correct_agent']/n,6),
        'patient_accuracy':round(sums['correct_patient']/n,6),
        'both_roles_correct':round(sums['both_correct']/n,6),
        'same_person_rate':round(sums['same_person']/n,6),
        'role_reversal_rate':round(sums['complete_reversal']/n,6),
        'both_from_wrong_event_rate':round(sums['wrong_event_both']/n,6),
        'wrong_event_answer_fraction':round(sums['wrong_event_any']/(2*n),6),
        'error_classes':{k.removeprefix('class_'):int(v) for k,v in sums.items() if k.startswith('class_')}}
    return ans,examples


def average_cell_metric(cells,key):
    return round(sum(x[key] for x in cells.values())/len(cells),6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('diagnose',))
    p.add_argument('--device',choices=('mps','cpu'),default='mps')
    args=p.parse_args()
    root=Path.cwd().resolve()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if commit!=SOURCE_COMMIT:raise PermissionError(f'Run only in original RC7 worktree at {SOURCE_COMMIT}; found {commit}')
    release_path=root/'reports/exp101/frozen_release.json'
    candidate=rel.require_approved_release(release_path)
    assert candidate['fingerprint']==SOURCE_FINGERPRINT
    assert candidate['dataset_manifest_sha256']==EXPECTED_DATA_HASH
    assert candidate['protected_evaluation_authorized'] is False
    data,train,dev,dose=prod.dataset_manifest()
    assert prod.stable_hash(data)==EXPECTED_DATA_HASH
    selected={'train':select_fixed(train,'train'),'dev':select_fixed(dev,'dev')}
    alignment_count=0;alignment_examples=[]
    for split in ('train','dev'):
        for cell in CELLS:
            for pkt in selected[split][cell]:
                proof=independent_alignment_check(pkt)
                alignment_count+=1
                if len(alignment_examples)<18 and all(not (x['split']==split and x['cell']==cell) for x in alignment_examples):
                    alignment_examples.append({'split':split,'cell':cell,**proof})
    import torch
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    if args.device!='mps':raise PermissionError('Use intended Mac MPS, not a changed backend')
    torch.set_grad_enabled(False)
    rtpolicy=runtime.configure_determinism(args.device)
    reports={}; representative=[]
    result_root=root/'results/exp101'
    for seed in prod.SEEDS:
        for arm in prod.ARMS:
            state=result_root/f'seed_{seed}'/arm/'state.pt'
            if not state.is_file():raise FileNotFoundError(str(state))
            model,opt,rg_base,rg_dose=runtime.init_state(seed,prod.make_model,device=args.device)
            step=runtime.load_state(state,model=model,optimizer=opt,gbase=rg_base,gdose=rg_dose,
                seed=seed,arm=arm,data_hash=EXPECTED_DATA_HASH,
                release_fingerprint=SOURCE_FINGERPRINT,device=args.device)
            if step!=prod.STEPS:raise PermissionError(f'{seed}/{arm}: checkpoint not at registered final step')
            model.eval()
            rows={'train':{},'dev':{}}
            examples_by_class={}
            for split in ('train','dev'):
                for cell in CELLS:
                    score,examples=evaluate_cell(model,selected[split][cell],args.device,0)
                    rows[split][cell]=score
                    if split=='dev':
                        for x in examples:
                            typ=x['category']
                            if typ not in examples_by_class:examples_by_class[typ]={'cell':cell,**x}
            offsets={0:rows['dev']}
            for offset in range(1,9):
                offsets[offset]={}
                for cell in CELLS:
                    result,_=evaluate_cell(model,selected['dev'][cell],args.device,offset)
                    offsets[offset][cell]=result
            offset_averages={str(o):{
                'four_choice_accuracy':average_cell_metric(offsets[o],'four_choice_accuracy'),
                'full_vocab_accuracy':average_cell_metric(offsets[o],'full_vocab_accuracy'),
                'full_vocab_mean_cross_entropy':average_cell_metric(offsets[o],'full_vocab_mean_cross_entropy')}
                for o in OFFSETS}
            summary={'train_offset_0':rows['train'],'development_offset_0':rows['dev'],
                     'development_same_samples_offsets_0_through_8':offset_averages,
                     'development_offset_0_through_8_per_cell':{
                         str(o):{c:offsets[o][c]['four_choice_accuracy'] for c in CELLS} for o in OFFSETS},
                     'offset_0_mean_dev_accuracy':offset_averages['0']['four_choice_accuracy'],
                     'dev_offset_accuracy_range':round(max(z['four_choice_accuracy'] for z in offset_averages.values())-min(z['four_choice_accuracy'] for z in offset_averages.values()),6),
                     'representative_development_predictions':list(examples_by_class.values())[:5],
                     'checkpoint_final_step':step,
                     'checkpoint_path':str(state.relative_to(root)),
                     'available_checkpoint_filenames':sorted(x.name for x in state.parent.glob('*.pt')),
                     'training_loss_history_available':False}
            reports[f'{seed}/{arm}']=summary
            print(f"DIAG {seed}/{arm}: train4={average_cell_metric(rows['train'],'four_choice_accuracy'):.4f} "+
                  f"dev4={summary['offset_0_mean_dev_accuracy']:.4f} "+
                  f"dev_fullCE={average_cell_metric(rows['dev'],'full_vocab_mean_cross_entropy'):.4f} "+
                  f"offset_span={summary['dev_offset_accuracy_range']:.4f}",flush=True)
            del model,opt
    log=result_root/'training.log'
    log_text=log.read_text(errors='replace') if log.exists() else ''
    logged_losses=re.findall(r'(?i)\b(?:loss|cross_entropy)\s*[=:]\s*([0-9.eE+-]+)',log_text)
    report={'study':'EXP-101','type':'POST_GATE_READ_ONLY_DIAGNOSTIC_NOT_REGISTERED_OUTCOME',
        'source_commit_sha':commit,'fingerprint':SOURCE_FINGERPRINT,
        'dataset_manifest_sha256':EXPECTED_DATA_HASH,
        'diagnostic_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protected_accessed':False,'protected_evaluation_authorized':False,
        'training_performed':False,'optimizer_step_performed':False,
        'registered_inference_offset_unchanged':0,'diagnostic_offsets':list(OFFSETS),
        'selection':'SHA256(packet_text|agent_question|patient_question), sorted ascending, first 48 per familiar cell, each split separately',
        'packets_per_cell_per_split':N_PER_CELL,
        'alignment_checked_packets':alignment_count,'alignment_failures':0,
        'sample_gold_readout_choice_alignment_examples':alignment_examples,
        'training_log_has_loss_values':len(logged_losses)>0,
        'training_log_loss_value_count':len(logged_losses),
        'training_loss_trajectory_inferable':False,
        'checkpoint_note':'Frozen runner overwrites state.pt every 800 updates; only step-12800 states were expected. This diagnostic never selects/loads alternative checkpoints.',
        'backend_determinism_policy':rtpolicy,
        'models':reports}
    target=result_root/'diagnostics'/'familiar_readonly_v1.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    rendered=json.dumps(report,sort_keys=True,indent=2)+'\n'
    if target.exists():
        if target.read_text()!=rendered:raise PermissionError(f'Existing diagnostic output differs; refusing overwrite {target}')
    else:target.write_text(rendered)
    print('EXP101_DIAGNOSTIC_JSON='+str(target),flush=True)
    print('ALL_16_CHECKPOINTS_CHECKED=TRUE',flush=True)
    print('PROTECTED_ACCESSED=FALSE',flush=True)
    print('TRAINING_PERFORMED=FALSE',flush=True)


if __name__=='__main__':main()
