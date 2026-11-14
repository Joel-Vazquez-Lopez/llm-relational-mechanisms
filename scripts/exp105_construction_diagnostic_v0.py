#!/usr/bin/env python3
"""EXP-105: read-only construction-specific error audit of the 3200-step pilot.

Never trains, selects checkpoints, scores L+I, or renders protected cases.
Regenerates the EXACT frozen 216 AA/PA/AP familiar-development examples.
Reads the single pilot checkpoint + saved step-3200 report and checks outputs
against the already archived accuracy/error metrics. Writes NEW derived analysis
files ONLY if --write is explicitly supplied (and never overwrites).

Usage from repository root:
  python3 scripts/exp105_construction_diagnostic_v0.py --device mps
  python3 scripts/exp105_construction_diagnostic_v0.py --device mps --write
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import torch

import exp105_generator_preflight as v0
import exp105_position_control_preflight_v1 as v1
import exp105_development_pilot_v0 as pilot

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / 'reports/exp105/development_pilot_v0'
CKPT = ROOT / 'results/exp105/development_pilot_v0/pilot_seed10801_step3200.pt'
REPORT = PILOT_DIR / 'pilot_seed10801_step3200.json'
PROTO = PILOT_DIR / 'protocol.json'
OUT = ROOT / 'reports/exp105/analysis'
VOICES = ('AA', 'PA', 'AP')
ARMS = ('familiar', 'target')


def stable_hash(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required saved pilot file: {path}')
    return json.loads(path.read_text())


def reconstruct_vocab_without_li():
    """Reconstruct pilot's sorted vocabulary from training syntax, no L+I.

    All fixed names/verbs, both layouts, all three train voices, all nine
    NON-LI modifier pairs, and all neutral frames are represented. This does
    not touch v3.pools(), whose design audit also constructs unused LI samples.
    """
    train_sets, _, _ = v0.partitions()
    specials = ('<pad>', '<bos>', '<sep>', '<answer>')
    words = set(v0.NAMES) | set(v0.VERBS)
    for q in v0.Q:
        words.update(v0.tokens(q))
    for frame in v1.FRAMES:
        if frame:
            words.update(v0.tokens(frame))
    for _, cid, events in list(v0.cores(train_sets[:12], 'train')):
        for voice in v0.VOICES:
            for mods in v0.TRAIN_MODS:
                s = v0.make_source(events, voice, mods, 'vocab-only', cid)
                words.update(v0.tokens(s.text))
    assert not (set(specials) & words)
    vocabulary = specials + tuple(sorted(words))
    return {word: i for i, word in enumerate(vocabulary)}


def check_provenance():
    protocol = read_json(PROTO)
    report = read_json(REPORT)
    assert protocol['study'] == 'EXP-105'
    assert protocol['stage'] == 'UNFROZEN_DEVELOPMENT_LEARNABILITY_PILOT_V0'
    assert protocol['no_LI_evaluation'] and protocol['no_confirmatory_seeds']
    assert report['study'] == 'EXP-105'
    assert report['phase'] == 'DISPOSABLE_DEVELOPMENT_PILOT'
    assert not report['protected_evaluated'] and not report['LI_evaluated']
    assert report['seed'] == pilot.PILOT_SEED == 10801
    assert report['steps'] == 3200
    assert report['protocol_sha256'] == stable_hash(protocol)
    assert protocol['pilot_seed'] == pilot.PILOT_SEED
    assert protocol['dev_familiar_sources'] == 216
    assert protocol['checkpoint_selection'] == 'last step only, no early stopping'
    for path in pilot.source_files():
        expected = protocol['code_hashes'][path.name]
        got = pilot.hash_file(path)
        assert got == expected, f'Source changed since pilot: {path.name}\nExpected {expected}\nGot {got}'
    dev = pilot.familiar_dev_sources()
    assert len(dev) == protocol['dev_familiar_sources']
    assert all(s.voice in VOICES and all(mod != 'LI' for mod in s.mods) for s in dev)
    assert stable_hash([(s.text, s.gold) for s in dev]) == protocol['dev_digest']
    assert Counter(s.voice for s in dev) == {'AA':72, 'PA':72, 'AP':72}
    vocab = reconstruct_vocab_without_li()
    assert len(vocab) == protocol['vocabulary_size'] == 50, f'Vocabulary mismatch: {len(vocab)}'
    assert stable_hash(sorted(vocab.items())) == protocol['vocabulary_sha256'], 'Vocab token IDs changed'
    checkpoint = torch.load(CKPT, map_location='cpu', weights_only=True)
    assert checkpoint['seed'] == report['seed']
    assert checkpoint['steps'] == report['steps']
    assert checkpoint['protocol_sha256'] == report['protocol_sha256']
    assert set(ARMS).issubset(checkpoint)
    assert report['history'] and report['history'][-1]['step'] == 3200
    return report, protocol, dev, vocab, checkpoint


@torch.inference_mode()
def get_predictions(model, encoded, device):
    model.eval()
    x, readout, labels, choices = encoded
    candidate_predictions = []
    full_correct_flags = []
    for st in range(0, len(labels), 128):
        en = min(st + 128, len(labels))
        logits = model(x[st:en].to(device), readout[st:en].to(device)).float()
        cand = choices[st:en].to(device)
        chosen = cand.gather(1, logits.gather(1, cand).argmax(1, keepdim=True)).flatten()
        candidate_predictions.append(chosen.cpu())
        full_correct_flags.extend((logits.argmax(1) == labels[st:en].to(device)).cpu().tolist())
    return (torch.cat(candidate_predictions).reshape(-1, 4).tolist(),
            labels.reshape(-1, 4).tolist(),
            [sum(full_correct_flags[i:i+4]) for i in range(0,len(full_correct_flags),4)])


def summarize_case_indices(indices, pred, truth, full_ok):
    n = len(indices)
    assert n
    errors = Counter()
    events = Counter()
    first = Counter()
    second = Counter()
    query_hits = [0, 0, 0, 0]
    for i in indices:
        p = pred[i]
        y = truth[i]
        assert len(set(y)) == 4
        assert len(p) == 4
        correct_source = sum(a == b for a, b in zip(p, y)) == 4
        errors['C_query'] += correct_source
        errors['full_vocab_correct'] += full_ok[i]
        for k in range(4):
            query_hits[k] += p[k] == y[k]
            e = k // 2
            correct_names = {y[2*e], y[2*e + 1]}
            if p[k] == y[k]:
                errors['correct'] += 1
            elif p[k] in correct_names:
                errors['wrong_role'] += 1
            else:
                errors['wrong_event'] += 1
        for e, destination in [(0, first), (1, second)]:
            start = 2*e
            a, b = p[start:start+2]
            ag, pat = y[start:start+2]
            in_event = (a in {ag, pat}, b in {ag, pat})
            events['total'] += 1
            destination['total'] += 1
            if a == b:
                events['same'] += 1
                destination['same'] += 1
            if a == ag and b == pat:
                events['both_correct'] += 1
                destination['both_correct'] += 1
            if all(in_event):
                events['both_from_correct_event'] += 1
                destination['both_from_correct_event'] += 1
                if a == ag and b == pat:
                    category = 'correct_assignment'
                elif a == pat and b == ag:
                    category = 'reversal'
                elif a == b:
                    category = 'collapse'
                else:
                    raise AssertionError('Unexpected pair category')
                events[category] += 1
                destination[category] += 1
            elif any(in_event):
                events['one_from_wrong_event'] += 1
                destination['one_from_wrong_event'] += 1
            else:
                events['both_from_wrong_event'] += 1
                destination['both_from_wrong_event'] += 1
    assert errors['correct'] + errors['wrong_role'] + errors['wrong_event'] == n*4
    assert events['total'] == 2*n
    assert events['both_from_correct_event'] == (events['correct_assignment'] + events['reversal'] + events['collapse'])
    assert (events['both_from_correct_event'] + events['one_from_wrong_event']
            + events['both_from_wrong_event'] == events['total'])
    def pct(k, denominator):
        return (100.0 * k / denominator) if denominator else None
    def event_record(c):
        return {
            'n':c['total'],
            'both_correct_pct':pct(c['both_correct'],c['total']),
            'same_answer_pct':pct(c['same'],c['total']),
            'both_from_correct_event_pct':pct(c['both_from_correct_event'], c['total']),
            'correct_assignment_conditional_pct':pct(c['correct_assignment'],c['both_from_correct_event']),
            'reversal_conditional_pct':pct(c['reversal'],c['both_from_correct_event']),
            'collapse_conditional_pct':pct(c['collapse'],c['both_from_correct_event']),
            'correct_assignment_count':c['correct_assignment'],
            'reversal_count':c['reversal'],
            'collapse_count':c['collapse'],
            'both_from_correct_event_count':c['both_from_correct_event'],
        }
    return {
        'sentences':n, 'questions':4*n,
        'accuracy4_pct':pct(errors['correct'],n*4),
        'C_query_pct':pct(errors['C_query'],n),
        'wrong_event_pct':pct(errors['wrong_event'],n*4),
        'wrong_role_pct':pct(errors['wrong_role'],n*4),
        'conditional_wrong_role_pct':pct(errors['wrong_role'],errors['correct']+errors['wrong_role']),
        'same_answer_pct':pct(events['same'],2*n),
        'both_correct_per_event_pct':pct(events['both_correct'],2*n),
        'full_vocab_accuracy_pct':pct(errors['full_vocab_correct'],n*4),
        'per_query_accuracy_pct':[pct(k,n) for k in query_hits],
        'events':event_record(events),
        'first_event':event_record(first),
        'second_event':event_record(second),
    }


def check_against_saved(derived, saved, arm):
    """Scoring code has reproduced the original pilot's *stored metrics*."""
    fields = {
        'C_query':'C_query_pct',
        'accuracy4':'accuracy4_pct',
        'wrong_event_rate':'wrong_event_pct',
        'wrong_role_rate':'wrong_role_pct',
        'same_answer_rate':'same_answer_pct',
        'full_vocab_accuracy':'full_vocab_accuracy_pct',
    }
    all_row = derived['ALL']
    for original, new in fields.items():
        d = abs(saved[original]*100 - all_row[new])
        if d > 1e-5:
            raise AssertionError(f'{arm}: saved {original} does not match checkpoint rerun: {saved[original]:.8f} vs {all_row[new]/100:.8f} (diff {d:.5f} pp)')
    for voice in VOICES:
        d = abs(saved['per_voice_C_query'][voice]*100 - derived[voice]['C_query_pct'])
        if d > 1e-5:
            raise AssertionError(f'{arm} {voice}: checkpoint per_voice_C_query mismatches report ({d:.5f} pp)')


def markdown(result):
    lines = [
        '# EXP-105 / construction-specific role-error diagnostic', '',
        '**Status:** development-only, read-only checkpoint re-scoring of fixed step-3200 pilot seed 10801; no L+I, reserved, or protected scoring.',
        '',
        'Reconstructed data and vocabulary match the original pilot hashes. All archived aggregate and per-voice metrics are reconciled with the saved pilot reports.',
        '',
        '## Per-construction native four-name decoding', '',
        '| Construction | Arm | n sentences | C_query | Per-question accuracy | Wrong event | Wrong role | Wrong role given correct event | Same-answer | Both correct / event |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for voice in ('ALL',)+VOICES:
        for arm in ARMS:
            r = result['arms'][arm][voice]
            def fmt(v):
                return 'N/A' if v is None else f'{v:.2f}%'
            lines.append('| '+voice+' | '+arm+' | '+str(r['sentences'])+' | '+
                         ' | '.join(fmt(r[k]) for k in ('C_query_pct','accuracy4_pct',
                         'wrong_event_pct','wrong_role_pct','conditional_wrong_role_pct',
                         'same_answer_pct','both_correct_per_event_pct'))+' |')
    lines += ['', 'Conditional wrong-role rate divides wrong-role questions by all questions selecting a member of the correct event. It is **not** the fraction of complete two-answer role reversals.', '',
              '## Oracle-free joint-response taxonomy', '',
              'The three disjoint categories below are calculated **only when both answers for an event name participants of that same correct event**. They are native model predictions *filtered by event membership*, not a re-decoding oracle that changes the candidate set.', '',
              '| Construction | Arm | Events (2 × n) | Both choose correct event | Correct assignment given both | Reversal given both | Collapse given both |',
              '|---|---|---:|---:|---:|---:|---:|']
    for voice in ('ALL',)+VOICES:
        for arm in ARMS:
            r = result['arms'][arm][voice]
            e = r['events']
            def fmt(x):
                return 'N/A' if x is None else f'{x:.2f}%'
            lines.append(f'| {voice} | {arm} | {e["n"]} | {fmt(e["both_from_correct_event_pct"])} ({e["both_from_correct_event_count"]}) | {fmt(e["correct_assignment_conditional_pct"])} | {fmt(e["reversal_conditional_pct"])} | {fmt(e["collapse_conditional_pct"])} |')
    lines += ['', '## Event-specific pair diagnostics', '',
              '| Construction | Arm | Event | Both roles correct | Same-answer | Both answers from correct event | Reversal among those | Collapse among those |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for voice in VOICES:
        for arm in ARMS:
            for name,field in (('first','first_event'),('second','second_event')):
                ev = result['arms'][arm][voice][field]
                def pformat(v):
                    return 'N/A' if v is None else f'{v:.2f}%'
                lines.append(f'| {voice} | {arm} | {name} | {pformat(ev["both_correct_pct"])} | {pformat(ev["same_answer_pct"])} | {pformat(ev["both_from_correct_event_pct"])} | {pformat(ev["reversal_conditional_pct"])} | {pformat(ev["collapse_conditional_pct"])} |')
    lines += ['', '## Target minus familiar: percentage-point contrasts', '',
              '| Construction | C_query | Question accuracy | Wrong event | Wrong role | Conditional wrong role | Same-answer |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for voice in ('ALL',)+VOICES:
        f=result['arms']['familiar'][voice]
        t=result['arms']['target'][voice]
        ks=['C_query_pct','accuracy4_pct','wrong_event_pct','wrong_role_pct','conditional_wrong_role_pct','same_answer_pct']
        values=[f'{t[k]-f[k]:+.2f}' for k in ks]
        lines.append('| '+voice+' | '+' | '.join(values)+' |')
    lines += ['', '## Per-query accuracy (first agent, first patient, second agent, second patient)', '',
              '| Construction | Arm | 1st agent | 1st patient | 2nd agent | 2nd patient |',
              '|---|---|---:|---:|---:|---:|']
    for voice in ('ALL',)+VOICES:
        for arm in ARMS:
            vals=[f'{x:.2f}%' for x in result['arms'][arm][voice]['per_query_accuracy_pct']]
            lines.append('| '+voice+' | '+arm+' | '+' | '.join(vals)+' |')
    lines += ['', '## Interpretation boundaries', '',
              '- One exploratory pilot seed; the denominators (72 sentences per construction) are not independent model-level replications.',
              '- Construction-level means can reveal tradeoffs hidden by overall accuracy. No claim of a transferable model mechanism follows from these summaries.',
              '- All per-construction outcomes use the original four-candidate decoding; unrestricted-vocabulary accuracy is separately recorded in JSON.',
              '- No checkpoint selection, training, L+I rendering, reserved-case construction, or confirmatory-seed use.',
              '- If a stored metric does not reproduce exactly, this audit raises an error rather than silently presenting an altered score.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='auto')
    parser.add_argument('--write',action='store_true',help='Create new Markdown+JSON analysis files only; never overwrite')
    args=parser.parse_args()
    report, protocol, dev, vocab, ckpt = check_provenance()
    device=pilot.get_device(args.device)
    encoded=pilot.encode_sources(dev,vocab)
    golds=None
    rows={}
    for arm in ARMS:
        model=pilot.Model(len(vocab)).to(device)
        model.load_state_dict(ckpt[arm],strict=True)
        pred, truth, full_correct_by_item=get_predictions(model,encoded,device)
        if golds is None:
            golds=truth
        else:
            assert golds==truth
        n=len(dev)
        all_idx=list(range(n))
        indices={v:[i for i,s in enumerate(dev) if s.voice==v] for v in VOICES}
        derived={'ALL':summarize_case_indices(all_idx,pred,truth,full_correct_by_item)}
        for voice in VOICES:
            assert len(indices[voice])==72
            derived[voice]=summarize_case_indices(indices[voice],pred,truth,full_correct_by_item)
        check_against_saved(derived,report['history'][-1][arm],arm)
        rows[arm]=derived
        print(f'{arm.upper():8s} PASS: exact checkpoint/report agreement (seed 10801, 3200 steps)',flush=True)
    result={
        'study':'EXP-105', 'status':'EXPLORATORY_DEVELOPMENT_ONLY',
        'no_training':True, 'no_LI_scoring_or_rendering':True,
        'no_reserved_or_protected_scoring':True,
        'seed':10801,'step':3200,'checkpoint_selection':'fixed_last_step',
        'pilot_protocol_sha256':report['protocol_sha256'],
        'pilot_report_sha256':pilot.hash_file(REPORT),
        'checkpoint_sha256':pilot.hash_file(CKPT),
        'dev_digest':protocol['dev_digest'],
        'arms':rows,
    }
    rendered=markdown(result)
    print(rendered)
    print('PASS: 216 familiar-development sources, exact checkpoint-to-report reconciliation; no training or L+I/protected access.')
    if args.write:
        OUT.mkdir(parents=True,exist_ok=True)
        paths=[OUT/'construction_error_pilot10801_step3200.json',OUT/'construction_error_pilot10801_step3200.md']
        if any(p.exists() for p in paths):
            raise FileExistsError(f'Analysis artifact already exists; will not overwrite: {paths}')
        paths[0].write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        paths[1].write_text(rendered+'\n')
        for p in paths:
            print('SAVED',p)


if __name__=='__main__':
    main()
