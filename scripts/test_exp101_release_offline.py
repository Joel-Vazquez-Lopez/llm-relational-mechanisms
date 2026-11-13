#!/usr/bin/env python3
"""Pretraining validation only. Tiny artificial tensors are NOT study runs."""
import json
from pathlib import Path
import tempfile
import unittest

import exp101_production_candidate as prod
import exp101_runtime as rt
import exp101_release as release
import exp101_protected_candidate as future
import exp101_scoring as scoring


class ReleaseOfflineTests(unittest.TestCase):
    def test_source_manifest_complete_and_locked(self):
        root=Path(__file__).resolve().parents[1]
        candidate=release.build_candidate(root)
        self.assertFalse(candidate['training_authorized'])
        self.assertEqual(candidate['status'],'PREFREEZE_REVIEW_ONLY_NO_TRAIN_AUTHORIZATION')
        self.assertTrue(candidate['position_coverage']['LI_within_arch_embedding_capacity'])
        self.assertTrue(candidate['position_coverage']['all_LI_eval_positions_covered'])
        self.assertEqual(candidate['position_coverage']['position_offsets'],list(range(9)))
        self.assertTrue(all(v>0 for v in candidate['position_coverage']['question_token_opportunities_per_index'].values()))
        self.assertEqual(candidate['position_coverage']['training_max_tokens'],34)
        self.assertGreater(candidate['position_coverage']['LI_development_static_max_tokens'],34)
        self.assertEqual(len(candidate['sources_sha256']),len(release.SOURCE_PATHS))
        self.assertEqual(release.validate_candidate(root/'reports/exp101/release_manifest_candidate.json'),candidate)
        self.assertEqual(candidate['intended_hardware_runtime_review'],'PENDING_MAC_MPS_ENGINEERING_FIXTURE')
        with self.assertRaises(PermissionError):
            release.require_approved_release(root/'reports/exp101/frozen_release.json')

    def test_protected_metadata_not_examples(self):
        planned=future.future_manifest_plan()
        self.assertEqual(planned['anchors'],288)
        self.assertEqual(planned['total_questions_per_model'],23040)
        self.assertEqual(planned['total_questions_across_models'],368640)
        self.assertTrue(planned['no_protected_text_generated'])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((FileNotFoundError,PermissionError)):
                future._require_unlock(Path(d),{'fingerprint':'does-not-exist'})

    def test_scoring_wrong_but_consistent(self):
        a={'gold_agent':'Aven','gold_patient':'Bero','pred_agent':'Cira','pred_patient':'Cira'}
        b=dict(a)
        x=scoring.score_voice_pair(a,b)
        self.assertEqual(x['S_voice'],1)
        self.assertEqual(x['C_voice'],0)
        rev={'gold_agent':'Bero','gold_patient':'Aven','pred_agent':'Cira','pred_patient':'Cira'}
        y=scoring.score_role_reverse(a,rev)
        self.assertEqual(y['S_reverse'],1)
        self.assertEqual(y['C_reverse'],0)
        ok={'gold_agent':'Aven','gold_patient':'Bero','pred_agent':'Aven','pred_patient':'Bero'}
        self.assertEqual(scoring.score_voice_pair(ok,ok)['C_voice'],1)
        q=[]
        for i,name in enumerate(('Aven','Bero','Cira','Dovan')):
            q.append({'source':'one identical source','event_position':i//2,
                      'role':'agent' if i%2==0 else 'patient',
                      'gold':name,'prediction':name})
        self.assertEqual(scoring.score_four_query(q)['C_query'],1)
        q[0]['prediction']='Cira'
        self.assertEqual(scoring.score_four_query(q)['C_query'],0)

    def test_cell_scoring_rejects_illegal_structure(self):
        rows=[]
        for i in range(3):
            rows.append({'source':f'example{i}', 'condition':'bare',
              'target_first':True,'voice':'active','correct_agent':'Aven','correct_patient':'Bero',
              'pred_agent':'Aven','pred_patient':'Bero' if i!=2 else 'Aven'})
        r=scoring.development_cell_scores(rows)['bare/first/active']
        self.assertEqual(r['events'],3)
        self.assertAlmostEqual(r['individual_accuracy'],5/6)
        self.assertAlmostEqual(r['both_correct'],2/3)
        with self.assertRaises(ValueError):scoring.development_cell_scores(rows+rows[:1])
        rows[0]['condition']='LI'
        with self.assertRaises(ValueError):scoring.development_cell_scores(rows)

    def test_saturation_not_poor_transfer(self):
        vals={c:{} for c in scoring.CONDITIONS}
        for s in scoring.SEEDS:
            for cond in ('bare','L','I'):
                vals[cond][s]={'target':.90,'familiar':.60}
            vals['LI'][s]={'target':.95,'familiar':.95}
        x=scoring.decision_rule(vals['LI'],vals)
        self.assertTrue(x['attenuation_of_incremental_benefit'])
        self.assertTrue(x['high_high_saturation'])
        self.assertFalse(x['poor_compositional_transfer'])
        self.assertEqual(x['decision'],'attenuated_incremental_advantage')
        with self.assertRaises(ValueError):scoring.paired_bootstrap([.1]*7)
        fake={c:[.1]*8 for c in scoring.CONDITIONS}
        self.assertEqual(scoring.synchronized_bootstrap(fake,reps=100)['bare']['ci95'],[.1,.1])

    def test_untrained_model_development_decode_smoke(self):
        import itertools
        import torch
        train,dev,sealed=prod.all_matchings()
        packets=list(itertools.islice(prod.examples(dev),2))
        self.assertTrue(all(p.condition!='LI' for p in packets))
        model=prod.make_model().eval()  # initialized, NEVER trained
        scores=prod.score_development_model(model,packets,batch_packets=2)
        cell=next(iter(scores.values()))
        self.assertEqual(cell['events'],2)
        self.assertGreaterEqual(cell['individual_accuracy'],0)
        self.assertLessEqual(cell['individual_accuracy'],1)
        self.assertGreaterEqual(cell['both_correct'],0)
        self.assertLessEqual(cell['both_correct'],1)

    def test_full_packet_scoring_on_synthetic_development_only(self):
        from exp101_generator_preflight import case_events,render_packet,other_event_query,eventblind_second,VOICES
        rows=[];blind=[]
        for cond in ('bare','L','I','LI'):
            for reverse in (0,1):
                for first in (True,False):
                    for voice in VOICES:
                        target,other,names=case_events(0,reverse,voice,'passive',0,1,0,0,0)
                        p=render_packet(target,other,names,cond,first)
                        for q,pkt in [('target',p)]+([('other',other_event_query(p))] if cond=='LI' else []):
                            key=(0,cond,reverse,first,voice)
                            base={'key':key,'query':q,'source':pkt.text,
                              'gold_agent':pkt.correct_agent,'gold_patient':pkt.correct_patient,
                              'target_first':pkt.target_first,'voice':pkt.event.voice}
                            rows.append(dict(base,pred_agent=pkt.correct_agent,
                              pred_patient=pkt.correct_patient))
                            bad=eventblind_second(pkt)
                            blind.append(dict(base,pred_agent=bad[0],pred_patient=bad[1]))
        perfect=future.analyze_prediction_rows(rows,anchor_count=1)
        self.assertEqual(perfect['C_voice_by_condition']['LI'],1)
        self.assertEqual(perfect['C_reverse_by_condition']['LI'],1)
        self.assertEqual(perfect['C_event_by_condition']['LI'],1)
        self.assertEqual(perfect['C_query_LI'],1)
        shortcut=future.analyze_prediction_rows(blind,anchor_count=1)
        self.assertEqual(shortcut['C_voice_by_condition']['LI'],1)
        self.assertEqual(shortcut['C_query_LI'],0)

    def test_paired_weights_batches_and_optimizer_resume(self):
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(301)
        vocab=len(prod.VOCAB_WORDS)
        # Random tokens/labels are artificial; not EXP-101 examples.
        x=torch.randint(0,vocab,(10,9))
        ro=torch.full((10,),8,dtype=torch.long)
        labels=torch.tensor([prod.VOCAB[n] for n in prod.NAMES[:10]])
        choices=torch.tensor([[prod.VOCAB[n] for n in prod.NAMES[:4]]]*10)
        base=(x,ro,labels,choices)
        dose=(x.flip(0),ro,labels.flip(0),choices)
        def make():return prod.make_model()
        (m1,o1,b1,d1)=rt.init_state(907,make)
        (m2,o2,b2,d2)=rt.init_state(907,make)
        for k,v in m1.state_dict().items():self.assertTrue(torch.equal(v,m2.state_dict()[k]),k)
        self.assertTrue(torch.equal(b1.get_state(),b2.get_state()))
        self.assertTrue(torch.equal(d1.get_state(),d2.get_state()))
        loss1,idx1=rt.one_optimizer_step(m1,o1,base,dose,b1,d1,
                  base_batch=3,dose_batch=2,pad_id=prod.VOCAB['<pad>'])
        loss2,idx2=rt.one_optimizer_step(m2,o2,base,dose,b2,d2,
                  base_batch=3,dose_batch=2,pad_id=prod.VOCAB['<pad>'])
        self.assertEqual(idx1,idx2);self.assertEqual(loss1,loss2)
        self.assertFalse(torch.equal(m1.embed.weight,rt.init_state(907,make)[0].embed.weight))
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'state.pt'
            rt.save_state(checkpoint,model=m2,optimizer=o2,gbase=b2,gdose=d2,
              seed=907,arm='target',step=1,release_fingerprint='dummy-frozen',
              data_hash='dummy-data',device='cpu')
            (m3,o3,b3,d3)=rt.init_state(907,make)
            step=rt.load_state(checkpoint,model=m3,optimizer=o3,gbase=b3,gdose=d3,
              seed=907,arm='target',release_fingerprint='dummy-frozen',
              data_hash='dummy-data',device='cpu')
            self.assertEqual(step,1)
            with self.assertRaises(PermissionError):
                rt.load_state(checkpoint,model=m3,optimizer=o3,gbase=b3,gdose=d3,
                   seed=907,arm='target',release_fingerprint='TAMPERED',
                   data_hash='dummy-data',device='cpu')
            l_uninterrupted,p_uninterrupted=rt.one_optimizer_step(m1,o1,base,dose,b1,d1,
                base_batch=3,dose_batch=2,pad_id=prod.VOCAB['<pad>'])
            l_resumed,p_resumed=rt.one_optimizer_step(m3,o3,base,dose,b3,d3,
                base_batch=3,dose_batch=2,pad_id=prod.VOCAB['<pad>'])
            self.assertEqual(p_uninterrupted,p_resumed)
            self.assertEqual(l_uninterrupted,l_resumed)
            for k,v in m1.state_dict().items():
                self.assertTrue(torch.equal(v,m3.state_dict()[k]),f'resume mismatch in {k}')
            for p1,p2 in zip(o1.param_groups,o3.param_groups):
                self.assertEqual(p1['lr'],p2['lr'])
            for va,vb in zip(o1.state.values(),o3.state.values()):
                for k in va:
                    if isinstance(va[k],torch.Tensor):
                        self.assertTrue(torch.equal(va[k],vb[k]),f'optimizer mismatch {k}')
                    else:self.assertEqual(va[k],vb[k])


    def test_position_offset_rejects_invalid_ids_and_preserves_no_shift_inference(self):
        import torch
        model=prod.make_model().eval()
        x=torch.full((2,8),prod.VOCAB['<bos>'],dtype=torch.long)
        r=torch.tensor([7,7])
        with torch.no_grad():
            legacy=model(x,r)
            shifted=model(x,r,position_offset=8)
            zero=model(x,r,position_offset=0)
        self.assertTrue(torch.equal(legacy,zero))
        self.assertFalse(torch.equal(zero,shifted))
        with self.assertRaises(ValueError):model(x,r,position_offset=9)

    def test_determinism_policy_is_explicit_and_strict_on_cpu(self):
        import torch
        cpu = rt.configure_determinism('cpu')
        self.assertEqual(cpu['mode'], 'cpu_strict')
        self.assertFalse(cpu['warn_only'])
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        if hasattr(torch, 'is_deterministic_algorithms_warn_only_enabled'):
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        if torch.backends.mps.is_available():
            mps = rt.configure_determinism('mps')
            self.assertEqual(mps['mode'], 'mps_warn_only')
            self.assertTrue(mps['warn_only'])
            if hasattr(torch, 'is_deterministic_algorithms_warn_only_enabled'):
                self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
            rt.configure_determinism('cpu')

    def test_disposable_cpu_hardware_fixture(self):
        # Three engineering optimizer steps, not study training; uses seed 101999.
        from exp101_hardware_fixture import run_fixture,FIXTURE_SEED
        self.assertNotIn(FIXTURE_SEED,prod.SEEDS)
        result=run_fixture('cpu')
        self.assertEqual(result['fixture_steps'],3)
        self.assertFalse(result['experiment_training_started'])
        self.assertEqual(result['resume_parameter_max_abs_difference'],0)
        self.assertTrue(result['resume_optimizer_within_tolerance'])
        self.assertTrue(all(x>1e-12 for x in result['position_gradient_norms_34_through_38'].values()))

    def test_checkpoint_schedule_identifier_guard(self):
        import torch, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            m,o,b,d=rt.init_state(101999,prod.make_model)
            state=Path(folder)/'state.pt'
            rt.save_state(state,model=m,optimizer=o,gbase=b,gdose=d,
               seed=101999,arm='fixture',step=1,release_fingerprint='fixture',
               data_hash='fixture',device='cpu')
            payload=torch.load(state,map_location='cpu',weights_only=True)
            payload['position_schedule']='UNAPPROVED_CHANGE'
            torch.save(payload,state)
            with self.assertRaises(PermissionError):
                rt.load_state(state,model=m,optimizer=o,gbase=b,gdose=d,
                  seed=101999,arm='fixture',release_fingerprint='fixture',
                  data_hash='fixture',device='cpu')

if __name__=='__main__':unittest.main()
