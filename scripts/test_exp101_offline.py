#!/usr/bin/env python3
"""Offline tests for EXP-101 R3; no torch, model training or protected cases."""
from __future__ import annotations
import dataclasses
import json
import unittest
from pathlib import Path

import exp101_generator_preflight as g
import exp101_production_candidate as prod
import exp101_render_validator as val


class R3OfflineTests(unittest.TestCase):
    def test_locative_slots_and_both_events_separately(self):
        a,b,nn=g.case_events(0,0,"active","passive",0,0,1,0,0)
        p=g.render_packet(a,b,nn,"LI",True)
        parsed=g.validate_case(p)
        self.assertEqual(parsed[0].gate_owner,a.by_name)
        self.assertEqual(parsed[0].near_person,a.near_name)
        self.assertEqual(parsed[1].gate_owner,b.by_name)
        self.assertEqual(parsed[1].near_person,b.near_name)

    def test_deliberate_corruptions_rejected(self):
        self.assertEqual(g.check_deliberate_corruptions(),6)

    def test_question_switch_same_source_distinct_people(self):
        a,b,nn=g.case_events(0,0,"active","passive",0,0,1,0,0)
        p=g.render_packet(a,b,nn,"LI",True)
        other=g.other_event_query(p)
        self.assertEqual(p.text,other.text)
        self.assertEqual({p.correct_agent,p.correct_patient,
                          other.correct_agent,other.correct_patient},set(nn))
        self.assertEqual(val.parse_query(other.questions[0]),(1,"agent"))
        g.validate_case(other)
        self.assertEqual(g.eventblind_second(other),(other.correct_agent,other.correct_patient))
        self.assertNotEqual(g.eventblind_second(p),(p.correct_agent,p.correct_patient))

    def test_one_factorization_and_no_cross_split_pairs(self):
        train,dev,sealed=prod.all_matchings()
        parts=[{pair for match in x for pair in match} for x in (train,dev,sealed)]
        self.assertEqual(tuple(map(len,parts)),(72,24,24))
        self.assertEqual(len(set.union(*parts)),120)
        self.assertFalse(parts[0]&parts[1]);self.assertFalse(parts[0]&parts[2])
        self.assertFalse(parts[1]&parts[2])

    def test_development_cells_and_training_arms(self):
        manifest,train,dev,doses=prod.dataset_manifest()
        self.assertEqual(len(manifest["dev_cells"]),9)
        self.assertTrue(all(n>=96 for n in manifest["dev_cells"].values()))
        self.assertEqual(len(doses["target"]),64)
        self.assertEqual(len(doses["familiar"]),64)
        self.assertEqual({p.condition for p in train},set(prod.DEV_CONDS))
        self.assertEqual({p.condition for p in dev},set(prod.DEV_CONDS))
        self.assertNotIn("LI",set(p.condition for p in train+dev))
        self.assertEqual(manifest["protected_pair_count"],24)
        self.assertNotIn("protected_text",manifest)
        self.assertEqual(manifest["train_examples"],62208)
        self.assertEqual(manifest["dev_examples"],20736)
        # Catch the R3 implementation bug discovered during production audit:
        # querying-event balance alone must not hide uneven distractor roles
        # or adjunct owner/near-person identity frequencies.
        for rows in (train,dev):
            for attr in ("agent","patient","by_name","near_name"):
                for ev in ("event","other_event"):
                    counts=__import__("collections").Counter(
                        getattr(getattr(pkt,ev),attr) for pkt in rows)
                    self.assertEqual(set(counts),set(prod.NAMES))
                    self.assertEqual(len(set(counts.values())),1)

    def test_competence_gate_joints_not_separate_arm_marginals(self):
        _,_,_,_=prod.dataset_manifest()
        required={f"{condition}/{('first' if first else 'second')}/{voice}"
                  for condition in prod.DEV_CONDS for first,voice in prod.FAMILIAR}
        passing={x:{"events":96,"individual_accuracy":.96,"both_correct":.91}
                 for x in required}
        failing={x:{"events":96,"individual_accuracy":.94,"both_correct":.91}
                 for x in required}
        # Seven passing in each arm, but only six jointly passing.
        scores={(seed,arm):passing.copy() for seed in prod.SEEDS for arm in prod.ARMS}
        scores[(prod.SEEDS[0],"familiar")]=failing
        scores[(prod.SEEDS[1],"target")]=failing
        gate=prod.competence_gate(scores)
        self.assertEqual(len(gate["jointly_passing"]),6)
        self.assertTrue(gate["eligible"])
        # One additional non-overlapping failure -> five joint; must stop.
        scores[(prod.SEEDS[2],"target")]=failing
        gate=prod.competence_gate(scores)
        self.assertEqual(len(gate["jointly_passing"]),5)
        self.assertFalse(gate["eligible"])

    def test_static_report_counts_are_computed(self):
        report = Path(__file__).resolve().parent.parent / "reports/exp101/static_preflight_R3.json"
        if not report.exists():
            report = Path(__file__).resolve().parent / "static_preflight_R3.json"
        r=json.loads(report.read_text())
        self.assertEqual(r["deliberate_corruptions_rejected"],6)
        for cond in g.CONDS:
            for row in r["conditions"][cond]["baseline_accuracy"].values():
                self.assertEqual(row["all_eight_perfect"],row["all_eight_perfect_packet_count"]>0)
                self.assertEqual(row["all_eight_packet_count"],3072)
        self.assertEqual(r["conditions"]["bare"]["baseline_accuracy"]["nearest_verb"]["all_eight_perfect_packet_count"],3072)
        second=r["conditions"]["LI"]["baseline_accuracy"]["event_blind_second"]
        self.assertEqual(second["second_event_voice_pair_correct"],6144)
        self.assertEqual(second["same_source_four_query_correct"],0)


if __name__ == "__main__":
    unittest.main()
