"""Check mirror-collapse weights and coverage for the IK ablation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from eval_imu_only_ik import MODELS, PRIMARY, group_name, reduce_rows
from dip_finetuning import MASKS


def fixture():
    p={'mask_order':list(MASKS),'masks':MASKS,'records':[{'sequence_id':0,'frames':10},{'sequence_id':1,'frames':20}]}
    rows=[]
    for mask,sensors in MASKS.items():
        for seq,n,extra in [(0,10,0),(1,20,3)]:
            models={}
            for i,model in enumerate(MODELS):
                v=10+len(sensors)+extra-i
                metrics={key:v for key in ['mpjre_deg','mpjpe_cm','mpjve_cm','local_rotation_deg','jitter_m_s3']}
                metrics['reference_jitter_m_s3']=5.
                models[model]={'sequence_id':str(seq),'frames':n,'jerk_samples':n-3,'metrics':metrics,'settings':{'fixed':'same'}}
            rows.append({'mask':mask,'sequence_id':seq,'models':models})
    return rows,p


class IkAggregationTests(unittest.TestCase):
    def test_24_masks_collapse_to_12_groups_with_known_multiplicities(self):
        rows,p=fixture();r=reduce_rows(rows,p)
        self.assertEqual(len(r['mirror_groups']),12)
        self.assertEqual(sorted(len(g['masks']) for g in r['mirror_groups'].values()),[1,1,1,1,2,2,2,2,2,2,4,4])
        self.assertEqual(group_name([0,3,4]),group_name([1,2,4]))
        self.assertNotEqual(group_name([0,1]),group_name([2,3]))

    def test_primary_weights_masks_and_auxiliary_weights_groups(self):
        rows,p=fixture();r=reduce_rows(rows,p)
        # 5 singles,10 pairs,9 triples. Frame-weighted extra is2.
        self.assertAlmostEqual(r['overall_mask_equal'][MODELS[0]]['mpjve_cm'],12+52/24)
        # Three single, five pair, four triple mirror groups.
        self.assertAlmostEqual(r['overall_12_groups_equal'][MODELS[0]]['mpjve_cm'],12+25/12)
        self.assertAlmostEqual(r['changes']['mpjve_cm']['after_minus_before'],-1.)

    def test_jitter_denominator_and_sequence_equal_are_distinct(self):
        rows,p=fixture();r=reduce_rows(rows,p)
        self.assertAlmostEqual(r['overall_mask_equal'][MODELS[0]]['jitter_100_m_s3'],(10+52/24+51/24)/100)
        self.assertAlmostEqual(r['overall_sequence_equal'][MODELS[0]]['mpjpe_cm'],10+52/24+1.5)

    def test_missing_duplicate_wrong_count_and_reference_change_fail(self):
        rows,p=fixture()
        cases=[rows[:-1],rows+[copy.deepcopy(rows[0])]]
        altered=copy.deepcopy(rows);altered[0]['models'][MODELS[1]]['jerk_samples']+=1;cases.append(altered)
        altered=copy.deepcopy(rows);altered[0]['models'][MODELS[1]]['metrics']['reference_jitter_m_s3']+=1;cases.append(altered)
        for values in cases:
            with self.assertRaises(ValueError):reduce_rows(values,p)


if __name__=='__main__':unittest.main()
