"""Paired test coverage and frozen 24-mask aggregation checks; no GPU needed."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_dip_finetuning import aggregate_pairs, MODEL_NAMES, PRIMARY, EXPECTED_TEST_SHA
from dip_finetuning import MASKS


def fixture():
    p={'mask_order':list(MASKS),'records':[{'sequence_id':0,'frames':10},{'sequence_id':1,'frames':20}]}
    rows=[]
    for i,mask in enumerate(p['mask_order']):
        for seq,frames,offset in [(0,10,0),(1,20,3)]:
            models={}
            for k,model in enumerate(MODEL_NAMES):
                value=10+i+offset-2*k
                metrics={key:value for key in ['mpjre_deg','mpjpe_cm','mpjve_cm','local_rotation_deg','jitter_m_s3']}
                metrics['reference_jitter_m_s3']=5.
                models[model]={'sequence_id':str(seq),'frames':frames,'jerk_samples':frames-3,
                               'metrics':metrics,'settings':{'fixed':'same'}}
            rows.append({'mask':mask,'sequence_id':seq,'models':models})
    return rows,p


class ComparisonAggregationTests(unittest.TestCase):
    def test_equal_mask_mean_and_frame_weighted_spatial_mean(self):
        rows,p=fixture()
        result=aggregate_pairs(rows,p)
        # mean(mask index)=11.5; frame-weighted offset=(10*0+20*3)/30=2.
        self.assertAlmostEqual(result['overall_mask_equal'][MODEL_NAMES[0]]['mpjve_cm'],23.5)
        self.assertAlmostEqual(result['overall_mask_equal'][MODEL_NAMES[1]]['mpjve_cm'],21.5)
        self.assertAlmostEqual(result['changes']['mpjve_cm']['after_minus_before'],-2.)
        self.assertAlmostEqual(result['changes']['mpjve_cm']['improvement_percent'],2/23.5*100)
        self.assertEqual(result['coverage'],{'masks':24,'recordings_per_mask':2,'frames_per_mask':30,'pairs':48})

    def test_jitter_uses_t_minus_three_and_100_unit_conversion(self):
        rows,p=fixture()
        r=aggregate_pairs(rows,p)
        # jerk counts 7 and17 differ from the spatial counts10 and20.
        self.assertAlmostEqual(r['overall_mask_equal'][MODEL_NAMES[0]]['jitter_100_m_s3'],(21.5+17*3/24)/100)
        self.assertEqual(r['reference_jitter_m_s3'],5.)

    def test_missing_duplicate_extra_and_global_results_rejected(self):
        rows,p=fixture()
        cases=[rows[:-1],rows+[copy.deepcopy(rows[0])]]
        extra=copy.deepcopy(rows[0]);extra['mask']='global';cases.append(rows+[extra])
        for altered in cases:
            with self.assertRaises(ValueError):
                aggregate_pairs(altered,p)

    def test_changed_recording_length_or_jerk_stencil_rejected(self):
        rows,p=fixture()
        for key in ['frames','jerk_samples']:
            changed=copy.deepcopy(rows)
            changed[0]['models'][MODEL_NAMES[1]][key]+=1
            with self.assertRaises(ValueError):
                aggregate_pairs(changed,p)

    def test_mismatched_target_trajectory_is_rejected(self):
        rows,p=fixture()
        rows[0]['models'][MODEL_NAMES[1]]['metrics']['reference_jitter_m_s3']+=1
        with self.assertRaises(ValueError):
            aggregate_pairs(rows,p)

    def test_result_order_does_not_change_spatial_aggregate(self):
        rows,p=fixture()
        a=aggregate_pairs(rows,p)
        b=aggregate_pairs(list(reversed(rows)),p)
        for model in MODEL_NAMES:
            for key in PRIMARY:
                self.assertAlmostEqual(a['overall_mask_equal'][model][key],b['overall_mask_equal'][model][key],places=12)
        self.assertEqual(len(EXPECTED_TEST_SHA),64)


if __name__ == '__main__':
    unittest.main()
