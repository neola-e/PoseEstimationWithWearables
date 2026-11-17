import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from experiment_temporal_ik import predict_interval,select_weight,GRID
from window_inference import infer_sequence


class Predictor:
    def predict(self,x,lengths=None):
        x=torch.as_tensor(x)
        if x.ndim==2:x=x[None]
        return x[:,:,:1].cumsum(1).repeat(1,1,144)


class TemporalExperimentTests(unittest.TestCase):
    def test_interval_matches_canonical_rolling_last(self):
        x=torch.arange(300,dtype=torch.float32)[:,None].repeat(1,60)
        all_outputs=infer_sequence(Predictor(),x,mode='rolling_last',batch_size=16)
        for a,b in [(0,4),(100,140),(240,300)]:
            torch.testing.assert_close(predict_interval(Predictor(),x,a,b,batch_size=7),all_outputs[a:b],atol=0,rtol=0)

    def test_fixed_spatial_guards_then_jitter_determine_selection(self):
        raw={'mpjre_deg':20.,'mpjpe_cm':10.,'mpjve_cm':12.,'jitter_100_m_s3':1.}
        ind={**raw,'mpjre_deg':19.,'mpjpe_cm':9.9,'mpjve_cm':11.8,'jitter_100_m_s3':1.2}
        metrics={'raw':raw,'framewise_ik':ind}
        for w in GRID:metrics['lambda_'+str(w)]={**ind,'mpjve_cm':11.85,'jitter_100_m_s3':1.1-w*.01}
        # Lowest jitter loses too much spatial gain and must be rejected.
        metrics['lambda_1.0']['mpjve_cm']=12.2
        result=select_weight(metrics)
        self.assertEqual(result['chosen_weight'],.1);self.assertTrue(result['eligible'])
        self.assertAlmostEqual(result['targets']['mpjve_cm'],11.9)


if __name__=='__main__':unittest.main()
