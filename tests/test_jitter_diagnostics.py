import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_ik_jitter import band_energy,third_difference


class JitterDiagnosticTests(unittest.TestCase):
    def test_third_difference_polynomials_and_no_cross_recording(self):
        t=np.arange(20,dtype=np.float64)/25
        for power in [0,1,2]:self.assertLess(np.max(np.abs(third_difference(t**power))),1e-12)
        np.testing.assert_allclose(third_difference(t**3)*25**3,6,atol=1e-10)
        a=np.zeros((10,24,3));b=np.ones_like(a)
        self.assertEqual(np.abs(third_difference(a)).sum()+np.abs(third_difference(b)).sum(),0)
        self.assertGreater(np.abs(third_difference(np.concatenate([a,b]))).sum(),0)

    def test_known_frequency_band_and_energy(self):
        t=np.arange(500)/25
        for f,inside in [(1,False),(7,True)]:
            x=np.zeros((500,1,3));x[:,0,0]=np.sin(2*np.pi*f*t)
            high,total=band_energy(x)
            # Linear detrending a finite sine has a small nonzero fitted slope;
            # the subsequent Hann weighting need not preserve exactly0.5.
            self.assertAlmostEqual(total[0],.5,delta=5e-4)
            self.assertTrue(high[0]/total[0]>.999 if inside else high[0]/total[0]<1e-6)

    def test_signed_contributions_add_with_shared_denominator(self):
        values=np.arange(2*3*4).reshape(2,3,4)-10
        denominator=values.size*100
        self.assertAlmostEqual(sum(values[:,j].sum()/denominator for j in range(3)),values.mean()/100)


if __name__=='__main__':unittest.main()
