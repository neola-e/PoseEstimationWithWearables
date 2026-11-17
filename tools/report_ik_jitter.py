"""Static scientific figures and concise diagnostics from completed J01 results."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_ik_jitter import rotation_vectors
from paper_metrics import local_rotations


def periodogram(x,fps=25):
    t=np.arange(len(x),dtype=np.float64);t-=t.mean();x=x-x.mean(0)
    x=x-t[:,None]*(x*t[:,None]).sum(0)/(t*t).sum()
    window=np.hanning(len(x));power=(np.abs(np.fft.rfft(x*window[:,None],axis=0))**2).sum(1)/(fps*(window*window).sum())
    power[1:]*=2
    if len(x)%2==0:power[-1]/=2
    return np.fft.rfftfreq(len(x),1/fps),power


def report(run,counterfactual):
    s=json.loads((run/'summary.json').read_text());p=json.loads((run/'PROTOCOL.json').read_text());total=s['overall_increase']
    fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
    for ax,name,label,title in zip(axes,['by_joint','by_mask','by_category'],['joint_name','mask','category'],['Receiving joints','Sensor combinations','Recorded motion categories']):
        rows=s['rankings'][name][:10]
        ax.barh([r[label] for r in rows][::-1],[100*r['overall_contribution']/total for r in rows][::-1],color='#2a8581')
        ax.set_title(title);ax.set_xlabel('Share of net jitter increase (%)');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Framewise IK: attribution of the full24-mask jitter increase')
    fig.savefig(run/'jitter_attribution.png',dpi=180);fig.savefig(run/'jitter_attribution.pdf');plt.close(fig)
    case=s['top_pairs'][0];name=f'{case["mask"]}_seq{case["sequence_id"]:02d}'
    with np.load(run/'time_series'/f'{name}.npz',allow_pickle=False) as a:
        t=np.arange(len(a['correction_rotvec']))/25;jt=(np.arange(len(a['jerk_norm_raw']))+3)/25
        peak=case['peak_output_frame']/25;view=(max(0,peak-1.6),min(t[-1],peak+1.6))
        fig,axes=plt.subplots(4,1,figsize=(12,10),constrained_layout=True)
        axes[0].plot(jt,a['jerk_norm_raw'].mean(1)/100,label='Network');axes[0].plot(jt,a['jerk_norm_ik'].mean(1)/100,label='Framewise IK',alpha=.8)
        axes[0].set_ylabel('Jitter (100 m/s³)');axes[0].legend()
        for i,label in enumerate(['x','y','z']):axes[1].plot(t,np.rad2deg(a['correction_rotvec'][:,0,i]),label='pelvis correction '+label)
        axes[1].set_ylabel('Correction (deg)');axes[1].legend(ncol=3)
        axes[2].plot(t[1:],a['global_step_raw_deg'][:,11],label='Right foot, network')
        axes[2].plot(t[1:],a['global_step_ik_deg'][:,11],label='Right foot, IK',alpha=.8)
        axes[2].set_ylabel('Global step (deg)');axes[2].legend()
        for ax in axes[:3]:ax.set_xlim(*view);ax.axvline(peak,color='gray',linestyle=':',alpha=.5)
        axes[2].set_xlabel('Time in processed recording (s)')
    base=Path(p['baseline_run']);ik=Path(p['ik_run'])
    with np.load(base/'predictions'/f'{name}.npz',allow_pickle=False) as a,np.load(ik/'refined'/f'{name}.npz',allow_pickle=False) as b:
        for pose,label in [(a['pose_dip_final_2ep'],'Network'),(b['pose'],'Framewise IK')]:
            r=local_rotations(pose).double().numpy()[:,0]
            velocity=rotation_vectors(np.swapaxes(r[:-1],-1,-2)@r[1:])*25
            f,power=periodogram(velocity)
            axes[3].semilogy(f,np.maximum(power,1e-15),label=label)
    axes[3].axvspan(5,12.5,color='#eee1c9',alpha=.6,label='Reported high-frequency band')
    axes[3].set_xlabel('Frequency (Hz)');axes[3].set_ylabel('Pelvis angular-velocity PSD');axes[3].legend()
    fig.suptitle(f'{case["subject"]}/{case["source_file"]}, {case["mask"]}, category={case["category"]}\nExample chosen by greatest contribution, not by visual appearance')
    fig.savefig(run/'rotation_time_frequency_example.png',dpi=170);fig.savefig(run/'rotation_time_frequency_example.pdf');plt.close(fig)
    cf=json.loads((counterfactual/'results.json').read_text());raw=cf['baseline']['network_final']['jitter_100_m_s3'];ikv=cf['baseline']['network_final_ik']['jitter_100_m_s3']
    enriched={'pelvis_restore_excess_reduction_percent':(ikv-cf['overall']['restore_pelvis'])/(ikv-raw)*100,
              'pocket_masks_share_net_increase_percent':sum(r['overall_contribution'] for r in s['rankings']['by_mask'] if 'lp' in r['mask'] or 'rp' in r['mask'])/total*100,
              'four_foot_ankle_joints_share_net_increase_percent':sum(r['overall_contribution'] for r in s['rankings']['by_joint'] if r['joint'] in [7,8,10,11])/total*100,
              'counterfactuals':cf['overall'],'example':case}
    (run/'report_summary.json').write_text(json.dumps(enriched,indent=2)+'\n');print(json.dumps(enriched,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--counterfactuals',type=Path,required=True)
    a=parser.parse_args();report(a.run,a.counterfactuals)
