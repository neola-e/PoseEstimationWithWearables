"""Render I02 frozen-test outcomes, category/subject/joint breakdowns."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_ik_jitter import CATEGORIES
from sensor_metrics import JOINT_NAMES

MODELS=['raw','framewise_ik','temporal_ik'];KEYS=['mpjre_deg','mpjpe_cm','mpjve_cm','jitter_100_m_s3']


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def render(root):
    run=root/'test';p=json.loads((run/'PROTOCOL.json').read_text());m=json.loads((run/'metrics.json').read_text())
    validation=json.loads((run/'independent_validation.json').read_text())
    if not validation['passed']:raise ValueError('Independent validation required')
    rows=[json.loads(f.read_text()) for f in sorted((run/'rows').glob('*.json'))]
    records={r['sequence_id']:r for r in p['records']}
    for row in rows:
        rec=records[row['sequence_id']];row['subject']=rec['subject'];row['source_file']=rec['source_file'];row['category']=CATEGORIES[int(rec['source_file'][:2])]
    breakdown={}
    for field in ['subject','category','sequence_id']:
        output=[]
        for value in sorted({r[field] for r in rows}):
            for model in MODELS:
                metrics={}
                for key in KEYS:
                    count='jerk_samples' if key=='jitter_100_m_s3' else 'frames';means=[]
                    for mask in p['mask_order']:
                        selected=[r['models'][model] for r in rows if r['mask']==mask and r[field]==value]
                        means.append(math.fsum(r['metrics'][key]*r[count] for r in selected)/sum(r[count] for r in selected))
                    metrics[key]=math.fsum(means)/24
                output.append({field:value,'model':model,**metrics})
        write_csv(run/f'by_{field}.csv',output);breakdown[field]=output
    joint_rows=[]
    for j in range(24):
        item={'joint':j,'joint_name':JOINT_NAMES[j]}
        for model in MODELS:
            item[model]=math.fsum(r['models'][model]['per_joint']['jitter_m_s3'][j]*r['models'][model]['jerk_samples'] for r in rows)/(24*24021*100)
        item['drop_vs_framewise']=item['framewise_ik']-item['temporal_ik'];joint_rows.append(item)
    write_csv(run/'joint_jitter.csv',joint_rows)
    counts={key:{'lower_than_framewise':sum(m['per_mask'][mask]['temporal_ik']['metrics']['frame_weighted'][key]<m['per_mask'][mask]['framewise_ik']['metrics']['frame_weighted'][key] for mask in p['mask_order']),
                 'lower_than_raw':sum(m['per_mask'][mask]['temporal_ik']['metrics']['frame_weighted'][key]<m['per_mask'][mask]['raw']['metrics']['frame_weighted'][key] for mask in p['mask_order'])} for key in KEYS}
    summary={'overall':m['overall'],'vertex_gain_retained_fraction':m['vertex_gain_retained_fraction'],
             'jitter_drop_vs_framewise_percent':m['jitter_drop_vs_framewise_percent'],'jitter_change_vs_raw_percent':m['jitter_change_vs_raw_percent'],
             'predeclared_success':m['predeclared_success'],'also_below_raw_jitter':m['also_below_raw_jitter'],
             'per_mask_counts':counts,'breakdown':breakdown,'joint_jitter':joint_rows,
             'selection':p['calibration_selection'],'causality':'5 past/current pose/IMU frames, no added future; original upstream dependencies remain'}
    (run/'report_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,4,figsize=(13.6,4),constrained_layout=True)
    for ax,key,title,unit in zip(axes,KEYS,['MPJRE','MPJPE','MPJVE (vertices)','Jitter'],['deg','cm','cm','100 m/s³']):
        values=[m['overall'][model][key] for model in MODELS]
        bars=ax.bar([0,1,2],values,color=['#66788a','#c68b4b','#1b8d88'],width=.65)
        ax.set_xticks([0,1,2],['Network','Framewise\nIK','Temporal\nIK'],fontsize=9)
        ax.set_title(title);ax.set_ylabel(unit);ax.set_ylim(0,max(values)*1.25);ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Separate experiment: causal 5-frame Temporal IK, development-selected lambda')
    fig.savefig(run/'temporal_four_metrics.png',dpi=180);fig.savefig(run/'temporal_four_metrics.pdf');plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['overall','vertex_gain_retained_fraction','jitter_drop_vs_framewise_percent','jitter_change_vs_raw_percent','predeclared_success','also_below_raw_jitter','per_mask_counts']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);render(parser.parse_args().run)
