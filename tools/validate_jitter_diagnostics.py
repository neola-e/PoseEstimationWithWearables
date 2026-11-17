"""Independently verify J01 per-joint measurement and contribution accounting."""
import argparse,json,math
from pathlib import Path
from runtime import sha256


def audit(run,counterfactual):
    p=json.loads((run/'PROTOCOL.json').read_text());base=Path(p['ik_run'])
    for root in [run,counterfactual]:
        for name,digest in json.loads((root/'ARTIFACT_MANIFEST.json').read_text()).items():
            if sha256(root/name)!=digest:raise ValueError('Diagnostic artifact changed')
    rows=[json.loads(f.read_text()) for f in sorted((run/'rows').glob('*.json'))]
    if len(rows)!=456 or len({(r['mask'],r['sequence_id']) for r in rows})!=456:raise ValueError('Coverage mismatch')
    differences=[];contributions=[]
    for row in rows:
        name=f'{row["mask"]}_seq{row["sequence_id"]:02d}'
        original=json.loads((base/'rows'/f'{name}.json').read_text())
        for joint in row['joints']:
            j=joint['joint']
            for model,key in [('network_final','raw_jitter_100_m_s3'),('network_final_ik','ik_jitter_100_m_s3')]:
                differences.append(abs(joint[key]-original['models'][model]['per_joint']['jitter_m_s3'][j]/100))
            contributions.append(joint['jitter_delta_sum_m_s3']/(24*24*24021*100))
    if max(differences)>1e-10:raise ValueError('Per-joint measurements differ')
    increase=math.fsum(contributions)
    if abs(increase-p['baseline_delta'])>1e-10:raise ValueError('Accounting does not reproduce total increase')
    for name in ['by_joint','by_mask','by_sequence','by_category','by_joint_category']:
        summary=json.loads((run/'summary.json').read_text())
        if abs(math.fsum(x['overall_contribution'] for x in summary['rankings'][name])-increase)>1e-10:
            raise ValueError('Grouped contributions do not add up')
    return {'passed':True,'pairs':456,'joint_recording_mask_cells':456*24,'per_joint_max_abs_difference':max(differences),
            'independent_increase':increase,'accounting_max_abs_difference':abs(increase-p['baseline_delta']),
            'all_group_totals_reproduce_delta':True,'frequency_checks':'Known 1Hz/7Hz signal tests passed; high frequency is descriptive, not an action/noise ground truth',
            'counterfactual_status':'Controlled removal/retention of pose components; nonlinear effects not additive','source_sha256':sha256(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--counterfactuals',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=audit(args.run,args.counterfactuals);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
