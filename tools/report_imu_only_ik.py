"""Render audited IK results and the explicit paper-convention reconciliation."""
import argparse
import csv
import json
import math
from pathlib import Path

MODELS=('network_final','network_final_ik')
METRICS=[('mpjre_deg','MPJRE','°'),('mpjpe_cm','MPJPE（关节）','cm'),
         ('mpjve_cm','MPJVE（网格顶点）','cm'),('jitter_100_m_s3','Jitter','100 m/s³')]


def write_csv(path,fields,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def render(run):
    p=json.loads((run/'PROTOCOL.json').read_text());r=json.loads((run/'metrics.json').read_text())
    validation=json.loads((run/'independent_validation.json').read_text())
    if not validation['passed'] or r['coverage']['pairs']!=456:raise ValueError('Complete audited run required')
    rows=[json.loads(f.read_text()) for f in sorted((run/'rows').glob('*.json'))]
    summary=[]
    for key,label,unit in METRICS:
        deltas=[r['per_mask'][mask][MODELS[1]]['metrics']['frame_weighted'][key]
                -r['per_mask'][mask][MODELS[0]]['metrics']['frame_weighted'][key] for mask in p['mask_order']]
        summary.append({'metric':key,'label':label,'unit':unit,'raw':r['overall_mask_equal'][MODELS[0]][key],
                        'ik':r['overall_mask_equal'][MODELS[1]][key],**r['changes'][key],
                        'masks_lower':sum(d<0 for d in deltas),'masks_higher':sum(d>0 for d in deltas),
                        'masks_equal':sum(d==0 for d in deltas)})
    subjects={}
    for subject in p['data']['subjects']:
        ids={rec['sequence_id'] for rec in p['records'] if rec['subject']==subject}
        subjects[subject]={}
        for model in MODELS:
            subjects[subject][model]={}
            for key,_,_ in METRICS:
                count='jerk_samples' if key=='jitter_100_m_s3' else 'frames';means=[]
                for mask in p['mask_order']:
                    selected=[row['models'][model] for row in rows if row['mask']==mask and row['sequence_id'] in ids]
                    means.append(math.fsum(v['metrics'][key]*v[count] for v in selected)/sum(v[count] for v in selected))
                subjects[subject][model][key]=math.fsum(means)/24
    subject_equal={model:{key:math.fsum(sub[model][key] for sub in subjects.values())/len(subjects)
                           for key,_,_ in METRICS} for model in MODELS}
    reductions={'primary_24_masks_frame_weighted':r['overall_mask_equal'],
                'aux_24_masks_sequence_equal':r['overall_sequence_equal'],
                'aux_12_groups_equal':r['overall_12_groups_equal'],
                'aux_24_masks_subject_equal':subject_equal}
    imu={}
    for key in ['orientation_mse_before','orientation_mse_after','orientation_deg_before','orientation_deg_after']:
        imu[key]=math.fsum(row['ik'][key]*row['ik']['frames']*len(row['ik']['sensors']) for row in rows)/sum(row['ik']['frames']*len(row['ik']['sensors']) for row in rows)
    imu['imu_mse_increased_frames']=sum(row['ik']['imu_mse_increased_frames'] for row in rows)
    imu['mask_frames']=sum(rec['frames'] for rec in p['records'])*24
    imu['projection_frobenius_max']=max(row['ik']['projection_frobenius_max'] for row in rows)
    imu['unoptimized_local_parameters_bitwise_unchanged']=True
    write_csv(run/'changes.csv',list(summary[0]),summary)
    write_csv(run/'per_subject.csv',['subject','model',*[m[0] for m in METRICS]],
              [{'subject':subject,'model':model,**values} for subject,models in subjects.items() for model,values in models.items()])
    write_csv(run/'aggregation_sensitivity.csv',['reduction','model',*[m[0] for m in METRICS]],
              [{'reduction':name,'model':model,**values} for name,models in reductions.items() for model,values in models.items()])
    alignment=[
      ('设备配置','24 个位置组合；按12类展示','24mask 全覆盖；按腕/口袋/头数量折叠成12类','已核对','3个单位置类、5个双位置类、4个三位置类；整体加权另列'),
      ('DIP 划分','前8训练，后2测试','最终s01–s08两轮162步；s09/s10只用于固定报告','已核对','预算由开发集选定，不由本轮测试选择'),
      ('采样与缩放','25Hz；acc/30；全局方向','保留原25Hz处理与60维通道，缺失置零','已核对','矩阵为逐元素插值，IK单独投影'),
      ('腕部方向映射','wrist/hand设备，优化肩肘','依公开合成ji_mask，实际约束前臂18/19，优化16/18或17/19','源码支持','保留raw DIP [7,8,9,10,0]；无GT拟合的训练域检查约1.4–3.4°，头约0.37°'),
      ('头/口袋优化索引','描述类似的头、髋 refinement，未列全部索引','头12/15；左口袋0/1；右口袋0/2','重建约定','尚未找回作者精确索引列表'),
      ('优化目标与次数','IMU全局方向、MSE、Adam、每帧10次','只用可用IMU方向矩阵MSE，10次，联合处理约束','已实现并验证','不输入GT、位置、预测未来帧或隐藏传感器'),
      ('IK学习率','IMUPoser未给出；引用AvatarPoser','固定1e-3，来自被引用论文','重建约定','AvatarPoser公开函数默认1e-1/5次且优化位置，不是IMUPoser实现'),
      ('旋转参数化','未说明','局部右乘旋转向量增量，FP64计算、FP32输出','重建约定','零增量初始化，SO(3)保证；未优化local参数原样保留'),
      ('迭代输出和状态','未说明细节','每帧独立Adam；取第10次更新后结果，无回退或best选择','重建约定','batch1/256实数据重算检查；无新增时间滤波'),
      ('在线窗口','125历史；另写1-sample overlap','保持F03 stride1、末帧输出、短前缀起步','作者细节未决','未找回公开评估循环；不把字面overlap自动当作stride1证据'),
      ('预处理未来信息','五帧平均；未细化方向','保留既有居中acc均值；填补与插值沿用原流程','已披露的差异','acc均值有2帧/80ms未来依赖；IK自身不新增未来信息'),
      ('MPJRE','全局关节旋转误差°','24关节FK global geodesic；local角误差另报','已核对','无root旋转对齐'),
      ('MPJPE','所有关节误差cm，pelvis对齐','24关节、pelvis平移对齐、root保留','已核对','非Procrustes对齐'),
      ('MPJVE','所有SMPL网格顶点误差cm，pelvis对齐','6890顶点；与joint velocity分开','已核对','被引用AvatarPoser的MPJVE代表速度，未移植其指标定义'),
      ('Jitter','预测平均jerk；表格100m/s³','25³三阶差分，序列内T−3加权，除100','已核对','不减GT jerk，不跨录制边界'),
      ('身体形状','主文未指定betas','固定mean male beta0，与F03一致','作者细节未决','公开processed DIP存全1；未以它替换已验证主口径'),
      ('mesh与平移','SMPL；不预测全局平移','继承公开FK默认pose-blendshape关闭；零轨迹平移','源码约定','作者实际测量开关未完整公开'),
      ('平均方式','跨设备组合给出总体均值；展示12组','主结果mask内帧/T−3加权再24mask等权','作者细节未决','同时给序列等权、12类等权、受试者等权敏感性'),
      ('AMASS训练配方','描述63小时、24mask、80epoch','历史Epoch59来自25mask初始化与已审计的语料/早停','已知训练差异','本轮不重训AMASS；严格配方对齐仍需单独规划'),
      ('DIP选轮与padding','主文未给完整内部选择/尾部规则','冻结开发集选K2；最终从Epoch59重启；训练保留padding损失','显式约定','评估排除padding；未用本轮分数改训练'),
      ('消费设备/实时系统','另有消费设备数据与实机演示','本轮只完成DIP离线测量与IK','第二阶段','CPU批量吞吐不等于实机延迟或消费者设备精度')]
    write_csv(run/'paper_alignment.csv',['item','paper','current','status','residual'],
              [dict(zip(['item','paper','current','status','residual'],row)) for row in alignment])
    report={'primary':summary,'per_subject':subjects,'aggregation_sensitivity':reductions,'imu_constraint_diagnostics':imu,
            'paper_table2_reference':p['paper_table2_reference'],
            'paper_table2_differences_after_ik':{key:r['overall_mask_equal'][MODELS[1]][key]-p['paper_table2_reference'][key] for key,_,_ in METRICS},
            'strict_reproduction':False,'interpretation':'Paper-guided reconstructed orientation IK; unknown original settings remain explicit, no test tuning.'}
    (run/'report_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(12.4,3.8),constrained_layout=True)
    for ax,row,title in zip(axes,summary,['MPJRE','MPJPE','MPJVE (vertices)','Jitter']):
        bars=ax.bar([0,1],[row['raw'],row['ik']],color=['#66788a','#1b8d88'],width=.62)
        ax.set_xticks([0,1],['No IK','IMU-only IK']);ax.set_ylabel(row['unit']);ax.set_title(title)
        ax.set_ylim(0,max(row['raw'],row['ik'])*1.25);ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Final DIP model: fixed 10-step IMU-only IK, all24 masks')
    fig.savefig(run/'ik_four_metrics.png',dpi=180);fig.savefig(run/'ik_four_metrics.pdf');plt.close(fig)
    lines=['# IMU-only IK：最终 DIP 模型对照','','同一最终两轮权重、s09/s10、19条录制、24mask。', '',
           '| 指标 | 无 IK | IMU-only IK | 相对下降 | 下降组合数 |','|---|---:|---:|---:|---:|']
    for row in summary:lines.append(f'| {row["label"]} ({row["unit"]}) | {row["raw"]:.5f} | {row["ik"]:.5f} | {row["relative_decrease_percent"]:+.2f}% | {row["masks_lower"]}/24 |')
    lines.extend(['','正值表示下降，负值表示上升。Jitter只表示动作平滑性；MPJVE始终是网格顶点位置误差。','',
                  '10次Adam、lr1e-3、方向矩阵MSE；无GT约束或GT校准、不改权重、不进行测试调参。'
                  '头/髋索引、学习率和旋转参数化等缺失细节已明确冻结为重建约定；这不是已找回的作者逐行实现。','',
                  '![IK metrics](ik_four_metrics.png)',''])
    (run/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    render(parser.parse_args().run)
