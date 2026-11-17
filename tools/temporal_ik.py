"""Separate causal rolling-window IK experiment with joint-jerk regularization.

For output t, optimize only prediction/IMU frames[max(0,t-W+1),t]. The fixed
mean-body skeleton supplies geometry, not observed positions. No GT, future
frame, optimizer state from another window, or post-hoc filter is used.
"""
from dataclasses import dataclass
import math as scalar_math
import torch
from imuposer import math
from imu_only_ik import (PARENTS,SENSOR_JOINTS,OPTIMIZED_BY_SENSOR,sensor_ids,
                         exp_map,global_rotations,project_orientations,_rotations)


@dataclass(frozen=True)
class TemporalIKSettings:
    iterations:int=10
    learning_rate:float=1e-3
    window:int=5
    jerk_weight:float=.01
    jerk_scale_m:float=.01
    frame_batch:int=128

    def __post_init__(self):
        for key in ['iterations','window','frame_batch']:
            x=getattr(self,key)
            if isinstance(x,bool) or not isinstance(x,int) or x<(0 if key=='iterations' else 1):raise ValueError('Invalid '+key)
        for key in ['learning_rate','jerk_scale_m','jerk_weight']:
            x=getattr(self,key)
            if isinstance(x,bool) or not scalar_math.isfinite(x) or x<(0 if key=='jerk_weight' else 1e-30):raise ValueError('Invalid '+key)
        if self.jerk_weight>0 and self.window<4:raise ValueError('Jerk window needs at least4 frames')


def world_kinematics(local,offsets):
    g=[local[:,0]];p=[offsets[0].expand(len(local),3)]
    for j in range(1,24):
        parent=PARENTS[j];g.append(g[parent]@local[:,j])
        p.append(p[parent]+(g[parent]@offsets[j,:,None]).squeeze(-1))
    return torch.stack(g,1),torch.stack(p,1)


def discrete_jerk_loss(position,valid,scale):
    if position.shape[1]<4:return position.sum((1,2,3))*0
    d=position[:,3:]-3*position[:,2:-1]+3*position[:,1:-2]-position[:,:-3]
    stencil=valid[:,3:]&valid[:,2:-1]&valid[:,1:-2]&valid[:,:-3]
    return ((d/scale).square().mean((2,3))*stencil).sum(1)


@torch.inference_mode(False)
def refine_temporal(predicted_pose,imu_orientations,sensors,offsets,settings=TemporalIKSettings()):
    ids=sensor_ids(sensors)
    pose=torch.as_tensor(predicted_pose,dtype=torch.float32,device='cpu').detach().clone()
    if pose.ndim!=2 or pose.shape[1]!=144 or not len(pose) or not torch.isfinite(pose).all():raise ValueError('Invalid predicted pose')
    base=_rotations(pose.double()).detach()
    ori=torch.as_tensor(imu_orientations,dtype=torch.float64,device='cpu').detach()
    offsets=torch.as_tensor(offsets,dtype=torch.float64,device='cpu').detach()
    if ori.shape!=(len(pose),5,3,3) or offsets.shape!=(24,3) or not torch.isfinite(offsets).all():raise ValueError('Invalid geometry or orientations')
    if not ids:return {'pose':pose,'summary':{'frames':len(pose),'sensors':[],'iterations':0}}
    target=project_orientations(ori[:,ids]);sensed=tuple(SENSOR_JOINTS[s] for s in ids)
    joints=tuple(sorted({j for s in ids for j in OPTIMIZED_BY_SENSOR[s]}))
    result=pose.reshape(-1,24,6).clone();histories=[];emitted_before=[];emitted_after=[]
    for first in range(0,len(pose),settings.frame_batch):
        stop=min(len(pose),first+settings.frame_batch);times=torch.arange(first,stop)
        indices=times[:,None]-torch.arange(settings.window-1,-1,-1)[None,:]
        valid=indices>=0;indices=indices.clamp_min(0)
        initial=base[indices];goal=target[indices];batch,window=indices.shape
        delta=torch.zeros(batch,window,len(joints),3,dtype=torch.float64,requires_grad=True)
        optimizer=torch.optim.Adam([delta],lr=settings.learning_rate,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)
        def objective():
            local=initial.clone();local[:,:,joints]=initial[:,:,joints]@exp_map(delta)
            flat=local.reshape(-1,24,3,3)
            if settings.jerk_weight:
                g,position=world_kinematics(flat,offsets)
                temporal=discrete_jerk_loss(position.reshape(batch,window,24,3),valid,settings.jerk_scale_m)
            else:
                g=global_rotations(flat);temporal=torch.zeros(batch,dtype=torch.float64)
            residual=(g.reshape(batch,window,24,3,3)[:,:,sensed]-goal).square().mean((2,3,4))
            data=(residual*valid).sum(1)
            return data+settings.jerk_weight*temporal,local,residual[:,-1],data,temporal
        history=[]
        with torch.enable_grad():
            for step in range(settings.iterations):
                loss,local,last,data,temporal=objective()
                if step==0:emitted_before.append(last.detach())
                history.append(torch.stack((loss,data,temporal),-1).detach())
                optimizer.zero_grad(set_to_none=True);loss.sum().backward()
                if delta.grad is None or not torch.isfinite(delta.grad).all():raise ValueError('Invalid temporal gradient')
                optimizer.step()
        with torch.no_grad():
            loss,local,last,data,temporal=objective()
            if settings.iterations==0:emitted_before.append(last.detach())
            emitted_after.append(last)
            history.append(torch.stack((loss,data,temporal),-1))
            histories.append(torch.stack(history,1))
            if settings.iterations:
                result[first:stop,joints]=math.rotation_matrix_to_r6d(local[:,-1,joints]).reshape(batch,len(joints),6).float()
    output=result.reshape(-1,144).detach();_rotations(output.double())
    fixed=[j for j in range(24) if j not in joints]
    if not torch.equal(result[:,fixed],pose.reshape(-1,24,6)[:,fixed]):raise AssertionError('Frozen local entries changed')
    histories=torch.cat(histories)
    return {'pose':output,'loss_history':histories,
            'summary':{'frames':len(pose),'sensors':list(ids),'optimized_joints':list(joints),'iterations':settings.iterations,
                       'window':settings.window,'jerk_weight':settings.jerk_weight,'jerk_scale_m':settings.jerk_scale_m,
                       'orientation_mse_before':torch.cat(emitted_before).mean().item(),
                       'orientation_mse_after':torch.cat(emitted_after).mean().item(),
                       'objective_before':histories[:,0,0].mean().item(),'objective_after':histories[:,-1,0].mean().item(),
                       'inputs':'past/current predictions and available orientations only; fixed mean-body skeleton',
                       'future_frames_added':0,'state_carry':False,'unoptimized_pose6d_max_abs_change':0.}}
