"""Bounded single-process CAtNIPP PPO training/evaluation, CPU or CUDA."""
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('--config',default=None)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--nodes',type=int,default=200)
    p.add_argument('--budget-m',type=float,default=180.)
    p.add_argument('--spacing-m',type=float,default=.6)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--output',default='runs/smartmagdl')
    p.add_argument('--ppo-epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--plot',action='store_true')
    p.add_argument('--trajectory-samples',type=int,default=1)
    p.add_argument('--plan-steps',type=int,default=15)
    p.add_argument('--reconstructor', choices=['gp', 'smartmagdl'], default=None)
    p.add_argument('--smartmag-checkpoint')
    p.add_argument('--smartmag-scaling')
    p.add_argument('--smartmag-device', default=None)
    p.add_argument('--plot-grid-size', type=int, default=None)
    p.add_argument('--no-gif', action='store_true')
    p.add_argument('--gif-every', type=int, default=1)
    p.add_argument('--gif-fps', type=float, default=5)
    a=p.parse_args()
    if a.gif_every < 1 or a.gif_fps <= 0:
        p.error('gif-every and gif-fps must be positive')
    if a.trajectory_samples<1 or a.plan_steps<1:
        p.error('trajectory-samples and plan-steps must be positive')
    if a.episodes<1 or a.nodes<32 or a.ppo_epochs<1 or a.batch_size<1 or a.spacing_m<0:
        p.error('Require episodes/epochs/batch-size >=1, nodes >=32, spacing-m >=0')
    if a.mode=='evaluate' and not a.checkpoint:
        p.error('Evaluation requires --checkpoint; an untrained policy is not an evaluation result')
    if a.config:
        os.environ['SMARTMAG_CONFIG']=str(Path(a.config).resolve())
    from magnetic_simulator import MagneticConfig
    from attention_net import AttentionNet
    from worker import Worker
    from env import Env
    from parameters import INPUT_DIM,EMBEDDING_DIM,K_SIZE,LR,GAMMA
    config=MagneticConfig.from_environment()
    for name in ('reconstructor', 'smartmag_checkpoint', 'smartmag_scaling', 'smartmag_device', 'plot_grid_size'):
        value = getattr(a, name)
        if value is not None:
            if name in ('smartmag_checkpoint', 'smartmag_scaling'):
                value = str(Path(value).resolve())
            setattr(config, name, value)
    if a.smartmag_device is None:
        config.smartmag_device = a.device
    config.__post_init__()
    if config.reconstructor == 'smartmagdl':
        from smartmag_backend import SmartMagDLForIPP
        # Validate weights/scaling before constructing or overwriting run outputs.
        SmartMagDLForIPP(np.zeros((1, 2)), config)
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    model=AttentionNet(INPUT_DIM,EMBEDDING_DIM).to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    if a.checkpoint:
        checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        if a.mode=='train' and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    from dataclasses import asdict
    (out/'run_config.json').write_text(json.dumps({'arguments':vars(a),'magnetic':asdict(config)},indent=2))
    budget=a.budget_m/config.domain_m
    for episode in range(a.episodes):
        seed=a.seed+episode
        model.eval()
        env = Env(sample_size=a.nodes, k_size=K_SIZE, start=(0, 0),
                  destination=(1, 1), budget_range=(budget, budget),
                  seed=seed, magnetic_config=config)
        if a.mode == 'evaluate' and not a.no_gif:
            env.enable_capture(out / f'episode_{episode:05d}_frames', episode, a.gif_every)
        if a.mode == 'evaluate' and a.trajectory_samples > 1:
            import sys
            sys.path.insert(0,str(Path(__file__).resolve().parent/'eval'))
            import test_worker as tw
            tw.SAMPLE_SIZE=a.nodes
            tw.NUM_SAMPLE_TEST=a.trajectory_samples
            tw.PLAN_STEP=a.plan_steps
            tw.TRAJECTORY_SAMPLING=True
            w=tw.WorkerTest(0,model,episode,(budget,budget),
                sample_length=a.spacing_m/config.domain_m,device=a.device,seed=seed,env=env)
            metrics=w.work(episode,0)
        else:
            w = Worker(0, model, episode, (budget, budget), sample_size=a.nodes,
                       sample_length=a.spacing_m/config.domain_m, device=a.device,
                       greedy=a.mode=='evaluate', save_image=False, env=env)
            metrics=w.run_episode(episode)
        metrics['episode_return'] = w.env.episode_return
        metrics['reconstructor'] = config.reconstructor
        metrics['uncertainty_source'] = 'gp'
        metrics['R2']=metrics.pop('F1Score')
        metrics['RMSE_uT']=metrics['RMSE']*config.scale_uT
        metrics['distance_m']=(w.env.budget0-w.env.budget)*config.domain_m
        metrics['remaining_budget_m'] = w.env.budget * config.domain_m
        metrics['roadmap_attempt'] = w.env.roadmap_attempt
        metrics['episode']=episode; metrics['scene_seed']=seed
        if a.mode=='train':
            model.train()
            b=[torch.stack(v).to(a.device) for v in w.experience]
            # Same TD advantage, clipped PPO loss and return targets as CAtNIPP.
            advantage=(b[5]+GAMMA*b[6]-b[4]).detach()
            with torch.no_grad():
                old,_,_,_=model(b[0],b[1],b[8],b[2],b[9],b[10],b[12],b[11])
                old=old.gather(1,b[3].squeeze(1)).unsqueeze(1)
            for _ in range(a.ppo_epochs):
                for idx in torch.randperm(len(b[0]),device=a.device).split(a.batch_size):
                    lp,v,_,_=model(b[0][idx],b[1][idx],b[8][idx],b[2][idx],b[9][idx],b[10][idx],b[12][idx],b[11][idx])
                    lp=lp.gather(1,b[3][idx].squeeze(1)).unsqueeze(1)
                    ratio=(lp-old[idx]).exp()
                    policy=-torch.minimum(ratio*advantage[idx],ratio.clamp(.8,1.2)*advantage[idx]).mean()
                    value=(v-b[7][idx]).square().mean()
                    loss=policy+.5*value
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite PPO loss')
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                    optimizer.step()
            metrics['loss']=float(loss.detach())
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'episode':episode+1,'magnetic_config':asdict(config)},out/'checkpoint.pth')
        route=[1]+w.env.route
        np.savez_compressed(out/f'episode_{episode:05d}.npz',
            route_xy_m=(w.env.node_coords[route]-.5)*config.domain_m,
            observations_xy_heading_magnitude_uT=np.asarray(w.env.underlying_distribution.observation_log).reshape(-1,4),
            ground_truth_uT=w.env.ground_truth*config.scale_uT+config.center_uT,
            prediction_uT=w.env.gp_ipp.predict(w.env.gp_ipp.grid())*config.scale_uT+config.center_uT)
        if a.plot or a.mode == 'evaluate':
            w.env.plot(route,episode,len(route),str(out))
        if a.mode == 'evaluate' and not a.no_gif:
            from visualization import save_gif
            metrics['gif'] = save_gif(w.env.frame_files, out / f'episode_{episode:05d}.gif', a.gif_fps)
        record={k:(v.item() if isinstance(v,np.generic) else v) for k,v in metrics.items()}
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record))

if __name__=='__main__':
    main()
