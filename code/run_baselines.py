'Run the Euler experiments: python code/run_baselines.py --out results/classical.'
from pathlib import Path
import argparse
import csv
import numpy as np
import burgers_reference as br
import metrics
from common import environment, median_seconds, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='results/classical')
    args=parser.parse_args(); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    nu,T=br.NU_BENCHMARK,1.; rows=[]; previous={}

    # Reference: explicit Euler on 8192 points. The 4096-point and half-step runs are
    # used only for the reference sensitivity checks saved in metrics.json.
    _, reference_8192=br.burgers_fd_explicit(8192,nu,T)
    _, reference_4096=br.burgers_fd_explicit(4096,nu,T)
    _, reference_4096_half_step=br.burgers_fd_explicit(4096,nu,T,cfl=.4)
    # Question 11(a): errors e_n against the reference restricted to the same n points.
    for n in (128,256,512,1024,2048):
        x,explicit=br.burgers_fd_explicit(n,nu,T)
        _,implicit=br.burgers_fd_implicit(n,nu,T)
        truth=reference_8192[::8192//n]
        for label,pred in [('FD explicit Euler',explicit),
                           ('FD implicit-diffusion Euler',implicit)]:
            error=metrics.rel_l2(pred,truth)
            row={'n':n,'method':label,'relative_l2':error,
                 'observed_order':None,
                 'mass_drift':metrics.mass_drift(pred,br.initial_condition(x),2/n)}
            previous[label]=error;rows.append(row); print(row,flush=True)

    # Observed rates p_n = log2(e_n / e_2n).
    for row in rows:
        finer = next((r for r in rows if r['method'] == row['method'] and r['n'] == 2*row['n']), None)
        if finer is not None:
            row['observed_order'] = float(np.log2(row['relative_l2']/finer['relative_l2']))
    # Timestep sensitivity at n=512 (halving the 0.8 safety factor).
    _,e1=br.burgers_fd_explicit(512,nu,T,cfl=.8)
    _,e2=br.burgers_fd_explicit(512,nu,T,cfl=.4)
    _,i1=br.burgers_fd_implicit(512,nu,T,cfl=.8)
    _,i2=br.burgers_fd_implicit(512,nu,T,cfl=.4)
    reference_checks={'reference_grid_n':8192,
                      'reference_grid_refinement_relative_l2_n4096_to_n8192':metrics.rel_l2(reference_4096,reference_8192[::2]),
                      'reference_fixed_grid_step_refinement_relative_l2_n4096':metrics.rel_l2(reference_4096,reference_4096_half_step),
                      'explicit_euler_step_refinement_relative_l2_n512':metrics.rel_l2(e1,e2),
                      'implicit_euler_step_refinement_relative_l2_n512':metrics.rel_l2(i1,i2),
                      'note':'These refinement differences measure sensitivity, not certified continuum error bounds.'}
    # Profiles for the zoomed front figure, interpolated to a common 1024-point plotting grid.
    common_x,_=br.grid(1024); common_true=reference_8192[::8]
    common_initial=br.initial_condition(common_x); synthesis=[]; timings={}
    for n in (128,512):
        _,values=br.burgers_fd_explicit(n,nu,T)
        pred=br.periodic_linear_resample(values,len(common_x))
        synthesis.append(metrics.evaluate(pred,common_true,common_initial,common_x,f'FD explicit Euler {n}, linear interpolation'))
        timings[f'FD explicit Euler {n}']=median_seconds(lambda n=n: br.burgers_fd_explicit(n,nu,T),repeats=5)
        np.savez_compressed(out/f'fd_explicit_{n}_prediction.npz',x=common_x,pred=pred,true=common_true,initial=common_initial)
    _,values=br.burgers_fd_implicit(512,nu,T)
    pred=br.periodic_linear_resample(values,len(common_x))
    synthesis.append(metrics.evaluate(pred,common_true,common_initial,common_x,'FD implicit-diffusion Euler 512, linear interpolation'))
    timings['FD implicit-diffusion Euler 512']=median_seconds(lambda:br.burgers_fd_implicit(512,nu,T),repeats=5)
    np.savez_compressed(out/'fd_implicit_512_prediction.npz',x=common_x,pred=pred,true=common_true,initial=common_initial)
    # Solution snapshots at t = 0, 0.3, 1.
    snapshot_times=(0.,.3,1.)
    reference_snapshots=[]; ex128_snapshots=[]; ex512_snapshots=[]; im512_snapshots=[]
    for snapshot_time in snapshot_times:
        if snapshot_time == T:
            reference_snapshots.append(common_true.copy())
        else:
            reference_snapshots.append(br.burgers_fd_reference(common_x,nu,snapshot_time,n_ref=8192))
        for n, snapshots in ((128,ex128_snapshots),(512,ex512_snapshots)):
            _,state=br.burgers_fd_explicit(n,nu,snapshot_time)
            snapshots.append(br.periodic_linear_resample(state,len(common_x)))
        _,state=br.burgers_fd_implicit(512,nu,snapshot_time)
        im512_snapshots.append(br.periodic_linear_resample(state,len(common_x)))
    np.savez_compressed(out/'euler_trajectory_snapshots.npz', x=common_x,
                        times=np.array(snapshot_times), reference=np.stack(reference_snapshots),
                        fd_explicit_128=np.stack(ex128_snapshots),
                        fd_explicit_512=np.stack(ex512_snapshots),
                        fd_implicit_512=np.stack(im512_snapshots))
    write_json(out/'metrics.json',{'environment':environment(),'convergence':rows,
               'reference_checks':reference_checks,
               'synthesis_rows':synthesis,'timings':timings})
    with (out/'convergence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print('wrote',out)


if __name__=='__main__': main()
