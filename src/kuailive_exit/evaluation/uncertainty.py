"""Paired user-cluster bootstrap, conditional on fixed trained models."""
import numpy as np
import pandas as pd


def paired_user_bootstrap(users, y, candidate, baseline, *, seed=20261001, resamples=2000):
    y=np.asarray(y); c=np.clip(candidate,1e-7,1-1e-7); b=np.clip(baseline,1e-7,1-1e-7)
    delta_ll=-y*np.log(c)-(1-y)*np.log1p(-c)+y*np.log(b)+(1-y)*np.log1p(-b)
    delta_brier=(c-y)**2-(b-y)**2
    grouped=pd.DataFrame({'user':users,'n':1,'ll':delta_ll,'brier':delta_brier}).groupby('user',sort=True).sum()
    n=grouped.n.to_numpy(); rng=np.random.default_rng(seed)
    samples=[]
    for start in range(0,resamples,100):
        ix=rng.integers(0,len(n),size=(min(100,resamples-start),len(n)))
        den=n[ix].sum(axis=1)
        samples.append(np.column_stack([grouped[k].to_numpy()[ix].sum(axis=1)/den for k in ['ll','brier']]))
    draws=np.vstack(samples)
    return {'users':len(grouped),'rows':len(y),'resamples':resamples,'seed':seed,
        'unit':'user; point-weighted metric within each resample',
        'scope':'conditional on fixed models; excludes training and room-level dependence',
        **{key:{'difference':float(grouped[k].sum()/n.sum()),
                 'ci95_lower':float(np.quantile(draws[:,j],.025)),
                 'ci95_upper':float(np.quantile(draws[:,j],.975))}
            for j,(key,k) in enumerate([('log_loss','ll'),('brier','brier')])}}
