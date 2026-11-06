"""Is top-down losing because the idea is wrong, or because my allocation was weak?
Compares a static hour-of-week share, a rolling share kept current from the last four
same weekday-hour occurrences, and the same rolling share fed the TRUE group total."""
import numpy as np, pandas as pd, time, sys
from core import Panel, FeatureBuilder, make_groups, predict_strategy, topdown_predict, score
import run_experiment as R
import paths

p = Panel.from_dir(paths.PANEL, name='smartds')
sel = np.sort(np.random.default_rng(3).choice(p.N, 400, replace=False))
p = p.subset(sel, name='smartds')
tr, va, te, scale = R.prep(p, 'winter')
groups, glab = make_groups('shape_kmeans', p, tr, scale, k=8)
fb = FeatureBuilder(p, scale, seed=11)
rows = []
def run(tag, **kw):
    t0 = time.time()
    pr, tt, rr, cc = topdown_predict(fb, p, groups, tr, va, te, scale, 300, **kw)
    m, _ = score(pr, tt, rr, cc, p, scale, tag=tag)
    m.update(strategy=tag, secs=round(time.time()-t0, 1))
    rows.append(m); print(f"{tag:26s} nMAE {100*m['nmae_med']:.1f}%  peakAPE {100*m['peak_ape_med']:.1f}%  "
                         f"bias {100*m['peak_bias_med']:+.1f}%  ({m['secs']}s)", flush=True)
pr, tt, rr, cc = predict_strategy('local', fb, tr, va, te, rounds=300)
m,_ = score(pr, tt, rr, cc, p, scale, tag='local'); m.update(strategy='local')
rows.append(m); print(f"{'local (reference)':26s} nMAE {100*m['nmae_med']:.1f}%  peakAPE {100*m['peak_ape_med']:.1f}%", flush=True)
run('topdown_static_how')
run('topdown_rolling4', share_mode='rolling')
run('topdown_rolling4_oracle', share_mode='rolling', oracle=True)
run('topdown_static_oracle', oracle=True)
pd.DataFrame(rows).to_json(f'{paths.RESULTS}/topdown_probe.jsonl', orient='records', lines=True)
