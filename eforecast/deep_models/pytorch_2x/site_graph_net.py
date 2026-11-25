# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The cross-site graph network: one node per plant, attention over the plants with a distance prior.

Input ``X (b, N, F)`` (b timestamps, N sites, F features each: the site's own row features, i.e.
observation lags, calendar and NWP statistics), output ``(b, N, n_out)``.  A shared node encoder
maps every site's features to ``hidden``; a learned site embedding tells the sites apart; ``depth``
graph-attention layers let every site attend over the others with an additive prior
``-d^2 / (2 l^2)`` on the attention logits (``l`` learned per head from the median distance) and an
optional k-nearest-neighbour mask, each followed by a feed-forward block (pre-norm, residuals); a
shared linear head gives the steps of every site.  Site embeddings and the distance prior are what
a per-site model cannot have: the neighbours' current observations and forecasts.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from eforecast.compute_context import active_worker, resident_backend
from eforecast.execution_policy import torch_device


def _cancel():
    from eforecast.bounded_pools import cancel_check
    check = cancel_check.get()
    if check is not None:
        check()


class SiteAttention(nn.Module):
    def __init__(self, hidden, heads, distances, dropout=0.1, k=None, length_scale=None):
        super().__init__()
        self.heads = heads
        self.hidden = hidden
        self.scale = (hidden // heads) ** -0.5
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        d = torch.as_tensor(np.asarray(distances, dtype=np.float32))
        self.register_buffer('d2', d ** 2)
        n = d.shape[0]
        off = d[~torch.eye(n, dtype=torch.bool)]
        base = float(length_scale) if length_scale else (float(off.median()) if off.numel() else 1.0)
        self.log_scale = nn.Parameter(torch.full((heads,), float(np.log(max(base, 1e-3)))))
        mask = torch.ones(n, n, dtype=torch.bool)
        if k is not None and k < n - 1:
            order = torch.argsort(d, dim=1)
            mask = torch.zeros(n, n, dtype=torch.bool)
            for i in range(n):
                mask[i, order[i, :int(k) + 1]] = True
        self.register_buffer('mask', mask)
        self.last_attention = None

    def forward(self, x):                                       # (b, N, hidden)
        b, n, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        split = lambda t: t.view(b, n, self.heads, -1).transpose(1, 2)      # (b, heads, N, dh)
        q, k, v = split(q), split(k), split(v)
        logits = (q @ k.transpose(-1, -2)) * self.scale                     # (b, heads, N, N)
        prior = -self.d2.unsqueeze(0) / (2 * torch.exp(self.log_scale).view(-1, 1, 1) ** 2)
        logits = logits + prior.unsqueeze(0)
        logits = logits.masked_fill(~self.mask, float('-inf'))
        attn = torch.softmax(logits, dim=-1)
        self.last_attention = attn.detach().mean(dim=1)                      # (b, N, N) averaged over the heads
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, self.hidden)
        return self.out(out)


class GridEncoder(nn.Module):
    """A site's NWP grid (lags, vars, lat, lon) -> ``hidden`` features, shared by every site: the lags and
    variables as channels, two convolutions with GELU and pooling, an adaptive 4x4 pool, a linear layer."""

    def __init__(self, grid_shape, hidden, channels=32, dropout=0.1):
        super().__init__()
        L, V, H, W = grid_shape
        self.conv = nn.Sequential(nn.Conv2d(L * V, channels, 3, padding=1), nn.GELU(), nn.AvgPool2d(2) if min(H, W) >= 8 else nn.Identity(),
                                  nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(channels * 16, hidden))

    def forward(self, g):                                       # (b, N, L, V, H, W) -> (b, N, hidden)
        b, n, L, V, H, W = g.shape
        return self.conv(g.reshape(b * n, L * V, H, W)).reshape(b, n, -1)


class SiteGraphNet(nn.Module):
    def __init__(self, n_features, n_out, coords, hidden=64, depth=2, heads=4, dropout=0.1, k=None, length_scale=None,
                 grid_shape=None, nodes='row'):
        super().__init__()
        from eforecast.datasets.site_graph import haversine_km
        distances = haversine_km(coords)
        n_sites = distances.shape[0]
        hidden = int(np.ceil(hidden / heads) * heads)
        self.n_out = int(n_out)
        self.nodes = nodes if grid_shape is not None else 'row'   # row | nwp | both
        self.grid = GridEncoder(grid_shape, hidden, dropout=dropout) if grid_shape is not None and self.nodes in ('nwp', 'both') else None
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)) \
            if self.nodes in ('row', 'both') else None
        self.fuse = nn.Linear(2 * hidden, hidden) if self.nodes == 'both' else None
        self.site_embed = nn.Parameter(torch.zeros(1, n_sites, hidden))
        nn.init.trunc_normal_(self.site_embed, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(int(depth)):
            self.layers.append(nn.ModuleDict({
                'norm_a': nn.LayerNorm(hidden), 'attn': SiteAttention(hidden, heads, distances, dropout, k, length_scale),
                'norm_f': nn.LayerNorm(hidden),
                'ffn': nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden)),
            }))
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, self.n_out)
        self.output_shape = [10, n_sites, self.n_out]

    def forward(self, x, g=None):                               # x (b, N, F), g (b, N, L, V, H, W) -> (b, N, n_out)
        if self.nodes == 'row':
            h = self.encoder(x)
        elif self.nodes == 'nwp':
            h = self.grid(g)
        else:
            h = self.fuse(torch.cat([self.encoder(x), self.grid(g)], dim=-1))
        h = h + self.site_embed
        for layer in self.layers:
            h = h + layer['attn'](layer['norm_a'](h))
            h = h + layer['ffn'](layer['norm_f'](h))
        return self.head(self.norm(h))

    @property
    def last_attention(self):
        return self.layers[-1]['attn'].last_attention if self.layers else None


def train_site_graph(data, train_idx, val_idx, params=None, device='cpu', log=print):
    """Train ``SiteGraphNet`` on ``data`` (``SiteGraphData``) with early stopping on the validation MAE.

    ``params``: hidden (64), depth (2), heads (4), dropout (0.1), k (None), length_scale (None), epochs (100),
    batch_size (256), learning_rate (1e-3), patience (10), seed (1).  Returns (model, history frame)."""
    import pandas as pd
    p = {'hidden': 64, 'depth': 2, 'heads': 4, 'dropout': 0.1, 'k': None, 'length_scale': None, 'epochs': 100,
         'batch_size': 256, 'learning_rate': 1e-3, 'patience': 10, 'seed': 1, 'nodes': 'row', **(params or {})}
    device = torch_device(device)
    p['batch_size'] = max(1, int(p['batch_size']))
    if not len(train_idx) or not len(val_idx):
        raise ValueError('site graph needs nonempty frozen training and validation coverage')
    torch.manual_seed(int(p['seed']))
    np.random.seed(int(p['seed']))
    nodes = p.get('nodes', 'row') if data.G is not None else 'row'
    model = SiteGraphNet(data.X.shape[2], data.y.shape[2], data.coords, p['hidden'], p['depth'], p['heads'], p['dropout'],
                         p['k'], p['length_scale'], grid_shape=data.grid_shape if nodes != 'row' else None, nodes=nodes).to(device)
    # Dataset/stack residence is host RAM. Only the current batch enters VRAM.
    X = torch.as_tensor(data.X)
    y = torch.as_tensor(data.y)
    from eforecast.datasets.tensor_views import TensorView, batch as tensor_batch
    G = data.G if isinstance(data.G, TensorView) else torch.as_tensor(data.G) if (data.G is not None and nodes != 'row') else None
    if nodes == 'row': G = None
    def run(idx):
        if isinstance(G, TensorView):
            private = tensor_batch({'grid': G}, idx.cpu().numpy(),
                extra_bytes=len(idx) * (int(np.prod(data.X.shape[1:])) + int(np.prod(data.y.shape[1:]))) * 8)['grid']
            grid = torch.from_numpy(private).to(device)
        else: grid = None if G is None else G[idx].to(device)
        return model(X[idx].to(device), grid)
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(p['learning_rate']), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=max(2, p['patience'] // 3))
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(int(p['epochs'])):
        _cancel()
        model.train()
        perm = np.random.permutation(train_idx)
        losses = []
        for start in range(0, len(perm), int(p['batch_size'])):
            _cancel()
            idx = torch.as_tensor(perm[start:start + int(p['batch_size'])])
            optimiser.zero_grad()
            loss = torch.mean((run(idx) - y[idx].to(device)) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            predictions = []
            for s in range(0, len(val_idx), min(512, p['batch_size'])):
                _cancel()
                predictions.append(run(torch.as_tensor(val_idx[s:s + min(512, p['batch_size'])])).cpu())
            pred = torch.cat(predictions)
            truth = y[torch.as_tensor(val_idx)]
            mae_val = float(torch.mean(torch.abs(pred - truth)))
            mae_sites = torch.mean(torch.abs(pred - truth), dim=(0, 2)).numpy()
        scheduler.step(mae_val)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'mae_val': mae_val, **{f'mae_val_{n}': float(m) for n, m in zip(data.names, mae_sites)}})
        improved = mae_val < best - 1e-6
        if improved:
            best, wait = mae_val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        log(f"epoch {epoch}: loss {np.mean(losses):.5f} mae_val {mae_val:.5f}{' *' if improved else ''} | wait {wait}/{p['patience']}")
        if wait >= int(p['patience']):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def predict_site_graph(model, data, idx=None, device='cpu', batch_size=512):
    """{site: frame (dates x target columns)} of the model on ``data`` (all rows or ``idx``)."""
    import pandas as pd
    idx = np.arange(len(data.dates)) if idx is None else np.asarray(idx)
    device = torch_device(device)
    batch_size = max(1, min(512, int(batch_size)))
    if not len(idx):
        return {name: pd.DataFrame(index=data.dates[:0], columns=data.targets, dtype=np.float32) for name in data.names}
    X = torch.as_tensor(data.X)
    from eforecast.datasets.tensor_views import TensorView, batch as tensor_batch
    nodes = getattr(model, 'nodes', 'row')
    G = data.G if isinstance(data.G, TensorView) else torch.as_tensor(data.G) if (data.G is not None and nodes != 'row') else None
    if nodes == 'row': G = None
    model.eval()
    with torch.no_grad():
        predictions = []
        for s in range(0, len(idx), batch_size):
            _cancel()
            batch = torch.as_tensor(idx[s:s + batch_size])
            if isinstance(G, TensorView):
                private = tensor_batch({'grid': G}, batch.cpu().numpy(),
                    extra_bytes=len(batch) * int(np.prod(data.X.shape[1:])) * 8)['grid']
                grid = torch.from_numpy(private).to(device)
            else: grid = None if G is None else G[batch].to(device)
            predictions.append(model(X[batch].to(device), grid).cpu())
        pred = torch.cat(predictions).numpy()
    pred = np.clip(pred, 0, None)
    return {name: pd.DataFrame(pred[:, k], index=data.dates[idx], columns=data.targets) for k, name in enumerate(data.names)}


class SiteGraphModel:
    """A trained site graph on disk (``site_graph.pt`` + ``meta.json``) and its online prediction.

    ``meta.json`` (written by ``run_site_graph.py``) carries the sites, their case files, the feature and
    target names, the read options and the network parameters, so the model can rebuild the online
    stack of every site for new dates without the training script."""

    def __init__(self, folder):
        import json
        from eforecast.common_utils.graph_bundle import bundle_root
        from eforecast.common_utils import artifact
        self.folder = str(bundle_root(folder))
        verdict, reason = artifact.inspect(self.folder)
        if verdict == 'rejected':
            raise ValueError(f'unusable site graph: {reason}')
        with open(os.path.join(self.folder, 'meta.json'), encoding='utf-8') as stream:
            self.meta = json.load(stream)
        self.names = list(self.meta['sites'])
        self.features = list(self.meta['features'])
        self.targets = list(self.meta['targets'])
        self.coords = np.asarray(self.meta['coords'], dtype=float)
        p = self.meta['params']
        self.nodes = self.meta.get('nodes', 'row')
        grid_shape = tuple(self.meta['grid_shape']) if self.meta.get('grid_shape') else None
        self._net = None

    @property
    def net(self):
        if self._net is None:
            if active_worker.get() is None and resident_backend('site_graph') is not None:
                raise RuntimeError('site-graph weights load inside the admitted prediction worker')
            from eforecast.common_utils import artifact
            verdict, reason = artifact.inspect(self.folder)
            if verdict == 'rejected':
                raise ValueError(f'unusable site graph: {reason}')
            if verdict == 'verified':
                valid, reason = artifact.verify_digests(self.folder)
                if not valid:
                    raise ValueError(f'site-graph payload verification failed: {reason}')
            p = self.meta['params']
            grid_shape = tuple(self.meta['grid_shape']) if self.meta.get('grid_shape') else None
            self._net = SiteGraphNet(len(self.features), len(self.targets), self.coords, p['hidden'], p['depth'], p['heads'],
                                    p['dropout'], p.get('k'), p.get('length_scale'), grid_shape=grid_shape, nodes=self.nodes)
            self._net.load_state_dict(torch.load(os.path.join(self.folder, 'site_graph.pt'), map_location='cpu', weights_only=True))
            self._net.eval()
        return self._net

    def read_options(self):
        r = self.meta.get('read', {})
        options = dict(data_tag=self.meta.get('data_tag', 'row_all'), merge=r.get('merge', 'all'), compress=r.get('compress', 'dense'),
                       scale_row_method=r.get('scale_row_method', 'minmax'), scale_nwp_method=r.get('scale_nwp_method', 'minmax'))
        if self.nodes != 'row':
            options.update(nwp=True, nwp_pool=int(self.meta.get('nwp_pool', 1)), nwp_lag_step=int(self.meta.get('nwp_lag_step', 1)))
        return options

    def predict_stack(self, data, idx=None, device='cpu'):
        adapter = resident_backend('site_graph')
        if active_worker.get() is None and adapter is not None:
            return adapter('stack', folder=self.folder, data=data, idx=idx, device=str(device))
        device = torch_device('cpu' if active_worker.get() is None else device)
        self.net.to(device)
        return predict_site_graph(self.net, data, idx, device, self.meta['params'].get('batch_size', 512))

    def predict_online(self, dates, statics=None, build_datasets=True, device='cpu', log=print):
        """{site: frame in the site's target scale} for ``dates`` (a DatetimeIndex of days or timestamps).

        Every site's online datasets are (re)built for the days spanned by ``dates`` (``build_datasets``),
        read through the site's own feeder, aligned on the common timestamps and pushed through the net.
        ``statics``: {site: static_data} to reuse initialised cases (else the case files of ``meta``)."""
        adapter = resident_backend('site_graph')
        if active_worker.get() is None and adapter is not None:
            return adapter('online', folder=self.folder, dates=dates, statics=statics,
                           build_datasets=build_datasets, device=str(device), inverse=False)
        import pandas as pd
        from eforecast.init.initialize import initializer
        from eforecast.datasets.online import create_online_datasets
        from eforecast.datasets.site_graph import assemble, site_frames
        dates = pd.DatetimeIndex(dates)
        if not len(dates):
            return {name: pd.DataFrame(columns=self.targets, dtype=np.float32) for name in self.names}
        configs = self.meta.get('configs') or {}
        from eforecast.compute_context import inference_sites
        statics = {**(inference_sites.get() or {}), **(statics or {})}
        for name in self.names:
            if name not in statics:
                if name not in configs:
                    raise ValueError(f'no case file for site {name} in {self.folder}/meta.json; pass statics')
                statics[name] = initializer(configs[name])
        days = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq='D')
        sites = {}
        for name in self.names:
            sd = statics[name]
            if build_datasets:
                log(f'{name}: building the online datasets for {days[0].date()} -> {days[-1].date()}')
                try:
                    create_online_datasets(sd, dates)
                except Exception as error:  # noqa: BLE001 - a site whose feed is down keeps its last online rows
                    if active_worker.get() is not None:
                        raise  # an admitted attempt must not publish stale/partial rebuilt inputs
                    log(f'{name}: online datasets not rebuilt ({str(error)[:160]}); using the existing online rows')
            X, _, coord, *grid = site_frames(sd, online=True, **self.read_options())
            sites[name] = (X, None, coord, sd.get('rated'), *grid)
            log(f'{name}: {X.shape[0]} online rows {X.index.min()} -> {X.index.max()}' + (f', grid {grid[0].shape[1:]}' if grid else ''))
        from eforecast.inference_inputs import record_online_inputs
        record_online_inputs({name: statics[name] for name in self.names}, 'site_graph')
        data = assemble(sites, features=self.features, targets=self.targets)
        if (dates != dates.normalize()).any():        # timestamps, including an initial midnight
            idx = data.slice(dates)
        else:
            idx = np.flatnonzero(data.dates.normalize().isin(dates.normalize()))
        return self.predict_stack(data, idx, device)


def predict_online_site_graph(folder, dates, statics=None, build_datasets=True, device='cpu', inverse=True, log=print):
    """Online forecast of every site of a trained site graph; ``inverse`` returns power units through each
    site's target scaler (else the scaled target space).  Returns {site: frame}."""
    adapter = resident_backend('site_graph')
    if active_worker.get() is None and adapter is not None:
        return adapter('online', folder=str(folder), dates=dates, statics=statics,
                       build_datasets=build_datasets, device=str(device), inverse=inverse)
    model = SiteGraphModel(folder)
    from eforecast.init.initialize import initializer
    statics = dict(statics or {})
    for name in model.names:
        if name not in statics:
            statics[name] = initializer(model.meta['configs'][name])
    frames = model.predict_online(dates, statics, build_datasets, device, log)
    if inverse:
        from eforecast.combine_predictions.ensemble import Targets
        frames = {name: Targets(statics[name], online=True).inverse(frame) for name, frame in frames.items()}
    return frames
