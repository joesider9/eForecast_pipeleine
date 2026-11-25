# eForecast project author: George Sideratos and Claude
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Bounded synthetic fit for the opt-in managed Kubernetes qualification.

``smoke_managed_kubernetes.py --compute cpu|gpu`` copies this module beside the script its Job
runs, in the shared folder, and puts that folder on the path of the Job and of the worker the
Job starts: no image has to carry it. The entry point coordinates only; torch is imported
inside the worker.
"""
import json
import os
from pathlib import Path


def fit(payload, context):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(42)
    gpu = context.device_uuid
    device = 'cuda:0' if gpu else 'cpu'
    if gpu:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        actual = str(torch.cuda.get_device_properties(0).uuid)
        assert actual.removeprefix('GPU-') == gpu.removeprefix('GPU-'), (actual, gpu)
        torch.cuda.reset_peak_memory_stats()
    x = torch.randn(128, 8, dtype=torch.float64)
    y = x.sum(dim=1, keepdim=True) * .2
    initial = torch.nn.Linear(8, 1, dtype=torch.float64).state_dict()
    def train(where):
        model = torch.nn.Linear(8, 1, dtype=torch.float64).to(where)
        model.load_state_dict(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        a, b = x.to(where), y.to(where)
        for _ in range(100):
            optimizer.zero_grad()
            loss = (model(a) - b).square().mean()
            loss.backward()
            optimizer.step()
        return model, model(a).detach().cpu()
    reference_model, reference = train('cpu')
    model, prediction = train(device) if gpu else (reference_model, reference)
    torch.testing.assert_close(prediction, reference, rtol=1e-5, atol=1e-5)
    folder = Path(context.result_directory) / 'artifacts'
    folder.mkdir()
    checkpoint = folder / 'weights.pt'
    torch.save(model.state_dict(), checkpoint)
    restored = torch.nn.Linear(8, 1, dtype=torch.float64).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    torch.testing.assert_close(restored(x.to(device)).detach().cpu(), prediction, rtol=0, atol=0)
    peak = torch.cuda.max_memory_allocated() / 2**20 if gpu else 0
    assert not gpu or peak > 0
    return {'device_uuid': gpu, 'peak_gpu_allocated_mb': peak,
            'max_absolute_error': (prediction-reference).abs().max().item(),
            'loss': (prediction-y).square().mean().item(),
            'artifacts': ['artifacts/weights.pt'], 'epochs': 100,
            'pod_uid': os.environ['EFORECAST_POD_UID']}


def main():
    """Coordinate one fit in a worker of its own, the way a stage does, and keep its result.

    The worker is given one thread and, when the check asks for a card, the one card this Job
    sees. Nothing is reserved for it: it is a process, started, watched and collected by
    ``ComputeWorkers`` (eforecast/config/compute_workers.py).
    """
    from eforecast.config import model_compute, run_reporting
    use_gpu = os.environ.get('EFORECAST_QUALIFY_GPU') == '1'
    workers = model_compute.model_workers()
    cards = model_compute.cards_for('auto') if use_gpu else []
    if use_gpu:
        # A Job that asks for a card is shown exactly one, and the fit must run on that one.
        assert len(cards) == 1, cards
    card = cards[0] if cards else None
    run_id = os.environ['EFORECAST_RUN_ID']
    request = workers.request(run_id, run_id + '-qualification-fit', run_id + '-fit', 1, device=card)
    handle = workers.prepare(request, 'managed_compute_probe:fit', {},
                             execution={'device': 'auto' if card else 'cpu', 'threads': 1})
    try:
        result = workers.wait(handle, timeout=180)
    except BaseException:
        workers.cancel(handle, force=True)
        raise
    assert result['state'] == 'success', result
    # The supervisor hashed what the worker left once its whole process tree had exited, and
    # ``wait`` compared the file with that record before it called this a success.
    assert 'artifacts/weights.pt' in result['files'], result
    report = Path(os.environ['EFORECAST_WORKSPACE']).parent / (run_id + '-compute.json')
    report.write_text(json.dumps(result, indent=2))
    workers.discard_result(handle)
    run_reporting.result('success', 'fit in a worker of its own, CPU parity and checkpoint reload verified')


if __name__ == '__main__':
    main()
