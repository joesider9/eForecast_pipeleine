# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A14-03: what a colleague hands a maintainer when a run goes wrong.

One report, assembled from what the installation already knows: the durable run record, the
A13 doctor, and a bounded tail of the run's own log. Nothing is recomputed and nothing new is
measured -- a second diagnostic system would be a second thing to be wrong.

What it deliberately does **not** contain: datasets, model weights, any other case, and any
credential. The first three are never read; the fourth is removed by
`secrets_source.redact`, which knows the values this installation stores and the shapes a
credential has even when the value is unknown here.

The report is case-scoped: the route that serves it resolves the case through the same
dependency as every other case route, so a colleague can only ask about their own runs.
"""
from __future__ import annotations

from typing import Any

#: Bounds. A support report is read by a person and pasted into a message, so every list in it
#: is short enough to read and the log tail is short enough to send.
LOG_BYTES = 16 * 1024
FINDINGS = 12
STAGES = 24
WARNINGS = 8

#: Fields of a run record that name where the work ran. A support report says which worker and
#: which image; it never says how to reach them.
EXECUTION = ('executor_adapter', 'image', 'image_pinned', 'allocation_id', 'fence', 'device',
             'device_uuid', 'host', 'namespace', 'name', 'pid', 'job_uid', 'pod_uid')

ERRORS = ('report_error', 'preparation_error', 'profile_error', 'error')


def _redact(value):
    from eforecast.config import secrets_source
    if isinstance(value, str):
        return secrets_source.redact(value)
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _stage_summary(run):
    """The stage a maintainer needs to look at, and enough of the rest to place it."""
    stages = run.get('stages') or []
    counted: dict[str, int] = {}
    failed = []
    # Every stage is counted -- a total of 60 beside counts that add up to 24 would be a
    # puzzle for the reader. Only the detail below is bounded.
    for stage in stages:
        state = str(stage.get('state') or 'unknown')
        counted[state] = counted.get(state, 0) + 1
    for stage in stages[:STAGES]:
        outcome = stage.get('outcome') or {}
        if outcome.get('kind') in {'failure', 'partial'} or stage.get('blocked'):
            failed.append({'stage': stage.get('stage'), 'scope': stage.get('scope'),
                           'state': str(stage.get('state') or 'unknown'),
                           'outcome': outcome.get('kind'),
                           'detail': _redact(outcome.get('detail')),
                           'blocked': [_redact(item.get('reason'))
                                       for item in (stage.get('blocked') or [])][:WARNINGS]})
    return {'counts': counted, 'total': len(stages), 'examined': min(len(stages), STAGES),
            'attention': failed[:WARNINGS]}


def assemble(*, case, run, revision=None, doctor=None, log=None, links=None):
    """The report, from pieces the caller has already read.

    Separated from the gathering so every rule here -- what is included, what is bounded, what
    is redacted -- can be exercised without a live run manager, a store or a doctor.
    """
    accepted = run.get('accepted_case_revision')
    execution = {key: run.get(key) for key in EXECUTION if run.get(key) is not None}
    dispatch = run.get('dispatch') or None
    errors = {key: _redact(run.get(key)) for key in ERRORS if run.get(key)}
    outcome = run.get('outcome') or {}
    report: dict[str, Any] = {
        'version': 1,
        'case': {'name': case, 'revision': revision, 'accepted_revision': accepted,
                 # The question behind "it worked yesterday": is this the case that ran?
                 'changed_since_the_run': bool(revision and accepted and revision != accepted)},
        'run': {'id': run.get('id'), 'script': run.get('script'), 'state': run.get('state'),
                'status': run.get('status'), 'accepted': run.get('accepted'),
                'finished': run.get('finished'), 'returncode': run.get('returncode'),
                'outcome': {'kind': outcome.get('kind'), 'detail': _redact(outcome.get('detail'))}
                if outcome else None,
                'cleanup': run.get('cleanup'), 'stop_requested': run.get('stop_requested'),
                'errors': errors},
        'stages': _stage_summary(run),
        'execution': execution,
        'dispatch': {'status': dispatch.get('status'), 'reason': _redact(dispatch.get('reason')),
                     'dispatch_id': dispatch.get('dispatch_id'), 'route': dispatch.get('route')}
        if isinstance(dispatch, dict) else None,
        'installation': {'verdict': (doctor or {}).get('verdict'),
                         'counts': (doctor or {}).get('counts'),
                         'release': (doctor or {}).get('release'),
                         'attention': [{'check': item.get('check'),
                                        'severity': item.get('severity'),
                                        'finding': _redact(item.get('finding')),
                                        'remedy': _redact(item.get('remedy'))}
                                       for item in ((doctor or {}).get('findings') or [])
                                       if item.get('severity') != 'ok'][:FINDINGS]}
        if doctor is not None else {'verdict': None, 'note': 'the doctor was not run'},
        # And what it looked like when this run was accepted, which is the comparison a
        # maintainer actually wants: "it is unhealthy now" and "it was unhealthy before this
        # run started" lead to different places. Stamped at acceptance from the cached
        # reading, so an absent or stale one says so.
        'when_accepted': _redact(run.get('health')) or {'verdict': None,
                                                        'note': 'no reading was stamped'},
        'links': dict(links or {}),
        'log': _log_section(log),
        'excludes': ('datasets, model weights, other cases and other accounts are never read '
                     'for this report; credentials are removed from everything that is'),
    }
    return report


def _log_section(log):
    if not log:
        return {'text': '', 'note': 'no log was available for this run'}
    text = log.get('text') or ''
    size = int(log.get('size') or len(text.encode()))
    tail = text[-LOG_BYTES:]
    return {'text': _redact(tail), 'bytes': len(tail.encode()), 'total_bytes': size,
            'truncated': bool(log.get('truncated') or len(tail) < len(text)),
            'note': (f'the last {len(tail.encode())} bytes of {size}; download the full log from '
                     'the run for more' if size > len(tail.encode()) else 'the whole log')}


def render(report) -> str:
    """The same report as text, for the colleague who pastes it into a message."""
    case, run, stages = report['case'], report['run'], report['stages']
    lines = [f"eforecast support report (version {report['version']})",
             f"case     {case['name']}  revision {str(case.get('revision'))[:12]}"
             + ('  -- EDITED SINCE THIS RUN' if case['changed_since_the_run'] else ''),
             f"run      {run['id']}  {run.get('script')}  {run.get('state')}/{run.get('status')}",
             f"cleanup  {run.get('cleanup')}"
             + ('  stop requested' if run.get('stop_requested') else '')]
    if run.get('outcome') and run['outcome'].get('kind'):
        lines.append(f"outcome  {run['outcome']['kind']}: {run['outcome'].get('detail') or ''}")
    for key, value in (run.get('errors') or {}).items():
        lines.append(f"error    {key}: {str(value)[:300]}")
    if stages['total']:
        counts = ', '.join(f'{count} {state}' for state, count in sorted(stages['counts'].items()))
        lines.append(f"stages   {stages['total']} ({counts})")
        for item in stages['attention']:
            lines.append(f"  stage  {item['stage']}/{item.get('scope')}: {item.get('outcome')}"
                         f" {str(item.get('detail') or '')[:200]}")
            for reason in item.get('blocked') or []:
                lines.append(f"         blocked: {str(reason)[:160]}")
    if report['execution']:
        lines.append('ran on   ' + ', '.join(f'{key}={value}'
                                             for key, value in sorted(report['execution'].items())))
    if report.get('dispatch'):
        dispatch = report['dispatch']
        lines.append(f"dispatch {dispatch.get('status')} {dispatch.get('dispatch_id') or ''} "
                     f"{dispatch.get('reason') or ''}".rstrip())
    installation = report['installation']
    lines.append(f"doctor   {installation.get('verdict')} {installation.get('counts') or ''}"
                 + (f" release {installation['release']}" if installation.get('release') else ''))
    for item in installation.get('attention') or []:
        lines.append(f"  {item['severity']:9s} {item['check']}: {item['finding']}")
    accepted_health = report.get('when_accepted') or {}
    if accepted_health.get('verdict') or accepted_health.get('note'):
        age = accepted_health.get('age_s')
        lines.append('accepted ' + (f"the installation was {accepted_health['verdict']} when this run "
                                    f"was accepted" if accepted_health.get('verdict')
                                    else accepted_health.get('note'))
                     + (f" (reading {age:.0f}s old)" if isinstance(age, (int, float)) else '')
                     + ('  -- READING WAS STALE' if accepted_health.get('stale') else ''))
        for severity, checks in sorted((accepted_health.get('checks') or {}).items()):
            lines.append(f"  {severity:9s} {', '.join(checks)}")
    for name, url in sorted((report.get('links') or {}).items()):
        lines.append(f"link     {name}: {url}")
    lines.append(f"excludes {report['excludes']}")
    lines.append('--- log (' + (report['log'].get('note') or '') + ') ---')
    lines.append(report['log'].get('text') or '')
    return '\n'.join(lines)
