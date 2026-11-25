# eforecast configuration web UI

For developers. Colleagues use the deployed lab URL; see [the user guide](../docs/user.md).
Run the backend in a separate terminal with an explicit development workspace and
`python -m eforecast.config.serve --host 127.0.0.1 --no-auth`. Never point fixture/fault tests
at a production workspace. Commands below run inside `webui/`.

Vite + React 18 + TypeScript + MUI, forms generated from the JSON Schema with react-jsonschema-form.
The FastAPI application (`eforecast/config/api.py`) serves the built app and the API.

```bash
cd webui
npm ci
npm run dev          # http://localhost:5173, proxies /api to http://localhost:8000
npm run build        # -> webui/dist, served by `python -m eforecast.config.serve` (see deploy/README.md)
```

Layout: `src/api` typed client, `src/lib` schema and YAML helpers, `src/forms` the schema-driven
section form and the YAML field, `src/components` review widgets, `src/pages` Login, Cases, CaseEditor.

## Tests and editing behavior (A06)

```bash
npm ci
npm test                 # Vitest / Testing Library, deterministic HTTP fixtures
npm run lint:baseline    # rejects findings beyond the recorded explicit-any baseline
npm run build            # TypeScript checks and production bundles
npm run test:e2e         # Playwright against the production build on port 5175
```

Playwright uses the installed `/opt/google/chrome/chrome` locally when available,
or `EFORECAST_CHROME=/path/to/chrome` when supplied. Otherwise install its browser
with `npx playwright install chromium`. CI installs the matching bundled Chromium
and runs the frontend job in `.github/workflows/tests.yml`. Browser fixtures mock
the HTTP boundary; they do not launch training. Component tests mock canvas/editor
layout; browser workflows use the real graph canvas and CodeMirror.

Full `npm run lint` is a stricter check than the accepted baseline.
`eslint-baseline.json` records per-file allowances (52 in the recorded file);
`npm run lint:baseline` checks the actual current findings and refuses regressions.
Do not treat an old count as the output of a fresh lint run. The broader generated-contract work follows this harness and is not replaced
by disabling TypeScript or ESLint rules.

Each identity has a separate disposable private query cache. Identity and case
changes cancel old requests. The editor keeps its saved baseline separate from its
draft. Refreshing a changed saved case preserves edits and shows differences;
revision conflicts offer reload or save-as-new. Edits made during a save remain
unsaved afterward. A transport failure makes the save outcome uncertain and requires
a read before retrying. Navigation and tab closing warn about unsaved document edits.

Deliberate logout offers **Save and log out**, **Discard and log out**, and **Cancel**
before sending the request. Saving uses the same revision-checked path; failures or
newer edits prevent logout. Unfinished YAML cannot be saved, and cancelling retains
the editor. A failed logout also retains its draft. Confirmed session expiry or an
account change still clears private state.

Raw form and experiment YAML buffers live in the case draft, so unfinished text
survives tab changes and counts as unsaved. A case-wide banner identifies parse
errors and blocks Save, copy, Validate, Run and Dry build until repaired or explicitly
discarded. Discarding a buffer restores its last valid parsed value; other parsed
edits remain. Valid formatting/comments survive tabs, but saving still sends the
parsed configuration through the existing backend serializer. Buffers are in-memory
editing state, not a durable recovery file.

Experiment YAML also checks branch/layer/parameter structure before updating the
canvas. Invalid text keeps the last valid graph and shows a field-path error.
**Normalize YAML** uses the backend's existing legacy conversion for tuples and
parameter shorthands. Stored experiments normalize on load. A local graph recovery
boundary preserves the case draft if rendering fails.

Experiments is for custom deep-learning methods: CNN, TIMM, LSTM, TRANSFORMER, MLP,
CrossViVit and Distributed. The method picker uses the server's supported method
catalogue and shows families already configured in the case. CPU, foundation,
site-GRAPH and clustering settings remain in Methods.

Choose the **method** first. Select an existing **experiment**, then **Assign to
<method>**, or choose **New**, enter a tag and **Create & assign**. **Duplicate**
also assigns the new copy to the selected method. The assigned chips show the
method's templates; click one to edit it. Assigning adds to existing templates.
**Unassign from <method>** removes only that link. Use **Save** to persist edits.
Assignment does not enable training: a disabled method is labelled, and can be
enabled for global or cluster training in Methods. Module/preset definitions can
be brought into the editor with **Load presets**.

Renaming updates references in all configured custom deep methods. Deletion is
blocked while one of those methods references the template. Shared templates are
labelled because edits affect every assigned method; duplicate for an independent
version.

In Experiments, drag a branch's header to arrange the canvas. Merge and head nodes
can also be moved. Dragging a node does not automatically pan the canvas at its
edges. **Fit graph** brings every node into view while keeping your arrangement.
**Reset layout** restores the selected template's automatic arrangement and
recenters the view, including after background panning. Positions survive layer edits, saving, Graph/YAML switches and case
tab switches while the case remains open. Reloading or leaving the case resets
the layout. Positions are presentation state: they are not saved in case YAML,
do not mark the case as changed, and do not alter layer order or connections.

New API responses include a content revision. The editor submits `expected_revision`
with PUT; stale PUT/PATCH writes return HTTP 409 and the current document without
rewriting it. The check is optional for existing clients. It serializes cooperating
API edits on this Linux host; CLI/external writes and complete multi-file transaction
recovery remain separate work. No stored case migration is required.

**Local timezone** is the timezone of the source timestamps. **Site timezone** is
the physical site's timezone and can differ. Readiness checks the source using
Local timezone. In the installed runtime, `CET` observes summer time; `Etc/GMT-1`
is fixed UTC+1 throughout the year. Confirm the source convention before changing
it. Stored zones outside the preset list remain visible and are preserved.

Run logs drain all pages, including completed runs. The visible tail retains at most
512 Ki UTF-16 code units and 4,000 lines. Earlier text is omitted with a notice;
**Download full log** streams the original file without accumulating it in React.
File replacement/truncation resets the cursor and failed reads offer retry.

## Runs and studies

The **Runs** tab submits the saved case to durable acceptance for datasets, training or combine.
An independent coordinator prepares immutable inputs, dispatches the selected local/managed
route, verifies stage artifacts and collects outcome plus cleanup. Active or unknown ownership
can hold a case even after the root process ends. Retry identity is retained across uncertain
responses. Continue/New revision/Rebuild previews show reuse and consequences.

The configured run store (SQLite outside the workspace or PostgreSQL) owns durable state;
case run folders retain logs/snapshots and compatibility records. Do not treat a log folder as
the entire run database. See [the administrator guide](../docs/administrator.md) and the optional
[resource](../docs/cpu_training_scopes.md) and
[command-line flags](../docs/cli.md).
The Jobs tab is omitted from the case editor; the complete saved `jobs` configuration
is preserved on save/copy. Nothing is reserved for a run: the Runs panel offers the four
settings that say how much CPU training starts at once (CPUs, studies at a time, threads per
study, foundation device), and a run's **Resources** show what is running and what its
processes were seen to hold. Settings of earlier versions that a stored case still names
(execution profiles, cache tuning, memory allowances) are ignored and no longer offered.
The Data form hides `nwp_selection_policy`; new cases inherit `as_of`. Stored
`legacy_merge` values are preserved with a warning about unavailable-at-issue weather
and the required rebuild. Publication delay remains editable. The backend compatibility
contract is unchanged; there is no automatic fallback to legacy weather selection.
Methods hides `env_name`, `path_env` and `runtime_ids`, including the object/YAML editor.
The full values still round-trip on save. Native execution uses the service interpreter;
existing family routing selects supported model images when needed. Readiness reports the
effective interpreter, without an attention finding for an inert legacy path. Real package,
image and approved-runtime failures remain visible. Stored explicit selections and accepted
bindings remain authoritative; hiding the control does not reroute existing models.
The **Studies & results** tab reads the dataset files, the clusters, every Optuna study (journal files or
`EFORECAST_OPTUNA_STORAGE`), the selected trials and `Results/*.csv`; links to optuna-dashboard, Grafana
and the orchestrator metrics come from `EFORECAST_OPTUNA_DASHBOARD_URL`, `EFORECAST_GRAFANA_URL` and
`EFORECAST_METRICS_URL` (defaults: the compose stack on localhost).

## Users and deployment

Authentication on (`EFORECAST_AUTH=on`, the default of `serve`): users live in `<workspace>/users.json`,
the first user created is the administrator. Personal cases live under
`<workspace>/users/<name>/cases`; shared tenant cases use `<workspace>/tenants/<id>/cases`
and tenant memberships/roles. A case marked *shared* (share icon on the cases page) is visible to
everybody read-only and can be copied.  Administrators see and edit every case and manage the users
on the **Users** page; everyone changes their own password from the app bar.  `deploy/` holds the
systemd units and the script that configures a machine; the procedure is
[docs/installation.md](../docs/installation.md).


## Contract and live integration checks

Two kinds of check the tests above cannot do: that the pages and the server agree on the shape
of the data they exchange (the **contract**), and that the pages work against a real server
instead of a pretend one (the **live** check).

### The contract

The **Runs** page shows what a run is doing with the machine — which processes are running and
what they hold — and offers four settings for CPU training (`n_cpus`, `cpu_workers`,
`cpu_study_threads`, `cpu_study_device`). The server sends all that as JSON, and the page is
written in TypeScript, which has to know the exact shape of that JSON. "The contract" is that
shape. It is written down once, in Python, and copied mechanically twice:

```
eforecast/config/resource_contract.py           the Python classes — the one place to edit
        │   python -m eforecast.config.resource_export        (from the repository root)
        ▼
schema/resource_contract.schema.json            the same shape as a JSON Schema
        │   npm run contracts:generate                         (inside webui/)
        ▼
webui/src/api/resourceContract.generated.ts     the TypeScript types the page is compiled against
```

All three files are committed. If one is changed and the next is not regenerated, the page and
the server drift apart without any error — a field the page no longer shows, a setting it
sends under an old name. One check guards each arrow:

| command | run it in | passes when | otherwise says |
| --- | --- | --- | --- |
| `python -m eforecast.config.resource_export --check` | the repository root | the schema file is exactly what the Python classes produce now | `resource schema stale: python -m eforecast.config.resource_export` |
| `npm run contracts:check` | `webui/` | the TypeScript file is exactly what the schema file produces now | `resource types stale: npm run contracts:generate` |

Each looks at its own arrow only: `contracts:check` reads the schema file and never runs
Python, so it cannot notice a Python change that was not exported. Run both — `pytest` does
(`tests/test_resource_controls.py`). Neither needs anything running.

To change the contract on purpose: edit the Python classes, run the two generating commands in
that order, read the diff of the two generated files, and commit all three together.

### What each browser check proves

| command | who answers the page's requests | what it needs | what a pass proves |
| --- | --- | --- | --- |
| `npm test` | pretend answers written in each test | nothing | a component behaves as intended (no real browser) |
| `npm run test:e2e` | pretend answers written in `webui/e2e/` | a build in `webui/dist`, port 5175 free, Chrome | the *built* pages work in a real browser: the graph canvas, the YAML editor, moving between pages |
| `npx playwright test --config playwright.private.config.ts` | the same | a build in a private folder, port 5176 free, Chrome | the same, without touching `webui/dist` (see below) |
| `pytest tests/test_run_integration.py tests/test_mixed_run_integration.py` | **the real server** | see below | the pages, the API, the run coordinator and the run database agree |
| `npm run typecheck:live` | — | nothing | only that the test files of the live check (`webui/e2e-live/`) still compile. It runs none of them |

The first three never start a run: every `/api/…` request is answered by the test itself. They
show that a page did not break; they say nothing about the server.

**The live check** is the one that does, and it is started by pytest, not by npm. Each test
starts two API processes and two run coordinators on free ports of `127.0.0.1`, over a
temporary workspace and a temporary run database with login switched off, and submits real
dataset runs of a tiny case to them. What it covers depends on what you allow it:

| with | it also does | otherwise |
| --- | --- | --- |
| nothing extra (any plain `pytest tests`) | accepts a run submitted twice only once, kills and restarts the service at three points and checks the run recovers, cancels a run | — |
| `EFORECAST_TEST_LIVE_BROWSER=1` | builds the pages into a temporary folder, serves them from the test's own API and drives Chrome through `e2e-live/runs.spec.ts` and `compute.spec.ts` (real RF and lasso studies) | those tests are skipped |
| `EFORECAST_TEST_RUN_STORE_DSN=postgresql://…` naming a **disposable** database | repeats every test on PostgreSQL | they run on SQLite only |
| `EFORECAST_TEST_LIVE_GPU=1` | trains one small network on GPU 0 | skipped |

A failure here means the real parts disagree: a run executed twice, a cleanup that never
finished, a page that did not survive a restart of the service. A test that was **skipped** has
proved nothing — `pytest -rs` lists the skips with their reasons. Never give it the
installation's own database or workspace; it creates and deletes rows.
`playwright.live.config.ts` refuses to load unless that pytest hands it a `127.0.0.1` address,
so the live tests cannot be pointed at a real installation by accident.

### Do not test on the pages people are using

`webui/dist` is what the service shows unless `EFORECAST_WEBUI_DIST` names another folder, and
`npm run build` replaces it at once — for everybody who has the page open. To try a change,
build somewhere else and let the private configuration preview that:

```bash
npm run build -- --outDir dist-private-a13 --emptyOutDir
npx playwright test --config playwright.private.config.ts
```

`EFORECAST_PREVIEW_DIR` and `EFORECAST_PREVIEW_PORT` name another folder or port. To put new
pages into service, build into a new dated folder and point `EFORECAST_WEBUI_DIST` at it:
[installation.md §2.2](../docs/installation.md).
