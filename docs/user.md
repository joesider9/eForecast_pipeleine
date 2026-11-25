# For the person running a case

Everything you need, in the order you need it. You work in a browser; the lab machine does the
computing.

If something has already gone wrong, go to [troubleshooting.md](troubleshooting.md) instead.

---

## 0. First: is the installation working?

The chip in the top bar, beside your name, says one of three things: **Installation ready**,
**Needs attention (n)**, or **Must be fixed**.

Click it. Each problem says what was checked, what follows from it, and the one thing to do —
marked **yours to do** or **the maintainer's**. Anything marked the maintainer's is not for you
to fix; the chip tells you how many there are so you know what to ask for.

The chip also says how old its reading is. One taken a while ago may have changed since.

Start here whenever anything is unclear. Most answers in this document end with "look at the
chip".

---

## 1. Configure the case

The tabs, in the order you need them:

| tab | what you do there |
| --- | --- |
| **Everyday settings** | the plant, the target file, the timezone, the resolution, the horizon |
| the section forms | the full configuration, section by section |
| **Experiments** | the model graph — which networks are tried |
| **Review & save** | see exactly what will change, before it changes |
| **Runs** | start work, and watch it |

The case is saved into your own workspace as a YAML document, split into part files. Editing
those by hand works too; the JSON Schema in `schema/` gives completion and validation in an
editor.

### Two fields people get wrong

**Local timezone describes the source timestamps. Site timezone describes the location.** They
are not the same question, and readiness checks the *Local* one. If your source stays at a fixed
offset all year, its name is a fixed-offset name like `Etc/GMT-1` — not `CET`, which this runtime
treats as UTC+1 in winter and UTC+2 in summer. Confirm which convention the source file actually
uses before changing this, because relabelling it changes how every timestamp is read.

**Weather selection uses `as_of`:** only weather runs that were available at the forecast's issue
time are eligible. The publication-delay field is there for the provider's real delay. If a case
carries an older selection rule, the UI warns you — ask the maintainer to arrange the rebuild
rather than changing it yourself.

### Weather that is not on this machine

A case names, for each weather provider, the folder its raw files are in. A machine that does not
hold the ECMWF archive can read it from the lab's shared Google Drive folder instead: on the
**Project** tab, under **Nwp Models**, set **source** to **Google Drive**. The link of the shared
folder is filled in. Any other folder works too, if it holds one `YYYYMMDD` folder a day and is
shared with "anyone with the link".

The datasets step then downloads each day that is not in the local folder, converts it to the
daily file the pipeline reads, and removes what it downloaded. The folder beside the choice is
only where a day is held meanwhile; nothing stays in it. Days that are already in that folder are
read from it and left alone.

Three things to know before choosing it:

* **It is slow the first time, and only then.** A day is 25 to 45 MB, so two and a half years are
  close to 40 GB through the connection — about an hour on the lab's line, where forty days took
  three and a half minutes. A day that has been converted is never downloaded again, and a
  finished extraction reaches no network at all.
* **A day that did not arrive stops the step; it is not skipped.** Start the datasets step again
  and it asks only for the days it lacks. See
  [troubleshooting](troubleshooting.md#6-the-inputs-are-not-what-the-case-expects).
* **Changing the case's weather variables downloads everything again,** because the files a
  different set of variables would be read from were not kept.

### Choosing deep-learning models

In **Experiments**, pick the **method** first. Then either choose an existing experiment and
**Assign to \<method\>**, or click **New**, name it, and **Create & assign**. A method can use
more than one experiment. Click an assigned name to edit its network, then **Save** the case.
**Duplicate** makes and assigns an independent copy.

Assigning an experiment does not enable a disabled method — turn on its global or cluster switch
in **Methods** when you are ready.

Click a layer to set its parameters. **add a parameter…** offers the ones that layer takes. One
that has a default starts at it; any other starts empty, and an empty field is a parameter that
is not set — the method's value stands, as the field's `default` hint says. A parameter that
takes branch names (`attention_query`) offers the branches of the graph. A value of the wrong
kind is marked under its field, and **Check readiness** names it too.

CPU and foundation models are configured in **Methods**, not here.

Dragging a branch header rearranges the diagram. **Fit graph** brings anything off-screen back
without changing the layout; **Reset layout** restores the automatic arrangement. Neither changes
the model.

### Before you start a long run: Check readiness

On the **Review & save** tab. It inspects the inputs *before* an hour is spent: coverage of the
target series, the declared variables against what the files actually hold, and timestamp
ambiguity. The verdict is *ready*, *needs attention* or *cannot start*, and each finding names
the field it belongs to.

Readiness and the health chip answer different questions. The chip is about the installation;
readiness is about this case's inputs. Check both before reporting a problem.

### Changing a case that has already built something

A case builds **one model**, and that model is the model of **one configuration**. So a setting
you change after something was built from it is not a note for later — it decides what has to
be done again:

| you change | what is done again |
| --- | --- |
| a variable, its lags or its parameters, the target, the horizon, a weather provider | the **datasets**, and after them every model |
| a parameter of one family, or what it searches over | **that family**, from the beginning; the other families are kept |
| how the models are combined afterwards — the combine methods and their settings, how many models are kept of each family, the prediction intervals | the **combination**, from the trials that are there; nothing is trained again |
| a budget (`n_trials`), a path, how many processes | nothing |

**The Save button asks before it saves a change that sends datasets or families back.** It shows the settings you changed, the
datasets a later run will build again and the families it will train from the beginning (their
trials are removed first), and it saves when you have typed the name of the case. A tick would
be given in passing; a name is not. *Cancel* saves nothing and keeps your draft, and *Save as
new case* gives the change a model of its own and leaves this one alone.

You can ask the same question without saving: **Review & save → What does saving this draft
cost?** With nothing unsaved, the same button asks the other question — **Is what is built this
configuration's?** — and names the datasets and families that are still waiting to be done again
for a change that was saved earlier.

Saving replaces nothing. The **run** that follows does, and it asks once more, the same way,
with the files it would replace (§2). Until that run has finished the case says so plainly:
nothing is trained on datasets that were built from other variables, and nothing is combined
with a family that was trained under other parameters. If the change was a mistake, put the
setting back and save — everything that was built is this case's again, as it is.

---

## 2. Run it

Three steps, in order, from **Runs**:

| step | what it does | how long |
| --- | --- | --- |
| **datasets** | reads the weather and the target, builds the model inputs, freezes the data split, fits the scalers | the slow one — hours, dominated by weather extraction |
| **training** | searches for good models, one trial at a time, on the GPUs (or the CPU for shallow families) | the long one — can be a day |
| **combine** | predicts, selects, combines, calibrates, and writes the evaluation | minutes to an hour |

Each run row shows one sentence and one action. A stage that is blocked says what it is waiting
for.

**Continue vs Run.** *Continue* reuses whatever is still valid and explains, in its preview, what
it would rebuild and why. Read that preview. It is the only place you see, before committing,
how far back a change has sent you.

**Clicking twice is safe.** The same request reaches the same run; it does not start a second
one. If you change an option first, that is a deliberate new request.

**What a change costs you** — this is the rule worth memorising:

| you changed | what runs again |
| --- | --- |
| a parameter of one model family, or its search space | that family trains from the beginning; the others are kept |
| a budget (`n_trials`) | nothing is thrown away: the study continues |
| added or removed a family | the other families are kept; combining runs again |
| a combine method or its settings, `keep_best_models`, the prediction intervals | combining runs again; nothing is trained |
| a variable, the target, the split, a scaling method | everything, from the datasets up |

The reason is in [how_it_works.md §11](how_it_works.md#11-what-makes-an-old-trial-reusable): the
deeper the thing you changed sits, the further back it sends you.

**A run that replaces what another configuration built asks first.** If the datasets or a
family's trials were built from settings the case no longer has, *Run* does not start: its
preview opens instead, with the settings that changed since the model was built, the datasets
it would build again and the trials it would remove. It starts when you have typed the name of
the case. Nothing of that can be had back except by doing the work again, which is why the
browser never does it unasked — and why putting the setting back, when the change was not
meant, costs nothing. (A run typed in a terminal is different: it prints the same list and goes
on, see [cli.md](cli.md).)

**Nothing you do here publishes anything.** A finished trial is a candidate. A finished run is
not permission to use it.

### Running a case on the cluster

Your runs execute on the lab machine. That is the default and, on most installations, all there
is. An installation can also be connected to a Kubernetes cluster, and then a case can be
*routed* there: you press **Run** on the same page, and the step executes in a container on the
cluster instead of on the lab machine. The pages, the stages, the preview, the log, Stop and
the results are the same, and the results are files in the same folders.

**Is it available here?** Ask the maintainer, or in a terminal:

```bash
bin/eforecast route <case>
```

```
scope    installation
venue    local  (default: nobody has decided)
waiting  0 outstanding dispatch(es)
```

`local` is the lab machine; `broker` is the cluster. `scope` says whose decision the answer is:
the installation's, until somebody decides for this case. On an installation that has no cluster,
choosing it is refused (`broker is not connected`) and there is nothing more to read here.

**Sending a case there** is a recorded decision, the maintainer's to make
([cli.md, `route`](cli.md#route--on-which-machine-does-a-run-execute)):

```bash
bin/eforecast route <case> broker --note "why"     # runs accepted from now on go to the cluster
bin/eforecast route <case> local  --note "back"    # ... and back to the lab machine
```

It applies to the runs you start afterwards. A run that is already going keeps the place it
was accepted for, and the decision is refused while one is (`dispatch(es) are outstanding`):
let it finish, or stop it, and decide again.

**What is different on the cluster:**

| | |
| --- | --- |
| waiting | the run row says what the cluster is waiting for — no machine free, the image not there yet — where a local run would say nothing |
| deep training | needs a GPU the cluster offers as soon as there is a trial to train. A cluster without one runs the datasets, the CPU families and the combine step, and passes the training step of a case whose training is finished |
| families in a container of their own | TiRex, TimesFM and training with `--docker` are started by the lab machine's Docker, which a cluster run cannot use. Such a run is refused when you press Run, and says so |
| after a code update | the cluster's image has to be rebuilt by the maintainer. Until then a run ends at once with `worker image Python source differs from the accepted source`: tell the maintainer, nothing is wrong with the case |
| one place at a time | nothing is retried by itself and nothing moves a running run from one place to the other |

**Trying it without deciding anything.** The maintainer can run one step of your case on a
small cluster made on the lab machine itself, which changes nothing of the installation — a
good way to find out whether your case can run in a container at all:
[deploy/kubernetes/README.md, "Try it on one workstation"](../deploy/kubernetes/README.md#try-it-on-one-workstation-start-to-finish).

---

## 3. Read the result before trusting it

The **Results** tab answers one question — *can I use this?* — and labels every number with the
question that number answers.

* **the result this case serves.** The predictor a forecast returns when nobody names one, in
  percent of rated power, from the holdout. The page names which predictor that is. If another
  one scored better over the same period, it says so — and warns you that choosing it *because*
  it won there turns the holdout into another selection slice, and the honest number is gone;
* **experimental or approved.** Until a maintainer approves a version, the page says
  *experimental*. That is a fact about permission, not about the numbers;
* **what is missing.** Gaps per slice, unsupported clusters, unfinished trials, tables that were
  never written. A missing coverage table is reported as *not ruled out* — never as full
  coverage;
* **which configuration produced it**, and whether the case has been edited since. The model
  keeps the one configuration it was built from, in its own folder
  (`.stage-artifacts/configuration/`);
* **the forecast**: what is published, and separately, whether the last attempt worked.

### The three slices, and why only one of them is honest

| slice | what it was used for |
| --- | --- |
| **selection** | choosing which trials to keep, and fitting the combiners |
| **calibration** | fitting the prediction bands |
| **holdout** | the final report, and nothing else |

A number measured on data that was used to make choices is optimistic. The holdout — `summary.csv`
— is the only honest estimate of how the model will behave on data it never influenced.

---

## 4. Getting a model into service

This part is the maintainer's, deliberately. Ask them, pointing at the Results tab.

They capture a reference bundle, which is replayed as part of being captured, and then approve
it — which copies it to an immutable snapshot, replays it *there*, and only then moves the
pointer the forecasts read. A bundle whose weights no longer load cannot become the one you use.

They can also roll back to the previous approved version. That is a pointer move with a
re-validation; it never retrains anything.

---

## 5. Where the forecast is, and whether it is current

```bash
bin/eforecast forecast-status <case>
```

Two separate answers, and the difference between them is usually the whole story:

```
forecast  issued 2026-10-27 06:00 UTC (1.2 h ago)
attempt   the last attempt was not published (empty)
          the published forecast above is unchanged; fix the attempt and retry
```

* **forecast** — what you are using. The last one that passed every check.
* **attempt** — whether the most recent *try* worked.

A published forecast with a failed attempt means the last accepted numbers remain, **and they may
be too old for the period you need**. Check the issue time against your period.

Exit codes, if you script it: `0` current, `1` stale or the last attempt failed, `2` nothing
published.

The files:

| file | what it is |
| --- | --- |
| `<path_group>/predictions/latest.csv` | the published forecast, as a file: a copy of the one the database points at |
| `.inference-generations/<launch id>/forecast.csv` | the forecast of one cycle, kept as it was accepted; `latest.csv` can be written again from it |
| `status.json` | what the last cycle did |

**A failed run never overwrites a good forecast.** That is a guarantee, not a coincidence.

---

## 6. When something goes wrong

Go to [troubleshooting.md](troubleshooting.md). The short version:

1. note the **run id**;
2. read the reason the run gives, not just "failed". If it names a **field**, that field is the
   thing to change — you do not need to read the stack trace underneath it;
3. check the health chip;
4. if it is the maintainer's, send them the **run id** and the **support report** from the run's
   log drawer (*Copy support report* or *Download support report*), plus what you were trying to
   do and what you had changed since it last worked.

Two things not to do:

* **do not delete the case or its folder to "start clean".** That throws away the frozen data
  split and the fitted scalers, and makes every model trained against them incomparable;
* **do not rebuild** because it sounds like a reset. The Review screen tells you what a rebuild
  would affect before it starts. Waiting to ask costs an hour; the alternative can cost a
  retraining campaign.

---

## 7. Want to know what is actually happening?

[how_it_works.md](how_it_works.md) walks the whole workflow in plain words: what runs, in what
order, what is written where, how the machine decides there is room to run, and how the training
is driven.
