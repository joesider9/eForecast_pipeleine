# The deep networks — `eforecast/deep_models/pytorch_2x`

For whoever draws a network in the graph editor, reads a training log, or adds a layer. It says
what every layer does and when to use it, how branches for different kinds of data are joined
into one network, what each branch is fed, how a network is trained, and how the cross-site
graph differs from all of that.

Everything here is what the code does today. Where the code has a default, the number is given.

| file | what is in it | chapter |
| --- | --- | --- |
| `layers.py` | every layer a graph can name | [3](#3-the-layers--layerspy) |
| `builders.py` | turns a graph into one network: branches, merge, head | [4](#4-how-a-network-is-assembled--builderspy) |
| `network.py` | `DeepNetwork`: loads what a trial needs, trains, saves, predicts | [2](#2-data-tags--what-a-branch-is-fed), [5](#5-the-training-loop--networkpy-trainerpy-optimizerspy-horizonpy) |
| `trainer.py` | batches, one optimisation step, one validation pass, the alignment loss | [5](#5-the-training-loop--networkpy-trainerpy-optimizerspy-horizonpy) |
| `optimizers.py` | optimizer, learning-rate schedule, loss and metrics | [5](#5-the-training-loop--networkpy-trainerpy-optimizerspy-horizonpy) |
| `horizon.py` | a loss weight per forecast step; the quantile loss | [5](#5-the-training-loop--networkpy-trainerpy-optimizerspy-horizonpy) |
| `global_builders.py` | the fuzzy rules of the clusterer's network | [4](#fuzzy-rules-the-clusterers-network) |
| `site_graph_net.py` | the cross-site graph | [6](#6-the-cross-site-graph--site_graph_netpy) |
| `concat_networks.py` | left from the TensorFlow version; nothing imports it | — |

## 1. The picture

Every network here has the same three parts:

```
 data tag           branch: a list of layers
 ───────────────    ──────────────────────────────────────────────
 nwp           ──►  cnn_att ─► conv ─► tokens ─► dense ─► dense ──┐
                                                                  ├─► merge ─► dense … ─► head ─► one value per forecast step
 row_calendar  ──►  dense ────────────────────────────────────────┘
```

* A **branch** takes one kind of data — the weather grid, the calendar, the past observations
  — and turns it into features. The name of the branch is a **data tag**: it says which data
  the branch is fed ([chapter 2](#2-data-tags--what-a-branch-is-fed)).
* The **merge** joins the branches. It is the first layer of the list called `output`; the
  layers after it work on the joined features.
* The **head** turns the result into the forecast. You do not write it: the builder adds it,
  with one output per forecast step.

That is what makes a network *multimodal* here: each kind of data gets the layers that suit it,
and one merge layer decides how they meet ([chapter 4](#4-how-a-network-is-assembled--builderspy)).

### How a graph is written

A graph — an *experiment* in the code — is a table from branch names to lists of
`(layer, size)`. This is the shipped `cnn1`, drawn above, from `configuration/experiments.py`:

```python
'cnn1': {
    'nwp':          [('cnn_att', 2), ('conv', 2), ('Flatten', []), ('dense', 512), ('dense', 128)],
    'row_calendar': [('dense', 64)],
    'output':       [('cross_attention', {1}), ('dense', {64})],
}
```

| part | meaning |
| --- | --- |
| a key such as `nwp`, `row_calendar` | a data tag: a branch, and the data it reads |
| the key `output` | the merge layer, then the layers after it |
| the key `hidden_layer` | optional: how the parts of one tag are joined before `output` (see [groups](#when-one-tag-becomes-several-branches)) |
| `('dense', 512)` | a layer and its **size**. What the size means depends on the layer: a width for `dense`, a kernel for `conv`, a factor for `lstm` — each layer below says |
| `{64}`, `{32, 64}` | a set: the study picks one of these values for each trial |
| `[]` | the layer takes no size |

The shipped graphs are templates. A case carries its own copies in its `experiments` part, and
that is what the web UI's **graph editor** edits. A method family names the graphs it trains in
its `experiment_tag`; the MLP family, for example, is given `mlp1` to `mlp4`, and its study
picks one of them for each trial.

**Layer parameters.** Besides its size a layer can be given parameters of its own — the number
of attention heads, a pooling factor, a model name. The editor's inspector writes them into
`experiment_layer_params`, each as a fixed value, a list of choices or a range the study
samples. For one layer, its own parameter stands in front of the family's parameter of the same
name, and that in front of the layer's default. A parameter left empty is not set.

> **A graph is shared by the whole case.** The fingerprint that decides whether a finished trial
> can be reused covers all the graphs of the case, for every family. Change any graph — add one,
> change a size, set a layer parameter — and every family's trials are trained again at the next
> run, not only those of the family that uses it. Settle the graphs before a case is trained, and
> try a new idea with the dry build or `debug_layer.py`
> ([chapter 7](#7-trying-a-layer-or-a-graph-and-adding-one)) or in a copy of the case.

## 2. Data tags — what a branch is fed

The name of a branch is not a label you choose. `DeepNetwork` reads it to decide which dataset
of the case to load and how, and the layers of the branch then receive exactly that.

| tag | what it holds | what the layers receive | family options that shape it |
| --- | --- | --- | --- |
| `nwp` | the weather fields around the site | one tensor `(samples, lags, variables, lat, lon)` | `merge`, `scale_nwp_method` |
| `lstm` | sequences: what was observed before the forecast, and what is known about the hours being forecast | a dictionary of tensors `(samples, steps, features)`, see below | `scale_row_method`, `feature_selection_method` |
| `row_obs` | the observation columns of the row dataset: past values of the target and of other measured series | `(samples, features)` | `scale_row_method`, `feature_selection_method` |
| `row_calendar` | the calendar columns | `(samples, features)` | the same |
| `row_obs_calendar` | both of the above, side by side | `(samples, features)` | the same |
| `row_nwp` | the weather **in row form**: each grid compressed to a few numbers per row | `(samples, features)` | the same, and `merge`, `compress`, `scale_nwp_method` |
| `row_obs_nwp` | observations and compressed weather | `(samples, features)` | the same |
| `row_all` | everything in row form: observations, calendar, compressed weather, and image statistics when the case has images | `(samples, features)` | the same |
| `images` | satellite images | read from disk batch by batch | `image_data_type`, `scale_img_method` |

**How a `row_…` tag is read.** The words after `row_` are taken one by one — `obs`, `calendar`,
`nwp`, `all` — and their columns are put side by side. A row tag that includes the weather
(`row_nwp`, `row_obs_nwp`, `row_all`) needs the family's `compress` option, because a grid
cannot stand beside table columns; without it the trial stops and says so, and readiness
reports it beforehand (`row_input_without_compress`).

**The calendar, as a network sees it.** `month`, `day`, `dayweek`, `hour` and `minute` each
become a sine and a cosine of their period (12, 30, 7, 24, 60), so that 23 h and 0 h are
neighbours. `dayofyear` is divided by 365, `sp_index` by 100, `horizon` by 72.

**The `lstm` dictionary.** A case with sequences gives up to five tensors, and a sequence layer
takes the ones it knows:

| key | what it is |
| --- | --- |
| `past_obs` | the observed series over the past steps |
| `past_data` | other variables over the past steps |
| `past_calendar` | the calendar of the past steps, as sine and cosine pairs |
| `future_data` | what is known ahead for the steps being forecast — the weather forecast at the site |
| `future_calendar` | the calendar of the steps being forecast |

**Which block a series lands in.** The case's `global_past_lags` and `global_future_lags` say which
lags make the past block and which the future block, and a series is put where it has those lags.
A series that has lags only in the past is an *observation* (`past_obs`). A series that has lags in
both blocks is no observation but a covariate: `past_data` in the past block and `future_data` in
the future one. That is what the weather at the site is, and it is also what a measured series
becomes when a case gives it on both sides of the row — the wind of a park whose power is
reconstructed from wind speed and direction measured before and after (lags −28 … −1 and
0 … 28). Such a case has no observations and, when it names no calendar, no calendar either: its
dictionary is `past_data` and `future_data`, and every sequence layer reads both
([3.2](#32-on-sequences--the-lstm-branch)). A case that puts all its lags in `global_past_lags`
and leaves `global_future_lags` empty gets one block, `past_obs`, and is read with `use_nwp` off.

A `long_term` case has no sequence dataset: its rows are (predicted hour, issue time) pairs, and
there is no single clock to run a sequence along. A graph with an `lstm` branch cannot be
trained there; readiness reports it (`no_sequences`).

**Feature selection.** When the family names a `feature_selection_method`, row and sequence
data pass through the selector that the `features` stage fitted for the cluster. The weather
tensor is never selected.

**Time step.** The weather comes hourly; for a 15-minute case it is interpolated to the case's
step, for a daily case it is aggregated.

### What happens between the tag and the first layer

`DeepNetwork.fit` does this once per trial (`load_data_deep_models`, then `get_slice_for_nets`):

1. every tag of the graph is loaded, scaled the way the trial's parameters say;
2. only the timestamps that **all** tags have are kept, and the target is aligned to them;
3. the three parts of the case's frozen split — training, validation, test — are cut out of
   every tag for the cluster the trial belongs to. A cluster that holds fewer samples than the
   case's `clustering.min_samples` trains on the whole case instead, with the case-wide split.

The network is then built from the shapes of the first batch, so a graph never states its input
shapes: they follow from the case.

### When one tag becomes several branches

Two things turn one tag of the graph into several branches of the network, each built from the
same list of layers and each with weights of its own:

* **The weather in groups.** With `merge: all` the family reads one weather tensor. With
  `by_variable`, `by_area`, `by_area_variable` or `by_nwp_provider` it reads one tensor per
  group, and the branch `nwp` becomes `nwp/Flux`, `nwp/Cloud`, … — one small network per
  variable or area.
* **Fuzzy rules.** The clusterer's own network repeats every branch once per rule
  ([chapter 4](#fuzzy-rules-the-clusterers-network)).

How the groups are joined again is part of [chapter 4](#grouped-branches).

## 3. The layers — `layers.py`

Every name a graph may use is registered in `layers_func()` at the end of `layers.py`. The web
UI's layer guide and its "+ layer" menu show the same list with the same descriptions; they
read it from `eforecast/config/layer_catalogue.py`.

Two rules hold for all of them:

* **Shapes are written without the batch.** `(lags, variables, lat, lon)` is one sample.
* **The activation is the family's `act_func`** (GELU in the shipped presets), and a layer may
  be given its own.

Three words describe what leaves a layer, and they decide what may follow it:

| word | shape | examples |
| --- | --- | --- |
| **flat** | `(features)` — one vector per sample | a `dense` branch, anything after `Flatten` |
| **sequence** / **tokens** | `(positions, features)` — a vector per lag, per step or per grid cell | recurrent and transformer layers, `tokens` |
| **grid** | `(…, lat, lon)` — still a map | convolutions, `cnn_att`, `afno` |

### 3.1 On the weather grid — the `nwp` branch

| layer | what it does | gives | `size` means |
| --- | --- | --- | --- |
| `conv` | one convolution block, the same kernels for every lag | grid; a vector per lag when it is the last convolution of a `short_term` branch | the kernel, made odd (2 → 3) |
| `conv_3d` | the same block with a kernel over (lags, lat, lon) | grid | the kernel on the grid; `kernel_time` (3) over the lags |
| `cnnT` | the `conv` block under an older name; it never takes the summary over the lags | grid | the kernel |
| `cnn_att` | attention **on** the fields: weights the variables and the places before anything else reads them | the same shape as its input | the kind of attention, as text; otherwise unused |
| `grid_transformer` | a small transformer whose tokens are the grid cells | tokens, one per cell | the token width (64 when under 8) |
| `afno` | a Fourier neural operator: mixes the whole grid at once in the frequency domain | grid (default) or tokens | the channel width (64 when under 8) |
| `bi_lstm` | reads the lags forwards and backwards, each lag's grid flattened to one vector; needs two lags or more | a short sequence (4 steps at most) | a **divisor** of the input width here |
| `vit_net`, `timm_net` | a pretrained vision model; lags × variables are its channels | flat | — |
| `time_distr_vit_net` | one pretrained vision model **per lag**, then an LSTM over the lags | flat | — |
| `unet` | a segmentation U-Net whose output map is flattened | flat | — |

**`conv` — the default reader of the grid.** One block is: convolution (same padding) →
normalisation (GroupNorm; BatchNorm with `batch_norm: true`) → activation → a residual
connection when the number of channels did not change and nothing is pooled → average pooling
(`pool`; 2 while the grid is at least 8 cells wide, otherwise none). The number of filters is the
family's `filters` × (position of the layer in the branch + 1), so the second layer of a branch
with `filters: 16` has 32; `filters_grow: false` keeps it constant. The lags are folded into
the batch, so the block learns one set of kernels for all lags and keeps the lag axis.

What the name builds depends on the case, and you do not choose it: `fit` renames every `conv`
of a graph to `time_distr_conv_2d` for a `short_term` case and to `conv_2d` otherwise (`_3d`
with `conv_dim: 3`). The two are the same block. The difference is one step the builder adds to
the **last** convolution of a `short_term` branch: the grid is averaged away and an LSTM runs
over the lags, leaving **one vector per lead time** — which is what a forecast of several steps
needs, and what an attention merge can then attend over.

*Use it* as the first choice for the weather. Two blocks are usually enough on a grid of a few
tens of cells: each pooling halves it.

**`cnn_att`** comes before the convolutions, not instead of them. `cbam` (the default) and
`cbam1` weight the channels and then the places — squeeze-and-excitation with a spatial map;
`att` computes a mask on a shifted and scaled copy of the input (a spatial transformer) and
multiplies the input by it; `geodatt` is a dual attention network over place and channel. The
kind comes from the family's `attention`, or from the size when the size is text.
*Use it* when the fields are many and only some matter, or when the weather that matters sits
off-centre.

**`grid_transformer`** makes every cell (or every `patch` × `patch` cells) a token carrying all
lags and variables of that place, adds a learned position, and runs `depth` (2) encoder layers
with `heads` (4). *Use it* when distant parts of the grid matter together — a front arriving —
and the grid is small; the cost grows with the square of the number of cells.

**`afno`** reaches the same goal more cheaply: `depth` (2) blocks, each a 2-D Fourier transform
of the grid, a small network per frequency (`num_blocks` 4), and back. Follow it with `tokens`
for an attention merge, with `Flatten`, or with a `conv`.

**Pretrained vision models** (`vit_net`, `timm_net` — one class under two names). `model_name`
is a timm model or a Hugging Face one; the fields are resized to the size the model expects.
`pretrained: finetune` **freezes** the pretrained weights so that only the layers after it
learn; any other value trains them too. The weights must already be on the machine unless
`EFORECAST_MODEL_CACHE_MODE` allows a download ([cli.md §4](cli.md#4-the-settings-that-matter)).
*Use them* with thousands of samples per cluster at least: they are the heaviest thing a graph
can hold, and a 17 × 17 field of physical values is far from the photographs they were trained
on. `time_distr_vit_net` holds a separate model for every lag and is heavier by that factor.

### 3.2 On sequences — the `lstm` branch

All of these take the `lstm` dictionary of [chapter 2](#2-data-tags--what-a-branch-is-fed). The
family's `use_nwp` decides whether the future covariates are read at all.

| layer | what it does | gives | `size` means |
| --- | --- | --- | --- |
| `lstm` | an LSTM over the past; with `use_nwp` a second one over the future, started from the state of the first, and a third over both | a vector per forecast step | hidden width = size × input width |
| `lstm_att` | the same two encoders, then every future step **attends** over the past steps | a vector per future step | hidden width = size × input width, rounded to a multiple of `heads` |
| `timexer` | the past cut into patches as tokens, every covariate as one token, a global token linking the two | tokens (default), the global token, or flat | the token width (64 when under 8) |
| `tide` | dense layers only: encode the flattened past and covariates, decode to one vector per future step | a vector per future step | features per step |
| `tsmixer` | dense layers only, mixing in turn along time and along the features | a vector per future step | features per step |
| `transformer` | a model of the time-series library under `time_series_library/models/`, named by the family's `model_name` (the file name: `Informer`, `Autoformer`, `PatchTST`, …), over the past | sequence | model width = size × input width |
| `bi_transformer` | two such models, past and future, joined by a linear layer | flat | the same |
| `bi_lstm` | two LSTMs, past and future, whose last 4 steps feed a bidirectional LSTM | a short sequence (4 steps) | hidden width = size × input width |
| `bi_spacetimeformer` | experimental | | |

**`lstm`** is the plain choice. With the past only it returns a vector for every past step;
with `use_nwp` it returns the last `n_out` steps, one per forecast step. On a tensor instead of
the dictionary — after convolutions, say — it is a single LSTM, and it returns only the last
step when a `Flatten` follows it.

**The past of `lstm` and `lstm_att`** is the observations and the past calendar. Where a case has
no observations, it is the case's past data (with the past calendar, when there is one) — the park
of the paragraph above. Until 24 November 2026 it was not: without observations `lstm` read the
past calendar alone, or with no calendar either the future block alone, and said nothing; and
`lstm_att` could not be built without one of the two. A network trained before that date is still
built the way it was trained, so its weights fit (`layers.PAST_DATA_IS_PAST`, absent from what it
saved and read as false; `network.reads_past_data`). Nothing changes for a case that has
observations, and nothing in the other sequence layers, which always read the past data.

> The past and the future block must differ in length when both are read. With equal lengths
> the graph cannot be built, and the layer says so: use `lstm_att`, `timexer`, `tide` or
> `tsmixer` there, or switch `use_nwp` off so that only the past is read.

**`lstm_att`** is `lstm` with the question "which past hours matter for this lead time" made
explicit: softmax attention of each future step over the encoded past (`heads` 4). *Use it*
when the recent past matters differently along the horizon — the first hours lean on the last
observation, the later ones on the daily pattern.

**`timexer`** separates what is forecast from what merely informs it. The observed series
becomes patches of `patch` (4) steps; each calendar column and, with `use_nwp`, each future
covariate becomes a single token; `depth` (2) blocks let the patches attend to each other and
a global token attend to the covariates. `output: tokens | global | flat`. *Use it* when there
are many covariates of uneven worth: its attention says which were used (`last_attention`).

**`tide`** and **`tsmixer`** have no recurrence and no attention, and train in a fraction of
the time of the others. `tide`: `hidden` 128, `depth` 2, covariates projected to `cov_dim` 8
per step, and a direct linear path from the past to the horizon (`skip`, on). `tsmixer`:
`hidden` 64, `depth` 2, `ff_ratio` 2. *Use them* first — as the baseline a heavier sequence
layer has to beat — and when there are many clusters to train.

### 3.3 On a row of numbers, and the helpers

| layer | what it does | `size` means |
| --- | --- | --- |
| `dense` | a linear layer and the activation, on the **last axis**: a flat vector gives a flat vector, a sequence gives a sequence with every token transformed alike | 8 or more: the width. Under 8: a multiple of the input width (`0.5` halves it, `2` doubles it). `linear`: halfway between the input width and the size of the next layer |
| `Flatten` | everything of a sample into one vector | — |
| `tokens` | the opposite decision: keep the positions, as `(positions, features)` | — |
| `Dropout` | drops values at random while training | the probability |
| `Reshape` | back to `(lags, features)` after an `lstm`; for older graphs | — |

How a shape is read as tokens (`token_view`):

| what arrives | read as |
| --- | --- |
| a flat vector `(f)` | 1 token of f features |
| a sequence `(n, f)` | n tokens |
| a feature map `(channels, lat, lon)` | one token per grid cell, the channels as its features |
| a weather tensor `(lags, variables, lat, lon)` | one token per grid cell, its lags and variables as features |

`Flatten` or `tokens` is the choice that matters most in a branch: after `Flatten` the network
no longer knows which number came from which lead time or which cell. You rarely have to make
it by hand — under an attention merge the builder turns a branch's `Flatten` into `tokens`
itself ([chapter 4](#4-how-a-network-is-assembled--builderspy)).

### 3.4 The merges — first layer of `output` (and of `hidden_layer`)

| layer | how the branches meet | gives | `size` means |
| --- | --- | --- | --- |
| `concatenate` | every branch is flattened and they are put end to end | flat | — |
| `gated_fusion` | a projection of the first branch is scaled, coordinate by coordinate, by the other; with more than two branches the others are concatenated first | flat, as wide as the first branch | — |
| `gated_fusion_both` | two such gates, one in each direction, end to end | flat | — |
| `cross_attention` | the flat branches **ask**, the sequence branches **answer**: each query takes a weighted mix of the positions | flat: queries × width | the common width (64 when under 8) |
| `cross_attention_both` | the same, plus a pooled summary of the sequences | flat | the same |
| `perceiver` | a small set of learned vectors reads all tokens of all branches; then one learned query per forecast step reads those | flat (default), or one value per step | the common width (64 when under 8) |

**`concatenate`** is the baseline and is hard to beat when every branch is already a short flat
vector. It cannot weigh one lead time or one cell against another: by the time it sees them
they are positions in one long vector.

**`cross_attention`** is what the shipped graphs use. Every branch is read as tokens, brought
to one width (`d_model`, or the size) and given a learned mark saying which branch it is and
where in it. The flat branches — calendar, row data — are the *queries*; the sequence branches
— weather cells or lead times, past steps — are what is attended over, with `heads` (4) and
`attention_dropout` (0.1). In words: *given this hour and this season, which lead times or
which places should I read?* The weights of the last pass are kept (`last_attention`).

* `attention_query` names the branches that ask, when it should not be "all the flat ones".
  It takes branch names; a value that names no branch is read as not set.
* When no branch is flat, one learned query pools the sequences.
* When every branch is flat, the learned query attends over the branches themselves — a weight
  per kind of data, per sample.

*Use it* whenever a branch has kept positions worth choosing among.

**`perceiver`** suits three or more branches of different kinds, and forecasts of many steps.
`latents` (16) learned vectors run `depth` (2) rounds of reading all tokens and then each
other; `horizon_queries` (one per forecast step unless set) then read the latents, so each lead
time gets its own mix of weather cells, past steps and calendar. Its cost depends on the
number of latents, not on the number of tokens, so a large grid and a long past can sit in one
merge.

**The gated fusions** are what `cross_attention` used to be: they see one flat vector per
branch and no positions, so there is nothing to attend *over*. They are kept under their real
name, and are a cheap step up from `concatenate` for two flat branches.

`clip` is not a layer you place. The builder adds it when the family sets `clip_loss_mode`
([chapter 5](#an-extra-loss-that-makes-the-branches-agree)).

### 3.5 Which one, when

| you have | start with | then try |
| --- | --- | --- |
| weather on a grid, a calendar | `conv` → `conv` in `nwp`, `dense` in `row_calendar`, `cross_attention` — the shipped `cnn1` and `mlp1` | `cnn_att` in front; `afno` or `grid_transformer` for the grid |
| a `short_term` case with a measured past | `tide` or `tsmixer` in `lstm` | `timexer`, `lstm_att` |
| weather, a measured past and a calendar together | `afno` → `tokens`, `timexer`, `dense`, merged by `perceiver` — the shipped `perceiver1` | |
| only row data (few inputs, no grid) | `dense` layers in `row_all`, `concatenate` | the CPU families are usually the better tool for this |
| very many samples and a card to spare | a pretrained vision model in `nwp` | |

## 4. How a network is assembled — `builders.py`

`build_graph(x, model_layers, params)` is given one batch of data (for its shapes), the graph
and the trial's parameters, and returns one `torch` module. It works in five steps.

**1. Adapt the branches to their merge.** If the merge is one that attends over positions —
`cross_attention`, `cross_attention_both`, `perceiver` — every `Flatten` in a branch it joins
becomes `tokens`, and the log says so (`nwp: Flatten -> tokens (merged by cross_attention)`).
The dense layers after it then work on each token separately. Under `concatenate` or a gated
fusion the `Flatten` stays. This is why one graph can be tried with either kind of merge
without rewriting its branches (`branches_for_merge`).

**2. Build every branch** (`build_layer`), in the alphabetical order of the tags. Each layer is
built from the output shape of the one before, so a mistake in a graph shows here, at the layer
that cannot take what it is given. Three things are decided for you:

* a `dense` size is turned into a width from the shape that reaches it (the rules of
  [3.3](#33-on-a-row-of-numbers-and-the-helpers));
* an `lstm` on a tensor returns its last step when a `Flatten` follows, otherwise every step;
* the last convolution of a `short_term` branch is given the summary over the lags
  ([3.1](#31-on-the-weather-grid--the-nwp-branch)).

A layer's parameters are the family's, with the layer's own in front
(`_merge_layer_params`).

**3. Join the groups**, if a tag came as several branches — see [below](#grouped-branches).

**4. Build the merge** from the list `output`: its first layer takes all branches, the rest
follow it.

**5. Add the head.** The family's `head` chooses it:

| `head` | what it is | when |
| --- | --- | --- |
| not set, `linear` | one linear layer with one output per forecast step | the default |
| `mlp` | a small network of its own for each step (`head_units` 32) on the shared features | several steps whose behaviour differs — the first hour and the last |
| `moe` | `experts` (4) small heads and a gate that mixes them per sample; a term in the loss keeps all experts in use (`moe_balance` 0.01) | regimes: clear and overcast days, low and high wind |
| `quantile` | every step gets `quantiles` (0.1, 0.5, 0.9 unless set), built so that they cannot cross | you want an interval. The median stands in wherever one number is needed — the study, the checkpoints, the combination |

### A graph, with its shapes

The shipped `cnn1` on a `short_term` case with weather of 16 lags and 2 variables on a
17 × 17 grid, 4 calendar columns and 16 forecast steps. This is what the dry build prints
([chapter 7](#7-trying-a-layer-or-a-graph-and-adding-one)):

```
nwp: Flatten -> tokens (merged by cross_attention)

[nwp] input (16, 2, 17, 17)
   cnn_att               -> (16, 2, 17, 17)          1,091 parameters
   time_distr_conv_2d    -> (16, 32)                 9,120    the last convolution: a vector per lag
   tokens                -> (16, 32)                     0    was Flatten
   dense 512             -> (16, 512)               16,896    on each of the 16 tokens
   dense 128             -> (16, 128)               65,664

[row_calendar] input (4)
   dense 64              -> (64)                       320

[merge]
   cross_attention       -> (64)                    47,168    the calendar asks, the 16 lags answer
   dense 64              -> (64)                     4,160

head: linear, 16 outputs                              1,040    145,459 parameters in all
```

Read from the bottom: sixteen numbers come from 64 features, which are the calendar's question
answered by a weighted mix of sixteen weather summaries, one per lead time.

### Grouped branches

When the weather is read `by_variable` (or by area), the branch `nwp` is built once per group:
`nwp/Cloud`, `nwp/Flux`, … Each has the same layers and its own weights. They are joined in two
stages:

```
nwp/Cloud ─┐
nwp/Flux  ─┼─► join the groups ─► nwp ─┐
nwp/…     ─┘                           ├─► output: merge ─► … ─► head
row_calendar ──────────────────────────┘
```

* The groups of a tag are joined by the list **`hidden_layer`** when the graph has one.
* When it has none, they are joined by the list **`output`** — which is then used twice: once
  for the groups, and once for the final merge. With an attention merge and no flat branch
  among the groups, that first use pools the groups with a learned query.
* A branch that did not come in groups — the calendar — waits for the final merge.

Give a graph a `hidden_layer` when the groups should be joined differently from the final
merge; `[('concatenate', []), ('dense', 64)]` is the plain choice.

> **Networks trained before 15 November 2026.** Until then the final merge of a graph with
> grouped weather took the joined groups and nothing else: a calendar branch was built and
> computed, and never read. Such a network forecasts from the weather alone. Nothing failed,
> and the trial looked like any other. A network saved before the correction is still built
> the way it was trained — its weights fit no other — so its forecasts are what they were;
> whatever is trained from now on uses all its branches. To give an existing case the
> correction, train the affected families again ([troubleshooting](troubleshooting.md#8-after-an-upgrade)).

### Fuzzy rules: the clusterer's network

One network is built differently: the one the clustering stage trains (graph `exp_fuzzy1`,
`global_builders.py`). It gets an extra input, `clustering` — the few variables the rules are
defined on — and an `RBFLayer` turns it into one activation per rule: a bell around the rule's
centre, 1 at the centre and falling with the distance measured in the rule's widths. Every
branch of the graph is repeated once per rule (`row_all/rule_0`, `row_all/rule_1`, …), and each
copy's output is multiplied by the activation of its rule before the merge, so a rule's branch
only speaks for the samples the rule covers. The centres stay where they were first put; the
widths are trained with the rest ([chapter 5](#the-clusterers-schedule)).

## 5. The training loop — `network.py`, `trainer.py`, `optimizers.py`, `horizon.py`

### When it runs

You never call it. `run_training.py` starts the GPU orchestrator; the orchestrator asks each
family's study for the next trial and starts a process for it
(`eforecast/training/command_line/trial_runner.py`) on a card; that process builds a
`DeepNetwork` for the trial's folder and calls `fit`, then predicts the evaluation period and
writes the trial's result. One trial is one network trained once. Which trials run together on a
card, and what happens when one runs out of memory, is in
[gpu_training_optuna.md](gpu_training_optuna.md).

What the study is told about a trial is its **best test MAE** (`best_mae_test`).

### What `fit` does, in order

1. **Nothing, if the trial is already trained** and a refit was not asked for: the stored score
   is returned.
2. **Names the convolutions** for the case's horizon ([3.1](#31-on-the-weather-grid--the-nwp-branch)).
3. **Settles the three sets of dates** — training, validation, test — and writes them beside
   the trial (`training_scope.json`), then loads and cuts the data
   ([chapter 2](#what-happens-between-the-tag-and-the-first-layer)).
4. **Puts the data where the batches are cut from.** If the training data takes less than 60 %
   of the card's free memory it is moved to the card once and every batch is a slice of it;
   otherwise it stays in main memory and each batch is moved
   (`TensorBatches: … GB of inputs stay on the host`). Training batches are shuffled, of
   `batch_size`. Validation and test are read in order.
5. **Fixes the seed** — the trial's `seed`, else the case's `split_seed`.
6. **Builds the network** from the first batch ([chapter 4](#4-how-a-network-is-assembled--builderspy)).
   With `tl_path_weights` it starts from another trial's weights instead of from scratch, and
   skips the first epoch.
7. **Creates the optimizer, the schedule, the loss and the two measures** (`optimize`):

   | parameter | values | what it gives |
   | --- | --- | --- |
   | `optimizer` | `adam`, `adamw` | started at `learning_rate` |
   | | `prodigy` | finds its own step size; `learning_rate` then only sets the floor of the schedule |
   | `scheduler` | `CosineAnnealing` | warm restarts: periods of 3, 6, 12, … epochs, down to a tenth of `learning_rate` |
   | | `ReduceLROnPlateau` | × 0.85 after more than 8 epochs without improvement, 4 epochs of rest after each cut, not below a tenth of `learning_rate`; every 20 epochs the rate is lifted back to 0.8 of where it last started, if it has fallen below that |
   | loss | | mean squared error over the steps; the pinball loss for a quantile head |
   | measures | | the mean absolute error — the MAE of every log line and of the study — and the summed squared error. The target is scaled as the case says (`scale_target_method`), so both are in scaled units: with `maxabs`, an MAE of 0.05 is 5 % of the largest value in the data |

8. **Runs the epochs**, at most `max_iterations` of them.
9. **Writes the result**: the best weights (`net_weights.pt`), everything needed to rebuild the
   network (`net_weights.pickle`), the table of all epochs (`results.csv`), and up to three
   earlier best epochs as `checkpoint_k.pt`. A fit that failed leaves `error.txt` with the
   traceback instead.

### One epoch

```
for every training batch (train_):
    forecast = network(batch)                 in half precision with mixed_precision: true
    loss = squared error  (+ alignment term)  (+ balance term of a moe head)
    gradients of the loss; their norm clipped at 1.0
    one optimizer step

validation set, then test set: MAE and squared error of each, no gradients
the schedule takes its step (the plateau one is given validation MAE + test MAE)
is this epoch the best so far?
    yes: keep a copy of the weights; the previous best becomes a checkpoint (the last 3 are kept)
    no:  wait = wait + 1
report to the study — it may prune the trial here
stop if wait > patience
```

**"Best" is judged on validation and test together.** Each of the two sets is measured in
`CVs` parts. The MAE of all parts (and, separately, their squared errors) are compared with
those of the last accepted epoch, each difference scaled by the range that part has shown so
far; an epoch is accepted when the sum of the differences is below zero for the MAE or for
the squared error. An epoch that improves validation at the cost of test is therefore not
automatically better.

**Stopping.** `patience` epochs without an accepted epoch — `max(8, max_iterations / 5)` when
the family does not set it — or the study pruning the trial, or `max_iterations`. A trial
whose test MAE is still above the case's `max_performance` after 100 epochs ends there with
what it has. A pruned trial keeps its best weights; it is a finished trial with a worse score,
not a failure.

**When the numbers go wrong.** If an epoch produces NaN in validation or test, the best weights
so far are put back and the two sets measured again; if that is NaN too, the trial fails with
`NaN in mae_val or mae_test`. An exception inside a training step is written to `error.txt` and
ends the trial.

**After the loop**, the trial's process predicts the evaluation period with the final weights
and with each checkpoint, and keeps whichever is best on the average of that error and the
training measures. So the weights a trial ends with are not always those of its last accepted
epoch.

### What you can set

All of these are parameters of the family (Methods tab), and the study can search any of them.

| parameter | default | effect |
| --- | --- | --- |
| `batch_size`, `max_iterations`, `learning_rate` | per family | as named |
| `patience` | `max(8, max_iterations / 5)` | epochs without an accepted epoch before stopping |
| `optimizer`, `scheduler` | per family | the table above |
| `mixed_precision` | off | the forward pass in half precision on a card: less memory and time for convolutions, numbers that are not those of a full-precision fit. Weights stay 32-bit |
| `horizon_weights` | none | a weight per forecast step in the loss, below |
| `head`, `quantiles`, `experts`, `head_units` | `linear` | the head ([chapter 4](#4-how-a-network-is-assembled--builderspy)) |
| `clip_loss_mode` | off | the alignment term, below |
| `seed` | the case's `split_seed` | initial weights and the order of the batches |
| `conv_dim` | 2 | 2-D or 3-D convolutions |

### A weight per forecast step

The error of a forecast grows with the lead time, so a plain squared error is mostly the error
of the last steps, and the network is trained mostly for them. `horizon_weights` gives each
step a weight proportional to 1 / (a reference error of that step)², scaled so that the weights
average one:

| value | the reference error of a step |
| --- | --- |
| not set | none: plain mean squared error |
| `persistence` | the error of "the last observed value holds", computed once from the training targets. Needs rows that follow each other in time; falls back to `model` otherwise |
| `model` | the network's own error on the training batches, smoothed over the epochs |
| a list of numbers | those weights |

`horizon_weight_floor` (0.05) bounds how small a reference error may be against the largest,
so no step weighs more than 400 times another. It only applies to forecasts of several steps.

### An extra loss that makes the branches agree

With `clip_loss_mode` the builder adds a `Clip` layer over the branches of the final merge:
each branch output is projected to `clip_dim` (128) numbers, and for every pair of branches the
similarity of every sample with every other sample of the batch is computed. The term added to
the loss rewards a sample's weather features for resembling **its own** calendar features more
than those of other samples — the idea of CLIP, between kinds of data instead of image and
text. Samples closer in time than `clip_neighbour_window` (the number of forecast steps unless
set) are not used against each other: they are nearly the same sample.

`clip_loss_mode`: `infonce` (the CLIP loss), `soft_margin`, `decoupled` or `barlow`.
`clip_loss_weight` (1.0) scales the term. It needs at least two branches at the final merge.

### The clusterer's schedule

The network of [fuzzy rules](#fuzzy-rules-the-clusterers-network) is trained in turns, because
the rules and the networks under them pull against each other (`train_schedule_fuzzy`). One
epoch of the loop is four passes over the training data: one that moves only the widths of
the rules, with an optimizer of their own, then three that train everything else with the
widths held. After every epoch the rules are checked against the data — how many samples
each covers, and whether a plain linear model on the activations still forecasts. With
`explode_clusters`, when that check fails the widths are set to a quarter more than the best
seen, and the next epoch begins with three extra passes of the networks alone. Mixed
precision is never used here.

### Predicting

`predict` loads the tags of the graph again for the dates asked, rebuilds the network from the
saved parameters, loads the best weights and runs it in batches without gradients. A quantile
head returns its median and keeps every quantile in `quantile_predictions`. The combine step
and the daily forecast both go through it.

## 6. The cross-site graph — `site_graph_net.py`

Everything above is one network for one case. The site graph is the one model that looks at
**several cases at once**: one network forecasts a group of plants together, each plant a node,
and lets every plant see what its neighbours are observing and forecasting at the same moment —
the one thing a per-site model cannot have. A front moving across a region reaches one wind
farm before the next; clouds over one park are over its neighbour a quarter of an hour later.

Each plant stays a case of its own, with its datasets, its scalers and its own models. The
graph is added to each of them as **one more candidate**: the global method `GRAPH`, which the
case's ensemble selects, combines and calibrates like any trial.

### What it reads

| | where it comes from |
| --- | --- |
| the sites | the case files given to `--configs`, two or more. A site's name is its `project_name` |
| a node's features | that site's own row data, already scaled by the site's scalers — the tag `--data-tag` (default `row_all`), no feature selection |
| the features in common | only the columns every site has, after the site's name is stripped from them. Fewer than half shared: the columns are matched by position instead |
| the timestamps | only those every site has |
| the target | each site's own scaled target; every site must have the same number of forecast steps |
| where the sites are | each case's `coord` (the centre of an area) |
| optionally, a weather grid per site | `--nodes nwp` or `both`: the site's scaled NWP tensor, thinned over the lags (`--nwp-lag-step`) and pooled (`--nwp-pool`); all sites need grids of one shape |
| the split | each site's **own** frozen partition. A timestamp trains the graph only if it is a training timestamp of *every* site; any day that is evaluation for some site, or that a site has in two parts, is left out. The graph never draws a partition of its own |

There is no list of edges. Every site attends to every site; how far apart two sites are enters
as a prior on that attention, and `--k` cuts it down to the k nearest.

### The network

Shapes: `b` timestamps in a batch, `N` sites, `F` features, `n_out` forecast steps.

```
features (b, N, F) ──► shared MLP ──┐
                                    ├─► + a learned embedding per site ──► (b, N, hidden)
grid (b, N, lags, vars, lat, lon) ──► shared small CNN ──┘ (with --nodes nwp|both)

repeat `depth` times:   h ← h + attention over the sites( norm(h) )
                        h ← h + feed-forward( norm(h) )

norm ──► one shared linear head ──► (b, N, n_out)      clipped at 0 afterwards
```

* **`SiteAttention`** is multi-head self-attention *over the sites of one timestamp*. To the
  usual attention score of two sites it adds `−d² / (2·l²)`, where `d` is their distance in km
  and `l` a length scale **learned per head** — started at `--length-scale`, or at the median
  distance between the sites. So a head begins by listening to near neighbours and is free to
  learn another range. The attention of the last forward pass is kept (`last_attention`) and
  written to `meta.json`: it shows which sites the model used for which.
* **`GridEncoder`** is one small CNN shared by all sites: two 3×3 convolutions, pooling,
  a linear layer.
* **`SiteGraphNet`** puts them together as drawn above. Defaults: `hidden` 64, `depth` 2,
  `heads` 4, `dropout` 0.1.
* **`SiteGraphModel`** is a trained graph on disk — `site_graph.pt` and `meta.json` — with the
  methods that predict from it.

The set of sites is fixed by training: the embedding and the distances have one row per site,
so a prediction needs all of them.

### How it is trained

AdamW (learning rate 1e-3, weight decay 1e-4), mean squared error over all sites and steps in
scaled units, gradients clipped at norm 1. A batch is 256 timestamps with every site of each.
Validation MAE is computed after every epoch; the learning rate is halved when it stops
improving, training stops after `--patience` (10) epochs without improvement, and the best
weights are put back. At most `--epochs` (100). The seed is fixed.

It runs in a process of its own, on the CPU unless `--gpu` names a card (`0`, `cuda:1`,
`auto`, a GPU id). While it runs it holds every site's case lock and the group folder's, so no
other run can write those cases meanwhile.

The result goes to `<group folder>/site_graph/<--name>/`: the weights, `meta.json` (sites,
coordinates, features, options, distances, attention, validation and test MAE per site),
`history.csv` (loss and validation MAE per epoch, per site too) and the scope it trained on.
Each run trains a new network and publishes a new generation; there is no resume.

### How and when it is executed

**Only a person starts the training** — from a terminal. The web pages do not offer it, and
`run_training.py` does not train it.

```bash
# 1. every site first needs its datasets, scalers and frozen partition
python run_datasets.py --config <case> --stage all

# 2. train the graph over the sites, and export it into each of them
python run_site_graph.py --configs caseA.yaml caseB.yaml caseC.yaml --gpu 0 --export

# 3. in each case, switch the global method GRAPH on (Methods tab), then combine as usual
python run_combine.py --config caseA.yaml
```

The sites' own trained models are not needed for step 2. `--export` writes into every site's
model folder a trial `global/GRAPH/test_0` holding the graph's predictions for that site's
evaluation period and a marker that names the trained graph. From then on the case treats
`GRAPH` like any other global method: `run_combine.py` reads those predictions as they are,
and selection and the combiners decide how much weight it gets.

**In a daily forecast it runs by itself.** A case with a `GRAPH` export forecasts with it:
the forecast loads the graph, builds the online inputs of *every* site of the group, and takes
its own node's output. The neighbour cases must therefore still be where they were when the
graph was trained.

Other forms of the command:

```bash
python run_site_graph.py --configs … --data-tag row_obs_calendar --k 5      # other inputs, 5 nearest sites
python run_site_graph.py --configs … --nodes both --nwp-pool 2              # with a weather grid per site
python run_site_graph.py --predict 2026-03-01 2026-03-02 --model <group>/site_graph/graph1 --csv out
                                                                            # forecast all sites for those days
python run_site_graph.py --recover-publication <group>/site_graph/graph1    # after an interrupted export
```

| option | default | meaning |
| --- | --- | --- |
| `--configs` | — | the case files, two or more |
| `--name` | `graph1` | the folder under `<group>/site_graph/` |
| `--data-tag` | `row_all` | which row features a node gets |
| `--nodes` | `row` | `row`, `nwp` (the grid only) or `both` |
| `--nwp-pool`, `--nwp-lag-step` | 2, 1 | pooling of the grid; keep every n-th lag |
| `--merge`, `--compress`, `--scale-row`, `--scale-nwp` | `all`, `dense`, `minmax`, `minmax` | how the site's inputs are read |
| `--hidden`, `--depth`, `--heads`, `--dropout` | 64, 2, 4, 0.1 | the network |
| `--k`, `--length-scale` | none | attend to the k nearest sites only; the starting range of the distance prior, km |
| `--epochs`, `--batch-size`, `--learning-rate`, `--patience` | 100, 256, 1e-3, 10 | training |
| `--gpu` | `cpu` | where it trains |
| `--export` | off | write `global/GRAPH/test_0` into every site |
| `--out`, `--model` | — | the model folder (for training; for `--predict`) |
| `--threads` | the cases' `n_cpus` | CPU threads |

### What to keep in mind

* All sites need the same number of forecast steps, shared timestamps, and — for grid nodes —
  grids of one shape. Resolution, target lags and weather variables are **not** checked:
  features that are not shared are simply dropped. Use sites that are configured alike.
* With `GRAPH` switched on and no export, a case just has no such trial; with nothing else it
  reports `no admissible completed model`.
* What it is worth is decided where everything else is: in each site's `Results/` after
  `run_combine.py`. The tests cover the mechanics on small generated cases, not accuracy on
  real plants.

## 7. Trying a layer or a graph, and adding one

Three tools, from the quickest to the most real. None of them touches a study.

| tool | what it runs on | what it tells you | time |
| --- | --- | --- | --- |
| the **dry build** in the graph editor | zeros in the shapes of your case | whether the graph can be built, the shape and the parameter count after every layer, which branches ask and which answer in an attention merge, and parameters that hold a value the layer does not take | under a second |
| `python debug_layer.py` | the same zeros, from a terminal | the same, for a single branch or a whole graph, plus a forward and a backward pass: NaN, and parameters that received no gradient | seconds |
| `python debug_experiment.py` | **the case's own datasets** | one real batch through the graph with the trial's scaling, merge and feature selection; with `--epochs N`, the real training loop for N epochs; with `--batches N`, the time per batch on a card | a minute or more |

```bash
# one branch: layers as name[:size], separated by commas
python debug_layer.py --branch nwp --layers "conv:3,conv:3,tokens,dense:64"
python debug_layer.py --branch lstm --layers "lstm_att:2,dense:32" --method LSTM
# that branch merged with a flat one
python debug_layer.py --branch nwp --layers "conv:3,tokens,dense:64" --with "row_calendar=dense:32" --merge cross_attention:64
# a whole graph of the case, as the editor's dry build
python debug_layer.py --config <case> --tag cnn1 --method CNN
# layer parameters and family parameters
python debug_layer.py --branch nwp --layers "grid_transformer:64" --layer-params '{"depth": 1, "heads": 2}' --params '{"act_func": "gelu"}'

# a graph on the case's real data; then five epochs of the real loop on card 0
python debug_experiment.py --config <case> --tag mlp1 --method MLP
python debug_experiment.py --config <case> --tag lstm1 --method LSTM --cluster RBF_rule_0 --epochs 5 --gpu 0
```

`debug_layer.py` takes its shapes from the case given with `--config` (or `$EFORECAST_CASE`),
and uses placeholder shapes when there is none; `--shape 6,4,17,17` sets them for the branch
under test. `debug_experiment.py` needs the
datasets and the clustering of the case (`run_datasets.py --stage all`) and writes under
`<cluster folder>/<method>/debug_<tag>`, never into a trial of a study.

The finding to look for is a parameter that received no gradient: a layer without one is a
layer the forecast does not depend on.

### Adding a layer

1. **Write the class** in `layers.py`, with the builder's contract:

   ```python
   class my_layer(nn.Module):
       def __init__(self, shape, params, size, name_scope, layer_id, train=True):
           super().__init__()
           self.name = f'{name_scope}_my_layer_{layer_id}'
           ...                                  # modules, from `shape` (the batch comes first; it is 10 here)
           self.apply(_init_weights)
           with torch.no_grad():                # the builder reads this to build the next layer
               self.output_shape = list(self.forward(torch.zeros(10, *shape[1:])).shape)

       def forward(self, x):
           ...
   ```

   `shape` is a list (a dictionary for the `lstm` branch, and for a merge: one shape per
   branch), `size` is the value written in the graph, `params` are the family's parameters with
   the layer's own in front.
2. **Register it** in `layers_func()` under the name graphs will use.
3. **Describe it** in `eforecast/config/layer_catalogue.py`: inputs, what the size means, its
   parameters, a sentence. The editor's menu and guide read that, and
   `tests/test_graph_service.py::test_catalogue_covers_registry` fails until it is there.
4. **Try it** in the order of the table above — `debug_layer.py`, then a graph through
   `--tag`, then `debug_experiment.py` on real data — and only then in a study:

   ```bash
   python run_training.py --config <case> --prepare-only      # writes one trial per study, and prints the command for each
   python eforecast/training/command_line/trial_runner.py --path-weights <cluster folder>/<method>/test_0 --gpu 0
   ```

Remember what a new graph costs on a trained case
([the note in chapter 1](#how-a-graph-is-written)).

---

Related: [gpu_training_optuna.md](gpu_training_optuna.md) for how trials share the cards,
[combine_evaluation.md](combine_evaluation.md) for what happens to a trained network afterwards,
[how_it_works.md](how_it_works.md) for where training sits in the whole.
