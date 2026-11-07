# Reproduction

## Environment

Use Julia 1.12.6 and run commands from the repository root. `make setup`
initializes the input/output directories and installs the dependency versions
in the Julia environment. Override the Julia executable with
`make reproduce JULIA=/path/to/julia`.

Without GNU Make:

```sh
julia --project=. scripts/init.jl
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --startup-file=no --threads=1 --project=. test/runtests.jl
julia --startup-file=no --threads=1 --project=. experiments/synthetic/run.jl
```

## Synthetic comparisons

```sh
make synthetic
```

The ranking experiment uses 40 players with true skills `1:40`, 1,000 uniformly
sampled non-self comparisons, seed 123, and corruption levels `0:0.02:0.20`.
The higher-skilled player wins each uncorrupted comparison; there are no draws.

- **Match corruption:** flip exactly `round(delta*n)` outcomes.
- **Player inconsistency:** choose `round(sqrt(delta)*k/2)` noisy players and
  flip incident outcomes with probability `sqrt(delta)`. Finite player counts
  mean that the realized corruption rate can differ from the nominal delta.

The skill prior is `N(k/2,(k/6)^2)`, vanilla performance-noise variance is
`(k/12)^2`, and the draw margin is zero. Both extensions use Gamma shape 1,
rate 1 precision priors and `s2=1`. `Gamma1D` stores shape minus one, then rate.

All comparisons are fitted. Kendall tau measures recovery of the true skill
order. AUROC measures flipped-match and noisy-player detection. At zero
corruption there are no positive detection examples, so the figure omits
the AUROC point and the CSV records `NaN`. The console prints 0.5 for this case.
The paper figure uses the default outer sweep caps of 200 for Robust and 100
for Consistent TrueSkill. At those caps, 21 of 44 extension fits stopped
above the residual tolerance. A sensitivity run with a 1,000-sweep cap
met the tolerance for all 44 extension fits on the same synthetic data.
The script writes
`output/synthetic/metrics.csv`, `corruption.png`, and `corruption.pdf`.

## Chess

### Source and preparation

The paper's source is [ChessBase Big Database 2025](https://shop.chessbase.com/en/products/big_database_2025),
a commercial chess database. Use a licensed copy of that edition. In ChessBase,
select blitz games for 2022–2024 (611,057 games, 13,037 players), or 2016–2024
(1,052,371 games, 26,149 players), and export the selected games as plain text.

```sh
make prepare-chess CHESS_EXPORT=/path/blitz.txt
```

This writes `chess_matches.csv` and `chess_players.csv` to
`output/preprocessed/`; set `CHESS_PREPARED=/path/directory` to choose another
destination. The parser expects ChessBase text blocks containing:

```text
Player A (1500) - Player B (1600) [A00]
Event name 01.01.2023
1. e4 e5 1-0
```

Results may be `1-0`, `0-1`, or `1/2-1/2`. Select the blitz time-control class
before exporting; the text parser does not infer it. This format differs from
standard bracket-header PGN.

### CSV inputs

| File | Required columns |
| --- | --- |
| Matches | `year,white_name,black_name,winner_name,is_draw` |
| Players | `name,max_elo` |

Years are integers, player identities are consistent strings, and the winner
must equal a participant on decisive games. Draw flags are booleans. Match CSVs
can additionally contain `date,match_id,event,white_elo,black_elo,result,eco`.
Dates such as `2023.01.01` are supported. Rows are filtered to the requested year
range, invalid outcomes and self-matches are removed, and rows are sorted by
year, date, and match ID when available.

### Fitting and figures

```sh
make chess CHESS_MATCHES=output/preprocessed/chess_matches.csv CHESS_PLAYERS=output/preprocessed/chess_players.csv
make figure-chess CHESS_MATCHES=output/preprocessed/chess_matches.csv
```

Use `FIRST_YEAR=2016 LAST_YEAR=2024` on `make chess` for the longer period.
The separate figure command fits the consistent model on 2022–2024 and writes
the vector PDF, PNG, plotted values, and fit diagnostics to `output/chess/figure/`.
Supply the recorded `white_elo` and `black_elo` columns to reproduce the Elo and
upset statistics.

Use `make figure-chess` for the camera-ready `blitz.pdf`. It shows the ten
highest posterior mean skills among players with at least 20 games and a posterior skill standard deviation of
at most 15 rating points. All games and players contribute to the fit.
`make chess` writes general fit diagnostics under their own filenames.

Settings: skill prior `N(1500,350^2)`, beta=220, nu=4, and the empirical draw
margin. Performance precision uses Gamma shape `nu/2`, rate
`target_variance*(nu/2-1)`, with `s2=1`; the robust target variance is
`2*beta^2` and the player-specific target variance is `beta^2`. Fits allow 100
inner/outer iterations and export convergence diagnostics. Skill bars show
three posterior standard deviations. The performance marker is the square root of the posterior
expected residual variance, in rating points. Its bars are the equal-tail
95% approximate posterior interval for the residual standard deviation.
Mean Elo averages available recorded ratings. Upset % counts wins over
higher-rated opponents and losses to lower-rated opponents together, as a share
of the player's decisive games with two recorded, unequal Elo ratings.
The CSV also includes the separate counts and rates.

The match CSV used for the paper figure has SHA-256
`1239c521689a6bb1f3778ecfb8ae74335217046110bb69a76c5da46e88eb2471`.
The fit reached the residual tolerance of `1e-4` after 64 outer iterations.

## Compar:IA

Run `make figure-comparia COMPARIA_INPUT=/path/to/comparia-fr-arena.parquet` with
the [pinned snapshot](https://huggingface.co/datasets/ministere-culture/comparia-fr-arena/tree/fd92db9d3b2f7b18c6024367711a4cabb16e7a34).
Accept the upstream access conditions on Hugging Face before downloading.
The dataset card lists Etalab 2.0 and CC BY 4.0, subject to third-party
rights on model outputs. The reproduction reads model IDs, vote labels,
conversation IDs, turn numbers, and timestamps.
The command verifies the Parquet SHA-256 and uses only votes from 2026.
The pinned snapshot covers January 1 through June 15 within that year.
It keeps decisive votes and `both_good` or `both_bad` votes between distinct
models, maps both verdicts to draws, removes duplicate rows and ambiguous turn
keys, and retains the highest eligible turn per conversation. Model identities
must have at least 100 votes in the retained comparison graph.
The figure fits all 60,010 eligible conversations among 66 model identities.

Consistent TrueSkill uses a skill prior of N(25, (25/3)^2), a shape–rate
precision prior of Gamma(16, 16), and a baseline performance scale of
16 rating points. The draw margin comes from the observed draw frequency.
The scale and precision prior were fixed from the exploratory 2026
analysis before changing the unit to one vote per conversation. The paper
shows the 15 identities with the highest posterior mean skill.
`protocol.csv` records the split counts and source checksum. `model_values.csv`
contains the exact model IDs and estimates.

## LMArena

### Skill and consistency figure

```sh
make figure-arena
```

This fits Consistent TrueSkill to all eligible Human Preference-140k votes
and shows the 15 model identities with the highest posterior mean skill.
Use `ARENA_INPUT=/path/to/parquet` for existing shards, or provide the CSV
projection described below. The command verifies the ordered input fingerprint.
It retains 135,622 labeled comparisons among 52 model identities, including
`tie` and `both_bad` as draws. Each identity has at least 100 comparisons
in the filtered snapshot. All labeled comparisons contribute to the fit,
including multiple votes from an evaluation session.
The pinned snapshot covers April 17 through July 24, 2025.

The skill prior is N(25, (25/3)^2), the shape–rate precision prior is
Gamma(1.25, 1.25), and the baseline performance scale is 40 rating points.
The draw margin is computed from all fitting votes. These settings match
the Consistent TrueSkill settings in the calibrated analysis below.
Outputs are written to `output/arena_figure/human_preference_140k/`.

### Archived calibrated comparison

```sh
make arena-paper
make arena-paper PAPER_DATASET=human_preference_55k ARENA_INPUT=/path/to/train.csv
```

The first command uses the pinned [Human Preference-140k snapshot](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k/tree/6322995ab34d7c2693e3f47dd13fa5caa0789a74).
Set `ARENA_INPUT=/path/to/parquet` to reuse its local shards. A CSV projection
of the same pinned rows with `timestamp,model_a,model_b,winner,evaluation_session_id`
is also accepted. Keep source row order and truncate timestamps to milliseconds
to match the Parquet loader. For the second
command, use [Arena 55k's pinned CSV](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/blob/18c298340948c0e7f7727399fd459cca6ce0ca6f/train.csv)
with SHA-256 `0692154aaf20fc6649090c3f49b6b5dd1e693a765ecb5f0856ae0da5946f5be2`.
140k prompts are CC BY 4.0, with responses governed by the model providers'
terms. The 55k dataset card specifies Apache-2.0. Obtain both from their
upstream source.

The split seed is 42. 140k keeps evaluation sessions together and retains
models with at least 100 votes in the source data. 55k groups by the hash of
the first prompt and retains models with at least 100 training votes. The
program checks the ordered input fingerprint and group separation. All ratings
use training votes only. Frozen `(beta, nu)` settings are `(24, –), (24, 10),
(40, 2.5)` for 140k and `(40, –), (40, 6), (40, 32)` for 55k, in vanilla,
robust, consistent order. Every model gets inverse-temperature and tie-intercept
calibration on validation, and predictive integration uses 64 quadrature nodes.
The paired bootstrap resamples held-out sessions or prompt groups 5,000 times,
conditional on the fitted models.

Outputs are in `output/arena_paper/<dataset>/`. The 140k figure displays the
15 model identities with the most training votes, ordered by fitted skill.
Both panels use model rating points. The CSV also records fitted Elo, which comes
from a Bradley–Terry fit on training votes, with half credit for each side
on ties or both-bad votes and mean rating anchored at 1,000. The CSV contains
all 52 model identities.

The CSV reports each model's upset percentage among decisive
held-out votes against opponents with a different unrounded fitted Elo.
A lower-rated win counts as an upset for both models involved. Elo uses
training votes only. The CSV gives both the numerator and denominator.
The other Arena commands below cover the exploratory protocol.

```sh
make arena DATASET=human_preference_140k
make arena DATASET=expert_5k
make arena DATASET=search_arena_24k
```

Sources:
[Human Preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k),
[Expert-5k](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k), and
[Search Arena](https://huggingface.co/datasets/lmarena-ai/search-arena-24k).
The dataset cards specify CC BY 4.0 for user prompts, with model outputs subject
to their providers' terms. Download revisions and shard URLs are fixed in
`src/datasets/arena.jl`. Files are downloaded into `data/arena/parquet/` when
needed. To reuse a download, set `HTS_ARENA_DATA_DIR` to a directory containing
the `human_preference_140k`, `expert_5k`, and/or `search_arena_24k` subdirectories.

Only labeled, non-self comparisons are retained. Models require at least 20
incident votes for Expert/Search or 100 for Human Preference, counted before
splitting. Identities are
categorical: response text and embeddings are not model features. `tie` and
`both_bad` votes are treated as draws.

The target split is 70% training, 15% validation, and 15% test, using seed 42.
Human Preference keeps complete `evaluation_session_id` groups together. The
Expert and Search entry points use vote-level splits. The final fit uses the
training and validation partitions, while predictive metrics are evaluated on
test. The draw margin is computed from fitting outcomes.

| Dataset | Vanilla beta | Robust beta / nu | Consistent beta / nu |
| --- | ---: | ---: | ---: |
| Human Preference-140k | 24 | 24 / 10 | 40 / 2.5 |
| Expert-5k | 20 | 24 / 32 | 20 / 128 |
| Search Arena | 24 | 24 / 6 | 24 / 16 |

The skill prior is `N(25,(25/3)^2)`. Normalized precision priors have Gamma
shape=rate=nu/2. Robust `s2=2*beta^2*(nu-2)/nu`; consistent `s2=beta^2`.
Final fits allow 60 inner/outer iterations. NLL integrates Gaussian skill
uncertainty and Gamma precision with 16-node Gauss–Laguerre quadrature. Accuracy
uses win/draw/loss predictions; Elo and Bradley–Terry provide ranking comparisons.
Outputs are saved under `output/arena/DATASET/selected_models/`.

## Inference

The models combine EP updates for outcome factors with variational updates for
Gaussian–Gamma factors. The Gaussian/Gamma posteriors are approximate, and the
combined schedule has no joint convergence guarantee. Numerical tests check
distribution updates, prediction semantics, data parsing, and experiment entry
points. Run them with `make test`.
