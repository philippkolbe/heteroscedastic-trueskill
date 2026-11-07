# Heteroscedastic TrueSkill

Code for **[Heteroscedastic TrueSkill: Modeling Match Noise and Player Consistency](https://openreview.net/forum?id=gNc29aLken)**, accepted at NeurIPS 2026.

**Authors:** Philipp Kolbe, Johann Ukrow, Anna Kazachkova, Rainer Schlosser, and Ralf Herbrich.

[TrueSkill](https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html)
(Herbrich, Minka, and Graepel, 2006) is a Bayesian rating system that estimates
player skill from match outcomes. This implementation provides vanilla TrueSkill
and two extensions: **robust TrueSkill**, with match-specific noise precision,
and **consistent TrueSkill**, with player-specific noise precision.

## Getting started

Install [Julia 1.12.6](https://julialang.org/downloads/) and GNU Make, then run:

```sh
make reproduce
```

This sets up the environment, runs the tests, and reproduces the synthetic
experiment. Results are saved in `output/synthetic/`. Use `make setup` to install
dependencies without running experiments, or `make help` to list all commands.

## Paper experiments

### Synthetic comparisons

The paper experiment generates **1,000 comparisons among 40 players**, using
seed 123 and corruption levels from 0% to 20%. It tests both corrupted matches
and inconsistent players, reporting skill-ranking recovery and noise detection.

```sh
make synthetic
```

### Blitz chess

The paper uses blitz games from **[ChessBase Big Database 2025](https://shop.chessbase.com/en/products/big_database_2025)**:

| Period | Games | Players |
| --- | ---: | ---: |
| 2022–2024 | 611,057 | 13,037 |
| 2016–2024 | 1,052,371 | 26,149 |

The database has a **commercial ChessBase license**. Use a licensed copy of this
edition and export the blitz games for the desired period. The preparation script
converts ChessBase plain-text exports into the match and player CSVs:

```sh
make prepare-chess CHESS_EXPORT=/path/blitz.txt
make chess CHESS_MATCHES=output/preprocessed/chess_matches.csv CHESS_PLAYERS=output/preprocessed/chess_players.csv
make figure-chess CHESS_MATCHES=output/preprocessed/chess_matches.csv
```

The default period is 2022–2024. Set `FIRST_YEAR=2016 LAST_YEAR=2024` on `make chess` for the
longer period. CSV schemas and export format are described in
[reproduction instructions](docs/reproduction.md#chess).

### Compar:IA human preferences

Download `comparia-fr-arena.parquet` from the [pinned Compar:IA release](https://huggingface.co/datasets/ministere-culture/comparia-fr-arena/tree/fd92db9d3b2f7b18c6024367711a4cabb16e7a34), then run:

```sh
make figure-comparia COMPARIA_INPUT=/path/to/comparia-fr-arena.parquet
```

The command uses only votes from 2026 and writes the paper figure, model
estimates, and cohort summary to `output/comparia_paper/`. See the
[Compar:IA protocol](docs/reproduction.md#comparia) for filtering and fit settings.

### LMArena human preferences

```sh
make figure-arena
```

The command downloads the pinned Human Preference-140k snapshot and writes
the figure and estimates to `output/arena_figure/human_preference_140k/`.
Set `ARENA_INPUT=/path/to/parquet` to use an existing download.
Both preference figures fit all eligible votes and show the 15 model
identities with the highest posterior mean skill.

## Additional experiments

The reproduction guide lists commands for [LMArena evaluations](docs/reproduction.md#lmarena).

## Code layout

```text
src/
  trueskill.jl          module entry point
  distributions/       Gaussian/Gamma distributions and belief containers
  inference/           factor messages and numerical functions
  models/              vanilla, robust, and consistent inference schedules
  evaluation/          predictive probabilities and rating baselines
  datasets/            chess and Arena data pipelines
  plots/               comparison and chess figures
  synthetic/           comparison generation and corruption
experiments/
  synthetic/           synthetic ranking experiment
  chess/               preprocessing, model fitting, and figures
  arena/               preference evaluation and validation selection
test/                  numerical and input-format tests
docs/reproduction.md   parameters, schemas, and detailed commands
```

## Citation

```bibtex
@inproceedings{ukrow2026heteroscedastic,
  title     = {Heteroscedastic {TrueSkill}: Modeling Match Noise and Player Consistency},
  author    = {Kolbe, Philipp and Ukrow, Johann and Kazachkova, Anna and Schlosser, Rainer and Herbrich, Ralf},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Accepted at NeurIPS 2026},
  url       = {https://openreview.net/forum?id=gNc29aLken}
}
```

Original TrueSkill reference: Ralf Herbrich, Tom Minka, and Thore Graepel.
**[TrueSkill: A Bayesian Skill Rating System](https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html)**.
Advances in Neural Information Processing Systems 19, 2006.

## License

The code is released under the [Apache License 2.0](LICENSE).
See the [reproduction guide](docs/reproduction.md) for dataset terms.
