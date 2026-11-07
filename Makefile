.DEFAULT_GOAL := help

JULIA ?= julia
JULIA_FLAGS ?= --startup-file=no --threads=1 --project=.
export GKSwstype ?= 100
DATASET ?= expert_5k
PAPER_DATASET ?= human_preference_140k
ARENA_INPUT ?=
COMPARIA_INPUT ?=
FIRST_YEAR ?= 2022
LAST_YEAR ?= 2024
CHESS_EXPORT ?=
CHESS_PREPARED ?= output/preprocessed
CHESS_MATCHES ?=
CHESS_PLAYERS ?=

.PHONY: help init setup test synthetic reproduce arena chess prepare-chess figure-chess \
        figure-comparia figure-arena comparia-paper arena-paper

help:
	$(info make setup       - initialize directories and install Julia dependencies)
	$(info make reproduce   - setup, tests, and the paper synthetic experiment)
	$(info make test        - run numerical and data-format tests)
	$(info make synthetic   - run the paper synthetic ranking experiment)
	$(info make prepare-chess CHESS_EXPORT=... - prepare ChessBase text exports)
	$(info make chess CHESS_MATCHES=... CHESS_PLAYERS=... - fit chess models)
	$(info make figure-chess CHESS_MATCHES=... - generate the paper's chess figure)
	$(info make figure-comparia COMPARIA_INPUT=... - generate the paper's Compar:IA figure)
	$(info make figure-arena ARENA_INPUT=... - generate the LMArena figure)
	$(info Additional experiments)
	$(info make arena DATASET=human_preference_140k - download and evaluate LMArena data)
	$(info make arena-paper  - reproduce the archived Human Preference-140k analysis)

init:
	"$(JULIA)" $(JULIA_FLAGS) scripts/init.jl

setup: init
	"$(JULIA)" $(JULIA_FLAGS) -e "import Pkg; Pkg.instantiate()"

test: setup
	"$(JULIA)" $(JULIA_FLAGS) test/runtests.jl

synthetic: setup
	"$(JULIA)" $(JULIA_FLAGS) experiments/synthetic/run.jl

reproduce: test synthetic

arena: setup
	"$(JULIA)" $(JULIA_FLAGS) experiments/arena/run.jl "$(DATASET)"

prepare-chess: setup
	$(if $(strip $(CHESS_EXPORT)),,$(error Set CHESS_EXPORT to your ChessBase text export))
	"$(JULIA)" $(JULIA_FLAGS) experiments/chess/prepare.jl "$(CHESS_EXPORT)" "$(CHESS_PREPARED)"

chess: setup
	$(if $(strip $(CHESS_MATCHES)),,$(error Set CHESS_MATCHES to your matches CSV))
	$(if $(strip $(CHESS_PLAYERS)),,$(error Set CHESS_PLAYERS to your players CSV))
	"$(JULIA)" $(JULIA_FLAGS) experiments/chess/run.jl "$(CHESS_MATCHES)" "$(CHESS_PLAYERS)" "$(FIRST_YEAR)" "$(LAST_YEAR)"

figure-chess: setup
	$(if $(strip $(CHESS_MATCHES)),,$(error Set CHESS_MATCHES to your matches CSV))
	"$(JULIA)" $(JULIA_FLAGS) experiments/chess/figure.jl "$(CHESS_MATCHES)"

arena-paper: setup
	"$(JULIA)" $(JULIA_FLAGS) experiments/arena/paper.jl "$(PAPER_DATASET)" $(if $(strip $(ARENA_INPUT)),"$(ARENA_INPUT)")

comparia-paper: figure-comparia

figure-comparia: setup
	$(if $(strip $(COMPARIA_INPUT)),,$(error Set COMPARIA_INPUT to the pinned comparia-fr-arena.parquet))
	"$(JULIA)" $(JULIA_FLAGS) experiments/arena/comparia.jl "$(COMPARIA_INPUT)"

figure-arena: setup
	"$(JULIA)" $(JULIA_FLAGS) experiments/arena/figure.jl $(if $(strip $(ARENA_INPUT)),"$(ARENA_INPUT)")
