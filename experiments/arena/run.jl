#=LMArena human-preference experiments. Pass expert_5k, human_preference_140k, or search_arena_24k.=#
include(joinpath(@__DIR__, "fit_selected.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    dataset = isempty(ARGS) ? :expert_5k : Symbol(only(ARGS))
    haskey(SELECTED_SETTINGS, dataset) || error("Unknown dataset: $dataset")
    run_fixed_predictive_llm_arena(dataset=dataset,
        min_games=dataset === :human_preference_140k ? 100 : 20,
        output_dir=joinpath(DEFAULT_OUTPUT_DIR, String(dataset), "selected_models"),
        settings_override=SELECTED_SETTINGS[dataset])
end
