# Create local input/output directories without downloading any data.
root = normpath(joinpath(@__DIR__, ".."))
for relative in ("output", joinpath("data", "chess"), joinpath("data", "arena", "parquet"))
    mkpath(joinpath(root, relative))
end
println("Local input/output directories initialized. No data downloaded.")
