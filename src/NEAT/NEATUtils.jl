#=
# Refer src/NEAT/NEAT.jl for NEAT
# Refer src/NEAT/NEATConfig.jl for NEATConfig, LogConfig
# Refer src/NEAT/Network/Network.jl for Network
# Refer src/NEAT/Network/NetworkConfig.jl for NetworkConfig
# Refer src/NEAT/Network/NetworkUtils.jl for GetNetworkDistance
=#

using Dates, JLD2
export NEAT, Log, Init, Evaluate, RemoveStagnantSpecies, UpdatePopulation, Speciate, Train, Save, Load

"""
```julia
@kwdef mutable struct NEAT{Network} <: AllNEATTypes where Network <: AllNEATTypes
NEAT(;kwargs...) = NEAT{Network}(;kwargs...)
```
*NEAT* is a mutable struct that has all the variables that are required for NEAT to run. Run `fieldnames(NEAT)` to list all the variables.

NEAT <: AllNEATTypes

# Usage
```jldoctest
julia> neat_config = NEATConfig(
       n_inputs = 10,
       n_outputs = 2,
       population_size = 500,
       max_generation = 1000,
       n_generations_to_pass = 2,
       )
julia> neat = NEAT(neat_config = neat_config)
julia> neat.comments = "NEAT"
```
__Note:__ *NEAT* is always meant to be initialised only with *NEATConfig*, other variables are not meant to be initialised by user.
"""
NEAT(;kwargs...) = NEAT{Network}(;kwargs...) # this is defined because, at the position where NEAT structure is defined Network structure is not defined, so, it uses parametric type system

"""
```julia
function Log(x::NEAT, start_time::Float64 = time(), stop_time::Float64 = time(); first_entry::Bool = false)::Nothing
```
*Log* function logs NEAT training to console and to a file. It refers to `NEAT.neat_config.log_config`
"""
function Log(x::NEAT, start_time::Float64 = time(), stop_time::Float64 = time(); first_entry::Bool = false)
  if x.neat_config.log_config.log_to_file
    path = abspath(x.neat_config.save_every_n_generations_path)
    mkpath(path)
    path = joinpath(path, x.neat_config.log_config.filename)
    deli::String= x.neat_config.log_config.delimeter
    text::String = ""
    if first_entry
      x.neat_config.log_config.timestamp && (text *= "timestamp$(deli)")
      x.neat_config.log_config.generation && (text *= "generation$(deli)")
      x.neat_config.log_config.best_networks && (text *= "best_networks$(deli)")
      x.neat_config.log_config.best_fitness && (text *= "best_fitness$(deli)")
      x.neat_config.log_config.time_taken && (text *= "time_taken (sec)$(deli)")
      x.neat_config.log_config.species && (text *= "species$(deli)")
      x.neat_config.log_config.max_GIN && (text *= "max_GIN$(deli)")
      text *= "\n"
      open(path, "w") do f
        write(f, text)
      end
    end

    text = ""
    x.neat_config.log_config.timestamp && (text *= "$(now())$(deli)")
    x.neat_config.log_config.generation && (text *= "$(x.generation)$(deli)")
    x.neat_config.log_config.best_networks && (text *= "$(join([i.idx for i in x.winners], ", "))$(deli)")
    x.neat_config.log_config.best_fitness && (text *= "$(join([x.fitness[i.idx] for i in x.winners], ", "))$(deli)")
    x.neat_config.log_config.time_taken && (text *= "$(stop_time - start_time)$(deli)")
    x.neat_config.log_config.species && (text *= "$(join(sort(collect(keys(x.species))), ", "))$(deli)")
    x.neat_config.log_config.max_GIN && (text *= "$(join([x.GIN.GIN[end], x.GIN.type[end], x.GIN.start_node[end], x.GIN.stop_node[end]], ", "))$(deli)")
    text *= "\n"
    open(path, "a") do f
      write(f, text)
    end
  end

  if x.neat_config.log_config.log_to_console
    x.neat_config.log_config.timestamp && (print("timestamp = $(now()); "))
    x.neat_config.log_config.generation && (print("generation = $(x.generation); "))
    x.neat_config.log_config.best_networks && (print("best_networks = $(join([i.idx for i in x.winners], ", ")); "))
    x.neat_config.log_config.best_fitness && (print("best_fitness = $(join([x.fitness[i.idx] for i in x.winners], ", ")); "))
    x.neat_config.log_config.time_taken && (print("time_taken = $(Dates.canonicalize(Dates.Nanosecond(Int128(round((stop_time - start_time)*1e9))))); "))
    x.neat_config.log_config.species && (print("species = $(join(sort(collect(keys(x.species))), ", ")); "))
    x.neat_config.log_config.max_GIN && (print("max_GIN = $(join([x.GIN.GIN[end], x.GIN.type[end], x.GIN.start_node[end], x.GIN.stop_node[end]], ", ")); "))
    println()
    println()
  end

  return
end

"""
```julia
function Init(x::NEAT)::Nothing

*Init* initialises NEAT struct according to `NEAT.neat_config`.
```
"""
function Init(x::NEAT)
  CheckConfig(x.neat_config)

  start_time = time()

  x.population = Vector{Network}(undef, x.neat_config.population_size)
  x.species[1] = Vector{Network}(undef, x.neat_config.population_size)
  x.fitness = Vector{Real}(undef, x.neat_config.population_size)
  for i = 1:x.neat_config.population_size
    x.population[i] = Network(idx = i, n_inputs = x.neat_config.n_inputs, n_outputs = x.neat_config.n_outputs, specie = 1, super = x)
    Init(x.population[i], x.neat_config.network_config)
    x.species[1][i] = x.population[i]
    x.fitness[i] = -Inf
  end
  push!.(getfield.((x.specie_info,), propertynames(x.specie_info)), [1, true, 0, 0, -Inf, -Inf, -Inf, 0, 0, -Inf])

  for i = 1:x.neat_config.n_inputs
    push!(x.GIN.GIN, i)
    push!(x.GIN.type, InputNode)
    push!(x.GIN.start_node, 0)
    push!(x.GIN.stop_node, 0)
  end
  for i = 1:x.neat_config.n_outputs
    push!(x.GIN.GIN, i+x.neat_config.n_inputs)
    push!(x.GIN.type, OutputNode)
    push!(x.GIN.start_node, 0)
    push!(x.GIN.stop_node, 0)
  end
  if x.neat_config.network_config.start_fully_connected
    push!(x.GIN.GIN, ((x.neat_config.n_inputs+x.neat_config.n_outputs) .+ (1:x.neat_config.n_inputs*x.neat_config.n_outputs))...)
    push!(x.GIN.type, fill(ForwardConnection, x.neat_config.n_inputs*x.neat_config.n_outputs)...)
    push!(x.GIN.start_node, repeat(1:x.neat_config.n_inputs, inner = x.neat_config.n_outputs)...)
    push!(x.GIN.stop_node, repeat(x.neat_config.n_inputs .+ (1:x.neat_config.n_outputs), outer = x.neat_config.n_inputs)...)
  end

  stop_time = time()
  Log(x, start_time, stop_time, first_entry = true)

  return
end

"""
```julia
function Evaluate(x::NEAT)::Vector{Unsigned}
```
*Evaluate* function evaluates every network, i.e. runs every network and calculates it's `fitness`. Based on fitness `specie_info` is updated and `winners` for this iteration are declared.
"""
function Evaluate(x::NEAT)
  CheckConfig(x.neat_config)

  Threads.@threads for i = 1:x.neat_config.population_size
    x.fitness[i] = x.neat_config.fitness_test_dict["fitness_function"](x.neat_config.fitness_test_dict, x.population[i], x)
  end

  winners = Iterators.filter(y -> !ismissing(y) && y >= x.neat_config.threshold_fitness, x.fitness) |> collect |> findall
  if isempty(winners)
    x.winners = Network[x.population[argmax(x.fitness)]]
    x.n_networks_passed = 0x0
  else
    x.winners = x.population[winners]
    x.n_networks_passed = length(winners)
  end

  temp = Dict{Unsigned, Real}(keys(x.species).=> -Inf)
  for i in collect(x.species)
    fitness = x.fitness[getfield.(i.second, :idx)]
    x.species[i.first] = x.species[i.first][sortperm(fitness, rev = true)] # sort the population based on fitness
    fitness = fitness[sortperm(fitness, rev = true)]
    x.specie_info.minimum_fitness[i.first] = fitness[end]
    x.specie_info.maximum_fitness[i.first] = fitness[1]
    x.specie_info.mean_fitness[i.first] = mean(fitness)
    temp[i.first] = x.specie_info.maximum_fitness[i.first]
    if x.specie_info.last_highest_maximum_fitness[i.first] < x.specie_info.maximum_fitness[i.first]
      x.specie_info.last_highest_maximum_fitness[i.first] = x.specie_info.maximum_fitness[i.first]
      x.specie_info.last_improved_generation[i.first] = x.generation
    end
  end
  specie = collect(keys(temp))[argmax(collect(values(temp)))]
  x.specie_info.last_topped_generation[specie] = x.generation

  if x.n_networks_passed >= x.neat_config.n_networks_to_pass
    x.n_generations_passed += 0x1
  else
    x.n_generations_passed = 0x0
  end

  return getfield.(x.winners, :idx)
end

"""
```julia
function RemoveStagnantSpecies(x::NEAT)::Vector{Unsigned}
```
*RemoveStagnantSpecies* checks and compares every active `specie_info` and removes stagnant species. Returns a list of removed species.
"""
function RemoveStagnantSpecies(x::NEAT)
  CheckConfig(x.neat_config)

  alive_species = collect(keys(x.species))
  alive_species = alive_species[sortperm(x.specie_info.maximum_fitness[alive_species])]
  ret = Unsigned[]
  to_delete = Unsigned[]
  # Making sure not to delete the fittest specie
  for i in alive_species[1:end-1]
    if ( (x.generation - x.specie_info.last_improved_generation[i])>x.neat_config.max_specie_stagnation && (x.generation - x.specie_info.last_topped_generation[i])>x.neat_config.max_specie_stagnation )
      x.specie_info.alive[i] = false
      x.specie_info.death_generation[i] = x.generation
      for j in x.species[i]
        push!(to_delete, j.idx)
      end
      delete!(x.species, i)
      push!(ret, i)
    end
  end
  x.population = x.population[ setdiff(1:x.neat_config.population_size, to_delete) ]
  x.fitness = x.fitness[ setdiff(1:x.neat_config.population_size, to_delete) ]
  for i in collect(enumerate(x.population))
    i[2].idx = i[1]
  end
  return ret
end

"""
```julia
function UpdatePopulation(x::NEAT)::Nothing
```
*UpdatePopulation* divides the population in every specie to good and bad, retains a subset of population from every specie, performs crossover according to `NEAT.neat_config.crossover_probability`, mutates every child and updates population.
"""
function UpdatePopulation(x::NEAT)
  CheckConfig(x.neat_config)

  RemoveStagnantSpecies(x)

  new_population = Network[]
  new_fitness = Real[]

  good_individuals = Dict{Unsigned, Vector{Network}}()
  bad_individuals = Dict{Unsigned, Vector{Network}}()

  for i in x.species
    if x.neat_config.n_individuals_to_retain >= 1
      n = Unsigned(min(length(i.second), floor(x.neat_config.n_individuals_to_retain)))
      for j = 0x1:n
        push!(new_population, i.second[j])
        push!(new_fitness, x.fitness[i.second[j].idx])
      end
    else
      n = Unsigned(ceil(length(i.second)*x.neat_config.n_individuals_to_retain))
      for j = 1:n
        push!(new_population, i.second[j])
        push!(new_fitness, x.fitness[i.second[j].idx])
      end
    end

    good_individuals[i.first] = Network[]
    bad_individuals[i.first] = Network[]
    if x.neat_config.n_individuals_considered_best >= 1
      n = Unsigned(min(length(i.second), floor(x.neat_config.n_individuals_considered_best)))
    else
      n = Unsigned(ceil(length(i.second)*x.neat_config.n_individuals_considered_best))
    end
    push!(good_individuals[i.first], i.second[1:n]...)
    push!(bad_individuals[i.first], i.second[n+1:length(i.second)]...)
  end

  new_population_length = length(new_population)
  append!(new_population, Vector{Network}(undef, x.neat_config.population_size-new_population_length))
  append!(new_fitness, Vector{Real}(undef, x.neat_config.population_size-new_population_length))
  for i = new_population_length+1:x.neat_config.population_size
    crossover_probability = [x.neat_config.crossover_probability[1:3]; length(good_individuals)>1 ? x.neat_config.crossover_probability[4:6] : [0,0,0]]
    crossover = sample(rng(), 1:6, Weights(crossover_probability[:]))

    if crossover == 1 # intraspecie good and good
      specie = rand(rng(), keys(good_individuals))

      parent1 = rand(rng(), good_individuals[specie])
      parent2 = rand(rng(), good_individuals[specie])

      child = Crossover(parent1, parent2, x.fitness[parent1.idx], x.fitness[parent2.idx])

      new_population[i] = child
      new_fitness[i] = -Inf
    elseif crossover == 2 # intraspecie good and bad
      specie = rand(rng(), keys(good_individuals))

      parent1 = rand(rng(), good_individuals[specie])
      parent2 = nothing
      if isempty(bad_individuals[specie])
        parent2 = rand(rng(), good_individuals[specie])
      else
        parent2 = rand(rng(), bad_individuals[specie])
      end

      child = Crossover(parent1, parent2, x.fitness[parent1.idx], x.fitness[parent2.idx])

      new_population[i] = child
      new_fitness[i] = -Inf
    elseif crossover == 3 # intraspecie bad and bad
      specie = rand(rng(), keys(good_individuals))

      parent1 = nothing
      parent2 = nothing
      if isempty(bad_individuals[specie])
        parent1 = rand(rng(), good_individuals[specie])
      else
        parent1 = rand(rng(), bad_individuals[specie])
      end
      if isempty(bad_individuals[specie])
        parent2 = rand(rng(), good_individuals[specie])
      else
        parent2 = rand(rng(), bad_individuals[specie])
      end

      child = Crossover(parent1, parent2, x.fitness[parent1.idx], x.fitness[parent2.idx])

      new_population[i] = child
      new_fitness[i] = -Inf
    elseif crossover == 4 # interspecie good and good
      specie1, specie2 = rand(rng(), keys(good_individuals), 2)

      parent1 = rand(rng(), good_individuals[specie1])
      parent2 = rand(rng(), good_individuals[specie2])

      child = Crossover(parent1, parent2, x.fitness[parent1.idx], x.fitness[parent2.idx])

      new_population[i] = child
      new_fitness[i] = -Inf
    elseif crossover == 5 # interspecie good and bad
      specie1, specie2 = rand(rng(), keys(good_individuals), 2)

      parent1 = rand(rng(), good_individuals[specie1])
      parent2 = nothing
      if isempty(bad_individuals[specie2])
        parent2 = rand(rng(), good_individuals[specie2])
      else
        parent2 = rand(rng(), bad_individuals[specie2])
      end

      child = Crossover(parent1, parent2, x.fitness[parent1.idx], x.fitness[parent2.idx])

      new_population[i] = child
      new_fitness[i] = -Inf
    elseif crossover == 6 # interspecie bad and bad
      specie1, specie2 = rand(rng(), keys(good_individuals), 2)

      parent1 = nothing
      parent2 = nothing
      if isempty(bad_individuals[specie1])
        parent1 = rand(rng(), good_individuals[specie1])
      else
        parent1 = rand(rng(), bad_individuals[specie1])
      end
      if isempty(bad_individuals[specie2])
        parent2 = rand(rng(), good_individuals[specie2])
      else
        parent2 = rand(rng(), bad_individuals[specie2])
      end

      child = Crossover(parent1, parent2, x.fitness[parent1.idx], x.fitness[parent2.idx])

      new_population[i] = child
      new_fitness[i] = -Inf
    end

    for m = 1:x.neat_config.n_mutations
      Mutate(new_population[i], x.neat_config.mutation_probability, x.neat_config.network_config)
    end
  end

  for i = 0x1:x.neat_config.population_size
    new_population[i].idx = i
  end

  x.population = new_population
  x.fitness = new_fitness

  return
end

"""
```julia
function Speciate(x::NEAT)::Nothing
```
*Speciate* categorises every individual into existing or new species. It uses `GetNetworkDistance` function to get the dissimilarity between 2 networks.
"""
function Speciate(x::NEAT)
  new_species = Dict{Unsigned, Vector{Network}}()

  for i in x.population
    min_distance = Inf
    specie = 0
    for j in x.species
      distance = GetNetworkDistance(i, j.second[1], x.neat_config.network_config)
      if (distance < x.neat_config.threshold_distance) && (distance < min_distance)
        min_distance = distance
        specie = j.first
      end
    end

    if specie > 0
      if haskey(new_species, specie)
        i.specie = specie
        push!(new_species[specie], i)
      else
        i.specie = specie
        new_species[specie] = Network[i]
      end
    else
      min_distance = Inf
      specie = 0
      condition_satisfied = false
      for j in new_species
        distance = GetNetworkDistance(i, j.second[1], x.neat_config.network_config)
        if distance < min_distance
          if distance < x.neat_config.threshold_distance
            condition_satisfied = true
          end
          min_distance = distance
          specie = j.first
        end
      end

      if condition_satisfied
        i.specie = specie
        push!(new_species[specie], i)
      else
        if length(new_species) < x.neat_config.n_species
          new_specie = x.specie_info.specie[end] + 0x1
          push!.(getfield.((x.specie_info,), propertynames(x.specie_info)), [new_specie, true, x.generation, 0, -Inf, -Inf, -Inf, 0, x.generation, -Inf])
          i.specie = new_specie
          new_species[new_specie] = Network[i]
        else
          i.specie = specie
          push!(new_species[specie], i)
        end
      end
    end
  end

  x.species = new_species
  return
end

"""
```julia
function Train(x::NEAT)::Nothing
```
*Train* function trains NEAT. It executes `Evaluate`, `RemoveStagnantSpecies`, `UpdatePopulation` and `Speciate` in sequential order to train NEAT. It also logs and saves the NEAT struct.
"""
function Train(x::NEAT)
  last_saved_generation = 0
  for itr = x.generation:x.neat_config.max_generation
    start_time = time()
    Evaluate(x)
    if x.n_generations_passed >= x.neat_config.n_generations_to_pass
      println("Congratulations NEAT is trained in $(x.generation) generations")
      println("Winners: $(join([i.idx for i in x.winners], ", "))")
      if x.neat_config.save_at_termination
        path = abspath(x.neat_config.save_every_n_generations_path)
        mkpath(path)
        if x.neat_config.save_every_n_generations_discard_previous
          if ispath(joinpath(path, x.neat_config.save_every_n_generations_filename*"-gen_$(last_saved_generation).jld2"))
            rm(joinpath(path, x.neat_config.save_every_n_generations_filename*"-gen_$(last_saved_generation).jld2"))
          end
        end
        last_saved_generation = x.generation
        path = joinpath(path, x.neat_config.save_at_termination_filename)
        Save(x, path)
      end
      return
    end

    if x.neat_config.save_every_n_generations > 0 && (x.generation % x.neat_config.save_every_n_generations == 0)
      path = abspath(x.neat_config.save_every_n_generations_path)
      mkpath(path)
      if x.neat_config.save_every_n_generations_discard_previous
        if ispath(joinpath(path, x.neat_config.save_every_n_generations_filename*"-gen_$(last_saved_generation).jld2"))
          rm(joinpath(path, x.neat_config.save_every_n_generations_filename*"-gen_$(last_saved_generation).jld2"))
        end
      end
      last_saved_generation = x.generation
      path = joinpath(path, x.neat_config.save_every_n_generations_filename*"-gen_$(x.generation).jld2")
      Save(x, path)
    end

    UpdatePopulation(x)
    Speciate(x)

    x.generation += 0x1
    stop_time = time()
    Log(x, start_time, stop_time)
  end

  Evaluate(x)
  if x.neat_config.save_at_termination
    path = abspath(x.neat_config.save_every_n_generations_path)
    mkpath(path)
    if x.neat_config.save_every_n_generations_discard_previous
      if ispath(joinpath(path, x.neat_config.save_every_n_generations_filename*"-gen_$(last_saved_generation).jld2"))
        rm(joinpath(path, x.neat_config.save_every_n_generations_filename*"-gen_$(last_saved_generation).jld2"))
      end
    end
    last_saved_generation = x.generation
    path = joinpath(path, x.neat_config.save_at_termination_filename*".jld2")
    Save(x, path)
  end
  println("Max generation reached, training terminated")
  println("Winners: $(join([i.idx for i in x.winners], ", "))")
  return
end

"""
```julia
function Save(x::NEAT, filepath::String = "")::String
```
*Save* saves the `NEAT` struct and returns the path of the file. The default filepath is "./NEAT.jld2"
"""
function Save(x::NEAT, filepath::String = "")
  isempty(filepath) && (filepath = "./NEAT.jld2")
  filepath = abspath(filepath)
  jldsave(filepath; x)
  return filepath
end

"""
```julia
function Load(filepath::String)::IdDict
```
*Load* opens and loads JLD2 file at `filepath` and returns an IdDict containing all the variables.
"""
function Load(filepath::String)
  ret = IdDict()
  filepath = abspath(filepath)
  jldopen(filepath, "r") do f
    for i in keys(f)
      ret[i] = f[i]
    end
  end
  return ret
end



