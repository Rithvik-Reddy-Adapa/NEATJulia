using Dates, JLD2
export NEAT, Log, Init, Evaluate, RemoveStagnantSpecies, UpdatePopulation, Speciate, Train, Save, Load

NEAT(;kwargs...) = NEAT{Network}(;kwargs...) # this is defined because, at the position where NEAT structure is defined Network structure is not defined, so, it uses parametric type system

function Log(x::NEAT, start_time::Float64 = time(), stop_time::Float64 = time(); first_entry::Bool = false)
  if x.neat_config.log_config.log_to_file
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
      open(x.neat_config.log_config.filename, "w") do f
        write(f, text)
      end
    end

    text = ""
    x.neat_config.log_config.timestamp && (text *= "$(now())$(deli)")
    x.neat_config.log_config.generation && (text *= "$(x.generation)$(deli)")
    x.neat_config.log_config.best_networks && (text *= "$([i.idx for i in x.winners])$(deli)")
    x.neat_config.log_config.best_fitness && (text *= "$([x.fitness[i.idx] for i in x.winners])$(deli)")
    x.neat_config.log_config.time_taken && (text *= "$(stop_time - start_time)$(deli)")
    x.neat_config.log_config.species && (text *= "$([i.first for i in x.species])$(deli)")
    x.neat_config.log_config.max_GIN && (text *= "$([x.GIN.GIN[end], x.GIN.type[end], x.GIN.start_node[end], x.GIN.stop_node[end]])$(deli)")
    text *= "\n"
    open(x.neat_config.log_config.filename, "a") do f
      write(f, text)
    end
  end

  if x.neat_config.log_config.log_to_console
    x.neat_config.log_config.timestamp && (print("timestamp = $(now()), "))
    x.neat_config.log_config.generation && (print("generation = $(x.generation), "))
    x.neat_config.log_config.best_networks && (print("best_networks = $([i.idx for i in x.winners]), "))
    x.neat_config.log_config.best_fitness && (print("best_fitness = $([x.fitness[i.idx] for i in x.winners]), "))
    x.neat_config.log_config.time_taken && (print("time_taken = $(Dates.canonicalize(Dates.Nanosecond(Int128(round((stop_time - start_time)*1e9))))), "))
    x.neat_config.log_config.species && (print("species = $([i.first for i in x.species]), "))
    x.neat_config.log_config.max_GIN && (print("max_GIN = $([x.GIN.GIN[end], x.GIN.type[end], x.GIN.start_node[end], x.GIN.stop_node[end]]), "))
    println()
    println()
  end

  return
end

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
    crossover = sample(rng(), Weights(crossover_probability[:]))

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

    Mutate(new_population[i], x.neat_config.mutation_probability, x.neat_config.network_config)
  end

  for i = 0x1:x.neat_config.population_size
    new_population[i].idx = i
  end

  x.population = new_population
  x.fitness = new_fitness

  return
end

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

function Train(x::NEAT)
  for itr = 1:x.neat_config.max_generation
    start_time = time()
    Evaluate(x)
    if x.n_generations_passed >= x.neat_config.n_generations_to_pass
      println("Congratulations NEAT is trained in $(x.generation) generations")
      println("Winners: $(join([i.idx for i in x.winners], ", "))")
      return
    end

    UpdatePopulation(x)
    Speciate(x)

    x.generation += 0x1
    stop_time = time()
    Log(x, start_time, stop_time)
  end

  Evaluate(x)
  println("Max generation reached, training terminated")
  println("Winners: $(join([i.idx for i in x.winners], ", "))")
  return
end

function Save(x::NEAT, filename::String = "")
  isempty(filename) && (filename = "NEAT.jld2")
  filename = abspath(filename)
  jldsave(filename; x)
  return filename
end

function Load(filename::String)
  ret = IdDict()
  jldopen(filename, "r") do f
    for i in keys(f)
      ret[i] = f[i]
    end
  end
  return ret
end



