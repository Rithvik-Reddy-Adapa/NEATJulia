export LogConfig, NEATConfig, CheckConfig

@kwdef mutable struct LogConfig <: Configs
  log_to_console::Bool = true
  log_to_file::Bool = true
  filename::String = "NEAT.log"
  delimeter::String = "<>"

  timestamp::Bool = true
  generation::Bool = true
  best_networks::Bool = true
  best_fitness::Bool = true
  time_taken::Bool = true
  species::Bool = false
  max_GIN::Bool = false
end

function Base.show(io::IO, x::LogConfig)
  println(io, summary(x))
  print(io, " log_to_console : $(x.log_to_console)
 log_to_file : $(x.log_to_file)
 filename : $(x.filename)
 delimeter : $(x.delimeter)

 timestamp : $(x.timestamp)
 generation : $(x.generation)
 best_networks : $(x.best_networks)
 best_fitness : $(x.best_fitness)
 time_taken : $(x.time_taken)
 species : $(x.species)
 max_GIN : $(x.max_GIN)
 ")
end

@kwdef mutable struct NEATConfig <: Configs
  const n_inputs::Unsigned
  const n_outputs::Unsigned
  const population_size::Unsigned = 50
  max_generation::Unsigned = 100
  n_species::Unsigned = 4 # number of species per generation
  max_specie_stagnation::Unsigned = 20 # for how many number of generations can a specie be alive without improvement
  n_networks_to_pass::Unsigned = 1
  n_generations_to_pass::Unsigned = 1
  n_individuals_considered_best::Real = 0.25 # number of individuals considered best in a specie in a generation. Takes real values >= 0. Number less than 1 is considered as ratio over total specie population, number >= 1 is considered as number of individuals.
  n_individuals_to_retain::Real = 1 # number of individuals to retain unchanged for next generation of a specie. Takes real values >= 0. Number less than 1 is considered as ratio over total specie population, number >= 1 is considered as number of individuals.
  threshold_fitness::Real = 0.0
  fitness_test_dict::Dict{String, Any} = Dict{String, Any}()
  threshold_distance::Real = 3

  crossover_probability::CrossoverProbability = CrossoverProbability()
  mutation_probability::MutationProbability = MutationProbability()

  network_config::NetworkConfig = NetworkConfig()
  log_config::LogConfig = LogConfig()
end

function CheckConfig(x::NEATConfig)
  x.n_inputs == 0 && error("NEATConfig : n_inputs should be greater than 0")
  x.n_outputs == 0 && error("NEATConfig : n_outputs should be greater than 0")
  x.population_size == 0 && error("NEATConfig : population_size should be greater than 0")
  x.n_species == 0 && error("NEATConfig : n_species should be greater than 0")
  x.max_specie_stagnation > 1 || error("NEATConfig : max_specie_stagnation should be greater than 1")
  (x.n_networks_to_pass > 0 && x.n_networks_to_pass <= x.population_size) || error("NEATConfig : n_networks_to_pass should be > 0 and <= population_size")
  (x.n_generations_to_pass > 0 && x.n_generations_to_pass <= x.max_generation) || error("NEATConfig : n_generations_to_pass should be > 0 and <= max_generation")
  isfinite(x.n_individuals_considered_best) || error("NEATConfig : got non finite value for n_individuals_considered_best")
  (x.n_individuals_considered_best <= 0 || x.n_individuals_considered_best > x.population_size) && error("NEATConfig : n_individuals_considered_best should be > 0 and < population_size")
  isfinite(x.n_individuals_to_retain) || error("NEATConfig : got non finite value for n_individuals_to_retain")
  (x.n_individuals_to_retain < 0 || x.n_individuals_to_retain > x.population_size) && error("NEATConfig : n_individuals_to_retain should be >= 0 and < population_size")
  haskey(x.fitness_test_dict, "fitness_function") || error("NEATConfig : fitness_test_dict should have \"fitness_function\" key that points to fitness function. The fitness function should take the fitness_test_dict, the network being tested and the NEAT structure the network is present in. It should return fitness of the network.")
  x.fitness_test_dict["fitness_function"] isa Function || error("NEATConfig : fitness_test_dict should have \"fitness_function\" key that points to fitness function. The fitness function should take the fitness_test_dict, the network being tested and the NEAT structure the network is present in It should return fitness of the network.")
  isfinite(x.threshold_distance) || error("NEATConfig : got non finite value for threshold_distance")
  x.threshold_distance >= 0 || error("NEATConfig : threshold_distance should be >= 0")
  
  CheckCrossoverProbability(x.crossover_probability)
  CheckMutationProbability(x.mutation_probability)

  CheckConfig(x.network_config)

  return
end

function Base.show(io::IO, x::NEATConfig)
  println(io, summary(x))
  print(io, " n_inputs : $(x.n_inputs)
 n_outputs : $(x.n_outputs)
 population_size : $(x.population_size)
 max_generation : $(x.max_generation)
 ")
  return
end


