include("./NEAT.jl")
using .NEAT
using Debugger, DataFrames, JSON, JSON3

fitness_function_dict = Dict{String, Any}()
# fitness_function_dict["train_inputs"] = [[0,0], [0,1], [1,0], [1,1]]
# fitness_function_dict["train_outputs"] = [[0], [1], [1], [0]]
fitness_function_dict["train_inputs"] = [[i] for i = -90:90]
fitness_function_dict["train_outputs"] = [[sind(i[1])] for i in fitness_function_dict["train_inputs"]]

function fitness_function(dict::Dict{String, Any}, network::N) where N <: Networks
  ret = 0.0
  for (ei,eo) in zip(dict["train_inputs"], dict["train_outputs"]) # is = input_sequence, os = output_sequence
    SetInput!(network, ei)
    output = Run(network)
    if any(ismissing.(output) .|| (output .== Inf) .|| (output .== -Inf) .|| isnan.(output))
      return -Inf
    end
    ret += sum( abs.(output[.!isnan.(eo)] .- eo[.!isnan.(eo)]) )
  end
  ret = -ret
  if isnan(ret) || ret == Inf || ret == -Inf
    return -Inf
  end
  return ret
end
fitness_function_dict["fitness_function"] = fitness_function

function main()
  global neat_ffnn_config = NEAT_FFNN_config(n_inputs = 1,
					     n_outputs = 1,
					     population_size = 100,
					     max_generation = 1_000,
					     threshold_fitness = -1,
					     fitness_function_dict = fitness_function_dict,
					     list_activation_functions = [Tanh],
					     threshold_distance = 5,
					     max_species_per_generation = 10,
					     normalise_distance = false,
					     initial_mutation_probability = FFNN_Mutation_Probability(add_forward_connection = 0.5,
					                                                              add_node = 0.5,
					                                                              enable_forward_connection = 0.5,
					                                                              enable_node = 0.5,
					                                                              disable_forward_connection = 0.5,
					                                                              disable_node = 0.5,)
					     )
  global neat_ffnn = NEAT_FFNN(config = neat_ffnn_config)
  neat_ffnn.config.log_config.species = true
  neat_ffnn.config.log_config.max_GIN = true
  Init(neat_ffnn)
end

main()
;
