include("./NEAT.jl")
using .NEAT
using Debugger, DataFrames, JSON, JSON3

function fitness_function(outputs, train_outputs; train_inputs = missing, network = missing, neat = missing)
  if any(ismissing.(outputs[1]))
    return -Inf
  end
  n_sets = length(outputs)
  n_inputs = length(outputs[1])
  difference = outputs.-train_outputs
  for i = 1:n_sets
    for j = 1:n_inputs
      difference[i][j] = abs(difference[i][j])
    end
  end
  return -sum( sum.(difference) )
end

function main()
  global neat_ffnn_config = NEAT_FFNN_config(n_inputs = 1,
					     n_outputs = 1,
					     population_size = 100,
					     max_generation = 1_000,
					     threshold_fitness = -1,
					     fitness_function = fitness_function,
					     list_activation_functions = [Sin],
					     threshold_distance = 5,
					     max_species_per_generation = 10,
					     normalise_distance = false,
					     start_fully_connected = true,
					     )
  global neat_ffnn = NEAT_FFNN(config = neat_ffnn_config)
  neat_ffnn.mutation_probability = FFNN_Mutation_Probability[FFNN_Mutation_Probability(add_forward_connection = 0.5, add_node = 0.5) for i = 1:neat_ffnn.config.population_size]
  # neat_ffnn.train_inputs = [[0,0], [0,1], [1,0], [1,1]]
  # neat_ffnn.train_outputs = [[0], [1], [1], [0]]
  neat_ffnn.train_inputs = [[i] for i = -90:90]
  neat_ffnn.train_outputs = [[sind(i[1])] for i in neat_ffnn.train_inputs]
  Init(neat_ffnn)
end

main()
;
