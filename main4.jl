include("./NEAT.jl")
using .NEAT
using Debugger, DataFrames, JSON, JSON3

function fitness_function(outputs, train_outputs; train_inputs = missing, network = missing, neat = missing)
  ret = 0.0
  for i = 1:length(outputs)
    for j = 1:length(outputs[i])
      for k = 1:length(outputs[i][j])
        if isnan(train_outputs[i][j][k])
          continue
        end
        if ismissing(outputs[i][j][k])
          return -Inf
        end
        if outputs[i][j][k] == Inf || outputs[i][j][k] == -Inf || outputs[i][j][k] == NaN
          # Save(neat) #
          # println("outputs[$(i)][$(j)][$(k)] = $(outputs[i][j][k]), hence exiting") #
          # println("network = $(network.ID), neat generation = $(neat.generation|>Int)") #
          # println() #
          # throw(error()) #
          return -Inf
        end
        ret += abs(outputs[i][j][k] - train_outputs[i][j][k])
      end
    end
  end
  ret = -ret
  if isnan(ret)
    return -Inf
  end
  return ret
end

function main()
  global neat_rnn_config = NEAT_RNN_config(n_inputs = 2,
                                           n_outputs = 1,
                                           population_size = 100,
                                           max_generation = 2_000,
                                           threshold_fitness = -0.5,
                                           fitness_function = fitness_function,
                                           list_activation_functions = [Tanh, Sigmoid, Sin, Relu, Identity],
                                           threshold_distance = 2,
                                           max_species_per_generation = 10,
                                           max_specie_stagnation = 20,
                                           distance_parameters = [1, 1, 1],
                                           normalise_distance = false,
                                           initial_mutation_probability = RNN_Mutation_Probability(),
                                          )
  global neat_rnn = NEAT_RNN(config = neat_rnn_config)

  train_inputs = []
  train_outputs = []

  input_sequence = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
  output_sequence = [[NaN], [NaN], [NaN], [0.0], [0.0], [0.0]]
  push!(train_inputs, input_sequence)
  push!(train_outputs, output_sequence)


  input_sequence = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
  output_sequence = [[NaN], [NaN], [NaN], [1.0], [1.0], [1.0]]
  push!(train_inputs, input_sequence)
  push!(train_outputs, output_sequence)


  input_sequence = [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
  output_sequence = [[NaN], [NaN], [NaN], [1.0], [0.0], [1.0]]
  push!(train_inputs, input_sequence)
  push!(train_outputs, output_sequence)


  input_sequence = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
  output_sequence = [[NaN], [NaN], [NaN], [1.0], [0.0], [0.0]]
  push!(train_inputs, input_sequence)
  push!(train_outputs, output_sequence)


  input_sequence = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
  output_sequence = [[NaN], [NaN], [NaN], [0.0], [1.0], [0.0]]
  push!(train_inputs, input_sequence)
  push!(train_outputs, output_sequence)


  input_sequence = [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
  output_sequence = [[NaN], [NaN], [NaN], [1.0], [1.0], [0.0]]
  push!(train_inputs, input_sequence)
  push!(train_outputs, output_sequence)

  neat_rnn.train_inputs = train_inputs
  neat_rnn.train_outputs = train_outputs

  Init(neat_rnn)
end

main()
;
