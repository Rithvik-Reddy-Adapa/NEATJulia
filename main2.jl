include("./NEAT.jl")
using .NEAT
using Debugger, DataFrames, JSON, JSON3

function fitness_function(outputs, train_outputs; train_inputs = missing, network = missing, neat = missing)
  ret = 0.0
  for i = 1:length(outputs)
    for j = 1:length(outputs[i])
      for k = 1:length(outputs[i][j])
        if ismissing(outputs[i][j][k])
          return -Inf
        end
        if outputs[i][j][k] == Inf || outputs[i][j][k] == -Inf || outputs[i][j][k] == NaN
          # println("outputs[$(i)][$(j)][$(k)] = $(outputs[i][j][k]), hence exiting")
          # println("network = $(network.ID), neat generation = $(neat.generation|>Int)")
          println()
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
                                           n_outputs = 3,
                                           population_size = 100,
                                           max_generation = 2_000,
                                           threshold_fitness = -1,
                                           fitness_function = fitness_function,
                                           list_activation_functions = [Tanh, Sigmoid, Sin, Relu, Identity],
                                           # list_activation_functions = [Identity],
                                           threshold_distance = 2,
                                           max_species_per_generation = 10,
                                           max_specie_stagnation = 20,
                                           distance_parameters = [1, 1, 1],
                                           normalise_distance = false,
                                           initial_mutation_probability = RNN_Mutation_Probability(shift_weight = 7, shift_bias = 3, add_forward_connection = 1, add_node = 0.1, add_recurrent_connection = 2, change_activation_function = 0.0),
                                          )
  global neat_rnn = NEAT_RNN(config = neat_rnn_config)

  train_inputs = []
  train_outputs = []

  input_sequence = []
  output_sequence = []
  t = 0:0.0002:0.1
  amp = 1
  phi = 30
  freq = 60
  for j = 1:length(t)
    push!(output_sequence, [amp,phi,freq])
    push!(input_sequence, [t[j], amp*sind(360*freq*t[j] + phi)])
  end
  push!(train_inputs, input_sequence)
  push!(train_outputs, output_sequence)

  input_sequence = []
  output_sequence = []
  t = 0:0.0002:0.1
  amp = 10
  phi = -30
  freq = 60
  for j = 1:length(t)
    push!(output_sequence, [amp,phi,freq])
    push!(input_sequence, [t[j], amp*sind(360*freq*t[j] + phi)])
  end
  push!(train_inputs, input_sequence)
  push!(train_outputs, output_sequence)

  input_sequence = []
  output_sequence = []
  t = 0:0.0002:0.1
  amp = 32.3124
  phi = -75.5
  freq = 55.625
  for j = 1:length(t)
    push!(output_sequence, [amp,phi,freq])
    push!(input_sequence, [t[j], amp*sind(360*freq*t[j] + phi)])
  end
  push!(train_inputs, input_sequence)
  push!(train_outputs, output_sequence)

  neat_rnn.train_inputs = train_inputs
  neat_rnn.train_outputs = train_outputs

  Init(neat_rnn)
end

main()
;
