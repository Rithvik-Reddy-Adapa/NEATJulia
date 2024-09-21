include("./NEAT.jl")
using .NEAT
using Debugger, DataFrames, JSON, JSON3

fitness_test_dict = Dict{String, Any}()

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

fitness_test_dict["train_inputs"] = train_inputs
fitness_test_dict["train_outputs"] = train_outputs

function fitness_function(dict::Dict{String, Any}, network::N) where N <: Networks
  ret = 0.0
  for (is,os) in zip(dict["train_inputs"], dict["train_outputs"]) # is = input_sequence, os = output_sequence
    Reset!(network)
    for (ei,eo) in zip(is, os) # ei = expected input, eo = expected output
      SetInput!(network, ei)
      output = Run(network)
      if any(ismissing.(output) .|| (output .== Inf) .|| (output .== -Inf) .|| isnan.(output))
        return -Inf
      end
      ret += sum( abs.(output[.!isnan.(eo)] .- eo[.!isnan.(eo)]) )
    end
  end
  ret = -ret
  if isnan(ret) || ret == Inf || ret == -Inf
    return -Inf
  end
  return ret
end
fitness_test_dict["fitness_function"] = fitness_function

function main()
  global neat_rnn_config = NEAT_RNN_config(n_inputs = 2,
                                           n_outputs = 3,
                                           population_size = 100,
                                           max_generation = 10_000,
                                           threshold_fitness = -1,
                                           fitness_test_dict = fitness_test_dict,
                                           # list_activation_functions = [Tanh, Sigmoid, Sin, Relu, Identity],
                                           list_activation_functions = [Relu, Identity],
                                           threshold_distance = 2,
                                           max_species_per_generation = 20,
                                           max_specie_stagnation = 20,
                                           distance_parameters = [1, 1, 1],
                                           normalise_distance = false,
                                           max_shift_weight = 0.01,
                                           max_shift_bias = 0.01,
                                           initial_mutation_probability = RNN_Mutation_Probability(no_mutation = 0,
                                                                                                   change_weight = 4,
                                                                                                   change_bias = 1,
                                                                                                   shift_weight = 10,
                                                                                                   shift_bias = 1,
                                                                                                   add_forward_connection = 2,
                                                                                                   add_node = 0.1,
                                                                                                   add_recurrent_connection = 0.1,
                                                                                                   disable_forward_connection = 0.1,
                                                                                                   disable_node = 1,
                                                                                                   disable_recurrent_connection = 0.1,
                                                                                                   enable_forward_connection = 0.1,
                                                                                                   enable_node = 0.1,
                                                                                                   enable_recurrent_connection = 0.1,
                                                                                                   change_activation_function = 0.1,
                                                                                                   add_self_connection = 1,
                                                                                                   enable_self_connection = 0.0,
                                                                                                   disable_self_connection = 0.0,
                                                                                                  ),
                                          )
  global neat_rnn = NEAT_RNN(config = neat_rnn_config)
  Init(neat_rnn)
end

main()
;
