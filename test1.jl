using Debugger, DataStructures, Plotly, StatsBase
include("src/NEATJulia.jl")
using .NEATJulia

function fitness_function(dict::Dict{String, Any}, network::Network, neat::NEAT)
  ret = 0.0
  for (amp_r, freq_r, phi_r) in zip(dict["amp"], dict["freq"], dict["phi"])
    T = 0:dict["dt"]:dict["max_time"]
    V = amp_r .* sin.(2 .*pi.*freq_r.*T .+ phi_r)
    buffer_t = CircularBuffer{Real}(dict["buffer_size"])
    buffer_v = CircularBuffer{Real}(dict["buffer_size"])
    ResetIO(network)
    for (t,v) in zip(T, V)
      push!(buffer_t, t)
      push!(buffer_v, v)
      SetInput(network, [t,v])
      amp, freq, phi = Run(network)
      (ismissing(amp) || !isfinite(amp) || ismissing(freq) || !isfinite(freq) || ismissing(phi) || !isfinite(phi)) && (return -Inf)
      v_out = amp .* sin.(2 .*pi.*freq.*buffer_t .+ phi)
      diff = v_out .- buffer_v
      diff = sum( abs.(diff) )
      isfinite(diff) || (return -Inf)
      ret -= diff
    end
  end
  return ret
end

fitness_test_dict = Dict("amp" => [12, 10, 3],
                         "freq" => [50, 55, 60],
                         "phi" => [0, pi/4, pi/3],
                         "max_time" => 0.2,
                         "buffer_size" => 256,
                         "dt" => 0.001,
                         "fitness_function" => fitness_function,
                        )

mutation_probability = MutationProbability(
                                           no_mutation = 0,

                                           global_shift_weight = 1.0,
                                           global_change_weight = 1.0,
                                           global_change_bias = 0.1,
                                           global_shift_bias = 1,
                                           global_change_activation_function = 1,
                                           global_toggle_enable = 1,
                                           global_enable_gene = 0,
                                           global_disable_gene = 0,

                                           add_backward_connection = 0.01,

                                           add_hidden_node_forward_connection = 0.05,
                                           add_hidden_node_backward_connection = 0.001,
                                           add_recurrent_hidden_node_forward_connection = 0.1,
                                           add_recurrent_hidden_node_backward_connection = 0.001,
                                           add_lstm_node_forward_connection = 0.1,
                                           add_lstm_node_backward_connection = 0.001,
                                           add_gru_node_forward_connection = 0.1,
                                           add_gru_node_backward_connection = 0.001,
                                          )

crossover_probability = CrossoverProbability(
                                             intraspecie_good_good = 1,
                                             intraspecie_good_bad = 0,
                                             intraspecie_bad_bad = 0,

                                             interspecie_good_good = 1,
                                             interspecie_good_bad = 0,
                                             interspecie_bad_bad = 0,
                                            )

neat_config = NEATConfig(
                         n_inputs = 2,
                         n_outputs = 3,
                         population_size = 100,
                         max_generation = 5000,
                         n_species = 10,
                         n_mutations = 3,
                         fitness_test_dict = fitness_test_dict,
                         threshold_distance = 5,
                         max_specie_stagnation = 50,

                         mutation_probability = mutation_probability,
                         crossover_probability = crossover_probability,
                        )

Clamp(x::Real) = isfinite(x) ? (return clamp(x, -pi/2, 65)) : (return 0)

neat_config.network_config.input_node_config.activation_functions[] = [Sigmoid, Relu, Identity, Tanh]
neat_config.network_config.output_node_config.activation_functions[] = [Clamp]
neat_config.network_config.hidden_node_config.activation_functions[] = [Sigmoid, Relu, Identity, Tanh]
neat_config.network_config.recurrent_hidden_node_config.activation_functions[] = [Sigmoid, Relu, Identity, Tanh]

neat_config.network_config.input_node_config.initial_activation_function[] = Identity
neat_config.network_config.output_node_config.initial_activation_function[] = Clamp
neat_config.network_config.hidden_node_config.initial_activation_function[] = Identity
neat_config.network_config.recurrent_hidden_node_config.initial_activation_function[] = Identity

neat_config.network_config.forward_connection_config.shift_weight[] = 0.1
neat_config.network_config.forward_connection_config.std_weight[] = 0.1
neat_config.network_config.backward_connection_config.shift_weight[] = 0.1
neat_config.network_config.backward_connection_config.std_weight[] = 0.1
neat_config.network_config.recurrent_hidden_node_config.shift_weight[] = 0.1
neat_config.network_config.recurrent_hidden_node_config.std_weight[] = 0.1
neat_config.network_config.lstm_node_config.shift_weight[] = 0.1
neat_config.network_config.lstm_node_config.std_weight[] = 0.1
neat_config.network_config.gru_node_config.shift_weight[] = 0.1
neat_config.network_config.gru_node_config.std_weight[] = 0.1

neat_config.log_config.species = true
neat_config.log_config.max_GIN = true

neat = NEAT(neat_config = neat_config)
Init(neat)

Train(neat)

Save(neat)

