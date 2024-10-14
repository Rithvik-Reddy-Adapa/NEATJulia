using Debugger

include("src/NEATJulia.jl")

using .NEATJulia

in_node = InputNode()
in_node_config = InputNodeConfig()

in_node_config.min_bias[] = -1
in_node_config.max_bias[] = 1
in_node_config.shift_bias[] = 0.1
in_node_config.std_bias[] = 0.01


Init(in_node, in_node_config)

network = Network(n_inputs = 2, n_outputs = 3)
network_config = NetworkConfig(start_fully_connected = false)
Init(network, network_config)

new_GIN = length(network.genes) + 1
AddForwardConnection(network, network.genes[1], network.genes[3], new_GIN, network_config)

new_GIN = length(network.genes) + 1
AddForwardConnection(network, network.genes[2], network.genes[5], new_GIN, network_config)

new_GIN = length(network.genes) + 1
AddBackwardConnection(network, network.genes[4], network.genes[1], new_GIN, network_config)

new_GIN = length(network.genes) + 1
AddHiddenNodeForwardConnection(network, network.genes[6], new_GIN, network_config)

new_GIN = length(network.genes) + 1
AddHiddenNodeBackwardConnection(network, network.genes[8], new_GIN, network_config)

new_GIN = length(network.genes) + 1
AddLSTMNodeBackwardConnection(network, network.genes[8], new_GIN, network_config)

new_GIN = length(network.genes) + 1
AddForwardConnection(network, network.genes[2], network.genes[4], new_GIN, network_config)

new_GIN = length(network.genes) + 1
AddGRUNodeForwardConnection(network, network.genes[18], new_GIN, network_config)

new_GIN = length(network.genes) + 1
AddRecurrentHiddenNodeBackwardConnection(network, network.genes[14], new_GIN, network_config)

mutation_probability = MutationProbability()


neat_config = NEATConfig(n_inputs = 2,
                         n_outputs = 3,
                         population_size = 100,
                         max_generation = 1000,
                         n_species = 10
                        )
neat = NEAT(neat_config = neat_config)
Init(neat)


