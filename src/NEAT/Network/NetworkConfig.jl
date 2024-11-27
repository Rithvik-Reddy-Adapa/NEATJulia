#=
# Refer src/NEAT/Probabilities.jl for MutationProbability, CheckMutationProbability, CrossoverProbability, CheckCrossoverProbability
# Refer src/AbstractTypes.jl for Genes, Nodes, Connections, RecurrentConnections, HiddenNodes, RecurrentHiddenNodes, AllNEATTypes
# Refer src/Reference.jl for Reference
# Refer src/NEAT/NEAT.jl for NEAT
=#

export NetworkConfig, CheckConfig

"""
```julia
@kwdef mutable struct NetworkConfig <: Configs
```
*NetworkConfig* has config for network and every type of nodes and connections.
"""
@kwdef mutable struct NetworkConfig <: Configs
  start_fully_connected::Bool = true
  distance_parameters::Vector{<:Real} = [1, 1, 1]
  normalise_distance::Bool = false

  input_node_config::InputNodeConfig = InputNodeConfig()
  output_node_config::OutputNodeConfig = OutputNodeConfig()
  hidden_node_config::HiddenNodeConfig = HiddenNodeConfig()
  forward_connection_config::ForwardConnectionConfig = ForwardConnectionConfig()
  backward_connection_config::BackwardConnectionConfig = BackwardConnectionConfig()
  recurrent_hidden_node_config::RecurrentHiddenNodeConfig = RecurrentHiddenNodeConfig()
  lstm_node_config::LSTMNodeConfig = LSTMNodeConfig()
  gru_node_config::GRUNodeConfig = GRUNodeConfig()
end

function CheckConfig(x::NetworkConfig)
  length(x.distance_parameters) == 3 || throw(error("NetworkConfig : distance_parameters should be of length 3"))

  CheckConfig(x.input_node_config)
  CheckConfig(x.output_node_config)
  CheckConfig(x.hidden_node_config)
  CheckConfig(x.forward_connection_config)
  CheckConfig(x.backward_connection_config)
  CheckConfig(x.recurrent_hidden_node_config)
  CheckConfig(x.lstm_node_config)
  CheckConfig(x.gru_node_config)
  return
end

function Base.show(io::IO, x::NetworkConfig)
  println(io, summary(x))
end

