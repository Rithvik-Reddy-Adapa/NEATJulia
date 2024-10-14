#=
# Refer src/NEAT/Network/NetworkConfig.jl for NetworkConfig, CheckConfig
# Refer src/NEAT/Probabilities.jl for MutationProbability, CheckMutationProbability, CrossoverProbability, CheckCrossoverProbability
# Refer src/AbstractTypes.jl for Genes, Nodes, Connections, RecurrentConnections, HiddenNodes, RecurrentHiddenNodes, AllNEATTypes
# Refer src/Reference.jl for Reference
# Refer src/NEAT/NEAT.jl for NEAT
=#

using StatsBase, JLD2

export Init, AddForwardConnection, AddBackwardConnection, AddHiddenNodeForwardConnection, AddHiddenNodeBackwardConnection, AddRecurrentHiddenNodeForwardConnection, AddRecurrentHiddenNodeBackwardConnection, AddLSTMNodeForwardConnection, AddLSTMNodeBackwardConnection, AddGRUNodeForwardConnection, AddGRUNodeBackwardConnection, GetNetworkDistance, Crossover, Mutate, Save, Visualise

function Init(x::Network, network_config::NetworkConfig)
  x.input = Reference{Real}[Reference{Real}() for i = 1:x.n_inputs]
  x.output = Reference{Real}[Reference{Real}() for i = 1:x.n_outputs]

  # Add input nodes and initialise
  layer = Vector{Nodes}(undef, x.n_inputs)
  for i = 0x1:x.n_inputs
    x.genes[i] = InputNode(GIN = i, serial_number = i, super = x)
    push!(x.genes[i].input, x.input[i])
    Init(x.genes[i], network_config.input_node_config)
    layer[i] = x.genes[i]
  end
  push!(x.layers, layer)

  # Add output nodes and initialise
  layer = Vector{Nodes}(undef, x.n_outputs)
  for i = 0x1:x.n_outputs
    x.genes[i+x.n_inputs] = OutputNode(GIN = i+x.n_inputs, serial_number = i, super = x)
    x.genes[i+x.n_inputs].output = x.output[i]
    Init(x.genes[i+x.n_inputs], network_config.output_node_config)
    layer[i] = x.genes[i+x.n_inputs]
  end
  push!(x.layers, layer)

  # Add forward connections from every input node to every output node
  if network_config.start_fully_connected
    for (i,j,k) in zip((x.n_inputs+x.n_outputs).+(1:(x.n_inputs*x.n_outputs)), repeat(1:x.n_inputs, inner = x.n_outputs), repeat(x.n_inputs.+(1:x.n_outputs), outer = x.n_inputs))
      x.genes[i] = ForwardConnection(GIN = i, in_node = x.genes[j], out_node = x.genes[k], super = x)
      Init(x.genes[i], network_config.forward_connection_config)
    end
  end
end

function AddForwardConnection(x::Network, start_node::Nodes, stop_node::Nodes, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddForwardConnection : new_GIN = $(new_GIN) already exists")
  (start_node == stop_node) && error("AddForwardConnection : got start_node == stop_node")
  (start_node in values(x.genes)) || error("AddForwardConnection : start_node does not belong to this network")
  (start_node.super == x) || error("AddForwardConnection : super field of start_node does not point to this network")
  (stop_node in values(x.genes)) || error("AddForwardConnection : stop_node does not belong to this network")
  (stop_node.super == x) || error("AddForwardConnection : super field of stop_node does not point to this network")
  (start_node isa OutputNode) && error("AddForwardConnection : start_node cannot be of type OutputNode")
  (stop_node isa InputNode) && error("AddForwardConnection : stop_node cannot be of type InputNode")

  start_layer = GetNodePosition(start_node)[1]
  stop_layer = GetNodePosition(stop_node)[1]

  (stop_layer > start_layer) || error("AddForwardConnection : layer of stop_node <= layer of start_node")

  (stop_node in getfield.(start_node.out_connections, :out_node)) && error("AddForwardConnection : connection exists from start_node to stop_node")

  CheckConfig(network_config.forward_connection_config)
  if (start_node isa LSTMNode) || (stop_node isa LSTMNode)
    CheckConfig(network_config.lstm_node_config)
  end
  if (start_node isa GRUNode) || (stop_node isa GRUNode)
    CheckConfig(network_config.gru_node_config)
  end
  
  x.genes[new_GIN] = ForwardConnection(GIN = new_GIN, in_node = start_node, out_node = stop_node, super = x)
  Init(x.genes[new_GIN], network_config.forward_connection_config)

  if stop_node isa LSTMNode
    push!(stop_node.weight_f, Ref{Real}())
    push!(stop_node.weight_i, Ref{Real}())
    push!(stop_node.weight_c, Ref{Real}())
    push!(stop_node.weight_o, Ref{Real}())
    if ismissing(network_config.lstm_node_config.initial_weight[]) || isnan(network_config.lstm_node_config.initial_weight[])
      stop_node.weight_f[end][] = network_config.lstm_node_config.min_weight[] + (network_config.lstm_node_config.max_weight[] - network_config.lstm_node_config.min_weight[]) * rand(rng())
      stop_node.weight_i[end][] = network_config.lstm_node_config.min_weight[] + (network_config.lstm_node_config.max_weight[] - network_config.lstm_node_config.min_weight[]) * rand(rng())
      stop_node.weight_c[end][] = network_config.lstm_node_config.min_weight[] + (network_config.lstm_node_config.max_weight[] - network_config.lstm_node_config.min_weight[]) * rand(rng())
      stop_node.weight_o[end][] = network_config.lstm_node_config.min_weight[] + (network_config.lstm_node_config.max_weight[] - network_config.lstm_node_config.min_weight[]) * rand(rng())
    else
      stop_node.weight_f[end][] = network_config.lstm_node_config.initial_weight[]
      stop_node.weight_i[end][] = network_config.lstm_node_config.initial_weight[]
      stop_node.weight_c[end][] = network_config.lstm_node_config.initial_weight[]
      stop_node.weight_o[end][] = network_config.lstm_node_config.initial_weight[]
    end
  end

  if stop_node isa GRUNode
    push!(stop_node.weight_r, Ref{Real}())
    push!(stop_node.weight_u, Ref{Real}())
    push!(stop_node.weight_c, Ref{Real}())
    if ismissing(network_config.gru_node_config.initial_weight[]) || isnan(network_config.gru_node_config.initial_weight[])
      stop_node.weight_r[end][] = network_config.gru_node_config.min_weight[] + (network_config.gru_node_config.max_weight[] - network_config.gru_node_config.min_weight[]) * rand(rng())
      stop_node.weight_u[end][] = network_config.gru_node_config.min_weight[] + (network_config.gru_node_config.max_weight[] - network_config.gru_node_config.min_weight[]) * rand(rng())
      stop_node.weight_c[end][] = network_config.gru_node_config.min_weight[] + (network_config.gru_node_config.max_weight[] - network_config.gru_node_config.min_weight[]) * rand(rng())
    else
      stop_node.weight_r[end][] = network_config.gru_node_config.initial_weight[]
      stop_node.weight_u[end][] = network_config.gru_node_config.initial_weight[]
      stop_node.weight_c[end][] = network_config.gru_node_config.initial_weight[]
    end
  end

  return x.genes[new_GIN]
end

function AddBackwardConnection(x::Network, start_node::Nodes, stop_node::Nodes, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddBackwardConnection : new_GIN = $(new_GIN) already exists")
  (start_node == stop_node) && error("AddBackwardConnection : got start_node == stop_node")
  (start_node in values(x.genes)) || error("AddBackwardConnection : start_node does not belong to this network")
  (start_node.super == x) || error("AddBackwardConnection : super field of start_node does not point to this network")
  (stop_node in values(x.genes)) || error("AddBackwardConnection : stop_node does not belong to this network")
  (stop_node.super == x) || error("AddBackwardConnection : super field of stop_node does not point to this network")
  (start_node isa InputNode) && error("AddBackwardConnection : start_node cannot be of type InputNode")
  (stop_node isa OutputNode) && error("AddBackwardConnection : stop_node cannot be of type OutputNode")

  start_layer = GetNodePosition(start_node)[1]
  stop_layer = GetNodePosition(stop_node)[1]

  (stop_layer < start_layer) || error("AddBackwardConnection : layer of stop_node >= layer of start_node")

  (stop_node in getfield.(start_node.out_connections, :out_node)) && error("AddForwardConnection : connection exists from start_node to stop_node")

  CheckConfig(network_config.backward_connection_config)
  if (start_node isa LSTMNode) || (stop_node isa LSTMNode)
    CheckConfig(network_config.lstm_node_config)
  end
  if (start_node isa GRUNode) || (stop_node isa GRUNode)
    CheckConfig(network_config.gru_node_config)
  end
  
  x.genes[new_GIN] = BackwardConnection(GIN = new_GIN, in_node = start_node, out_node = stop_node, super = x)
  Init(x.genes[new_GIN], network_config.backward_connection_config)

  if stop_node isa LSTMNode
    push!(stop_node.weight_f, Ref{Real}())
    push!(stop_node.weight_i, Ref{Real}())
    push!(stop_node.weight_c, Ref{Real}())
    push!(stop_node.weight_o, Ref{Real}())
    if ismissing(network_config.lstm_node_config.initial_weight[]) || isnan(network_config.lstm_node_config.initial_weight[])
      stop_node.weight_f[end][] = network_config.lstm_node_config.min_weight[] + (network_config.lstm_node_config.max_weight[] - network_config.lstm_node_config.min_weight[]) * rand(rng())
      stop_node.weight_i[end][] = network_config.lstm_node_config.min_weight[] + (network_config.lstm_node_config.max_weight[] - network_config.lstm_node_config.min_weight[]) * rand(rng())
      stop_node.weight_c[end][] = network_config.lstm_node_config.min_weight[] + (network_config.lstm_node_config.max_weight[] - network_config.lstm_node_config.min_weight[]) * rand(rng())
      stop_node.weight_o[end][] = network_config.lstm_node_config.min_weight[] + (network_config.lstm_node_config.max_weight[] - network_config.lstm_node_config.min_weight[]) * rand(rng())
    else
      stop_node.weight_f[end][] = network_config.lstm_node_config.initial_weight[]
      stop_node.weight_i[end][] = network_config.lstm_node_config.initial_weight[]
      stop_node.weight_c[end][] = network_config.lstm_node_config.initial_weight[]
      stop_node.weight_o[end][] = network_config.lstm_node_config.initial_weight[]
    end
  end

  if stop_node isa GRUNode
    push!(stop_node.weight_r, Ref{Real}())
    push!(stop_node.weight_u, Ref{Real}())
    push!(stop_node.weight_c, Ref{Real}())
    if ismissing(network_config.gru_node_config.initial_weight[]) || isnan(network_config.gru_node_config.initial_weight[])
      stop_node.weight_r[end][] = network_config.gru_node_config.min_weight[] + (network_config.gru_node_config.max_weight[] - network_config.gru_node_config.min_weight[]) * rand(rng())
      stop_node.weight_u[end][] = gru_node_confg.gru_node_config.min_weight[] + (network_config.gru_node_config.max_weight[] - network_config.gru_node_config.min_weight[]) * rand(rng())
      stop_node.weight_c[end][] = gru_node_confg.gru_node_config.min_weight[] + (network_config.gru_node_config.max_weight[] - network_config.gru_node_config.min_weight[]) * rand(rng())
    else
      stop_node.weight_r[end][] = network_config.gru_node_config.initial_weight[]
      stop_node.weight_u[end][] = network_config.gru_node_config.initial_weight[]
      stop_node.weight_c[end][] = network_config.gru_node_config.initial_weight[]
    end
  end

  return x.genes[new_GIN]
end

function AddHiddenNodeForwardConnection(x::Network, connection::ForwardConnection, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddHiddenNodeForwardConnection : new_GIN = $(new_GIN) already exists")
  (connection in values(x.genes)) || error("AddHiddenNodeForwardConnection : connection does not belong to this network")
  (connection.super == x) || error("AddHiddenNodeForwardConnection : super field of connection does not point to this network")
  CheckConfig(network_config.hidden_node_config)
  CheckConfig(network_config.forward_connection_config)

  node = HiddenNode(GIN = new_GIN, super = x)
  Init(node, network_config.hidden_node_config)
  x.genes[new_GIN] = node

  start_layer = GetNodePosition(connection.in_node)[1]
  stop_layer = GetNodePosition(connection.out_node)[1]
  layer = rand(rng(), start_layer+0.5:0.5:stop_layer-0.5)
  if floor(layer) == layer # add node to an existing layer
    layer = Int(layer)
    push!(x.layers[layer], node)
  else # add node to a new layer
    layer = Int(ceil(layer))
    insert!(x.layers, layer, Nodes[node])
  end

  connection1 = AddForwardConnection(x, connection.in_node, node, new_GIN+0x1, network_config)
  connection2 = AddForwardConnection(x, node, connection.out_node, new_GIN+0x2, network_config)

  connection.enabled = false

  return node, connection1, connection2
end

function AddHiddenNodeBackwardConnection(x::Network, connection::BackwardConnection, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddHiddenNodeBackwardConnection : new_GIN = $(new_GIN) already exists")
  (connection in values(x.genes)) || error("AddHiddenNodeBackwardConnection : connection does not belong to this network")
  (connection.super == x) || error("AddHiddenNodeBackwardConnection : super field of connection does not point to this network")
  CheckConfig(network_config.hidden_node_config)
  CheckConfig(network_config.backward_connection_config)

  node = HiddenNode(GIN = new_GIN, super = x)
  Init(node, network_config.hidden_node_config)
  x.genes[new_GIN] = node

  start_layer = GetNodePosition(connection.in_node)[1]
  stop_layer = GetNodePosition(connection.out_node)[1]
  layer = rand(rng(), stop_layer+0.5:0.5:start_layer-0.5)
  if floor(layer) == layer # add node to an existing layer
    layer = Int(layer)
    push!(x.layers[layer], node)
  else # add node to a new layer
    layer = Int(ceil(layer))
    insert!(x.layers, layer, Nodes[node])
  end

  connection1 = AddBackwardConnection(x, connection.in_node, node, new_GIN+0x1, network_config)
  connection2 = AddBackwardConnection(x, node, connection.out_node, new_GIN+0x2, network_config)

  connection.enabled = false

  return node, connection1, connection2
end

function AddRecurrentHiddenNodeForwardConnection(x::Network, connection::ForwardConnection, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddRecurrentHiddenNodeForwardConnection : new_GIN = $(new_GIN) already exists")
  (connection in values(x.genes)) || error("AddRecurrentHiddenNodeForwardConnection : connection does not belong to this network")
  (connection.super == x) || error("AddRecurrentHiddenNodeForwardConnection : super field of connection does not point to this network")
  CheckConfig(network_config.recurrent_hidden_node_config)
  CheckConfig(network_config.forward_connection_config)

  node = RecurrentHiddenNode(GIN = new_GIN, super = x)
  Init(node, network_config.recurrent_hidden_node_config)
  x.genes[new_GIN] = node

  start_layer = GetNodePosition(connection.in_node)[1]
  stop_layer = GetNodePosition(connection.out_node)[1]
  layer = rand(rng(), start_layer+0.5:0.5:stop_layer-0.5)
  if floor(layer) == layer # add node to an existing layer
    layer = Int(layer)
    push!(x.layers[layer], node)
  else # add node to a new layer
    layer = Int(ceil(layer))
    insert!(x.layers, layer, Nodes[node])
  end

  connection1 = AddForwardConnection(x, connection.in_node, node, new_GIN+0x1, network_config)
  connection2 = AddForwardConnection(x, node, connection.out_node, new_GIN+0x2, network_config)

  connection.enabled = false

  return node, connection1, connection2
end

function AddRecurrentHiddenNodeBackwardConnection(x::Network, connection::BackwardConnection, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddRecurrentHiddenNodeBackwardConnection : new_GIN = $(new_GIN) already exists")
  (connection in values(x.genes)) || error("AddRecurrentHiddenNodeBackwardConnection : connection does not belong to this network")
  (connection.super == x) || error("AddRecurrentHiddenNodeBackwardConnection : super field of connection does not point to this network")
  CheckConfig(network_config.recurrent_hidden_node_config)
  CheckConfig(network_config.backward_connection_config)

  node = RecurrentHiddenNode(GIN = new_GIN, super = x)
  Init(node, network_config.recurrent_hidden_node_config)
  x.genes[new_GIN] = node

  start_layer = GetNodePosition(connection.in_node)[1]
  stop_layer = GetNodePosition(connection.out_node)[1]
  layer = rand(rng(), stop_layer+0.5:0.5:start_layer-0.5)
  if floor(layer) == layer # add node to an existing layer
    layer = Int(layer)
    push!(x.layers[layer], node)
  else # add node to a new layer
    layer = Int(ceil(layer))
    insert!(x.layers, layer, Nodes[node])
  end

  connection1 = AddBackwardConnection(x, connection.in_node, node, new_GIN+0x1, network_config)
  connection2 = AddBackwardConnection(x, node, connection.out_node, new_GIN+0x2, network_config)

  connection.enabled = false

  return node, connection1, connection2
end

function AddLSTMNodeForwardConnection(x::Network, connection::ForwardConnection, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddLSTMNodeForwardConnection : new_GIN = $(new_GIN) already exists")
  (connection in values(x.genes)) || error("AddLSTMNodeForwardConnection : connection does not belong to this network")
  (connection.super == x) || error("AddLSTMNodeForwardConnection : super field of connection does not point to this network")
  CheckConfig(network_config.lstm_node_config)
  CheckConfig(network_config.forward_connection_config)

  node = LSTMNode(GIN = new_GIN, super = x)
  Init(node, network_config.lstm_node_config)
  x.genes[new_GIN] = node

  start_layer = GetNodePosition(connection.in_node)[1]
  stop_layer = GetNodePosition(connection.out_node)[1]
  layer = rand(rng(), start_layer+0.5:0.5:stop_layer-0.5)
  if floor(layer) == layer # add node to an existing layer
    layer = Int(layer)
    push!(x.layers[layer], node)
  else # add node to a new layer
    layer = Int(ceil(layer))
    insert!(x.layers, layer, Nodes[node])
  end

  connection1 = AddForwardConnection(x, connection.in_node, node, new_GIN+0x1, network_config)
  connection2 = AddForwardConnection(x, node, connection.out_node, new_GIN+0x2, network_config)

  connection.enabled = false

  return node, connection1, connection2
end

function AddLSTMNodeBackwardConnection(x::Network, connection::BackwardConnection, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddLSTMNodeBackwardConnection : new_GIN = $(new_GIN) already exists")
  (connection in values(x.genes)) || error("AddLSTMNodeBackwardConnection : connection does not belong to this network")
  (connection.super == x) || error("AddLSTMNodeBackwardConnection : super field of connection does not point to this network")
  CheckConfig(network_config.lstm_node_config)
  CheckConfig(network_config.backward_connection_config)

  node = LSTMNode(GIN = new_GIN, super = x)
  Init(node, network_config.lstm_node_config)
  x.genes[new_GIN] = node

  start_layer = GetNodePosition(connection.in_node)[1]
  stop_layer = GetNodePosition(connection.out_node)[1]
  layer = rand(rng(), stop_layer+0.5:0.5:start_layer-0.5)
  if floor(layer) == layer # add node to an existing layer
    layer = Int(layer)
    push!(x.layers[layer], node)
  else # add node to a new layer
    layer = Int(ceil(layer))
    insert!(x.layers, layer, Nodes[node])
  end

  connection1 = AddBackwardConnection(x, connection.in_node, node, new_GIN+0x1, network_config)
  connection2 = AddBackwardConnection(x, node, connection.out_node, new_GIN+0x2, network_config)

  connection.enabled = false

  return node, connection1, connection2
end

function AddGRUNodeForwardConnection(x::Network, connection::ForwardConnection, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddGRUNodeForwardConnection : new_GIN = $(new_GIN) already exists")
  (connection in values(x.genes)) || error("AddGRUNodeForwardConnection : connection does not belong to this network")
  (connection.super == x) || error("AddGRUNodeForwardConnection : super field of connection does not point to this network")
  CheckConfig(network_config.gru_node_config)
  CheckConfig(network_config.forward_connection_config)

  node = GRUNode(GIN = new_GIN, super = x)
  Init(node, network_config.gru_node_config)
  x.genes[new_GIN] = node

  start_layer = GetNodePosition(connection.in_node)[1]
  stop_layer = GetNodePosition(connection.out_node)[1]
  layer = rand(rng(), start_layer+0.5:0.5:stop_layer-0.5)
  if floor(layer) == layer # add node to an existing layer
    layer = Int(layer)
    push!(x.layers[layer], node)
  else # add node to a new layer
    layer = Int(ceil(layer))
    insert!(x.layers, layer, Nodes[node])
  end

  connection1 = AddForwardConnection(x, connection.in_node, node, new_GIN+0x1, network_config)
  connection2 = AddForwardConnection(x, node, connection.out_node, new_GIN+0x2, network_config)

  connection.enabled = false

  return node, connection1, connection2
end

function AddGRUNodeBackwardConnection(x::Network, connection::BackwardConnection, new_GIN::Integer, network_config::NetworkConfig)
  new_GIN = Unsigned(new_GIN)

  (new_GIN in keys(x.genes)) && error("AddGRUNodeBackwardConnection : new_GIN = $(new_GIN) already exists")
  (connection in values(x.genes)) || error("AddGRUNodeBackwardConnection : connection does not belong to this network")
  (connection.super == x) || error("AddGRUNodeBackwardConnection : super field of connection does not point to this network")
  CheckConfig(network_config.gru_node_config)
  CheckConfig(network_config.backward_connection_config)

  node = GRUNode(GIN = new_GIN, super = x)
  Init(node, network_config.gru_node_config)
  x.genes[new_GIN] = node

  start_layer = GetNodePosition(connection.in_node)[1]
  stop_layer = GetNodePosition(connection.out_node)[1]
  layer = rand(rng(), stop_layer+0.5:0.5:start_layer-0.5)
  if floor(layer) == layer # add node to an existing layer
    layer = Int(layer)
    push!(x.layers[layer], node)
  else # add node to a new layer
    layer = Int(ceil(layer))
    insert!(x.layers, layer, Nodes[node])
  end

  connection1 = AddBackwardConnection(x, connection.in_node, node, new_GIN+0x1, network_config)
  connection2 = AddBackwardConnection(x, node, connection.out_node, new_GIN+0x2, network_config)

  connection.enabled = false

  return node, connection1, connection2
end

function GetNetworkDistance(x::Network, y::Network, network_config::NetworkConfig)
  CheckConfig(network_config)

  x_GINs = Unsigned[i.GIN for i in values(x.genes) if i.enabled[]]
  y_GINs = Unsigned[i.GIN for i in values(y.genes) if i.enabled[]]
  # x_GINs = collect(keys(x.genes))
  # y_GINs = collect(keys(y.genes))

  common_GINs = intersect(x_GINs, y_GINs)
  max_common_GIN = maximum(common_GINs)

  uncommon_GINs = union(setdiff(x_GINs, y_GINs), setdiff(y_GINs, x_GINs))

  n_disjoint_GINs = isempty(uncommon_GINs) ? 0 : sum(uncommon_GINs .< max_common_GIN)
  n_excess_GINs = isempty(uncommon_GINs) ? 0 : sum(uncommon_GINs .> max_common_GIN)

  common_connection_GINs = common_GINs[typeof.(getindex.((x.genes,), common_GINs)).<:Connections]
  sum_weights_difference = isempty(common_connection_GINs) ? 0 : sum( abs.(  getindex.(getfield.(getindex.((x.genes,), common_connection_GINs), (:weight,)))  .-  getindex.(getfield.(getindex.((y.genes,), common_connection_GINs), (:weight,)))  ) )

  distance = Real[n_disjoint_GINs, n_excess_GINs, sum_weights_difference] .* network_config.distance_parameters
  if network_config.normalise_distance
    max_number_of_GINs = max(length(x_GINs), length(y_GINs))
    distance ./= [max_number_of_GINs, max_number_of_GINs, 1]
  end
  distance = sum(distance)

  return distance
end

function Crossover(x::Network, y::Network, fitness_x::Real = Inf, fitness_y::Real = -Inf)
  parent1 = nothing
  parent2 = nothing
  if fitness_x >= fitness_y
    parent1 = x
    parent2 = y
  else
    parent1 = y
    parent2 = x
  end
  super = parent1.super
  parent1.super = missing
  ret = deepcopy(parent1)
  ret.super = super
  parent1.super = super
  ret.idx = 0
  ret.specie = 0
  for i in collect(keys(ret.genes))
    if haskey(parent2.genes, i) && (ret.genes[i] isa Connections) && rand(rng(), [true, false])
      ret.genes[i].weight[] = parent2.genes[i].weight[]
    elseif haskey(parent2.genes, i) && any( isa.((ret.genes[i],), [InputNode, OutputNode, HiddenNode, RecurrentHiddenNode]) ) && rand(rng(), [true, false])
      ret.genes[i].bias[] = parent2.genes[i].bias[]
      ret.genes[i].activation_function[] = parent2.genes[i].activation_function[]
    elseif haskey(parent2.genes, i) && (ret.genes[i] isa LSTMNode) && rand(rng(), [true, false])
      ret.genes[i].bias_f[] = parent2.genes[i].bias_f[]
      ret.genes[i].bias_i[] = parent2.genes[i].bias_i[]
      ret.genes[i].bias_c[] = parent2.genes[i].bias_c[]
      ret.genes[i].bias_o[] = parent2.genes[i].bias_o[]
    elseif haskey(parent2.genes, i) && (ret.genes[i] isa GRUNode) && rand(rng(), [true, false])
      ret.genes[i].bias_r[] = parent2.genes[i].bias_r[]
      ret.genes[i].bias_u[] = parent2.genes[i].bias_u[]
      ret.genes[i].bias_c[] = parent2.genes[i].bias_c[]
    end
  end
  return ret
end

function _global_change_weight(x::Ref{Real}, config::Configs)
  CheckConfig(config)

  x[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
  return x[]
end

function _global_shift_weight(x::Ref{Real}, config::Configs)
  CheckConfig(config)

  method = (config.shift_weight_method[] == "Random") ? rand(rng(), ["Uniform", "Gaussian"]) : config.shift_weight_method[]

  if method == "Uniform"
    x[] += rand(rng()) * config.shift_weight[] * rand(rng(), [1, -1])
  elseif method == "Gaussian"
    x[] += randn(rng()) * config.std_weight[]
  end
  x[] = clamp(x[], config.min_weight[], config.max_weight[])
  return x[]
end

function _global_change_bias(x::Ref{Real}, config::Configs)
  CheckConfig(config)

  x[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  return x[]
end

function _global_shift_bias(x::Ref{Real}, config::Configs)
  CheckConfig(config)

  method = (config.shift_bias_method[] == "Random") ? rand(rng(), ["Uniform", "Gaussian"]) : config.shift_bias_method[]

  if method == "Uniform"
    x[] += rand(rng()) * config.shift_bias[] * rand(rng(), [1, -1])
  elseif method == "Gaussian"
    x[] += randn(rng()) * config.std_bias[]
  end
  x[] = clamp(x[], config.min_bias[], config.max_bias[])
  return x[]
end

function _global_change_activation_function(x::Ref{Function}, config::Configs)
  CheckConfig(config)

  x[] = rand(rng(), config.activation_functions[])
  return x[]
end

function _global_toggle_enable(x::Ref{Bool})
  x[] = !(x[])
  return x[]
end

function _global_enable_gene(x::Ref{Bool})
  x[] = true
  return
end

function _global_disable_gene(x::Ref{Bool})
  x[] = false
  return
end

function Mutate(x::Network, prob::MutationProbability, network_config::NetworkConfig; new_GIN::Integer = 0x0)
  new_GIN = Unsigned(new_GIN)
  CheckMutationProbability(prob)
  mutation_number = sample(rng(), 1:length(prob), Weights(prob[:]))

  if mutation_number == 1 # no mutation
    return 1
  elseif mutation_number == 2 # global change weight
    selected_weights = []
    for i in collect(values(x.genes))
      if i.enabled[]
        if i isa ForwardConnection
          push!(selected_weights, [i.GIN, i.weight, network_config.forward_connection_config])
        elseif i isa BackwardConnection
          push!(selected_weights, [i.GIN, i.weight, network_config.backward_connection_config])
        elseif i isa RecurrentHiddenNode
          push!(selected_weights, [i.GIN, i.weight, network_config.recurrent_hidden_node_config])
        elseif i isa LSTMNode
          for (wf, wi, wc, wo) in zip(i.weight_f, i.weight_i, i.weight_c, i.weight_o)
            push!(selected_weights, [i.GIN, wf, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wi, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wc, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wo, network_config.lstm_node_config])
          end
        elseif i isa GRUNode
          for (wr, wu, wc) in zip(i.weight_r, i.weight_u, i.weight_c)
            push!(selected_weights, [i.GIN, wr, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wu, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wc, network_config.lstm_node_config])
          end
        end
      end
    end
    if isempty(selected_weights)
      new_prob = deepcopy(prob)
      new_prob.global_change_weight = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_weight = rand(rng(), selected_weights)
    _global_change_weight(random_weight[2], random_weight[3])
    return 2, random_weight[1], random_weight[2][]
  elseif mutation_number == 3 # global shift weight
    selected_weights = []
    for i in collect(values(x.genes))
      if i.enabled[]
        if i isa ForwardConnection
          push!(selected_weights, [i.GIN, i.weight, network_config.forward_connection_config])
        elseif i isa BackwardConnection
          push!(selected_weights, [i.GIN, i.weight, network_config.backward_connection_config])
        elseif i isa RecurrentHiddenNode
          push!(selected_weights, [i.GIN, i.weight, network_config.recurrent_hidden_node_config])
        elseif i isa LSTMNode
          for (wf, wi, wc, wo) in zip(i.weight_f, i.weight_i, i.weight_c, i.weight_o)
            push!(selected_weights, [i.GIN, wf, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wi, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wc, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wo, network_config.lstm_node_config])
          end
        elseif i isa GRUNode
          for (wr, wu, wc) in zip(i.weight_r, i.weight_u, i.weight_c)
            push!(selected_weights, [i.GIN, wr, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wu, network_config.lstm_node_config])
            push!(selected_weights, [i.GIN, wc, network_config.lstm_node_config])
          end
        end
      end
    end
    if isempty(selected_weights)
      new_prob = deepcopy(prob)
      new_prob.global_change_weight = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_weight = rand(rng(), selected_weights)
    _global_shift_weight(random_weight[2], random_weight[3])
    return 3, random_weight[1], random_weight[2][]
  elseif mutation_number == 4 # global change bias
    selected_biases = []
    for i in collect(values(x.genes))
      if i.enabled[]
        if i isa InputNode
          push!(selected_biases, [i.GIN, i.bias, network_config.input_node_config])
        elseif i isa OutputNode
          push!(selected_biases, [i.GIN, i.bias, network_config.output_node_config])
        elseif i isa HiddenNode
          push!(selected_biases, [i.GIN, i.bias, network_config.hidden_node_config])
        elseif i isa RecurrentHiddenNode
          push!(selected_biases, [i.GIN, i.bias, network_config.recurrent_hidden_node_config])
        elseif i isa LSTMNode
          push!(selected_biases, [i.GIN, i.bias_f, network_config.lstm_node_config])
          push!(selected_biases, [i.GIN, i.bias_i, network_config.lstm_node_config])
          push!(selected_biases, [i.GIN, i.bias_c, network_config.lstm_node_config])
          push!(selected_biases, [i.GIN, i.bias_o, network_config.lstm_node_config])
        elseif i isa GRUNode
          push!(selected_biases, [i.GIN, i.bias_r, network_config.gru_node_config])
          push!(selected_biases, [i.GIN, i.bias_u, network_config.gru_node_config])
          push!(selected_biases, [i.GIN, i.bias_c, network_config.gru_node_config])
        end
      end
    end
    random_bias = rand(rng(), selected_biases)
    _global_change_bias(random_bias[2], random_bias[3])
    return 4, random_bias[1], random_bias[2][]
  elseif mutation_number == 5 # global shift bias
    selected_biases = []
    for i in collect(values(x.genes))
      if i.enabled[]
        if i isa InputNode
          push!(selected_biases, [i.GIN, i.bias, network_config.input_node_config])
        elseif i isa OutputNode
          push!(selected_biases, [i.GIN, i.bias, network_config.output_node_config])
        elseif i isa HiddenNode
          push!(selected_biases, [i.GIN, i.bias, network_config.hidden_node_config])
        elseif i isa RecurrentHiddenNode
          push!(selected_biases, [i.GIN, i.bias, network_config.recurrent_hidden_node_config])
        elseif i isa LSTMNode
          push!(selected_biases, [i.GIN, i.bias_f, network_config.lstm_node_config])
          push!(selected_biases, [i.GIN, i.bias_i, network_config.lstm_node_config])
          push!(selected_biases, [i.GIN, i.bias_c, network_config.lstm_node_config])
          push!(selected_biases, [i.GIN, i.bias_o, network_config.lstm_node_config])
        elseif i isa GRUNode
          push!(selected_biases, [i.GIN, i.bias_r, network_config.gru_node_config])
          push!(selected_biases, [i.GIN, i.bias_u, network_config.gru_node_config])
          push!(selected_biases, [i.GIN, i.bias_c, network_config.gru_node_config])
        end
      end
    end
    random_bias = rand(rng(), selected_biases)
    _global_shift_bias(random_bias[2], random_bias[3])
    return 5, random_bias[1], random_bias[2][]
  elseif mutation_number == 6 # global change activation function
    selected_activation_functions = []
    for i in collect(values(x.genes))
      if i.enabled[]
        if i isa InputNode
          push!(selected_activation_functions, [i.GIN, i.activation_function, network_config.input_node_config])
        elseif i isa OutputNode
          push!(selected_activation_functions, [i.GIN, i.activation_function, network_config.output_node_config])
        elseif i isa HiddenNode
          push!(selected_activation_functions, [i.GIN, i.activation_function, network_config.hidden_node_config])
        elseif i isa RecurrentHiddenNode
          push!(selected_activation_functions, [i.GIN, i.activation_function, network_config.recurrent_hidden_node_config])
        end
      end
    end
    random_activation_function = rand(rng(), selected_activation_functions)
    _global_change_activation_function(random_activation_function[2], random_activation_function[3])
    return 6, random_activation_function[1], random_activation_function[2][]
  elseif mutation_number == 7 # global toggle enable
    selected_genes = [i for i in values(x.genes) if i isa HiddenNodes || i isa Connections]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.global_toggle_enable = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = !random_gene.enabled[]
    return 7, random_gene.GIN, random_gene.enabled[]
  elseif mutation_number == 8 # global enable gene
    selected_genes = [i for i in values(x.genes) if (i isa HiddenNodes || i isa Connections) && !i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.global_enable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = true
    return 8, random_gene.GIN
  elseif mutation_number == 9 # global disable gene
    selected_genes = [i for i in values(x.genes) if (i isa HiddenNodes || i isa Connections) && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.global_enable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = false
    return 9, random_gene.GIN
  elseif mutation_number == 10 # input node change bias
    selected_genes = x.layers[1]
    random_gene = rand(rng(), selected_genes)
    ChangeBias(random_gene, network_config.input_node_config)
    return 10, random_gene.GIN, random_gene.bias[]
  elseif mutation_number == 11 # input node shift bias
    selected_genes = x.layers[1]
    random_gene = rand(rng(), selected_genes)
    ShiftBias(random_gene, network_config.input_node_config)
    return 11, random_gene.GIN, random_gene.bias[]
  elseif mutation_number == 12 # input node change activation function
    selected_genes = x.layers[1]
    random_gene = rand(rng(), selected_genes)
    ChangeActivationFunction(random_gene, network_config.input_node_config)
    return 12, random_gene.GIN, random_gene.activation_function[]
  elseif mutation_number == 13 # output node change bias
    selected_genes = x.layers[end]
    random_gene = rand(rng(), selected_genes)
    ChangeBias(random_gene, network_config.output_node_config)
    return 13, random_gene.GIN, random_gene.bias[]
  elseif mutation_number == 14 # output node shift bias
    selected_genes = x.layers[end]
    random_gene = rand(rng(), selected_genes)
    ShiftBias(random_gene, network_config.output_node_config)
    return 14, random_gene.GIN, random_gene.bias[]
  elseif mutation_number == 15 # output node change activation function
    selected_genes = x.layers[end]
    random_gene = rand(rng(), selected_genes)
    ChangeActivationFunction(random_gene, network_config.output_node_config)
    return 15, random_gene.GIN, random_gene.activation_function[]
  elseif mutation_number == 16 # add forward connection
    network_layers = GetLayers(x, simple = true)
    non_empty_network_layers = [i for i = 1:length(network_layers) if !(isempty(network_layers[i]))]
    add_connection = false
    start_layer = 0
    stop_layer = 0
    start_node = missing
    stop_node = missing
    for i = 1:50
      start_layer = rand(rng(), 1:length(non_empty_network_layers)-1)
      stop_layer = rand(rng(), start_layer+1:length(non_empty_network_layers))
      start_layer = non_empty_network_layers[start_layer]
      stop_layer = non_empty_network_layers[stop_layer]

      start_node = x.genes[ rand(rng(), network_layers[start_layer]) ]
      stop_node = x.genes[ rand(rng(), network_layers[stop_layer]) ]

      stop_node in getfield.(start_node.out_connections, :out_node) ? nothing : (add_connection = true; break;)
    end
    if add_connection
      if ismissing(x.super)
        new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      else
        new_GIN = findfirst( (x.super.GIN.start_node .== start_node.GIN .&& x.super.GIN.stop_node .== stop_node.GIN) )
      end

      if isnothing(new_GIN)
        new_GIN = x.super.GIN.GIN[end] + 0x1
        AddForwardConnection(x, start_node, stop_node, new_GIN, network_config)
        push!(x.super.GIN.GIN, new_GIN)
        push!(x.super.GIN.type, ForwardConnection)
        push!(x.super.GIN.start_node, start_node.GIN)
        push!(x.super.GIN.stop_node, stop_node.GIN)
        return 16, "new", new_GIN
      else
        AddForwardConnection(x, start_node, stop_node, new_GIN, network_config)
        if ismissing(x.super)
          return 16, new_GIN
        else
          return 16, "old", new_GIN
        end
      end
    else
      new_prob = deepcopy(prob)
      new_prob.add_forward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
  elseif mutation_number == 17 # forward connection change weight
    selected_genes = [i for i in values(x.genes) if i isa ForwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.forward_connection_change_weight = 0.0
      new_prob.forward_connection_shift_weight = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ChangeWeight(random_gene, network_config.forward_connection_config)
    return 17, random_gene.GIN, random_gene.weight[]
  elseif mutation_number == 18 # forward connection shift weight
    selected_genes = [i for i in values(x.genes) if i isa ForwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.forward_connection_shift_weight = 0.0
      new_prob.forward_connection_change_weight = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ShiftWeight(random_gene, network_config.forward_connection_config)
    return 18, random_gene.GIN, random_gene.weight[]
  elseif mutation_number == 19 # forward connection toggle enable
    selected_genes = [i for i in values(x.genes) if i isa ForwardConnection]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.forward_connection_change_weight = 0.0
      new_prob.forward_connection_shift_weight = 0.0
      new_prob.forward_connection_toggle_enable = 0.0
      new_prob.forward_connection_enable_gene = 0.0
      new_prob.forward_connection_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = !random_gene.enabled[]
    return 19, random_gene.GIN, random_gene.enabled[]
  elseif mutation_number == 20 # forward connection enable gene
    selected_genes = [i for i in values(x.genes) if i isa ForwardConnection && !i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.forward_connection_enable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = true
    return 20, random_gene.GIN
  elseif mutation_number == 21 # forward connection disable gene
    selected_genes = [i for i in values(x.genes) if i isa ForwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.forward_connection_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = false
    return 21, random_gene.GIN
  elseif mutation_number == 22 # add backward connection
    network_layers = GetLayers(x, simple = true)
    non_empty_network_layers = [i for i = 1:length(network_layers) if !(isempty(network_layers[i]))]
    add_connection = false
    start_layer = 0
    stop_layer = 0
    start_node = missing
    stop_node = missing
    for i = 1:50
      stop_layer = rand(rng(), 1:length(non_empty_network_layers)-1)
      start_layer = rand(rng(), stop_layer+1:length(non_empty_network_layers))
      stop_layer = non_empty_network_layers[stop_layer]
      start_layer = non_empty_network_layers[start_layer]

      start_node = x.genes[ rand(rng(), network_layers[start_layer]) ]
      stop_node = x.genes[ rand(rng(), network_layers[stop_layer]) ]

      stop_node in getfield.(start_node.out_connections, :out_node) ? nothing : (add_connection = true; break;)
    end
    if add_connection
      if ismissing(x.super)
        new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      else
        new_GIN = findfirst( (x.super.GIN.start_node .== start_node.GIN .&& x.super.GIN.stop_node .== stop_node.GIN) )
      end

      if isnothing(new_GIN)
        new_GIN = x.super.GIN.GIN[end] + 0x1
        AddBackwardConnection(x, start_node, stop_node, new_GIN, network_config)
        push!(x.super.GIN.GIN, new_GIN)
        push!(x.super.GIN.type, BackwardConnection)
        push!(x.super.GIN.start_node, start_node.GIN)
        push!(x.super.GIN.stop_node, stop_node.GIN)
        return 22, "new", new_GIN
      else
        AddBackwardConnection(x, start_node, stop_node, new_GIN, network_config)
        if ismissing(x.super)
          return 22, new_GIN
        else
          return 22, "old", new_GIN
        end
      end
    else
      new_prob = deepcopy(prob)
      new_prob.add_backward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
  elseif mutation_number == 23 # backward connection change weight
    selected_genes = [i for i in values(x.genes) if i isa BackwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.backward_connection_change_weight = 0.0
      new_prob.backward_connection_shift_weight = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ChangeWeight(random_gene, network_config.backward_connection_config)
    return 23, random_gene.GIN, random_gene.weight[]
  elseif mutation_number == 24 # backward connection shift weight
    selected_genes = [i for i in values(x.genes) if i isa BackwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.backward_connection_shift_weight = 0.0
      new_prob.backward_connection_change_weight = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ShiftWeight(random_gene, network_config.backward_connection_config)
    return 24, random_gene.GIN, random_gene.weight[]
  elseif mutation_number == 25 # backward connection toggle enable
    selected_genes = [i for i in values(x.genes) if i isa BackwardConnection]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.backward_connection_change_weight = 0.0
      new_prob.backward_connection_shift_weight = 0.0
      new_prob.backward_connection_toggle_enable = 0.0
      new_prob.backward_connection_enable_gene = 0.0
      new_prob.backward_connection_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = !random_gene.enabled[]
    return 25, random_gene.GIN, random_gene.enabled[]
  elseif mutation_number == 26 # backward connection enable gene
    selected_genes = [i for i in values(x.genes) if i isa BackwardConnection && !i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.backward_connection_enable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = true
    return 26, random_gene.GIN
  elseif mutation_number == 27 # backward connection disable gene
    selected_genes = [i for i in values(x.genes) if i isa BackwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.backward_connection_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = false
    return 27, random_gene.GIN
  elseif mutation_number == 28 # add hidden node forward connection
    selected_genes = [i for i in values(x.genes) if i isa ForwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.add_hidden_node_forward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    if ismissing(x.super)
      new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      AddHiddenNodeForwardConnection(x, random_gene, new_GIN, network_config)
    else
      new_GIN = x.super.GIN.GIN[end] + 0x1
      AddHiddenNodeForwardConnection(x, random_gene, new_GIN, network_config)
      push!(x.super.GIN.GIN, new_GIN)
      push!(x.super.GIN.type, HiddenNode)
      push!(x.super.GIN.start_node, 0x0)
      push!(x.super.GIN.stop_node, 0x0)
      push!(x.super.GIN.GIN, new_GIN+0x1)
      push!(x.super.GIN.type, ForwardConnection)
      push!(x.super.GIN.start_node, random_gene.in_node.GIN)
      push!(x.super.GIN.stop_node, new_GIN)
      push!(x.super.GIN.GIN, new_GIN+0x2)
      push!(x.super.GIN.type, ForwardConnection)
      push!(x.super.GIN.start_node, new_GIN)
      push!(x.super.GIN.stop_node, random_gene.out_node.GIN)
    end
    return 28, new_GIN, new_GIN+0x1, new_GIN+0x2
  elseif mutation_number == 29 # add hidden node backward connection
    selected_genes = [i for i in values(x.genes) if i isa BackwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.add_hidden_node_backward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    if ismissing(x.super)
      new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      AddHiddenNodeBackwardConnection(x, random_gene, new_GIN, network_config)
    else
      new_GIN = x.super.GIN.GIN[end] + 0x1
      AddHiddenNodeBackwardConnection(x, random_gene, new_GIN, network_config)
      push!(x.super.GIN.GIN, new_GIN)
      push!(x.super.GIN.type, HiddenNode)
      push!(x.super.GIN.start_node, 0x0)
      push!(x.super.GIN.stop_node, 0x0)
      push!(x.super.GIN.GIN, new_GIN+0x1)
      push!(x.super.GIN.type, BackwardConnection)
      push!(x.super.GIN.start_node, random_gene.in_node.GIN)
      push!(x.super.GIN.stop_node, new_GIN)
      push!(x.super.GIN.GIN, new_GIN+0x2)
      push!(x.super.GIN.type, BackwardConnection)
      push!(x.super.GIN.start_node, new_GIN)
      push!(x.super.GIN.stop_node, random_gene.out_node.GIN)
    end
    return 29, new_GIN, new_GIN+0x1, new_GIN+0x2
  elseif mutation_number == 30 # hidden node change bias
    selected_genes = [i for i in values(x.genes) if i isa HiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.hidden_node_change_bias = 0.0
      new_prob.hidden_node_shift_bias = 0.0
      new_prob.hidden_node_change_activation_function = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ChangeBias(random_gene, network_config.hidden_node_config)
    return 30, random_gene.GIN, random_gene.bias[]
  elseif mutation_number == 31 # hidden node shift bias
    selected_genes = [i for i in values(x.genes) if i isa HiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.hidden_node_change_bias = 0.0
      new_prob.hidden_node_shift_bias = 0.0
      new_prob.hidden_node_change_activation_function = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ShiftBias(random_gene, network_config.hidden_node_config)
    return 31, random_gene.GIN, random_gene.bias[]
  elseif mutation_number == 32 # hidden node change activation function
    selected_genes = [i for i in values(x.genes) if i isa HiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.hidden_node_change_bias = 0.0
      new_prob.hidden_node_shift_bias = 0.0
      new_prob.hidden_node_change_activation_function = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ChangeActivationFunction(random_gene, network_config.hidden_node_config)
    return 32, random_gene.GIN, random_gene.activation_function[]
  elseif mutation_number == 33 # hidden node toggle enable
    selected_genes = [i for i in values(x.genes) if i isa HiddenNode]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.hidden_node_change_bias = 0.0
      new_prob.hidden_node_shift_bias = 0.0
      new_prob.hidden_node_change_activation_function = 0.0
      new_prob.hidden_node_toggle_enable = 0.0
      new_prob.hidden_node_enable_gene = 0.0
      new_prob.hidden_node_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = !random_gene.enabled[]
    return 33, random_gene.GIN, random_gene.enabled[]
  elseif mutation_number == 34 # hidden node enable gene
    selected_genes = [i for i in values(x.genes) if i isa HiddenNode && !i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.hidden_node_enable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = true
    return 34, random_gene.GIN
  elseif mutation_number == 35 # hidden node disable gene
    selected_genes = [i for i in values(x.genes) if i isa HiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.hidden_node_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = false
    return 35, random_gene.GIN
  elseif mutation_number == 36 # add recurrent hidden node forward connection
    selected_genes = [i for i in values(x.genes) if i isa ForwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.add_recurrent_hidden_node_forward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    if ismissing(x.super)
      new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      AddRecurrentHiddenNodeForwardConnection(x, random_gene, new_GIN, network_config)
    else
      new_GIN = x.super.GIN.GIN[end] + 0x1
      AddRecurrentHiddenNodeForwardConnection(x, random_gene, new_GIN, network_config)
      push!(x.super.GIN.GIN, new_GIN)
      push!(x.super.GIN.type, RecurrentHiddenNode)
      push!(x.super.GIN.start_node, 0x0)
      push!(x.super.GIN.stop_node, 0x0)
      push!(x.super.GIN.GIN, new_GIN+0x1)
      push!(x.super.GIN.type, ForwardConnection)
      push!(x.super.GIN.start_node, random_gene.in_node.GIN)
      push!(x.super.GIN.stop_node, new_GIN)
      push!(x.super.GIN.GIN, new_GIN+0x2)
      push!(x.super.GIN.type, ForwardConnection)
      push!(x.super.GIN.start_node, new_GIN)
      push!(x.super.GIN.stop_node, random_gene.out_node.GIN)
    end
    return 36, new_GIN, new_GIN+0x1, new_GIN+0x2
  elseif mutation_number == 37 # add recurrent hidden node backward connection
    selected_genes = [i for i in values(x.genes) if i isa BackwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.add_recurrent_hidden_node_backward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    if ismissing(x.super)
      new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      AddRecurrentHiddenNodeBackwardConnection(x, random_gene, new_GIN, network_config)
    else
      new_GIN = x.super.GIN.GIN[end] + 0x1
      AddRecurrentHiddenNodeBackwardConnection(x, random_gene, new_GIN, network_config)
      push!(x.super.GIN.GIN, new_GIN)
      push!(x.super.GIN.type, RecurrentHiddenNode)
      push!(x.super.GIN.start_node, 0x0)
      push!(x.super.GIN.stop_node, 0x0)
      push!(x.super.GIN.GIN, new_GIN+0x1)
      push!(x.super.GIN.type, BackwardConnection)
      push!(x.super.GIN.start_node, random_gene.in_node.GIN)
      push!(x.super.GIN.stop_node, new_GIN)
      push!(x.super.GIN.GIN, new_GIN+0x2)
      push!(x.super.GIN.type, BackwardConnection)
      push!(x.super.GIN.start_node, new_GIN)
      push!(x.super.GIN.stop_node, random_gene.out_node.GIN)
    end
    return 37, new_GIN, new_GIN+0x1, new_GIN+0x2
  elseif mutation_number == 38 # recurrent hidden node change weight
    selected_genes = [i for i in values(x.genes) if i isa RecurrentHiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.recurrent_hidden_node_change_weight = 0.0
      new_prob.recurrent_hidden_node_shift_weight = 0.0
      new_prob.recurrent_hidden_node_change_bias = 0.0
      new_prob.recurrent_hidden_node_shift_bias = 0.0
      new_prob.recurrent_hidden_node_change_activation_function = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ChangeWeight(random_gene, network_config.recurrent_hidden_node_config)
    return 38, random_gene.GIN, random_gene.weight[]
  elseif mutation_number == 39 # recurrent hidden node shift weight
    selected_genes = [i for i in values(x.genes) if i isa RecurrentHiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.recurrent_hidden_node_change_weight = 0.0
      new_prob.recurrent_hidden_node_shift_weight = 0.0
      new_prob.recurrent_hidden_node_change_bias = 0.0
      new_prob.recurrent_hidden_node_shift_bias = 0.0
      new_prob.recurrent_hidden_node_change_activation_function = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ShiftWeight(random_gene, network_config.recurrent_hidden_node_config)
    return 39, random_gene.GIN, random_gene.weight[]
  elseif mutation_number == 40 # recurrent hidden node change bias
    selected_genes = [i for i in values(x.genes) if i isa RecurrentHiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.recurrent_hidden_node_change_weight = 0.0
      new_prob.recurrent_hidden_node_shift_weight = 0.0
      new_prob.recurrent_hidden_node_change_bias = 0.0
      new_prob.recurrent_hidden_node_shift_bias = 0.0
      new_prob.recurrent_hidden_node_change_activation_function = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ChangeBias(random_gene, network_config.recurrent_hidden_node_config)
    return 40, random_gene.GIN, random_gene.bias[]
  elseif mutation_number == 41 # recurrent hidden node shift bias
    selected_genes = [i for i in values(x.genes) if i isa RecurrentHiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.recurrent_hidden_node_change_weight = 0.0
      new_prob.recurrent_hidden_node_shift_weight = 0.0
      new_prob.recurrent_hidden_node_change_bias = 0.0
      new_prob.recurrent_hidden_node_shift_bias = 0.0
      new_prob.recurrent_hidden_node_change_activation_function = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ShiftBias(random_gene, network_config.recurrent_hidden_node_config)
    return 41, random_gene.GIN, random_gene.bias[]
  elseif mutation_number == 42 # recurrent hidden node change activation function
    selected_genes = [i for i in values(x.genes) if i isa RecurrentHiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.recurrent_hidden_node_change_weight = 0.0
      new_prob.recurrent_hidden_node_shift_weight = 0.0
      new_prob.recurrent_hidden_node_change_bias = 0.0
      new_prob.recurrent_hidden_node_shift_bias = 0.0
      new_prob.recurrent_hidden_node_change_activation_function = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ChangeActivationFunction(random_gene, network_config.recurrent_hidden_node_config)
    return 42, random_gene.GIN, random_gene.activation_function[]
  elseif mutation_number == 43 # recurrent hidden node toggle enable
    selected_genes = [i for i in values(x.genes) if i isa RecurrentHiddenNode]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.recurrent_hidden_node_change_weight = 0.0
      new_prob.recurrent_hidden_node_shift_weight = 0.0
      new_prob.recurrent_hidden_node_change_bias = 0.0
      new_prob.recurrent_hidden_node_shift_bias = 0.0
      new_prob.recurrent_hidden_node_change_activation_function = 0.0
      new_prob.recurrent_hidden_node_toggle_enable = 0.0
      new_prob.recurrent_hidden_node_enable_gene = 0.0
      new_prob.recurrent_hidden_node_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = !random_gene.enabled[]
    return 43, random_gene.GIN, random_gene.enabled[]
  elseif mutation_number == 44 # recurrent hidden node enable gene
    selected_genes = [i for i in values(x.genes) if i isa RecurrentHiddenNode && !i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.recurrent_hidden_node_enable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = true
    return 44, random_gene.GIN
  elseif mutation_number == 45 # recurrent hidden node disable gene
    selected_genes = [i for i in values(x.genes) if i isa RecurrentHiddenNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.recurrent_hidden_node_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = false
    return 45, random_gene.GIN
  elseif mutation_number == 46 # add lstm node forward connection
    selected_genes = [i for i in values(x.genes) if i isa ForwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.add_lstm_node_forward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    if ismissing(x.super)
      new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      AddLSTMNodeForwardConnection(x, random_gene, new_GIN, network_config)
    else
      new_GIN = x.super.GIN.GIN[end] + 0x1
      AddLSTMNodeForwardConnection(x, random_gene, new_GIN, network_config)
      push!(x.super.GIN.GIN, new_GIN)
      push!(x.super.GIN.type, LSTMNode)
      push!(x.super.GIN.start_node, 0x0)
      push!(x.super.GIN.stop_node, 0x0)
      push!(x.super.GIN.GIN, new_GIN+0x1)
      push!(x.super.GIN.type, ForwardConnection)
      push!(x.super.GIN.start_node, random_gene.in_node.GIN)
      push!(x.super.GIN.stop_node, new_GIN)
      push!(x.super.GIN.GIN, new_GIN+0x2)
      push!(x.super.GIN.type, ForwardConnection)
      push!(x.super.GIN.start_node, new_GIN)
      push!(x.super.GIN.stop_node, random_gene.out_node.GIN)
    end
    return 46, new_GIN, new_GIN+0x1, new_GIN+0x2
  elseif mutation_number == 47 # add lstm node backward connection
    selected_genes = [i for i in values(x.genes) if i isa BackwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.add_lstm_node_backward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    if ismissing(x.super)
      new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      AddLSTMNodeBackwardConnection(x, random_gene, new_GIN, network_config)
    else
      new_GIN = x.super.GIN.GIN[end] + 0x1
      AddLSTMNodeBackwardConnection(x, random_gene, new_GIN, network_config)
      push!(x.super.GIN.GIN, new_GIN)
      push!(x.super.GIN.type, LSTMNode)
      push!(x.super.GIN.start_node, 0x0)
      push!(x.super.GIN.stop_node, 0x0)
      push!(x.super.GIN.GIN, new_GIN+0x1)
      push!(x.super.GIN.type, BackwardConnection)
      push!(x.super.GIN.start_node, random_gene.in_node.GIN)
      push!(x.super.GIN.stop_node, new_GIN)
      push!(x.super.GIN.GIN, new_GIN+0x2)
      push!(x.super.GIN.type, BackwardConnection)
      push!(x.super.GIN.start_node, new_GIN)
      push!(x.super.GIN.stop_node, random_gene.out_node.GIN)
    end
    return 47, new_GIN, new_GIN+0x1, new_GIN+0x2
  elseif mutation_number == 48 # lstm node change weight
    selected_genes = [i for i in values(x.genes) if i isa LSTMNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.lstm_node_change_weight = 0.0
      new_prob.lstm_node_shift_weight = 0.0
      new_prob.lstm_node_change_bias = 0.0
      new_prob.lstm_node_shift_bias = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ret = ChangeWeight(random_gene, network_config.lstm_node_config)
    return 48, random_gene.GIN, ret
  elseif mutation_number == 49 # lstm node shift weight
    selected_genes = [i for i in values(x.genes) if i isa LSTMNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.lstm_node_change_weight = 0.0
      new_prob.lstm_node_shift_weight = 0.0
      new_prob.lstm_node_change_bias = 0.0
      new_prob.lstm_node_shift_bias = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ret = ShiftWeight(random_gene, network_config.lstm_node_config)
    return 49, random_gene.GIN, ret
  elseif mutation_number == 50 # lstm node change bias
    selected_genes = [i for i in values(x.genes) if i isa LSTMNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.lstm_node_change_weight = 0.0
      new_prob.lstm_node_shift_weight = 0.0
      new_prob.lstm_node_change_bias = 0.0
      new_prob.lstm_node_shift_bias = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ret = ChangeBias(random_gene, network_config.lstm_node_config)
    return 50, random_gene.GIN, ret
  elseif mutation_number == 51 # lstm node shift bias
    selected_genes = [i for i in values(x.genes) if i isa LSTMNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.lstm_node_change_weight = 0.0
      new_prob.lstm_node_shift_weight = 0.0
      new_prob.lstm_node_change_bias = 0.0
      new_prob.lstm_node_shift_bias = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ret = ShiftBias(random_gene, network_config.lstm_node_config)
    return 51, random_gene.GIN, ret
  elseif mutation_number == 52 # lstm node toggle enable
    selected_genes = [i for i in values(x.genes) if i isa LSTMNode]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.lstm_node_change_weight = 0.0
      new_prob.lstm_node_shift_weight = 0.0
      new_prob.lstm_node_change_bias = 0.0
      new_prob.lstm_node_shift_bias = 0.0
      new_prob.lstm_node_toggle_enable = 0.0
      new_prob.lstm_node_enable_gene = 0.0
      new_prob.lstm_node_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = !random_gene.enabled[]
    return 52, random_gene.GIN, random_gene.enabled[]
  elseif mutation_number == 53 # lstm node enable gene
    selected_genes = [i for i in values(x.genes) if i isa LSTMNode && !i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.lstm_node_enable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = true
    return 53, random_gene.GIN
  elseif mutation_number == 54 # lstm node disable gene
    selected_genes = [i for i in values(x.genes) if i isa LSTMNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.lstm_node_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = false
    return 54, random_gene.GIN
  elseif mutation_number == 55 # add gru node forward connection
    selected_genes = [i for i in values(x.genes) if i isa ForwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.add_gru_node_forward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    if ismissing(x.super)
      new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      AddGRUNodeForwardConnection(x, random_gene, new_GIN, network_config)
    else
      new_GIN = x.super.GIN.GIN[end] + 0x1
      AddGRUNodeForwardConnection(x, random_gene, new_GIN, network_config)
      push!(x.super.GIN.GIN, new_GIN)
      push!(x.super.GIN.type, GRUNode)
      push!(x.super.GIN.start_node, 0x0)
      push!(x.super.GIN.stop_node, 0x0)
      push!(x.super.GIN.GIN, new_GIN+0x1)
      push!(x.super.GIN.type, ForwardConnection)
      push!(x.super.GIN.start_node, random_gene.in_node.GIN)
      push!(x.super.GIN.stop_node, new_GIN)
      push!(x.super.GIN.GIN, new_GIN+0x2)
      push!(x.super.GIN.type, ForwardConnection)
      push!(x.super.GIN.start_node, new_GIN)
      push!(x.super.GIN.stop_node, random_gene.out_node.GIN)
    end
    return 55, new_GIN, new_GIN+0x1, new_GIN+0x2
  elseif mutation_number == 56 # add gru node backward connection
    selected_genes = [i for i in values(x.genes) if i isa BackwardConnection && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.add_gru_node_backward_connection = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    if ismissing(x.super)
      new_GIN == 0 && error("Mutate : got 0 for new_GIN")
      AddGRUNodeBackwardConnection(x, random_gene, new_GIN, network_config)
    else
      new_GIN = x.super.GIN.GIN[end] + 0x1
      AddGRUNodeBackwardConnection(x, random_gene, new_GIN, network_config)
      push!(x.super.GIN.GIN, new_GIN)
      push!(x.super.GIN.type, GRUNode)
      push!(x.super.GIN.start_node, 0x0)
      push!(x.super.GIN.stop_node, 0x0)
      push!(x.super.GIN.GIN, new_GIN+0x1)
      push!(x.super.GIN.type, BackwardConnection)
      push!(x.super.GIN.start_node, random_gene.in_node.GIN)
      push!(x.super.GIN.stop_node, new_GIN)
      push!(x.super.GIN.GIN, new_GIN+0x2)
      push!(x.super.GIN.type, BackwardConnection)
      push!(x.super.GIN.start_node, new_GIN)
      push!(x.super.GIN.stop_node, random_gene.out_node.GIN)
    end
    return 56, new_GIN, new_GIN+0x1, new_GIN+0x2
  elseif mutation_number == 57 # gru node change weight
    selected_genes = [i for i in values(x.genes) if i isa GRUNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.gru_node_change_weight = 0.0
      new_prob.gru_node_shift_weight = 0.0
      new_prob.gru_node_change_bias = 0.0
      new_prob.gru_node_shift_bias = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ret = ChangeWeight(random_gene, network_config.gru_node_config)
    return 57, random_gene.GIN, ret
  elseif mutation_number == 58 # gru node shift weight
    selected_genes = [i for i in values(x.genes) if i isa GRUNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.gru_node_change_weight = 0.0
      new_prob.gru_node_shift_weight = 0.0
      new_prob.gru_node_change_bias = 0.0
      new_prob.gru_node_shift_bias = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ret = ShiftWeight(random_gene, network_config.gru_node_config)
    return 58, random_gene.GIN, ret
  elseif mutation_number == 59 # gru node change bias
    selected_genes = [i for i in values(x.genes) if i isa GRUNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.gru_node_change_weight = 0.0
      new_prob.gru_node_shift_weight = 0.0
      new_prob.gru_node_change_bias = 0.0
      new_prob.gru_node_shift_bias = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ret = ChangeBias(random_gene, network_config.gru_node_config)
    return 59, random_gene.GIN, ret
  elseif mutation_number == 60 # gru node shift bias
    selected_genes = [i for i in values(x.genes) if i isa GRUNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.gru_node_change_weight = 0.0
      new_prob.gru_node_shift_weight = 0.0
      new_prob.gru_node_change_bias = 0.0
      new_prob.gru_node_shift_bias = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    ret = ShiftBias(random_gene, network_config.gru_node_config)
    return 60, random_gene.GIN, ret
  elseif mutation_number == 61 # gru node toggle enable
    selected_genes = [i for i in values(x.genes) if i isa GRUNode]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.gru_node_change_weight = 0.0
      new_prob.gru_node_shift_weight = 0.0
      new_prob.gru_node_change_bias = 0.0
      new_prob.gru_node_shift_bias = 0.0
      new_prob.gru_node_toggle_enable = 0.0
      new_prob.gru_node_enable_gene = 0.0
      new_prob.gru_node_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = !random_gene.enabled[]
    return 61, random_gene.GIN, random_gene.enabled[]
  elseif mutation_number == 62 # gru node enable gene
    selected_genes = [i for i in values(x.genes) if i isa GRUNode && !i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.gru_node_enable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = true
    return 62, random_gene.GIN
  elseif mutation_number == 63 # gru node disable gene
    selected_genes = [i for i in values(x.genes) if i isa GRUNode && i.enabled[]]
    if isempty(selected_genes)
      new_prob = deepcopy(prob)
      new_prob.gru_node_disable_gene = 0.0
      return Mutate(x, new_prob, network_config, new_GIN = new_GIN)
    end
    random_gene = rand(rng(), selected_genes)
    random_gene.enabled[] = false
    return 63, random_gene.GIN
  end
end

function Save(x::Network, filename::String = ""; super::Union{Missing, NEAT} = missing)
  super = ismissing(x.super) ? super : x.super
  if isempty(filename)
    if ismissing(super)
      filename = "Network.jld2"
    else
      filename = "Network-idx_$(x.idx)-gen_$(super.generation).jld2"
    end
  end
  filename = abspath(filename)
  jldsave(filename; x)
  return filename
end

function Visualise(x::Network; filename::String = "", export_type::String = "svg", super::Union{Missing, NEAT} = missing, rankdir_LR::Bool = true, simple::Bool = true, as_subgraph::Bool = false, ranksep::Integer = 5, nodesep::Float64 = 0.05, spline::String = "true")
  ranksep = Unsigned(ranksep)
  nodesep < 0 && error("Visualise : got nodesep < 0")
  nodesep = round(nodesep, digits = 2)
  super = ismissing(x.super) ? super : x.super
  export_types = ["svg", "png", "jpg", "jpeg", "none"]
  export_type in export_types || error("Visualise : export_type should be one among $(export_types)")
  splines = ["true", "false", "line", "polyline", "curved", "spline", "ortho"]
  spline in splines || error("Visualise : spline should be one among $(splines)")

  if isempty(filename)
    filename = "Network-idx_$(x.idx)"
    if ismissing(super)
      filename *= "-gen_$(super.generation)"
    end
  end

  graphviz_code = ["digraph G {", "}"]

  rankdir_LR && insert!(graphviz_code, lastindex(graphviz_code), "\trankdir = LR;")
  
  insert!(graphviz_code, lastindex(graphviz_code), "\tsplines = $(spline);")
  insert!(graphviz_code, lastindex(graphviz_code), "\tranksep=$(ranksep);")
  insert!(graphviz_code, lastindex(graphviz_code), "\tnodesep=$(nodesep);")

  insert!(graphviz_code, lastindex(graphviz_code), "")

  network_layers = GetLayers(x, simple = simple)
  network_layers = network_layers[.!isempty.(network_layers)]

  for (li,l) in enumerate(network_layers)
    subgraph_code = ["\tsubgraph cluster_$(li-1) {", "\t}\n"]
    if l == network_layers[1]
      if as_subgraph
        for n in l
          insert!(subgraph_code, lastindex(subgraph_code), "\t\t$(n) [shape = diamond];")
        end
        insert!(subgraph_code, lastindex(subgraph_code), "\t\tlabel = \"Input Layer\";")
        insert!(subgraph_code, lastindex(subgraph_code), "\t\trank = \"min\";")
      else
        temp = ["\t{rank=min;", "}\n"]
        for n in l
          insert!(graphviz_code, lastindex(graphviz_code), "\t$(n) [shape=diamond, label=\"$(n)\n$(typeof(x.genes[n]))\"]")
          insert!(temp, lastindex(temp), "$(n);")
        end
        insert!(graphviz_code, lastindex(graphviz_code), join(temp, " "))
      end
    elseif l == network_layers[end]
      if as_subgraph
        for n in l
          insert!(subgraph_code, lastindex(subgraph_code), "\t\t$(n) [shape = box];")
        end
        insert!(subgraph_code, lastindex(subgraph_code), "\t\tlabel = \"Output Layer\";")
        insert!(subgraph_code, lastindex(subgraph_code), "\t\trank = \"max\";")
      else
        temp = ["\t{rank=max;", "}\n"]
        for n in l
          insert!(graphviz_code, lastindex(graphviz_code), "\t$(n) [shape=box, label=\"$(n)\n$(typeof(x.genes[n]))\"]")
          insert!(temp, lastindex(temp), "$(n);")
        end
        insert!(graphviz_code, lastindex(graphviz_code), join(temp, " "))
      end
    else
      if as_subgraph
        for n in l
          insert!(subgraph_code, lastindex(subgraph_code), "\t\t$(n) [shape = circle, label = \"$(n)\n$(typeof(x.genes[n]))\"];")
        end
        insert!(subgraph_code, lastindex(subgraph_code), "\t\tlabel = \"Hidden Layer\";")
        insert!(subgraph_code, lastindex(subgraph_code), "\t\trank = \"same\";")
      else
        temp = ["\t{rank=same;", "}\n"]
        for n in l
          insert!(graphviz_code, lastindex(graphviz_code), "\t$(n) [shape=circle, label=\"$(n)\n$(typeof(x.genes[n]))\"]")
          insert!(temp, lastindex(temp), "$(n);")
        end
        insert!(graphviz_code, lastindex(graphviz_code), join(temp, " "))
      end
    end
    as_subgraph && insert!(graphviz_code, lastindex(graphviz_code), join(subgraph_code, "\n"))
  end

  for g in values(x.genes)
    if g isa Connections
      if simple && (!g.enabled[] || !g.in_node.enabled[] || !g.out_node.enabled[])
        continue
      end
      if g.enabled[]
        if g.weight[] >= 0
          insert!(graphviz_code, lastindex(graphviz_code), "\t$(g.in_node.GIN)->$(g.out_node.GIN) [label = \"$(g.GIN),   $(round(g.weight[], digits=3))\", color = green]")
        else
          insert!(graphviz_code, lastindex(graphviz_code), "\t$(g.in_node.GIN)->$(g.out_node.GIN) [label = \"$(g.GIN),   $(round(g.weight[], digits=3))\", color = red]")
        end
      else
        insert!(graphviz_code, lastindex(graphviz_code), "\t$(g.in_node.GIN)->$(g.out_node.GIN) [label = \"$(g.GIN),   $(round(g.weight[], digits=3))\", color = grey]")
      end
    end
  end

  graphviz_code = join(graphviz_code, "\n")

  open(filename*".dot", "w") do file
    write(file, graphviz_code)
  end

  if export_type != "none"
    try
      run(`dot -T$(export_type) $(filename*".dot") -o $(filename).$(export_type)`)
    catch e
      if typeof(e) <: Base.IOError
        println("`dot` command not found skipping export to .$(export_type)")
      end
    end
  end

  return graphviz_code
end






