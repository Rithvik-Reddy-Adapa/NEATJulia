
module NEAT

using Random, StatsBase, JLD2, DataFrames, Dates, Base.Threads
import Base:show
import Base:display
import Base:getindex
import Base:setindex!
import Base:lastindex
import Base:length
# include("Reference.jl")
Reference{T} = Ref{Union{Missing, T}}
Reference(x) = Reference{typeof(x)}(x)
Reference{T}() where T = Reference{T}(missing)
Reference{T}() where T<:Real = Reference{T}(NaN)

export Genes, Nodes, Connections, InputNode, HiddenNode, OutputNode, ForwardConnection, Networks, FFNN, NEATs, NEAT_FFNN, NEAT_configs, NEAT_FFNN_config, Probabilities, CrossoverProbability, FFNN_Mutation_Probability, RecurrentInputNode, RecurrentOutputNode, RecurrentConnection, RNN_Mutation_Probability, RNN, NEAT_RNN_config, NEAT_RNN, NEAT_Log_config
export GIN, SpecieInfo
export Init, Run, GetInput, SetInput!, GetOutput, GetLayers, GetNodePosition, GetNetworkInfo, AddForwardConnection, AddHiddenNode, Crossover, Mutate, Evaluate, RemoveStagnantSpecies, UpdatePopulation, GetNetworkDistance, Speciate, Train, toDict, Save, Load, Visualise, AddRecurrentConnection, Reset!
export Reference, show, display, getfield, getindex, setindex!, lastindex
export Identity, Relu, Sigmoid, Tanh, Sin

abstract type Genes end
abstract type Nodes <: Genes end
abstract type Connections <: Genes end
abstract type Networks end
abstract type NEAT_configs end
abstract type NEATs end
abstract type Probabilities end

@kwdef mutable struct InputNode <: Nodes
  GIN::Unsigned = 0 # Global Innovation Number
  input_number::Unsigned = 0 # position of this input node in the entire input set
  out_connections::Vector{Connections} = Connections[] # points to all the out connections of this input node
  input::Reference{Real} = Reference{Real}() # points to the input value for this input node
  output::Reference{Real} = Reference{Real}() # points to the output value of this input node
  processed::Bool = false # to check if output has been computed or not
  const enabled::Bool = true # to check if this input node is enabled or not, all input nodes are enabled always
  super::Union{Missing, Networks} = missing # points to the network containing this input node
end
function display(x::InputNode)
  println(typeof(x))
  print(" GIN : $(x.GIN)
 input_number : $(x.input_number)
 out_connections : $(Int[i.GIN for i in x.out_connections])
 input : $(x.input[])
 output : $(x.output[])
 processed : $(x.processed)
 enabled : $(x.enabled)
 super_type : $(typeof(x.super))
 super : $(ismissing(x.super) ? missing : x.super.ID)")
end
function toDict(x::InputNode)
  ret = Dict("instance" => typeof(x),
             "GIN" => x.GIN,
             "input_number" => x.input_number,
             "out_connections" => Unsigned[i.GIN for i in x.out_connections],
             "input" => x.input[],
             "output" => x.output[],
             "processed" => x.processed,
             "enabled" => x.enabled,
             "super_type" => typeof(x.super),
             "super" => ismissing(x.super) ? missing : x.super.ID,
            )
  return ret
end
function Reset!(x::InputNode)
  x.input[] = missing
  x.output[] = missing
  x.processed = false
  return
end
function Run(x::InputNode)
  x.output[] = x.input[]
  x.processed = true

  return x.output[]
end

@kwdef mutable struct RecurrentInputNode <: Nodes
  GIN::Unsigned = 0 # Global Innovation Number
  input_number::Unsigned = 0 # position of this input node in the entire input set
  in_connections::Vector{Connections} = Connections[] # points to all the in connections of this node
  out_connections::Vector{Connections} = Connections[] # points to all the out connections of this node
  input::Vector{Reference{Real}} = Reference{Real}[] # points to all the input values for this node
  output::Reference{Real} = Reference{Real}() # points to the output value of this node
  activation_function::Function = Sigmoid
  bias::Real = 0
  processed::Bool = false # to check if output has been computed or not
  const enabled::Bool = true # to check if this node is enabled or not
  super::Union{Missing, Networks} = missing # points to the network containing this node
end
function display(x::RecurrentInputNode)
  println(typeof(x))
  print(" GIN : $(x.GIN)
 input_number : $(x.input_number)
 in_connections : $(Int[i.GIN for i in x.in_connections])
 out_connections : $(Int[i.GIN for i in x.out_connections])
 input : $([i[] for i in x.input])
 output : $(x.output[])
 activation_function : $(x.activation_function)
 bias : $(x.bias)
 processed : $(x.processed)
 enabled : $(x.enabled)
 super_type : $(typeof(x.super))
 super : $(ismissing(x.super) ? missing : x.super.ID)")
end
function toDict(x::RecurrentInputNode)
  ret = Dict("instance" => typeof(x),
             "GIN" => x.GIN,
             "input_number" => x.input_number,
             "in_connections" => Unsigned[i.GIN for i in x.in_connections],
             "out_connections" => Unsigned[i.GIN for i in x.out_connections],
             "input" => [i[] for i in x.input],
             "output" => x.output[],
             "activation_function" => string(x.activation_function),
             "bias" => x.bias,
             "processed" => x.processed,
             "enabled" => x.enabled,
             "super_type" => typeof(x.super),
             "super" => ismissing(x.super) ? missing : x.super.ID,
            )
  return ret
end
function Reset!(x::RecurrentInputNode)
  for i in x.input
    i[] = missing
  end
  x.output[] = missing
  x.processed = false
  return
end
function Run(x::RecurrentInputNode)
  x.output[] = missing
  for i in x.input
    if !ismissing(i[])
      if ismissing(x.output[])
        x.output[] = i[]
      else
        x.output[] += i[]
      end
    end
  end
  x.output[] = ismissing(x.output[]) ? missing : x.activation_function(x.output[] + x.bias)
  x.processed = true
  return x.output[]
end

@kwdef mutable struct HiddenNode <: Nodes
  GIN::Unsigned = 0 # Global Innovation Number
  in_connections::Vector{Connections} = Connections[] # points to all the in connections of this hidden node
  out_connections::Vector{Connections} = Connections[] # points to all the out connections of this hidden node
  input::Vector{Reference{Real}} = Reference{Real}[] # points to all the input values for this hidden node
  output::Reference{Real} = Reference{Real}() # points to the output value of this hidden node
  activation_function::Function = Sigmoid
  bias::Real = 0
  processed::Bool = false # to check if output has been computed or not
  enabled::Bool = true # to check if this hidden node is enabled or not
  super::Union{Missing, Networks} = missing # points to the network containing this hidden node
end
function display(x::HiddenNode)
  println(typeof(x))
  print(" GIN : $(x.GIN)
 in_connections : $(Int[i.GIN for i in x.in_connections])
 out_connections : $(Int[i.GIN for i in x.out_connections])
 input : $([i[] for i in x.input])
 output : $(x.output[])
 activation_function : $(x.activation_function)
 bias : $(x.bias)
 processed : $(x.processed)
 enabled : $(x.enabled)
 super_type : $(typeof(x.super))
 super : $(ismissing(x.super) ? missing : x.super.ID)")
end
function toDict(x::HiddenNode)
  ret = Dict("instance" => typeof(x),
             "GIN" => x.GIN,
             "in_connections" => Unsigned[i.GIN for i in x.in_connections],
             "out_connections" => Unsigned[i.GIN for i in x.out_connections],
             "input" => [i[] for i in x.input],
             "output" => x.output[],
             "activation_function" => string(x.activation_function),
             "bias" => x.bias,
             "processed" => x.processed,
             "enabled" => x.enabled,
             "super_type" => typeof(x.super),
             "super" => ismissing(x.super) ? missing : x.super.ID,
            )
  return ret
end
function Reset!(x::HiddenNode)
  for i in x.input
    i[] = missing
  end
  x.output[] = missing
  x.processed = false
  return
end
function Run(x::HiddenNode)
  if !x.enabled
    x.output[] = missing
    x.processed = true
    return x.output[]
  end
  if isempty(x.in_connections)
    x.output[] = missing
    x.processed = true
    return x.output[]
  end
  enabled_connections = [i for i in x.in_connections if i.enabled]
  if isempty(enabled_connections)
    x.output[] = missing
    x.processed = true
    return x.output[]
  end
  processed_connections = [i for i in enabled_connections if i.processed]
  if length(processed_connections) == length(enabled_connections)
    x.output[] = missing
    for i in x.input
      if !ismissing(i[])
        if ismissing(x.output[])
          x.output[] = i[]
        else
          x.output[] += i[]
        end
      end
    end
    x.output[] = ismissing(x.output[]) ? missing : x.activation_function(x.output[] + x.bias)
    x.processed = true
    return x.output[]
  else
    return nothing
  end
end

@kwdef mutable struct OutputNode <: Nodes
  GIN::Unsigned = 0 # Global Innovation Number
  in_connections::Vector{Connections} = Connections[] # points to all the in connections of this output node
  output_number::Unsigned = 0 # position of this output node in the entire output set
  input::Vector{Reference{Real}} = Reference{Real}[] # points to all the inputs for this output node
  output::Reference{Real} = Reference{Real}() # points to the output value of this output node
  activation_function::Function = Sigmoid
  bias::Real = 0
  processed::Bool = false # to check if output has been computed or not
  const enabled::Bool = true # to check if this hidden node is enabled or not, all output nodes are enabled always
  super::Union{Missing, Networks} = missing # points to the network containing this output node
end
function display(x::OutputNode)
  println(typeof(x))
  print(" GIN : $(x.GIN)
 in_connections : $(Int[i.GIN for i in x.in_connections])
 output_number : $(x.output_number)
 input : $([i[] for i in x.input])
 output : $(x.output[])
 activation_function : $(x.activation_function)
 bias : $(x.bias)
 processed : $(x.processed)
 enabled : $(x.enabled)
 super_type : $(typeof(x.super))
 super : $(ismissing(x.super) ? missing : x.super.ID)")
end
function toDict(x::OutputNode)
  ret = Dict("instance" => typeof(x),
             "GIN" => x.GIN,
             "in_connections" => Unsigned[i.GIN for i in x.in_connections],
             "output_number" => x.output_number,
             "input" => [i[] for i in x.input],
             "output" => x.output[],
             "activation_function" => string(x.activation_function),
             "bias" => x.bias,
             "processed" => x.processed,
             "enabled" => x.enabled,
             "super_type" => typeof(x.super),
             "super" => ismissing(x.super) ? missing : x.super.ID,
            )
  return ret
end
function Reset!(x::OutputNode)
  for i in x.input
    i[] = missing
  end
  x.output[] = missing
  x.processed = false
  return
end
function Run(x::OutputNode)
  if isempty(x.in_connections)
    x.output[] = missing
    x.processed = true
    return x.output[]
  end
  enabled_connections = [i for i in x.in_connections if i.enabled]
  if isempty(enabled_connections)
    x.output[] = missing
    x.processed = true
    return x.output[]
  end
  processed_connections = [i for i in enabled_connections if i.processed]
  if length(processed_connections) == length(enabled_connections)
    x.output[] = missing
    for i in x.input
      if !ismissing(i[])
        if ismissing(x.output[])
          x.output[] = i[]
        else
          x.output[] += i[]
        end
      end
    end
    x.output[] = ismissing(x.output[]) ? missing : x.activation_function(x.output[] + x.bias)
    x.processed = true
    return x.output[]
  else
    return nothing
  end
end

@kwdef mutable struct RecurrentOutputNode <: Nodes
  GIN::Unsigned = 0 # Global Innovation Number
  in_connections::Vector{Connections} = Connections[] # points to all the in connections of this node
  output_number::Unsigned = 0 # position of this output node in the entire output set
  out_connections::Vector{Connections} = Connections[] # points to all the out connections of this node
  input::Vector{Reference{Real}} = Reference{Real}[] # points to all the input values for this node
  output::Reference{Real} = Reference{Real}() # points to the output value of this node
  activation_function::Function = Sigmoid
  bias::Real = 0
  processed::Bool = false # to check if output has been computed or not
  const enabled::Bool = true # to check if this node is enabled or not
  super::Union{Missing, Networks} = missing # points to the network containing this node
end
function display(x::RecurrentOutputNode)
  println(typeof(x))
  print(" GIN : $(x.GIN)
 in_connections : $(Int[i.GIN for i in x.in_connections])
 output_number : $(x.output_number)
 out_connections : $(Int[i.GIN for i in x.out_connections])
 input : $([i[] for i in x.input])
 output : $(x.output[])
 activation_function : $(x.activation_function)
 bias : $(x.bias)
 processed : $(x.processed)
 enabled : $(x.enabled)
 super_type : $(typeof(x.super))
 super : $(ismissing(x.super) ? missing : x.super.ID)")
end
function toDict(x::RecurrentInputNode)
  ret = Dict("instance" => typeof(x),
             "GIN" => x.GIN,
             "in_connections" => Unsigned[i.GIN for i in x.in_connections],
             "output_number" => x.output_number,
             "out_connections" => Unsigned[i.GIN for i in x.out_connections],
             "input" => [i[] for i in x.input],
             "output" => x.output[],
             "activation_function" => string(x.activation_function),
             "bias" => x.bias,
             "processed" => x.processed,
             "enabled" => x.enabled,
             "super_type" => typeof(x.super),
             "super" => ismissing(x.super) ? missing : x.super.ID,
            )
  return ret
end
function Reset!(x::RecurrentOutputNode)
  for i in x.input
    i[] = missing
  end
  x.output[] = missing
  x.processed = false
  return
end
function Run(x::RecurrentOutputNode)
  if isempty(x.in_connections)
    x.output[] = missing
    x.processed = true
    return x.output[]
  end
  enabled_connections = [i for i in x.in_connections if i.enabled]
  if isempty(enabled_connections)
    x.output[] = missing
    x.processed = true
    return x.output[]
  end
  processed_connections = [i for i in enabled_connections if i.processed]
  if length(processed_connections) == length(enabled_connections)
    x.output[] = missing
    for i in x.input
      if !ismissing(i[])
        if ismissing(x.output[])
          x.output[] = i[]
        else
          x.output[] += i[]
        end
      end
    end
    x.output[] = ismissing(x.output[]) ? missing : x.activation_function(x.output[] + x.bias)
    x.processed = true
    return x.output[]
  else
    return nothing
  end
end

@kwdef mutable struct ForwardConnection <: Connections
  GIN::Unsigned = 0 # Global Innovation Number
  in_node::Nodes # points to the input node
  out_node::Nodes # points to the output node
  input::Reference{Real} = Reference{Real}() # points to the input value for this connection
  output::Reference{Real} = Reference{Real}() # points to the output value of this connection
  weight::Real = 1.0
  processed::Bool = false # to check if the output has been computed or not
  enabled::Bool = true # to check if this connection is enabled or not
  super::Union{Missing, Networks} = missing # points to the network containing this connection
end
function Init(x::ForwardConnection)
  x.input = x.in_node.output
  push!(x.in_node.out_connections, x)

  push!(x.out_node.in_connections, x)
  push!(x.out_node.input, x.output)
  return
end
function display(x::ForwardConnection)
  println(typeof(x))
  print(" GIN : $(x.GIN)
 in_node : $(x.in_node.GIN)
 out_node : $(x.out_node.GIN)
 input : $(x.input[])
 output : $(x.output[])
 weight : $(x.weight)
 processed : $(x.processed)
 enabled : $(x.enabled)
 super_type : $(typeof(x.super))
 super : $(ismissing(x.super) ? missing : x.super.ID)")
end
function toDict(x::ForwardConnection)
  ret = Dict("instance" => typeof(x),
             "GIN" => x.GIN,
             "in_node" => x.in_node.GIN,
             "out_node" => x.out_node.GIN,
             "input" => x.input[],
             "output" => x.output[],
             "weight" => x.weight,
             "processed" => x.processed,
             "enabled" => x.enabled,
             "super_type" => typeof(x.super),
             "super" => ismissing(x.super) ? missing : x.super.ID,
            )
  return ret
end
function Reset!(x::ForwardConnection)
  x.input[] = missing
  x.output[] = missing
  x.processed = false
  return
end
function Run(x::ForwardConnection)
  if !(x.enabled)
    x.output[] = missing
    x.processed = true
    return x.output[]
  end
  if !x.in_node.enabled
    x.output[] = missing
    x.processed = true
    return x.output[]
  end
  if !x.in_node.processed
    return nothing
  end
  x.output[] = x.input[] * x.weight
  x.processed = true
  return x.output[]
end

@kwdef mutable struct RecurrentConnection <: Connections
  GIN::Unsigned = 0 # Global Innovation Number
  in_node::Nodes # points to the input node
  out_node::Nodes # points to the output node
  input::Reference{Real} = Reference{Real}() # points to the input value for this connection
  output::Reference{Real} = Reference{Real}() # points to the output value of this connection
  weight::Real = 1.0
  const processed::Bool = true # to check if the output has been computed or not
  enabled::Bool = true # to check if this connection is enabled or not
  super::Union{Missing, Networks} = missing # points to the network containing this connection
end
function Init(x::RecurrentConnection)
  x.input = x.in_node.output
  push!(x.in_node.out_connections, x)

  push!(x.out_node.in_connections, x)
  push!(x.out_node.input, x.output)
  return
end
function display(x::RecurrentConnection)
  println(typeof(x))
  print(" GIN : $(x.GIN)
 in_node : $(x.in_node.GIN)
 out_node : $(x.out_node.GIN)
 input : $(x.input[])
 output : $(x.output[])
 weight : $(x.weight)
 processed : $(x.processed)
 enabled : $(x.enabled)
 super_type : $(typeof(x.super))
 super : $(ismissing(x.super) ? missing : x.super.ID)")
end
function toDict(x::RecurrentConnection)
  ret = Dict("instance" => typeof(x),
             "GIN" => x.GIN,
             "in_node" => x.in_node.GIN,
             "out_node" => x.out_node.GIN,
             "input" => x.input[],
             "output" => x.output[],
             "weight" => x.weight,
             "processed" => x.processed,
             "enabled" => x.enabled,
             "super_type" => typeof(x.super),
             "super" => ismissing(x.super) ? missing : x.super.ID,
            )
  return ret
end
function Reset!(x::RecurrentConnection)
  x.input[] = missing
  x.output[] = missing
  return
end
function Run(x::RecurrentConnection)
  if !(x.enabled)
    x.output[] = missing
    return x.output[]
  end
  if !x.in_node.enabled
    x.output[] = missing
    return x.output[]
  end
  if !x.in_node.processed
    return nothing
  end
  x.output[] = x.input[] * x.weight
  return x.output[]
end

@kwdef mutable struct FFNN_Mutation_Probability <: Probabilities
  no_mutation::Real = 0.0
  change_weight::Real = 1.0
  change_bias::Real = 1.0
  shift_weight::Real = 5.0
  shift_bias::Real = 2.0
  add_forward_connection::Real = -1e15
  add_node::Real = -1e10
  disable_forward_connection::Real = 0.0
  disable_node::Real = 0.0
  enable_forward_connection::Real = 0.5
  enable_node::Real = 0.5
  change_activation_function::Real = 0.0
end

@kwdef mutable struct RNN_Mutation_Probability <: Probabilities
  no_mutation::Real = 0.0
  change_weight::Real = 1.0
  change_bias::Real = 1.0
  shift_weight::Real = 5.0
  shift_bias::Real = 2.0
  add_forward_connection::Real = -1e15
  add_node::Real = -1e10
  add_recurrent_connection::Real = -1e14
  disable_forward_connection::Real = 0.0
  disable_node::Real = 0.0
  disable_recurrent_connection::Real = 0.0
  enable_forward_connection::Real = 0.5
  enable_node::Real = 0.5
  enable_recurrent_connection::Real = 0.5
  change_activation_function::Real = 0.0
  add_self_connection::Real = 1
  enable_self_connection::Real = 0.5
  disable_self_connection::Real = 0.5
end

@kwdef mutable struct GIN
  GIN::Unsigned
  type::Type
  start_node::Unsigned
  stop_node::Unsigned
end
function display(x::GIN)
  println(typeof(x))
  print(" GIN : $(x.GIN)\n type : $(x.type)\n start_node : $(x.start_node)\n stop_node : $(x.stop_node)\n")
  return
end
function display(x::Vector{GIN})
  println("$(length(x))-element $(typeof(x)):")
  for i in x
    display(i)
  end
end
function toDict(x::GIN)
  ret = Dict("GIN" => x.GIN,
             "type" => x.type,
             "start_node" => x.start_node,
             "stop_node" => x.stop_node,
            )
  return ret
end

function display(x::Networks)
  println(typeof(x))
  print(" ID : $(x.ID)\n")
end
function display(x::Vector{T}) where T <: Networks
  println("$(length(x))-element $(typeof(x)):")
  for i in x
    display(i)
  end
end
function display(x::Dict{U, Vector{T}}) where {T <: Networks, U <: Unsigned}
  println("$(typeof(x)) with $(length(x)) entries:")
  for i = sort(collect(keys(x)))
    println(" $(i) => $(Int[j.ID for j in x[i]])")
  end
end

@kwdef mutable struct FFNN <: Networks # FeedForward Neural Network
  ID::Unsigned = 0 # it's index in the population
  const n_inputs::Unsigned # number of inputs
  const n_outputs::Unsigned # number of outputs
  input::Vector{Reference{Real}} = Reference{Real}[] # points to all input values for this network
  output::Vector{Reference{Real}} = Reference{Real}[] # points to all output values of this network
  genes::Dict{Unsigned, Genes} = Dict{Unsigned, Genes}() # represents all genes in this network, Dict( GIN (Global Innovation Number) => Node / Connection )
  specie::Unsigned = 0
  layers::Vector{Vector{Nodes}} = Vector{Vector{Nodes}}()
  list_activation_functions::Vector{Function} = Function[Sigmoid]
  super::Union{Missing, NEATs} = missing # points to the NEAT containing this network
end
function Init(x::FFNN, start_fully_connected::Bool = false)
  layer = Vector{Nodes}(undef, x.n_inputs)
  x.input = Vector{Reference{Real}}(undef, x.n_inputs)
  @threads for i = 1:x.n_inputs
    x.input[i] = Reference{Real}()
    x.genes[i] = InputNode(GIN = i, input_number = i, input = x.input[i], super = x)
    layer[i] = x.genes[i]
  end
  push!(x.layers, layer)

  layer = Vector{Nodes}(undef, x.n_outputs)
  x.output = Vector{Reference{Real}}(undef, x.n_outputs)
  @threads for i = 1:x.n_outputs
    x.output[i] = Reference{Real}()
    x.genes[i+x.n_inputs] = OutputNode(GIN = i+x.n_inputs, output_number = i, output = x.output[i], activation_function = rand(rng(), x.list_activation_functions), super = x)
    layer[i] = x.genes[i+x.n_inputs]
  end
  push!(x.layers, layer)

  if start_fully_connected
    @threads for i = 1:x.n_inputs
      for j = 1:x.n_outputs
        AddForwardConnection(x, i,j+x.n_inputs, x.n_inputs+x.n_outputs + (i-1)*x.n_outputs + j )
      end
    end
  end

  return
end
function toDict(x::FFNN)
  genes = Dict{Unsigned, Dict}()
  for i = sort(collect(keys(x.genes)))
    genes[i] = toDict(x.genes[i])
  end
  ret = Dict("instance" => typeof(x),
             "ID" => x.ID,
             "n_inputs" => x.n_inputs,
             "n_outputs" => x.n_outputs,
             "input" => [i[] for i in x.input],
             "output" => [i[] for i in x.output],
             "genes" => genes,
             "specie" => x.specie,
             "layers" => [[n.GIN for n in l] for l in x.layers],
             "list_activation_functions" => string.(x.list_activation_functions),
             "super" => typeof(x.super),
            )
  return ret
end
function Run(x::FFNN)
  for i in x.genes
    i.second.processed = false
  end
  for l in x.layers
    for n in l
      Run(n)
      if l != x.layers[end]
        for c in n.out_connections
          Run(c)
        end
      end
    end
  end

  return [i[] for i in x.output]
end
function GetInput(x::FFNN)
  return [i[] for i in x.input]
end
function SetInput!(x::FFNN, args::Vector{T}) where T<:Real
  length(args) == x.n_inputs || throw(error("Invalid number of input arguments"))

  for i = 1:x.n_inputs
    x.input[i][] = args[i]
  end
  return
end
function GetOutput(x::FFNN)
  return [i[] for i in x.output]
end
function GetLayers(x::FFNN, simple::Bool = true)
  return [Unsigned[n.GIN for n in l if (!simple || n.enabled)] for l in x.layers]
end
function GetNodePosition(x::FFNN, y::Integer)
  y = Unsigned(y)
  layer_number = 0
  node_number = 0
  for (i,l) in enumerate(x.layers)
    is_break = false
    for (j,n) in enumerate(l)
      if n.GIN == y
        layer_number = i
        node_number = j
        is_break = true
        break
      end
    end
    if is_break
      break
    end
  end
  return (layer_number, node_number)
end
function GetNodePosition(x::T) where T <: Nodes
  return GetNodePosition(x.super, x.GIN)
end
function AddForwardConnection(x::FFNN, start_node_GIN::Integer, stop_node_GIN::Integer, new_GIN::Integer)
  start_node_GIN = Unsigned(start_node_GIN)
  stop_node_GIN = Unsigned(stop_node_GIN)
  new_GIN = Unsigned(new_GIN)

  new_GIN in keys(x.genes) && throw(error("new_GIN = $(new_GIN) already exists"))

  start_layer = GetNodePosition(x, start_node_GIN)[1]
  start_layer == 0 && throw(error("start node is not present"))

  stop_layer = GetNodePosition(x, stop_node_GIN)[1]
  stop_layer == 0 && throw(error("stop node is not present"))

  stop_layer <= start_layer && throw(error("AddForwardConnection: stop_layer <= start_layer"))

  stop_node_GIN in getfield.(getfield.(x.genes[start_node_GIN].out_connections, :out_node), :GIN) && throw(error("forward connection exists from $(start_node_GIN) to $(stop_node_GIN)"))

  connection = ForwardConnection(in_node = x.genes[start_node_GIN], out_node = x.genes[stop_node_GIN], GIN = new_GIN, super = x)
  Init(connection)
  x.genes[new_GIN] = connection

  return connection
end
function AddHiddenNode(x::FFNN, connection_GIN::Integer, new_GIN::Integer)
  connection_GIN = Unsigned(connection_GIN)
  new_GIN = Unsigned(new_GIN)

  new_GIN in keys(x.genes) && throw(error("new_GIN = $(new_GIN) already exists"))

  connection_GIN in keys(x.genes) || throw(error("connection_GIN = $(connection_GIN) not present"))

  typeof(x.genes[connection_GIN])<:ForwardConnection || throw(error("connection_GIN = $(connection_GIN) is not of type ForwardConnection"))

  node = HiddenNode(GIN = new_GIN, activation_function = rand(rng(), x.list_activation_functions), super = x)
  x.genes[new_GIN] = node

  layer = rand(rng(), GetNodePosition(x, x.genes[connection_GIN].in_node.GIN)[1]+0.5:0.5:GetNodePosition(x, x.genes[connection_GIN].out_node.GIN)[1]-0.5)
  if floor(layer) == layer # existing layer
    layer = Int(layer)
    push!(x.layers[layer], node)
  else # create a new layer
    layer = Int(ceil(layer))
    insert!(x.layers, layer, Nodes[node])
  end

  connection1 = AddForwardConnection(x, x.genes[connection_GIN].in_node.GIN, new_GIN, new_GIN+1)
  connection2 = AddForwardConnection(x, new_GIN, x.genes[connection_GIN].out_node.GIN, new_GIN+2)

  x.genes[connection_GIN].enabled = false

  return node, connection1, connection2
end
function Crossover(x::FFNN, y::FFNN, fitness_x::Real = Inf, fitness_y::Real = -Inf)
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
  ret.list_activation_functions = parent1.list_activation_functions
  ret.ID = 0
  ret.specie = 0
  for i in keys(ret.genes)
    if haskey(parent2.genes, i) && (typeof(ret.genes[i])<:ForwardConnection) && rand(rng(), [true, false])
      ret.genes[i].weight = parent2.genes[i].weight
    elseif haskey(parent2.genes, i) && (typeof(ret.genes[i])<:HiddenNode) && rand(rng(), [true, false])
      ret.genes[i].bias = parent2.genes[i].bias
      ret.genes[i].activation_function = parent2.genes[i].activation_function
    elseif haskey(parent2.genes, i) && (typeof(ret.genes[i])<:OutputNode) && rand(rng(), [true, false])
      ret.genes[i].bias = parent2.genes[i].bias
      ret.genes[i].activation_function = parent2.genes[i].activation_function
    end
  end
  return ret
end
function Mutate(x::FFNN, y::FFNN_Mutation_Probability)
  mutation = sample(rng(), 1:length(y), Weights(abs.(y[:])))
  if (y.add_forward_connection < 0) && (mutation != 6 && mutation != 1)
    y.add_forward_connection *= 1.1
  end
  if (y.add_node < 0) && (mutation != 7 && mutation != 1)
    y.add_node *= 1.1
  end

  if mutation == 1 # no mutation
    # do nothing
    return 1
  elseif mutation == 2 # change weight
    network_info = GetNetworkInfo(x, true)
    GINs = network_info[(network_info[:,:type].<:Connections), :GIN]
    if isempty(GINs)
      return 2
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].weight = x.super.config.min_weight + (x.super.config.max_weight - x.super.config.min_weight)*rand(rng())
    return 2, random_connection
  elseif mutation == 3 # change bias
    network_info = GetNetworkInfo(x, true)
    GINs = network_info[(network_info[:,:type].<:HiddenNode).||(network_info[:,:type].<:OutputNode), :GIN]
    if isempty(GINs)
      return 3
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].bias = x.super.config.min_bias + (x.super.config.max_bias - x.super.config.min_bias)*rand(rng())
    return 3, random_node
  elseif mutation == 4 # shift weight
    network_info = GetNetworkInfo(x, true)
    GINs = network_info[(network_info[:,:type].<:Connections), :GIN]
    if isempty(GINs)
      return 4
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].weight += x.super.config.max_shift_weight * rand(rng()) * rand(rng(), [-1, 1])
    if x.genes[random_connection].weight > x.super.config.max_weight
      x.genes[random_connection].weight = x.super.config.min_weight
    end
    if x.genes[random_connection].weight < x.super.config.min_weight
      x.genes[random_connection].weight = x.super.config.max_weight
    end
    return 4, random_connection
  elseif mutation == 5 # shift bias
    network_info = GetNetworkInfo(x, true)
    GINs = network_info[(network_info[:,:type].<:HiddenNode).||(network_info[:,:type].<:OutputNode), :GIN]
    if isempty(GINs)
      return 5
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].bias += x.super.config.max_shift_bias * rand(rng()) * rand(rng(), [-1, 1])
    if x.genes[random_node].bias > x.super.config.max_bias
      x.genes[random_node].bias = x.super.config.min_bias
    end
    if x.genes[random_node].bias < x.super.config.min_bias
      x.genes[random_node].bias = x.super.config.max_bias
    end
    return 5, random_node
  elseif mutation == 6 # add forward connection
    network_layers = GetLayers(x, true)
    non_empty_network_layers = [i for i = 1:length(network_layers) if !(isempty(network_layers[i]))]
    add_connection = false
    start_layer = 0
    stop_layer = 0
    start_node_GIN = 0
    stop_node_GIN = 0
    for i = 1:50
      start_layer = rand(rng(), 1:length(non_empty_network_layers)-1)
      stop_layer = rand(rng(), start_layer+1:length(non_empty_network_layers))
      start_layer = non_empty_network_layers[start_layer]
      stop_layer = non_empty_network_layers[stop_layer]

      start_node_GIN = rand(rng(), network_layers[start_layer])
      stop_node_GIN = rand(rng(), network_layers[stop_layer])

      stop_node_GIN in getfield.(getfield.(x.genes[start_node_GIN].out_connections, :out_node), :GIN) ? (add_connection = false; continue) : (add_connection = true; break)
    end
    if add_connection
      new_GIN = all( [getfield.(x.super.GIN, :start_node) getfield.(x.super.GIN, :stop_node)] .== [start_node_GIN stop_node_GIN], dims = 2 )[:]
      new_GIN = findfirst(new_GIN)
      if isnothing(new_GIN)
        new_GIN = x.super.GIN[end].GIN+1
        new_connection = AddForwardConnection(x, start_node_GIN, stop_node_GIN, new_GIN)
        push!(x.super.GIN, GIN(new_connection.GIN, typeof(new_connection), new_connection.in_node.GIN, new_connection.out_node.GIN))

        if y.add_forward_connection < 0
          n_connections = sum(typeof.(values(x.genes)).<:Connections)
          y.add_forward_connection = -1/(n_connections^2)
        end

        return 6, "new", new_connection.GIN
      else
        new_connection = AddForwardConnection(x, start_node_GIN, stop_node_GIN, new_GIN)

        if y.add_forward_connection < 0
          n_connections = sum(typeof.(values(x.genes)).<:Connections)
          y.add_forward_connection = -1/(n_connections^2)
        end

        return 6, "old", new_connection.GIN
      end
    else
      if y.add_forward_connection < 0
        n_connections = sum(typeof.(values(x.genes)).<:Connections)
        y.add_forward_connection = -1/(n_connections^2)
      end
      return 6
    end
  elseif mutation == 7 # add node
    network_info = GetNetworkInfo(x, true)
    GINs = network_info[(network_info[:,:type].<:Connections), :GIN]
    if isempty(GINs)
      return 7
    end
    random_connection = rand(rng(), GINs)
    new_GIN = x.super.GIN[end].GIN+1
    new_node, new_connection1, new_connection2 = AddHiddenNode(x, random_connection, new_GIN)
    push!(x.super.GIN, GIN(new_node.GIN, typeof(new_node), 0, 0))
    push!(x.super.GIN, GIN(new_connection1.GIN, typeof(new_connection1), new_connection1.in_node.GIN, new_connection1.out_node.GIN))
    push!(x.super.GIN, GIN(new_connection2.GIN, typeof(new_connection2), new_connection2.in_node.GIN, new_connection2.out_node.GIN))

    if y.add_node < 0
      n_nodes = sum(typeof.(values(x.genes)).<:Nodes)
      y.add_node = -1/(n_nodes^2)
    end

    return 7, new_node.GIN, new_connection1.GIN, new_connection2.GIN
  elseif mutation == 8 # disable forward connection
    network_info = GetNetworkInfo(x, true)
    GINs = network_info[(network_info[:,:type].<:Connections), :GIN]
    if isempty(GINs)
      return 8
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].enabled = false
    return 8, random_connection
  elseif mutation == 9 # disable node
    network_info = GetNetworkInfo(x, true)
    GINs = network_info[(network_info[:,:type].<:HiddenNode), :GIN]
    if isempty(GINs)
      return 9
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].enabled = false
    return 9, random_node
  elseif mutation == 10 # enable forward connection
    network_info = GetNetworkInfo(x, false)
    GINs = network_info[(network_info[:,:type].<:Connections).||(network_info[:,:enabled].==false), :GIN]
    if isempty(GINs)
      return 10
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].enabled = true
    return 10, random_connection
  elseif mutation == 11 # enable node
    network_info = GetNetworkInfo(x, true)
    GINs = network_info[(network_info[:,:type].<:HiddenNode).||(network_info[:,:enabled].==false), :GIN]
    if isempty(GINs)
      return 11
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].enabled = true 
    return 11, random_node
  elseif mutation == 12 # change activation function
    network_info = GetNetworkInfo(x, true)
    GINs = network_info[(network_info[:,:type].<:HiddenNode).||(network_info[:,:type].<:OutputNode), :GIN]
    if isempty(GINs)
      return 12
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].activation_function = rand(rng(), x.list_activation_functions)
    return 12, random_node
  end
end
function GetNetworkDistance(x::FFNN, y::FFNN, distance_parameters::Vector{T} = [1,1,1], normalise::Bool = true) where T <: Real
  length(distance_parameters) == 3 || throw(error("distance_parameters should be of length 3"))

  x_GINs = collect(keys(x.genes))
  y_GINs = collect(keys(y.genes))

  common_GINs = intersect(x_GINs, y_GINs)
  max_common_GIN = maximum(common_GINs)

  uncommon_GINs = union(setdiff(x_GINs, y_GINs), setdiff(y_GINs, x_GINs))

  n_disjoint_GINs = isempty(uncommon_GINs) ? 0 : sum(uncommon_GINs .< max_common_GIN)
  n_excess_GINs = isempty(uncommon_GINs) ? 0 : sum(uncommon_GINs .> max_common_GIN)

  common_connection_GINs = common_GINs[typeof.(getindex.((x.genes,), common_GINs)).<:Connections]
  sum_weights_difference = isempty(common_connection_GINs) ? 0 : sum( abs.(  getfield.(getindex.((x.genes,), common_connection_GINs), (:weight,))  .-  getfield.(getindex.((y.genes,), common_connection_GINs), (:weight,))  ) )

  if normalise
    max_number_of_GINs = max(length(x_GINs), length(y_GINs))
    distance = [n_disjoint_GINs, n_excess_GINs, sum_weights_difference] .* distance_parameters ./ [max_number_of_GINs, max_number_of_GINs, 1]
  else
    distance = [n_disjoint_GINs, n_excess_GINs, sum_weights_difference] .* distance_parameters
  end
  distance = sum(distance)

  return distance
end
# function GetNetworkDistance1(x::FFNN, y::FFNN, distance_parameters::Vector{T} = [1,1,1], normalise::Bool = true) where T<:Real
#   length(distance_parameters) == 3 || throw(error("distance_parameters should be of length 3"))
#   max_GIN = max(maximum(keys(x.genes)), maximum(keys(y.genes)))
#   x_GINs = keys(x.genes)
#   y_GINs = keys(y.genes)
#
#   x_connection_GINs = [i for i in x_GINs if typeof(x.genes[i])<:Connections]
#   y_connection_GINs = [i for i in y_GINs if typeof(y.genes[i])<:Connections]
#
#   x_connection_GINs = Bool[i in x_connection_GINs for i = 1:max_GIN]
#   y_connection_GINs = Bool[i in y_connection_GINs for i = 1:max_GIN]
#
#   x_GINs = Bool[i in x_GINs for i = 1:max_GIN]
#   y_GINs = Bool[i in y_GINs for i = 1:max_GIN]
#
#   common_GINs = x_GINs .&& y_GINs
#   idx_last_common_GIN = findlast(common_GINs)
#   non_common_GINs = xor.(x_GINs, y_GINs)
#
#   n_disjoint_GINs = sum(non_common_GINs[1:idx_last_common_GIN])
#   n_excess_GINs = sum(non_common_GINs[idx_last_common_GIN:end])
#
#   common_connection_GINs = x_connection_GINs .&& y_connection_GINs
#   sum_weights_difference = 0
#   for i in findall(common_connection_GINs)
#     sum_weights_difference += abs( x.genes[i].weight - y.genes[i].weight )
#   end
#
#   if normalise
#     max_number_of_GINs = max(sum(x_GINs), sum(y_GINs))
#     distance = [n_disjoint_GINs, n_excess_GINs, sum_weights_difference] .* distance_parameters ./ [max_number_of_GINs, max_number_of_GINs, 1]
#   else
#     distance = [n_disjoint_GINs, n_excess_GINs, sum_weights_difference] .* distance_parameters
#   end
#   distance = sum(distance)
#
#   return distance
# end
function GetNetworkInfo(x::FFNN, simple::Bool = true)
  ret = DataFrame(GIN = Unsigned[],
                  type = Type[],
                  in_node = Union{Missing, Unsigned}[],
                  out_node = Union{Missing, Unsigned}[],
                  enabled = Bool[],
                 )
  for i = sort(collect(keys(x.genes)))
    if simple && !x.genes[i].enabled
      continue
    end
    if typeof(x.genes[i]) <: Nodes
      push!(ret, [i, typeof(x.genes[i]), missing, missing, x.genes[i].enabled])
    elseif typeof(x.genes[i]) <: Connections
      push!(ret, [i, typeof(x.genes[i]), x.genes[i].in_node.GIN, x.genes[i].out_node.GIN, x.genes[i].enabled])
    end
  end

  return ret
end

function Reset!(x::Networks)
  for i in x.genes
    Reset!(i.second)
  end
  return
end

@kwdef mutable struct RNN <: Networks # Recurrent Neural Network
  ID::Unsigned = 0 # it's index in the population
  const n_inputs::Unsigned # number of inputs
  const n_outputs::Unsigned # number of outputs
  input::Vector{Reference{Real}} = Reference{Real}[] # points to all input values for this network
  output::Vector{Reference{Real}} = Reference{Real}[] # points to all output values of this network
  genes::Dict{Unsigned, Genes} = Dict{Unsigned, Genes}() # represents all genes in this network, Dict( GIN (Global Innovation Number) => Node / Connection )
  specie::Unsigned = 0
  layers::Vector{Vector{Nodes}} = Vector{Vector{Nodes}}()
  list_activation_functions::Vector{Function} = Function[Sigmoid]
  super::Union{Missing, NEATs} = missing # points to the NEAT containing this network
end
function Init(x::RNN, start_fully_connected = true)
  layer = Vector{Nodes}(undef, x.n_inputs)
  x.input = Vector{Reference{Real}}(undef, x.n_inputs)
  @threads for i = 1:x.n_inputs
    x.input[i] = Reference{Real}()
    x.genes[i] = RecurrentInputNode(GIN = i, input_number = i, activation_function = rand(rng(), x.list_activation_functions), super = x)
    push!(x.genes[i].input, x.input[i])
    layer[i] = x.genes[i]
  end
  push!(x.layers, layer)

  layer = Vector{Nodes}(undef, x.n_outputs)
  x.output = Vector{Reference{Real}}(undef, x.n_outputs)
  @threads for i = 1:x.n_outputs
    x.output[i] = Reference{Real}()
    x.genes[i+x.n_inputs] = RecurrentOutputNode(GIN = i+x.n_inputs, output_number = i, output = x.output[i], activation_function = rand(rng(), x.list_activation_functions), super = x)
    layer[i] = x.genes[i+x.n_inputs]
  end
  push!(x.layers, layer)

  if start_fully_connected
    @threads for i = 1:x.n_inputs
      for j = 1:x.n_outputs
        AddForwardConnection(x, i,j+x.n_inputs, x.n_inputs+x.n_outputs + (i-1)*x.n_outputs + j )
      end
    end
  end

  return
end
function toDict(x::RNN)
  genes = Dict{Unsigned, Dict}()
  for i = sort(collect(keys(x.genes)))
    genes[i] = toDict(x.genes[i])
  end
  ret = Dict("instance" => typeof(x),
             "ID" => x.ID,
             "n_inputs" => x.n_inputs,
             "n_outputs" => x.n_outputs,
             "input" => [i[] for i in x.input],
             "output" => [i[] for i in x.output],
             "genes" => genes,
             "specie" => x.specie,
             "layers" => [[n.GIN for n in l] for l in x.layers],
             "list_activation_functions" => string.(x.list_activation_functions),
             "super" => typeof(x.super),
            )
  return ret
end
function Run(x::RNN)
  for i in x.genes
    if !(typeof(i.second) <: RecurrentConnection)
      i.second.processed = false
    end
  end
  for l in x.layers
    for n in l
      Run(n)
      for c in n.out_connections
        Run(c)
      end
    end
  end

  return [i[] for i in x.output]
end
function GetInput(x::RNN)
  return [i[] for i in x.input]
end
function SetInput!(x::RNN, args::Vector{T}) where T<:Real
  length(args) == x.n_inputs || throw(error("Invalid number of input arguments"))

  for i = 1:x.n_inputs
    x.input[i][] = args[i]
  end
  return
end
function GetOutput(x::RNN)
  return [i[] for i in x.output]
end
function GetLayers(x::RNN, simple::Bool = true)
  return [Unsigned[n.GIN for n in l if (!simple || n.enabled)] for l in x.layers]
end
function GetNetworkInfo(x::RNN, simple::Bool = true)
  ret = DataFrame(GIN = Unsigned[],
                  type = Type[],
                  in_node = Union{Missing, Unsigned}[],
                  out_node = Union{Missing, Unsigned}[],
                  enabled = Bool[],
                 )
  for i = sort(collect(keys(x.genes)))
    if simple && !x.genes[i].enabled
      continue
    end
    if typeof(x.genes[i]) <: Nodes
      push!(ret, [i, typeof(x.genes[i]), missing, missing, x.genes[i].enabled])
    elseif typeof(x.genes[i]) <: Connections
      push!(ret, [i, typeof(x.genes[i]), x.genes[i].in_node.GIN, x.genes[i].out_node.GIN, x.genes[i].enabled])
    end
  end

  return ret
end
function GetNodePosition(x::RNN, y::Integer)
  y = Unsigned(y)
  layer_number = 0
  node_number = 0
  for (i,l) in enumerate(x.layers)
    is_break = false
    for (j,n) in enumerate(l)
      if n.GIN == y
        layer_number = i
        node_number = j
        is_break = true
        break
      end
    end
    if is_break
      break
    end
  end
  return (layer_number, node_number)
end
function AddForwardConnection(x::RNN, start_node_GIN::Integer, stop_node_GIN::Integer, new_GIN::Integer)
  start_node_GIN = Unsigned(start_node_GIN)
  stop_node_GIN = Unsigned(stop_node_GIN)
  new_GIN = Unsigned(new_GIN)

  new_GIN in keys(x.genes) && throw(error("new_GIN = $(new_GIN) already exists"))

  start_layer = GetNodePosition(x, start_node_GIN)[1]
  start_layer == 0 && throw(error("start node is not present"))

  stop_layer = GetNodePosition(x, stop_node_GIN)[1]
  stop_layer == 0 && throw(error("stop node is not present"))

  stop_layer <= start_layer && throw(error("stop_layer <= start_layer"))

  stop_node_GIN in getfield.(getfield.(x.genes[start_node_GIN].out_connections, :out_node), :GIN) && throw(error("connection exists from $(start_node_GIN) to $(stop_node_GIN)"))

  connection = ForwardConnection(in_node = x.genes[start_node_GIN], out_node = x.genes[stop_node_GIN], GIN = new_GIN, super = x)
  Init(connection)
  x.genes[new_GIN] = connection

  return connection
end
function AddRecurrentConnection(x::RNN, start_node_GIN::Integer, stop_node_GIN::Integer, new_GIN::Integer)
  start_node_GIN = Unsigned(start_node_GIN)
  stop_node_GIN = Unsigned(stop_node_GIN)
  new_GIN = Unsigned(new_GIN)

  new_GIN in keys(x.genes) && throw(error("new_GIN = $(new_GIN) already exists"))

  start_layer = GetNodePosition(x, start_node_GIN)[1]
  start_layer == 0 && throw(error("start node is not present"))

  stop_layer = GetNodePosition(x, stop_node_GIN)[1]
  stop_layer == 0 && throw(error("stop node is not present"))

  stop_layer > start_layer && throw(error("AddRecurrentConnection: stop_layer > start_layer"))

  stop_node_GIN in getfield.(getfield.(x.genes[start_node_GIN].out_connections, :out_node), :GIN) && throw(error("connection exists from $(start_node_GIN) to $(stop_node_GIN)"))

  connection = RecurrentConnection(in_node = x.genes[start_node_GIN], out_node = x.genes[stop_node_GIN], GIN = new_GIN, super = x)
  Init(connection)
  x.genes[new_GIN] = connection

  return connection
end
function AddHiddenNode(x::RNN, connection_GIN::Integer, new_GIN::Integer)
  connection_GIN = Unsigned(connection_GIN)
  new_GIN = Unsigned(new_GIN)

  new_GIN in keys(x.genes) && throw(error("new_GIN = $(new_GIN) already exists"))

  connection_GIN in keys(x.genes) || throw(error("connection_GIN = $(connection_GIN) not present"))

  if typeof(x.genes[connection_GIN]) <: ForwardConnection
    node = HiddenNode(GIN = new_GIN, activation_function = rand(rng(), x.list_activation_functions), super = x)
    x.genes[new_GIN] = node

    layer = rand(rng(), GetNodePosition(x, x.genes[connection_GIN].in_node.GIN)[1]+0.5:0.5:GetNodePosition(x, x.genes[connection_GIN].out_node.GIN)[1]-0.5)
    if floor(layer) == layer # existing layer
      layer = Int(layer)
      push!(x.layers[layer], node)
    else # create a new layer
      layer = Int(ceil(layer))
      insert!(x.layers, layer, Nodes[node])
    end

    connection1 = AddForwardConnection(x, x.genes[connection_GIN].in_node.GIN, new_GIN, new_GIN+1)
    connection2 = AddForwardConnection(x, new_GIN, x.genes[connection_GIN].out_node.GIN, new_GIN+2)
  elseif typeof(x.genes[connection_GIN]) <: RecurrentConnection
    layer = GetNodePosition(x, x.genes[connection_GIN].in_node.GIN)[1]-0.5:-0.5:GetNodePosition(x, x.genes[connection_GIN].out_node.GIN)[1]+0.5
    if isempty(layer)
      return nothing, nothing, nothing
    end

    node = HiddenNode(GIN = new_GIN, activation_function = rand(rng(), x.list_activation_functions), super = x)
    x.genes[new_GIN] = node

    layer = rand(rng(), layer)
    if floor(layer) == layer # existing layer
      layer = Int(layer)
      push!(x.layers[layer], node)
    else # create a new layer
      layer = Int(ceil(layer))
      insert!(x.layers, layer, Nodes[node])
    end

    connection1 = AddRecurrentConnection(x, x.genes[connection_GIN].in_node.GIN, new_GIN, new_GIN+1)
    connection2 = AddRecurrentConnection(x, new_GIN, x.genes[connection_GIN].out_node.GIN, new_GIN+2)
  else
    throw(error("connection_GIN = $(connection_GIN) is not of type Connections"))
  end

  x.genes[connection_GIN].enabled = false

  return node, connection1, connection2
end
function GetNetworkDistance(x::RNN, y::RNN, distance_parameters::Vector{T} = [1,1,1], normalise::Bool = true) where T <: Real
  length(distance_parameters) == 3 || throw(error("distance_parameters should be of length 3"))

  x_GINs = collect(keys(x.genes))
  y_GINs = collect(keys(y.genes))

  common_GINs = intersect(x_GINs, y_GINs)
  max_common_GIN = maximum(common_GINs)

  uncommon_GINs = union(setdiff(x_GINs, y_GINs), setdiff(y_GINs, x_GINs))

  n_disjoint_GINs = isempty(uncommon_GINs) ? 0 : sum(uncommon_GINs .< max_common_GIN)
  n_excess_GINs = isempty(uncommon_GINs) ? 0 : sum(uncommon_GINs .> max_common_GIN)

  common_connection_GINs = common_GINs[typeof.(getindex.((x.genes,), common_GINs)).<:Connections]
  sum_weights_difference = isempty(common_connection_GINs) ? 0 : sum( abs.(  getfield.(getindex.((x.genes,), common_connection_GINs), (:weight,))  .-  getfield.(getindex.((y.genes,), common_connection_GINs), (:weight,))  ) )

  if normalise
    max_number_of_GINs = max(length(x_GINs), length(y_GINs))
    distance = [n_disjoint_GINs, n_excess_GINs, sum_weights_difference] .* distance_parameters ./ [max_number_of_GINs, max_number_of_GINs, 1]
  else
    distance = [n_disjoint_GINs, n_excess_GINs, sum_weights_difference] .* distance_parameters
  end
  distance = sum(distance)

  return distance
end
# function GetNetworkDistance1(x::RNN, y::RNN, distance_parameters::Vector{T} = [1,1,1], normalise::Bool = true) where T<:Real
#   length(distance_parameters) == 3 || throw(error("distance_parameters should be of length 3"))
#   max_GIN = max(maximum(keys(x.genes)), maximum(keys(y.genes)))
#   x_GINs = keys(x.genes)
#   y_GINs = keys(y.genes)
#
#   x_connection_GINs = [i for i in x_GINs if typeof(x.genes[i])<:Connections]
#   y_connection_GINs = [i for i in y_GINs if typeof(y.genes[i])<:Connections]
#
#   x_connection_GINs = Bool[i in x_connection_GINs for i = 1:max_GIN]
#   y_connection_GINs = Bool[i in y_connection_GINs for i = 1:max_GIN]
#
#   x_GINs = Bool[i in x_GINs for i = 1:max_GIN]
#   y_GINs = Bool[i in y_GINs for i = 1:max_GIN]
#
#   common_GINs = x_GINs .&& y_GINs
#   idx_last_common_GIN = findlast(common_GINs)
#   non_common_GINs = xor.(x_GINs, y_GINs)
#
#   n_disjoint_GINs = sum(non_common_GINs[1:idx_last_common_GIN])
#   n_excess_GINs = sum(non_common_GINs[idx_last_common_GIN:end])
#
#   common_connection_GINs = x_connection_GINs .&& y_connection_GINs
#   sum_weights_difference = 0
#   for i in findall(common_connection_GINs)
#     sum_weights_difference += abs( x.genes[i].weight - y.genes[i].weight )
#   end
#
#   if normalise
#     max_number_of_GINs = max(sum(x_GINs), sum(y_GINs))
#     distance = [n_disjoint_GINs, n_excess_GINs, sum_weights_difference] .* distance_parameters ./ [max_number_of_GINs, max_number_of_GINs, 1]
#   else
#     distance = [n_disjoint_GINs, n_excess_GINs, sum_weights_difference] .* distance_parameters
#   end
#   distance = sum(distance)
#
#   return distance
# end
function Crossover(x::RNN, y::RNN, fitness_x::Real = Inf, fitness_y::Real = -Inf)
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
  ret.list_activation_functions = parent1.list_activation_functions
  ret.ID = 0
  ret.specie = 0
  for i in keys(ret.genes)
    if haskey(parent2.genes, i) && (typeof(ret.genes[i])<:Connections) && rand(rng(), [true, false])
      ret.genes[i].weight = parent2.genes[i].weight
    elseif haskey(parent2.genes, i) && (typeof(ret.genes[i])<:Nodes) && rand(rng(), [true, false])
      ret.genes[i].bias = parent2.genes[i].bias
      ret.genes[i].activation_function = parent2.genes[i].activation_function
    end
  end
  return ret
end
function Mutate(x::RNN, y::RNN_Mutation_Probability)
  mutation = sample(rng(), 1:length(y), Weights(abs.(y[:])))
  if (y.add_forward_connection < 0) && (mutation != 6 && mutation != 1)
    y.add_forward_connection *= 1.1
  end
  if (y.add_node < 0) && (mutation != 7 && mutation != 1)
    y.add_node *= 1.1
  end
  if (y.add_recurrent_connection < 0) && (mutation != 8 && mutation != 1)
    y.add_recurrent_connection *= 1.1
  end

  if mutation == 1 # no mutation
    # do nothing
    return 1
  elseif mutation == 2 # change weight
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:Connections) && (i.second.enabled == true)]
    if isempty(GINs)
      return 2
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].weight = x.super.config.min_weight + (x.super.config.max_weight - x.super.config.min_weight)*rand(rng())
    return 2, random_connection
  elseif mutation == 3 # change bias
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:Nodes) && (i.second.enabled == true)]
    if isempty(GINs)
      return 3
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].bias = x.super.config.min_bias + (x.super.config.max_bias - x.super.config.min_bias)*rand(rng())
    return 3, random_node
  elseif mutation == 4 # shift weight
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:Connections) && (i.second.enabled == true)]
    if isempty(GINs)
      return 4
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].weight += x.super.config.max_shift_weight * rand(rng()) * rand(rng(), [-1, 1])
    if x.genes[random_connection].weight > x.super.config.max_weight
      x.genes[random_connection].weight = x.super.config.min_weight
    end
    if x.genes[random_connection].weight < x.super.config.min_weight
      x.genes[random_connection].weight = x.super.config.max_weight
    end
    return 4, random_connection
  elseif mutation == 5 # shift bias
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:Nodes) && (i.second.enabled == true)]
    if isempty(GINs)
      return 5
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].bias += x.super.config.max_shift_bias * rand(rng()) * rand(rng(), [-1, 1])
    if x.genes[random_node].bias > x.super.config.max_bias
      x.genes[random_node].bias = x.super.config.min_bias
    end
    if x.genes[random_node].bias < x.super.config.min_bias
      x.genes[random_node].bias = x.super.config.max_bias
    end
    return 5, random_node
  elseif mutation == 6 # add forward connection
    network_layers = GetLayers(x, true)
    non_empty_network_layers = [i for i = 1:length(network_layers) if !(isempty(network_layers[i]))]
    add_connection = false
    start_layer = 0
    stop_layer = 0
    start_node_GIN = 0
    stop_node_GIN = 0
    for i = 1:50
      start_layer = rand(rng(), 1:length(non_empty_network_layers)-1)
      stop_layer = rand(rng(), start_layer+1:length(non_empty_network_layers))
      start_layer = non_empty_network_layers[start_layer]
      stop_layer = non_empty_network_layers[stop_layer]

      start_node_GIN = rand(rng(), network_layers[start_layer])
      stop_node_GIN = rand(rng(), network_layers[stop_layer])

      stop_node_GIN in getfield.(getfield.(x.genes[start_node_GIN].out_connections, :out_node), :GIN) ? (add_connection = false; continue) : (add_connection = true; break)
    end
    if add_connection
      new_GIN = all( [getfield.(x.super.GIN, :start_node) getfield.(x.super.GIN, :stop_node)] .== [start_node_GIN stop_node_GIN], dims = 2 )[:]
      new_GIN = findfirst(new_GIN)
      if isnothing(new_GIN)
        new_GIN = x.super.GIN[end].GIN+1
        new_connection = AddForwardConnection(x, start_node_GIN, stop_node_GIN, new_GIN)
        push!(x.super.GIN, GIN(new_connection.GIN, typeof(new_connection), new_connection.in_node.GIN, new_connection.out_node.GIN))

        if y.add_forward_connection < 0
          n_connections = sum(typeof.(values(x.genes)).<:ForwardConnection)
          y.add_forward_connection = -1/(n_connections^2)
        end

        return 6, "new", new_connection.GIN
      else
        new_connection = AddForwardConnection(x, start_node_GIN, stop_node_GIN, new_GIN)

        if y.add_forward_connection < 0
          n_connections = sum(typeof.(values(x.genes)).<:ForwardConnection)
          y.add_forward_connection = -1/(n_connections^2)
        end

        return 6, "old", new_connection.GIN
      end
    else
      if y.add_forward_connection < 0
        n_connections = sum(typeof.(values(x.genes)).<:ForwardConnection)
        y.add_forward_connection = -1/(n_connections^2)
      end
      return 6
    end
  elseif mutation == 7 # add node
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:Connections) && (i.second.enabled == true)]
    if isempty(GINs)
      return 7
    end
    random_connection = rand(rng(), GINs)
    new_GIN = x.super.GIN[end].GIN+1
    new_node, new_connection1, new_connection2 = AddHiddenNode(x, random_connection, new_GIN)
    if isnothing(new_node)
      if y.add_node < 0
        n_nodes = sum(typeof.(values(x.genes)).<:Nodes)
        y.add_node = -1/(n_nodes^2)
      end
      return 7
    end
    push!(x.super.GIN, GIN(new_node.GIN, typeof(new_node), 0, 0))
    push!(x.super.GIN, GIN(new_connection1.GIN, typeof(new_connection1), new_connection1.in_node.GIN, new_connection1.out_node.GIN))
    push!(x.super.GIN, GIN(new_connection2.GIN, typeof(new_connection2), new_connection2.in_node.GIN, new_connection2.out_node.GIN))

    if y.add_node < 0
      n_nodes = sum(typeof.(values(x.genes)).<:Nodes)
      y.add_node = -1/(n_nodes^2)
    end

    return 7, new_node.GIN, new_connection1.GIN, new_connection2.GIN
  elseif mutation == 8 # add recurrent connection
    network_layers = GetLayers(x, true)
    non_empty_network_layers = [i for i = 1:length(network_layers) if !(isempty(network_layers[i]))]
    add_connection = false
    start_layer = 0
    stop_layer = 0
    start_node_GIN = 0
    stop_node_GIN = 0
    for i = 1:50
      start_layer = rand(rng(), 1:length(non_empty_network_layers))
      stop_layer = rand(rng(), 1:start_layer)
      start_layer = non_empty_network_layers[start_layer]
      stop_layer = non_empty_network_layers[stop_layer]

      start_node_GIN = rand(rng(), network_layers[start_layer])
      stop_node_GIN = rand(rng(), network_layers[stop_layer])

      stop_node_GIN in getfield.(getfield.(x.genes[start_node_GIN].out_connections, :out_node), :GIN) ? (add_connection = false; continue) : (add_connection = true; break)
    end
    if add_connection
      new_GIN = all( [getfield.(x.super.GIN, :start_node) getfield.(x.super.GIN, :stop_node)] .== [start_node_GIN stop_node_GIN], dims = 2 )[:]
      new_GIN = findfirst(new_GIN)
      if isnothing(new_GIN)
        new_GIN = x.super.GIN[end].GIN+1
        new_connection = AddRecurrentConnection(x, start_node_GIN, stop_node_GIN, new_GIN)
        push!(x.super.GIN, GIN(new_connection.GIN, typeof(new_connection), new_connection.in_node.GIN, new_connection.out_node.GIN))

        if y.add_recurrent_connection < 0
          n_connections = sum(typeof.(values(x.genes)).<:RecurrentConnection)
          y.add_recurrent_connection = -1/(n_connections^2)
        end

        return 8, "new", new_connection.GIN
      else
        new_connection = AddRecurrentConnection(x, start_node_GIN, stop_node_GIN, new_GIN)

        if y.add_recurrent_connection < 0
          n_connections = sum(typeof.(values(x.genes)).<:RecurrentOutputNode)
          y.add_recurrent_connection = -1/(n_connections^2)
        end

        return 8, "old", new_connection.GIN
      end
    else
      if y.add_recurrent_connection < 0
        n_connections = sum(typeof.(values(x.genes)).<:RecurrentConnection)
        y.add_recurrent_connection = -1/(n_connections^2)
      end
      return 8
    end
  elseif mutation == 9 # disable forward connection
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:ForwardConnection) && (i.second.enabled == true)]
    if isempty(GINs)
      return 9
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].enabled = false
    return 9, random_connection
  elseif mutation == 10 # disable node
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:HiddenNode) && (i.second.enabled == true)]
    if isempty(GINs)
      return 10
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].enabled = false
    return 10, random_node
  elseif mutation == 11 # disable recurrent connection
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:RecurrentConnection) && (i.second.enabled == true)]
    if isempty(GINs)
      return 11
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].enabled = false
    return 11, random_connection
  elseif mutation == 12 # enable forward connection
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:ForwardConnection) && (i.second.enabled == false)]
    if isempty(GINs)
      return 12
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].enabled = true
    return 12, random_connection
  elseif mutation == 13 # enable node
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:HiddenNode) && (i.second.enabled == false)]
    if isempty(GINs)
      return 13
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].enabled = true 
    return 13, random_node
  elseif mutation == 14 # enable recurrent connection
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:RecurrentConnection) && (i.second.enabled == false)]
    if isempty(GINs)
      return 14
    end
    random_connection = rand(rng(), GINs)
    x.genes[random_connection].enabled = true
    return 14, random_connection
  elseif mutation == 15 # change activation function
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:Nodes) && (i.second.enabled == true)]
    if isempty(GINs)
      return 15
    end
    random_node = rand(rng(), GINs)
    x.genes[random_node].activation_function = rand(rng(), x.list_activation_functions)
    return 15, random_node
  elseif mutation == 16 # add self connection
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:Nodes) && (i.second.enabled == true)]
    if isempty(GINs)
      return 16
    end
    start_node_GIN = 0
    add_connection = false
    for i = 1:50
      start_node_GIN = rand(GINs)
      start_node_GIN in getfield.(getfield.(x.genes[start_node_GIN].out_connections, :out_node), :GIN) ? (add_connection = false; continue) : (add_connection = true; break)
    end

    if add_connection
      new_GIN = all( [getfield.(x.super.GIN, :start_node) getfield.(x.super.GIN, :stop_node)] .== [start_node_GIN start_node_GIN], dims = 2 )[:]
      new_GIN = findfirst(new_GIN)
      if isnothing(new_GIN)
        new_GIN = x.super.GIN[end].GIN+1
        new_connection = AddRecurrentConnection(x, start_node_GIN, start_node_GIN, new_GIN)
        push!(x.super.GIN, GIN(new_connection.GIN, typeof(new_connection), new_connection.in_node.GIN, new_connection.out_node.GIN))

        return 16, "new", new_connection.GIN
      else
        new_connection = AddRecurrentConnection(x, start_node_GIN, start_node_GIN, new_GIN)

        return 16, "old", new_connection.GIN
      end
    else
      return 16
    end
  elseif mutation == 17 # enable self connection
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:RecurrentConnection) && (i.second.enabled == false) && (i.second.in_node == i.second.out_node)]
    if isempty(GINs)
      return 17
    end
    random_connection = rand(GINs)
    x.genes[random_connection].enabled = true
    return 17, random_connection
  elseif mutation == 18 # disable self connection
    GINs = Unsigned[i.first for i in x.genes if (typeof(i.second)<:RecurrentConnection) && (i.second.enabled == true) && (i.second.in_node == i.second.out_node)]
    if isempty(GINs)
      return 18
    end
    random_connection = rand(GINs)
    x.genes[random_connection].enabled = false
    return 18, random_connection
  end
end

@kwdef mutable struct CrossoverProbability <: Probabilities
  intraspecie_good_and_good::Real = 1
  intraspecie_good_and_bad::Real = 1
  intraspecie_bad_and_bad::Real = 1
  interspecie_good_and_good::Real = 1
  interspecie_good_and_bad::Real = 1
  interspecie_bad_and_bad::Real = 1
end

function display(x::Probabilities)
  println(typeof(x))
  for i in fieldnames(typeof(x))
    print(" $(string(i)) : $(getfield(x, i))\n")
  end
  return
end
function display(x::Vector{T}) where T <: Probabilities
  println("$(length(x))-element $(typeof(x)):")
  for i in x
    display(i)
  end
end
function lastindex(x::Probabilities)
  return length(fieldnames(typeof(x)))
end
function getindex(x::Probabilities, idx::Union{T, Vector{T}, OrdinalRange{T, T}}) where T<:Integer
  names = fieldnames(typeof(x))
  ret = Real[getfield(x, names[i]) for i in idx]
  if length(ret) == 1
    return ret[1]
  else
    return ret
  end
end
function getindex(x::Probabilities, idx::Symbol)
  return getfield(x, idx)
end
function getindex(x::Probabilities, idx::Colon)
  return x[1:end]
end
function setindex!(x::Probabilities, v::Real, idx::Integer)
  names = fieldnames(typeof(x))
  for i in idx
    setfield!(x, names[i], v)
  end
  return
end
function length(x::Probabilities)
  return length(fieldnames(typeof(x)))
end
function toDict(x::Probabilities)
  ret = Dict()
  for i in fieldnames(typeof(x))
    ret[string(i)] = x[i]
  end
  return ret
end

@kwdef mutable struct SpecieInfo
  specie::Unsigned
  alive::Bool = true
  birth_generation::Unsigned
  death_generation::Unsigned = 0
  minimum_fitness::Real = -Inf
  maximum_fitness::Real = -Inf
  mean_fitness::Real = -Inf
  last_topped_generation::Unsigned = 0
  last_improved_generation::Unsigned = 0
  last_highest_maximum_fitness::Real = -Inf
end
function display(x::SpecieInfo)
  println(typeof(x))
  for i in fieldnames(typeof(x))
    print(" $(string(i)) : $(getfield(x, i))\n")
  end
  return
end
function display(x::Vector{SpecieInfo})
  println("$(length(x))-element $(typeof(x)):")
  for i in x
    display(i)
  end
end
function toDict(x::SpecieInfo)
  ret = Dict()
  for i in fieldnames(typeof(x))
    ret[string(i)] = getfield(x, i)
  end
  return ret
end

function Reset!(x::NEATs)
  for i in x.population
    Reset!(i)
  end
  return
end


@kwdef mutable struct NEAT_Log_config
  log_to_console::Bool = true
  log_to_file::Bool = true
  filename::String = ""
  delimeter::String = "<>"

  log_initialisation::Bool = true

  timestamp::Bool = true
  NEAT_type::Bool = true
  generation::Bool = true
  best_networks::Bool = true
  best_fitness::Bool = true
  time_taken::Bool = true
  species::Bool = false
  max_GIN::Bool = false
end
function display(x::NEAT_Log_config)
  println(typeof(x))
  for i in fieldnames(NEAT_Log_config)
    println(" $(i) : $(getfield(x, i))")
  end
  return
end
function toDict(x::NEAT_Log_config)
  ret = Dict(string(i) => getfield(x, i) for i in fieldnames(typeof(x)))
  return ret
end

@kwdef mutable struct NEAT_FFNN_config <: NEAT_configs
  const n_inputs::Unsigned
  const n_outputs::Unsigned
  max_generation::Unsigned = 100
  const population_size::Unsigned = 20
  threshold_fitness::Real
  n_networks_to_pass::Unsigned = 1
  n_generations_to_pass::Unsigned = 1
  fitness_test_dict::Dict{String, Any}
  max_weight::Real = 2.0
  min_weight::Real = -2.0
  max_shift_weight::Real = 0.1 # the maximum amount weights can be shifted during shift weight mutation
  max_bias::Real = 1.0
  min_bias::Real = -1.0
  max_shift_bias::Real = 0.1 # the maximum amount biases can be shifted during shift bias mutation
  n_individuals_considered_best::Real = 0.25 # number of individuals considered best in a specie in a generation. Takes real values >= 0. Number less than 1 is considered as ratio over total specie population, number >= 1 is considered as number of individuals.
  n_individuals_to_retain::Real = 1 # number of individuals to retain unchanged for next generation of a specie. Takes real values >= 0. Number less than 1 is considered as ratio over total specie population, number >= 1 is considered as number of individuals.
  crossover_probability::CrossoverProbability = CrossoverProbability() # [intraspecie good and good, intraspecie good and bad, intraspecie bad and bad, interspecie good and good, interspecie good and bad, interspecie bad and bad]
  max_specie_stagnation::Unsigned = 20
  distance_parameters::Vector{Real} = [1,1,1]
  threshold_distance::Real = 0.9
  normalise_distance::Bool = true
  list_activation_functions::Vector{Function} = [Sigmoid]
  start_fully_connected::Bool = false
  max_species_per_generation::Unsigned = 4
  initial_mutation_probability::FFNN_Mutation_Probability = FFNN_Mutation_Probability()
  log_config::NEAT_Log_config = NEAT_Log_config()
end
function display(x::NEAT_FFNN_config)
  println(typeof(x))
  return
end
function toDict(x::NEAT_FFNN_config)
  ret = Dict("n_inputs" => x.n_inputs,
             "n_outputs" => x.n_outputs,
             "max_generation" => x.max_generation,
             "population_size" => x.population_size,
             "threshold_fitness" => x.threshold_fitness,
             "n_networks_to_pass" => x.n_networks_to_pass,
             "n_generations_to_pass" => x.n_generations_to_pass,
             "fitness_test_dict" => string(x.fitness_test_dict["fitness_function"]),
             "max_weight" => x.max_weight,
             "min_weight" => x.min_weight,
             "max_shift_weight" => x.max_shift_weight,
             "max_bias" => x.max_bias,
             "min_bias" => x.min_bias,
             "max_shift_bias" => x.max_shift_bias,
             "n_individuals_considered_best" => x.n_individuals_considered_best,
             "n_individuals_to_retain" => x.n_individuals_to_retain,
             "crossover_probability" => toDict(x.crossover_probability),
             "max_specie_stagnation" => x.max_specie_stagnation,
             "distance_parameters" => x.distance_parameters,
             "threshold_distance" => x.threshold_distance,
             "normalise_distance" => x.normalise_distance,
             "list_activation_functions" => string.(x.list_activation_functions),
             "start_fully_connected" => x.start_fully_connected,
             "max_species_per_generation" => x.max_species_per_generation,
             "initial_mutation_probability" => toDict(x.initial_mutation_probability),
             "log_config" => toDict(x.log_config),
            )
  return ret
end

@kwdef mutable struct NEAT_FFNN <: NEATs
  config::NEAT_FFNN_config
  GIN::Vector{GIN} = GIN[]
  population::Vector{Networks} = Networks[]
  generation::Unsigned = 0
  winners::Vector{Networks} = Networks[]
  fitness::Vector{Real} = Real[]
  mutation_probability::Vector{FFNN_Mutation_Probability} = FFNN_Mutation_Probability[]
  species::Dict{Unsigned, Vector{Networks}} = Dict{Unsigned, Vector{Networks}}()
  specie_info::Vector{SpecieInfo} = SpecieInfo[]
  n_generations_passed::Unsigned = 0
  n_networks_passed::Unsigned = 0
end
function Init(x::NEAT_FFNN)
  isempty(x.config.log_config.filename) && (x.config.log_config.filename = "$(typeof(x)).log")
  start_time = time()

  x.population = Vector{Networks}(undef, x.config.population_size)
  x.species[1] = Vector{Networks}(undef, x.config.population_size)
  x.fitness = Vector{Real}(undef, x.config.population_size)
  x.mutation_probability = Vector{FFNN_Mutation_Probability}(undef, x.config.population_size)
  @threads for i = 1:x.config.population_size
    x.population[i] = FFNN(ID = i, n_inputs = x.config.n_inputs, n_outputs = x.config.n_outputs, specie = 1, list_activation_functions = x.config.list_activation_functions, super = x)
    Init(x.population[i], x.config.start_fully_connected)
    x.species[1][i] = x.population[i]
    x.fitness[i] = -Inf
    x.mutation_probability[i] = deepcopy(x.config.initial_mutation_probability)
  end
  push!(x.specie_info, SpecieInfo(specie = 1, birth_generation = x.generation, last_improved_generation = x.generation))

  for i = 1:x.config.n_inputs
    push!(x.GIN, GIN(i, InputNode, 0, 0))
  end
  for i = 1:x.config.n_outputs
    push!(x.GIN, GIN(i+x.config.n_inputs, OutputNode, 0, 0))
  end
  if x.config.start_fully_connected
    append!(x.GIN, Vector{GIN}(undef, x.config.n_inputs*x.config.n_outputs))
    @threads for i = 1:x.config.n_inputs
      @threads for j = 1:x.config.n_outputs
        x.GIN[x.config.n_inputs+x.config.n_outputs + (i-1)*x.config.n_outputs + j] = GIN(x.config.n_inputs+x.config.n_outputs + (i-1)*x.config.n_outputs + j , ForwardConnection, i, j+x.config.n_inputs)
      end
    end
  end

  stop_time = time()
  Log(x, start_time, stop_time, first_entry = true)

  return x
end
function display(x::NEAT_FFNN)
  println(typeof(x))
  print(" generation : $(x.generation)\n")
  return
end
function toDict(x::NEAT_FFNN)
  ret = Dict("config" => toDict(x.config),
             "GIN" => toDict.(x.GIN),
             "population" => toDict.(x.population),
             "generation" => x.generation,
             "winners" => [i.ID for i in x.winners],
             "fitness" => x.fitness,
             "mutation_probability" => toDict.(x.mutation_probability),
             "species" => Dict(i.first => [j.ID for j in i.second] for i in x.species),
             "specie_info" => toDict.(x.specie_info),
             "n_generations_passed" => x.n_generations_passed,
             "n_networks_passed" => x.n_networks_passed,
            )
  return ret
end

@kwdef mutable struct NEAT_RNN_config <: NEAT_configs
  const n_inputs::Unsigned
  const n_outputs::Unsigned
  max_generation::Unsigned = 100
  const population_size::Unsigned = 20
  threshold_fitness::Real
  n_networks_to_pass::Unsigned = 1
  n_generations_to_pass::Unsigned = 1
  fitness_test_dict::Dict{String, Any}
  max_weight::Real = 2.0
  min_weight::Real = -2.0
  max_shift_weight::Real = 0.1 # the maximum amount weights can be shifted during shift weight mutation
  max_bias::Real = 1.0
  min_bias::Real = -1.0
  max_shift_bias::Real = 0.1 # the maximum amount biases can be shifted during shift bias mutation
  n_individuals_considered_best::Real = 0.25 # number of individuals considered best in a specie in a generation. Takes real values >= 0. Number less than 1 is considered as ratio over total specie population, number >= 1 is considered as number of individuals.
  n_individuals_to_retain::Real = 1 # number of individuals to retain unchanged for next generation of a specie. Takes real values >= 0. Number less than 1 is considered as ratio over total specie population, number >= 1 is considered as number of individuals.
  crossover_probability::CrossoverProbability = CrossoverProbability() # [intraspecie good and good, intraspecie good and bad, intraspecie bad and bad, interspecie good and good, interspecie good and bad, interspecie bad and bad]
  max_specie_stagnation::Unsigned = 20
  distance_parameters::Vector{Real} = [1,1,1]
  threshold_distance::Real = 0.9
  normalise_distance::Bool = true
  list_activation_functions::Vector{Function} = [Sigmoid]
  start_fully_connected::Bool = true
  max_species_per_generation::Unsigned = 4
  initial_mutation_probability::RNN_Mutation_Probability = RNN_Mutation_Probability()
  log_config::NEAT_Log_config = NEAT_Log_config()
end
function display(x::NEAT_RNN_config)
  println(typeof(x))
  return
end
function toDict(x::NEAT_RNN_config)
  ret = Dict("n_inputs" => x.n_inputs,
             "n_outputs" => x.n_outputs,
             "max_generation" => x.max_generation,
             "population_size" => x.population_size,
             "threshold_fitness" => x.threshold_fitness,
             "n_networks_to_pass" => x.n_networks_to_pass,
             "n_generations_to_pass" => x.n_generations_to_pass,
             "fitness_test_dict" => string(x.fitness_test_dict["fitness_function"]),
             "max_weight" => x.max_weight,
             "min_weight" => x.min_weight,
             "max_shift_weight" => x.max_shift_weight,
             "max_bias" => x.max_bias,
             "min_bias" => x.min_bias,
             "max_shift_bias" => x.max_shift_bias,
             "n_individuals_considered_best" => x.n_individuals_considered_best,
             "n_individuals_to_retain" => x.n_individuals_to_retain,
             "crossover_probability" => toDict(x.crossover_probability),
             "max_specie_stagnation" => x.max_specie_stagnation,
             "distance_parameters" => x.distance_parameters,
             "threshold_distance" => x.threshold_distance,
             "normalise_distance" => x.normalise_distance,
             "list_activation_functions" => string.(x.list_activation_functions),
             "start_fully_connected" => x.start_fully_connected,
             "max_species_per_generation" => x.max_species_per_generation,
             "initial_mutation_probability" => toDict(x.initial_mutation_probability),
             "log_config" => toDict(x.log_config)
            )
  return ret
end

@kwdef mutable struct NEAT_RNN <: NEATs
  config::NEAT_RNN_config
  GIN::Vector{GIN} = GIN[]
  population::Vector{Networks} = Networks[]
  generation::Unsigned = 0
  winners::Vector{Networks} = Networks[]
  fitness::Vector{Real} = Real[]
  mutation_probability::Vector{RNN_Mutation_Probability} = RNN_Mutation_Probability[]
  species::Dict{Unsigned, Vector{Networks}} = Dict{Unsigned, Vector{Networks}}()
  specie_info::Vector{SpecieInfo} = SpecieInfo[]
  n_generations_passed::Unsigned = 0
  n_networks_passed::Unsigned = 0
end
function Init(x::NEAT_RNN)
  isempty(x.config.log_config.filename) && (x.config.log_config.filename = "$(typeof(x)).log")
  start_time = time()

  x.population = Vector{Networks}(undef, x.config.population_size)
  x.species[1] = Vector{Networks}(undef, x.config.population_size)
  x.fitness = Vector{Real}(undef, x.config.population_size)
  x.mutation_probability = Vector{RNN_Mutation_Probability}(undef, x.config.population_size)
  @threads for i = 1:x.config.population_size
    x.population[i] = RNN(ID = i, n_inputs = x.config.n_inputs, n_outputs = x.config.n_outputs, specie = 1, list_activation_functions = x.config.list_activation_functions, super = x)
    Init(x.population[i], x.config.start_fully_connected)
    x.species[1][i] = x.population[i]
    x.fitness[i] = -Inf
    x.mutation_probability[i] = deepcopy(x.config.initial_mutation_probability)
  end
  push!(x.specie_info, SpecieInfo(specie = 1, birth_generation = x.generation, last_improved_generation = x.generation))

  for i = 1:x.config.n_inputs
    push!(x.GIN, GIN(i, RecurrentInputNode, 0, 0))
  end
  for i = 1:x.config.n_outputs
    push!(x.GIN, GIN(i+x.config.n_inputs, RecurrentOutputNode, 0, 0))
  end
  if x.config.start_fully_connected
    append!(x.GIN, Vector{GIN}(undef, x.config.n_inputs*x.config.n_outputs))
    @threads for i = 1:x.config.n_inputs
      @threads for j = 1:x.config.n_outputs
        x.GIN[x.config.n_inputs+x.config.n_outputs + (i-1)*x.config.n_outputs + j] = GIN(x.config.n_inputs+x.config.n_outputs + (i-1)*x.config.n_outputs + j , ForwardConnection, i, j+x.config.n_inputs)
      end
    end
  end

  stop_time = time()
  Log(x, start_time, stop_time, first_entry = true)

  return x
end
function display(x::NEAT_RNN)
  println(typeof(x))
  print(" generation : $(x.generation)\n")
  return
end
function toDict(x::NEAT_RNN)
  ret = Dict("config" => toDict(x.config),
             "GIN" => toDict.(x.GIN),
             "population" => toDict.(x.population),
             "generation" => x.generation,
             "winners" => [i.ID for i in x.winners],
             "fitness" => x.fitness,
             "mutation_probability" => toDict.(x.mutation_probability),
             "species" => Dict(i.first => [j.ID for j in i.second] for i in x.species),
             "specie_info" => toDict.(x.specie_info),
             "n_generations_passed" => x.n_generations_passed,
             "n_networks_passed" => x.n_networks_passed,
            )
  return ret
end

function Evaluate(x::NEATs)
  @threads for i = 1:x.config.population_size
    x.fitness[i] = x.config.fitness_test_dict["fitness_function"](x.config.fitness_test_dict, x.population[i])
  end

  winners = findall(x.fitness[.!ismissing.(x.fitness)].>=x.config.threshold_fitness)
  if isempty(winners)
    x.winners = Networks[x.population[argmax(x.fitness)]]
    x.n_networks_passed = 0x0
  else
    x.winners = x.population[winners]
    x.n_networks_passed = length(winners)
  end

  temp = Dict{Unsigned, Real}(keys(x.species).=>-Inf)
  @threads for i in collect(x.species)
    fitness = x.fitness[getfield.(i.second, :ID)]
    x.species[i.first] = x.species[i.first][sortperm(fitness, rev = true)] # sort the population based on fitness
    fitness = fitness[sortperm(fitness, rev = true)]
    x.specie_info[i.first].minimum_fitness = fitness[end]
    x.specie_info[i.first].maximum_fitness = fitness[1]
    x.specie_info[i.first].mean_fitness = mean(fitness)
    temp[i.first] = x.specie_info[i.first].maximum_fitness
    if x.specie_info[i.first].last_highest_maximum_fitness < x.specie_info[i.first].maximum_fitness
      x.specie_info[i.first].last_highest_maximum_fitness = x.specie_info[i.first].maximum_fitness
      x.specie_info[i.first].last_improved_generation = x.generation
    end
    x.species[i.first] = x.species[i.first][sortperm(fitness, rev = true)]
  end
  specie = collect(keys(temp))[argmax(collect(values(temp)))]
  x.specie_info[specie].last_topped_generation = x.generation

  return x.winners
end
function RemoveStagnantSpecies(x::NEATs)
  alive_species = [i for i in x.specie_info if i.alive]
  alive_species = alive_species[sortperm([i.mean_fitness for i in alive_species])]
  ret = Unsigned[]
  to_delete = Unsigned[]
  # Making sure not to delete the fittest specie
  @threads for i in alive_species[1:end-1]
    if ( (x.generation - i.last_improved_generation) > x.config.max_specie_stagnation ) && ( (x.generation - i.last_topped_generation) > x.config.max_specie_stagnation )
      i.alive = false
      i.death_generation = x.generation
      for j in x.species[i.specie]
        push!(to_delete, j.ID)
      end
      delete!(x.species, i.specie)
      push!(ret, i.specie)
    end
  end
  x.population = x.population[ setdiff(1:length(x.population), to_delete) ]
  x.fitness = x.fitness[ setdiff(1:length(x.fitness), to_delete) ]
  x.mutation_probability = x.mutation_probability[ setdiff(1:length(x.mutation_probability), to_delete) ]
  @threads for i = collect(enumerate(x.population))
    i[2].ID = i[1]
  end
  return ret
end
function UpdatePopulation(x::NEATs)
  new_population = Networks[]
  new_fitness = Real[]
  new_mutation_probability = Probabilities[]

  good_individuals = Dict{Unsigned, Vector{Networks}}()
  bad_individuals = Dict{Unsigned, Vector{Networks}}()

  (x.config.n_individuals_to_retain < 0) && throw(error("n_individuals_to_retain should be a real value >= 0"))
  (x.config.n_individuals_considered_best < 0) && throw(error("n_individuals_considered_best should be a real value >= 0"))

  for i in x.species
    if x.config.n_individuals_to_retain >= 1
      n = Int(min(length(i.second), floor(x.config.n_individuals_to_retain)))
      for j = 1:n
        push!(new_population, i.second[j])
        push!(new_fitness, x.fitness[i.second[j].ID])
        push!(new_mutation_probability, x.mutation_probability[i.second[j].ID])
        new_population[end].ID = length(new_population)
      end
    else
      n = Int(ceil(length(i.second)*x.config.n_individuals_to_retain))
      for j = 1:n
        push!(new_population, i.second[j])
        push!(new_fitness, x.fitness[i.second[j].ID])
        push!(new_mutation_probability, x.mutation_probability[i.second[j].ID])
        new_population[end].ID = length(new_population)
      end
    end

    good_individuals[i.first] = Networks[]
    bad_individuals[i.first] = Networks[]
    if x.config.n_individuals_considered_best >= 1
      n = Int(min(length(i.second), floor(x.config.n_individuals_considered_best)))
      for j = 1:n
        push!(good_individuals[i.first], i.second[j])
      end
      for j = n+1:length(i.second)
        push!(bad_individuals[i.first], i.second[j])
      end
    else
      n = Int(ceil(length(i.second)*x.config.n_individuals_considered_best))
      for j = 1:n
        push!(good_individuals[i.first], i.second[j])
      end
      for j = n+1:length(i.second)
        push!(bad_individuals[i.first], i.second[j])
      end
    end
  end

  new_population_length = length(new_population)
  append!(new_population, Vector{Networks}(undef, x.config.population_size-new_population_length))
  append!(new_fitness, Vector{Real}(undef, x.config.population_size-new_population_length))
  append!(new_mutation_probability, Vector{typeof(x.mutation_probability[1])}(undef, x.config.population_size-new_population_length))
  for i = new_population_length+1:x.config.population_size
    crossover_probability = [x.config.crossover_probability[1:3]; length(good_individuals)>1 ? x.config.crossover_probability[4:6] : [0,0,0]]
    crossover = sample(rng(), 1:length(x.config.crossover_probability), Weights(abs.(crossover_probability)))

    if crossover == 1 # intraspecie good and good
      specie1 = rand(rng(), keys(good_individuals))
      specie2 = specie1

      parent1 = rand(rng(), good_individuals[specie1])
      parent2 = rand(rng(), good_individuals[specie2])

      child = Crossover(parent1, parent2, x.fitness[parent1.ID], x.fitness[parent2.ID])

      new_population[i] = child
      new_fitness[i] = -Inf
      new_mutation_probability[i] = typeof(x.mutation_probability[1])( (x.mutation_probability[parent1.ID][:] .+ x.mutation_probability[parent2.ID][:])./2.0... )
    elseif crossover == 2 # intraspecie good and bad
      specie1 = rand(rng(), keys(good_individuals))
      specie2 = specie1

      parent1 = rand(rng(), good_individuals[specie1])
      if isempty(bad_individuals[specie2])
        parent2 = rand(rng(), good_individuals[specie2])
      else
        parent2 = rand(rng(), bad_individuals[specie2])
      end

      child = Crossover(parent1, parent2, x.fitness[parent1.ID], x.fitness[parent2.ID])

      new_population[i] = child
      new_fitness[i] = -Inf
      new_mutation_probability[i] = typeof(x.mutation_probability[1])( (x.mutation_probability[parent1.ID][:] .+ x.mutation_probability[parent2.ID][:])./2.0... )
    elseif crossover == 3 # intraspecie bad and bad
      specie1 = rand(rng(), keys(good_individuals))
      specie2 = specie1

      if isempty(bad_individuals[specie1])
        parent1 = rand(rng(), good_individuals[specie1])
      else
        parent1 = rand(rng(), bad_individuals[specie1])
      end
      if isempty(bad_individuals[specie2])
        parent2 = rand(rng(), good_individuals[specie2])
      else
        parent2 = rand(rng(), bad_individuals[specie2])
      end

      child = Crossover(parent1, parent2, x.fitness[parent1.ID], x.fitness[parent2.ID])

      new_population[i] = child
      new_fitness[i] = -Inf
      new_mutation_probability[i] = typeof(x.mutation_probability[1])( (x.mutation_probability[parent1.ID][:] .+ x.mutation_probability[parent2.ID][:])./2.0... )
    elseif crossover == 4 # interspecie good and good
      specie1 = rand(rng(), keys(good_individuals))
      specie2 = rand(rng(), setdiff(keys(good_individuals), specie1))

      parent1 = rand(rng(), good_individuals[specie1])
      parent2 = rand(rng(), good_individuals[specie2])

      child = Crossover(parent1, parent2, x.fitness[parent1.ID], x.fitness[parent2.ID])

      new_population[i] = child
      new_fitness[i] = -Inf
      new_mutation_probability[i] = typeof(x.mutation_probability[1])( (x.mutation_probability[parent1.ID][:] .+ x.mutation_probability[parent2.ID][:])./2.0... )
    elseif crossover == 5 # interspecie good and bad
      specie1 = rand(rng(), keys(good_individuals))
      specie2 = rand(rng(), setdiff(keys(good_individuals), specie1))

      parent1 = rand(rng(), good_individuals[specie1])
      if isempty(bad_individuals[specie2])
        parent2 = rand(rng(), good_individuals[specie2])
      else
        parent2 = rand(rng(), bad_individuals[specie2])
      end

      child = Crossover(parent1, parent2, x.fitness[parent1.ID], x.fitness[parent2.ID])

      new_population[i] = child
      new_fitness[i] = -Inf
      new_mutation_probability[i] = typeof(x.mutation_probability[1])( (x.mutation_probability[parent1.ID][:] .+ x.mutation_probability[parent2.ID][:])./2.0... )
    elseif crossover == 6 # interspecie bad and bad
      specie1 = rand(rng(), keys(good_individuals))
      specie2 = rand(rng(), setdiff(keys(good_individuals), specie1))

      if isempty(bad_individuals[specie1])
        parent1 = rand(rng(), good_individuals[specie1])
      else
        parent1 = rand(rng(), bad_individuals[specie1])
      end
      if isempty(bad_individuals[specie2])
        parent2 = rand(rng(), good_individuals[specie2])
      else
        parent2 = rand(rng(), bad_individuals[specie2])
      end

      child = Crossover(parent1, parent2, x.fitness[parent1.ID], x.fitness[parent2.ID])

      new_population[i] = child
      new_fitness[i] = -Inf
      new_mutation_probability[i] = typeof(x.mutation_probability[1])( (x.mutation_probability[parent1.ID][:] .+ x.mutation_probability[parent2.ID][:])./2.0... )
    end

    new_population[i].ID = i
    Mutate(new_population[i], new_mutation_probability[i])
  end

  x.population = new_population
  x.fitness = new_fitness
  x.mutation_probability = new_mutation_probability

  return
end
function Speciate(x::NEATs)
  new_species = Dict{Unsigned, Vector{Networks}}()

  for i in x.population
    min_distance = Inf
    specie = 0
    for j in x.species
      distance = GetNetworkDistance(i, j.second[1], x.config.distance_parameters, x.config.normalise_distance)
      if (distance < x.config.threshold_distance) && (distance < min_distance)
        min_distance = distance
        specie = j.first
      end
    end

    if specie > 0
      if haskey(new_species, specie)
        i.specie = specie
        push!(new_species[specie], i)
      else
        i.specie = specie
        new_species[specie] = Networks[i]
      end
    else
      min_distance = Inf
      specie = 0
      condition_satisfied = false
      for j in new_species
        distance = GetNetworkDistance(i, j.second[1], x.config.distance_parameters, x.config.normalise_distance)
        if (distance < min_distance)
          if distance < x.config.threshold_distance
            condition_satisfied = true
          end
          min_distance = distance
          specie = j.first
        end
      end

      if condition_satisfied
        i.specie = specie
        push!(new_species[specie], i)
      else
        if length(keys(new_species)) < x.config.max_species_per_generation
          new_specie = length(x.specie_info)+1
          push!(x.specie_info, SpecieInfo(specie = new_specie, birth_generation = x.generation, last_improved_generation = x.generation))
          i.specie = new_specie
          new_species[new_specie] = Networks[i]
        else
          i.specie = specie
          push!(new_species[specie], i)
        end
      end
    end
  end

  x.species = new_species
  return
end
function Train(x::NEATs)
  for itr = 1:x.config.max_generation
    start_time = time()
    Evaluate(x)
    if x.n_networks_passed >= x.config.n_networks_to_pass
      x.n_generations_passed += 0x1
    else
      x.n_generations_passed = 0x0
    end

    if x.n_generations_passed >= x.config.n_generations_to_pass
      println("Congrats NEAT is trained in $(x.generation) generations")
      println("Winners: $([i.ID for i in x.winners])")
      return
    end

    RemoveStagnantSpecies(x)
    UpdatePopulation(x)
    Speciate(x)

    x.generation += 0x1
    stop_time = time()
    Log(x, start_time, stop_time)
  end

  Evaluate(x)
  println("Max generation reached, training terminated")
  println("Winners: $([i.ID for i in x.winners])")
  return
end
function Log(x::NEATs, start_time::Float64 = time(), stop_time::Float64 = time(); first_entry::Bool = false)
  if x.config.log_config.log_to_file
    deli::String= x.config.log_config.delimeter
    text::String = ""
    if first_entry
      x.config.log_config.timestamp && (text *= "timestamp$(deli)")
      x.config.log_config.NEAT_type && (text *= "NEAT_type$(deli)")
      x.config.log_config.generation && (text *= "generation$(deli)")
      x.config.log_config.best_networks && (text *= "best_networks$(deli)")
      x.config.log_config.best_fitness && (text *= "best_fitness$(deli)")
      x.config.log_config.time_taken && (text *= "time_taken (sec)$(deli)")
      x.config.log_config.species && (text *= "species$(deli)")
      x.config.log_config.max_GIN && (text *= "max_GIN$(deli)")
      text *= "\n"
      open(x.config.log_config.filename, "w") do f
        write(f, text)
      end
    end

    text = ""
    x.config.log_config.timestamp && (text *= "$(now())$(deli)")
    x.config.log_config.NEAT_type && (text *= "$(typeof(x))$(deli)")
    x.config.log_config.generation && (text *= "$(x.generation)$(deli)")
    x.config.log_config.best_networks && (text *= "$([i.ID for i in x.winners])$(deli)")
    x.config.log_config.best_fitness && (text *= "$([x.fitness[i.ID] for i in x.winners])$(deli)")
    x.config.log_config.time_taken && (text *= "$(stop_time - start_time)$(deli)")
    x.config.log_config.species && (text *= "$([i.first for i in x.species])$(deli)")
    x.config.log_config.max_GIN && (text *= "$([x.GIN[end].GIN, x.GIN[end].type, x.GIN[end].start_node, x.GIN[end].stop_node])$(deli)")
    text *= "\n"
    open(x.config.log_config.filename, "a") do f
      write(f, text)
    end
  end

  if x.config.log_config.log_to_console
    x.config.log_config.timestamp && (print("timestamp = $(now()), "))
    x.config.log_config.NEAT_type && (print("NEAT_type = $(typeof(x)), "))
    x.config.log_config.generation && (print("generation = $(x.generation), "))
    x.config.log_config.best_networks && (print("best_networks = $([i.ID for i in x.winners]), "))
    x.config.log_config.best_fitness && (print("best_fitness = $([x.fitness[i.ID] for i in x.winners]), "))
    x.config.log_config.time_taken && (print("time_taken (sec) = $(Dates.canonicalize(Dates.Nanosecond(Int128(round((stop_time - start_time)*1e9))))), "))
    x.config.log_config.species && (print("species = $([i.first for i in x.species]), "))
    x.config.log_config.max_GIN && (print("max_GIN = $([x.GIN[end].GIN, x.GIN[end].type, x.GIN[end].start_node, x.GIN[end].stop_node]), "))
    println()
    println()
  end

  return
end

function display(x::Tuple{Vararg{Genes}})
  println("$(length(x))-element $(typeof(x)):")
  for i in x
    display(i)
  end
  return
end
function display(x::Tuple{Vararg{Networks}})
  println("$(length(x))-element $(typeof(x)):")
  for i in x
    display(i)
    if i != x[end]
      println()
    end
  end
  return
end
function display(x::Dict{Unsigned, T}) where T<:Genes
  println("$(typeof(x)) with $(length(x)) entries:")
  for i = sort(collect(keys(x)))
    print(" $(i) => $(typeof(x[i]))\n")
  end
  return
end
function display(x::Vector{T}) where T <: Genes
  println("$(length(x))-element $(typeof(x)):")
  for i in x
    display(i)
    if i != x[end]
      println()
    end
  end
end

function Save(x::NEATs, filename::String = "")
  isempty(filename) && (filename = "$(typeof(x))-$(x.generation).jld2")
  jldsave(filename; x)
  return filename
end
function Save(x::Networks, filename::String = "")
  isempty(filename) && (filename = "$(typeof(x))-$(x.super.generation)-$(x.ID).jld2")
  jldsave(filename; x)
  return filename
end

function Load(filename::String)
  ret = IdDict()
  jldopen(filename, "r") do f
    for i in keys(f)
      ret[i] = f[i]
    end
  end
  return ret
end

function Visualise(x::Networks; directed::Bool = true, rankdir_LR::Bool = true, connection_label::Bool = true, pen_width::Real = 1.0, export_type::String = "svg", simple::Bool = true, filename::String = "")
  graphviz_code = ""

  if rankdir_LR
    graphviz_code *= "\trankdir=LR;\n"
  end
  graphviz_code *= "\tnode [shape=circle];\n"
  for (li,l) in enumerate(x.layers)
    # graphviz_code *= "\tsubgraph cluster_$(li-1) {\n\t\tlabel=\"Layer $(li)\";\n\t\tstyle=invisible;\n"
    if l == x.layers[1]
      temp = "\t\t{rank=min; "
    elseif l == x.layers[end]
      temp = "\t\t{rank=max; "
    else
      temp = "\t\t{rank=same; "
    end
    for n in l
      if simple && !n.enabled
        continue
      end
      if l == x.layers[1]
        graphviz_code *= "\t\t$(n.GIN) [color=green, shape=diamond];\n"
      elseif l == x.layers[end]
        graphviz_code *= "\t\t$(n.GIN) [color=green, shape=box];\n"
      else
        graphviz_code *= n.enabled ? "\t\t$(n.GIN) [color=green];\n" : "\t\t$(n.GIN) [color=red];\n"
      end
      temp *= "$(n.GIN), "
      if !(typeof(n) <: OutputNode)
        for c in n.out_connections
          if simple && !c.enabled
            continue
          end
          if directed
            graphviz_code *= "\t\t$(n.GIN)->$(c.out_node.GIN)"
          else
            graphviz_code *= "\t\t$(n.GIN)--$(c.out_node.GIN)"
          end
          graphviz_code *= "["
          if c.enabled
            graphviz_code *= "color=green"
          else
            graphviz_code *= "color=red"
          end
          graphviz_code *= ","
          if connection_label
            graphviz_code *= "label=\"$(c.GIN),  $(round(c.weight,digits=3))\""
          end
          graphviz_code *= ","
          graphviz_code *= "penwidth=$(pen_width)"
          graphviz_code *= "];\n"
        end
      end
    end
    temp = temp[1:end-2] * "};\n"
    graphviz_code *= temp
    # graphviz_code *= "\t}\n"
  end

  if directed
    graphviz_code = "digraph {\n" * graphviz_code * "}"
  else
    graphviz_code = "graph {\n" * graphviz_code * "}"
  end

  if isempty(filename)
    dot_filename = simple ? "$(typeof(x))-$(x.super.generation)-$(x.ID)-simple.dot" : "$(typeof(x))-$(x.super.generation)-$(x.ID).dot"
  else
    dot_filename = simple ? "$(filename)-simple.dot" : "$(filename).dot"
  end
  open(dot_filename, "w") do file
    write(file, graphviz_code)
  end

  export_types = ["svg", "png", "jpg", "jpeg", "none"]
  if (export_type in export_types) && (export_type != "none")
    export_filename = dot_filename[1:end-3] * export_type
    try
      run(`dot -T$(export_type) $(dot_filename) -o $(export_filename)`)
    catch e
      if typeof(e) <: Base.IOError
        println("`dot` command not found skipping export to $(export_type)")
      end
    end
  elseif export_type == "none"
  else
    throw(error("Invalid export_type, got $(export_type), accepted values = $(export_types)"))
  end

  return graphviz_code
end

function Identity(x::Real)
  return x
end
function Relu(x::Real)
  return max(x, 0.0)
end
function Sigmoid(x::Real)
  return 1/(1+exp(-x))
end
function Tanh(x::Real)
  return tanh(x)
end
function Sin(x::Real)
  if x == Inf
    return 1.0
  elseif x == -Inf
    return -1.0
  else
    return sin(x)
  end
end

rng() = Random.seed!(time_ns())

end
;
