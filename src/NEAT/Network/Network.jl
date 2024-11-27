#=
# Refer src/AbstractTypes.jl for Genes, Nodes, Connections, AllNEATTypes
# Refer src/Reference.jl for Reference
# Refer src/NEAT/NEAT.jl for NEAT
# Refer src/NEAT/Network/Genes/Nodes/InputNode.jl for InputNode, InputNodeConfig
# Refer src/NEAT/Network/Genes/Nodes/OutputNode.jl for OutputNode, OutputNodeConfig
# Refer src/NEAT/Network/Genes/Nodes/HiddenNodes/HiddenNode.jl for HiddenNode, HiddenNodeConfig
# Refer src/NEAT/Network/Genes/Connections/ForwardConnection.jl for ForwardConnection, ForwardConnectionConfig
=#

export Network, ResetIO, Run, GetInput, SetInput, GetOutput, GetLayers, GetNodePosition, GetNetworkSummary

using DataFrames

@kwdef mutable struct Network <: AllNEATTypes
  idx::Unsigned = 0 # index of this network in the NEAT population
  const n_inputs::Unsigned # number of inputs
  const n_outputs::Unsigned # number of outputs
  genes::Dict{Unsigned, Genes} = Dict{Unsigned, Genes}() # this variable stores all genes (nodes and connections) of this network | Dict{ GIN (Global Innovation Number) => gene}
  input::Vector{Reference{Real}} = Reference{Real}[] # points to the input set of this network
  output::Vector{Reference{Real}} = Reference{Real}[] # points to the output set of this network
  specie::Unsigned = 0
  layers::Vector{Vector{Nodes}} = Vector{Vector{Nodes}}() # all the nodes are arranged as layers and are represented in this variable
  comments::Any = nothing # A variable to store info about this network
  super::Union{Missing, NEAT} = missing # points to the NEAT structure this Network is present in
end

function Base.show(io::IO, x::Network)
  println(io, summary(x))
  print(io, " idx : $(x.idx)
 input : $(join(Union{Missing, Real}[i[] for i in x.input], ", "))
 output : $(join(Union{Missing, Real}[i[] for i in x.output], ", "))
 number of genes : $(length(x.genes))
 specie : $(x.specie|>Int)
 number of layers : $(length(x.layers))
")
  return
end

function ResetIO(x::Network)
  Threads.@threads for i in x.input
    i[] = missing
  end

  Threads.@threads for i in x.output
    i[] = missing
  end

  Threads.@threads for i in collect(values(x.genes))
    ResetIO(i)
  end

  return
end

function Run(x::Network)
  for l in x.layers
    Threads.@threads for n in l
      Run(n)
      for c in n.out_connections
        Run(c)
      end
    end
  end

  return [i[] for i in x.output]
end

function GetInput(x::Network)
  return [i[] for i in x.input]
end

function SetInput(x::Network, args::Vector{Union{Missing, <:Real}})
  length(args) == x.n_inputs || error("SetInput : Invalid number of arguments, got $(length(args)) required $(x.n_inputs)")

  for i = 1:x.n_inputs
    x.input[i][] = args[i]
  end
end

function GetOutput(x::Network)
  return [i[] for i in x.output]
end

function GetLayers(x::Network; simple::Bool = true)
  return [Unsigned[n.GIN for n in l if (!simple || n.enabled[])] for l in x.layers]
end

function GetNodePosition(x::Nodes)
  ismissing(x.super) && error("GetNodePosition : super field of this node is missing")
  (x in values(x.super.genes)) || error("GetNodePosition : this node is not present in the genes of it's network")

  layer_number = 0
  node_number = 0
  for (i,l) in enumerate(x.super.layers)
    is_break = false
    for (j,n) in enumerate(l)
      if x === n
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

function GetNetworkSummary(x::Network; simple::Bool = true)
  selected_GINS = [i.GIN for i in values(x.genes) if (!simple || i.enabled[])]
  sort!(selected_GINS)
  length_selected_GINs = length(selected_GINS)
  ret = DataFrame(GIN = fill(UInt(0x0), length_selected_GINs),
                  type = fill(Any, length_selected_GINs),
                  in_node = fill(UInt(0x0), length_selected_GINs),
                  out_node = fill(UInt(0x0), length_selected_GINs),
                  enabled = fill(true, length_selected_GINs),
                 )
  Threads.@threads for (i,j) = collect(enumerate(selected_GINS))
    if x.genes[j] isa Nodes
      ret[i, :GIN] = x.genes[j].GIN
      ret[i, :type] = typeof(x.genes[j])
      ret[i, :enabled] = x.genes[j].enabled[]
    elseif x.genes[j] isa Connections
      ret[i, :GIN] = x.genes[j].GIN
      ret[i, :type] = typeof(x.genes[j])
      ret[i, :in_node] = x.genes[j].in_node.GIN
      ret[i, :out_node] = x.genes[j].out_node.GIN
      ret[i, :enabled] = x.genes[j].enabled[]
    end
  end

  return ret
end


