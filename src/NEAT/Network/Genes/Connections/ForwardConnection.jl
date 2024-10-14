#=
# Refer src/NEAT/Network/Genes/Connections/ForwardConnectionConfig.jl for ForwardConnectionConfig, CheckConfig
# Refer src/Reference.jl for Reference
# Refer src/AbstractTypes.jl for Nodes, Connections, HiddenNodes, Configs
# Refer src/NEAT/Network/Gene/Node/InputNode.jl for InputNode
# Refer src/NEAT/Network/Gene/Node/OutputNode.jl for OutputNode
# Refer src/ActivationFunctions.jl for Sigmoid
# Refer src/RNG.jl for rng
=#

export ForwardConnection, Init, Run, ResetIO, ChangeWeight, ShiftWeight, ToggleEnable, EnableGene, DisableGene

@kwdef mutable struct ForwardConnection <: Connections
  GIN::Unsigned = 0 # Global Innovation Number
  in_node::Union{InputNode, HiddenNodes} # points to the input node
  out_node::Union{OutputNode, HiddenNodes} # points to the output node
  input::Reference{Real} = Reference{Real}() # points to the input to this connection
  output::Reference{Real} = Reference{Real}() # points to the output of this connection
  weight::Ref{Real} = Ref{Real}(1)
  enabled::Ref{Bool} = Ref{Bool}(true)
  super::Union{Missing, Network} = missing # points to the network this connection is present in
end

function Base.show(io::IO, x::ForwardConnection)
  println(io, summary(x))
  print(io, " GIN : $(x.GIN)
 in_node : $(x.in_node.GIN|>Int128)
 out_node : $(x.out_node.GIN|>Int128)
 input : $(x.input[])
 output : $(x.output[])
 weight : $(x.weight[])
 enabled : $(x.enabled[])
 super : $(ismissing(x.super) ? missing : x.super.idx)
")
  return
end

function Init(x::ForwardConnection, config::ForwardConnectionConfig)
  CheckConfig(config)

  x.input = x.in_node.output
  push!(x.in_node.out_connections, x)

  push!(x.out_node.input, x.output)
  push!(x.out_node.in_connections, x)

  if ismissing(config.initial_weight[]) || isnan(config.initial_weight[])
    x.weight[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
  else
    x.weight[] = config.initial_weight[]
  end

  return
end

function Run(x::ForwardConnection)
  if x.enabled[]
    x.output[] = x.input[] * x.weight[]
  else
    x.output[] = missing
  end

  return x.output[]
end

function ResetIO(x::ForwardConnection)
  x.input[] = missing
  x.output[] = missing
  return
end

function ChangeWeight(x::ForwardConnection, config::ForwardConnectionConfig)
  CheckConfig(config)

  x.weight[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
  return x.weight[]
end

function ShiftWeight(x::ForwardConnection, config::ForwardConnectionConfig)
  CheckConfig(config)

  method = (config.shift_weight_method[] == "Random") ? rand(rng(), ["Uniform", "Gaussian"]) : config.shift_weight_method[]

  if method == "Uniform"
    x.weight[] += rand(rng()) * config.shift_weight[] * rand(rng(), [1, -1])
  elseif method == "Gaussian"
    x.weight[] += randn(rng()) * config.std_weight[]
  end
  x.weight[] = clamp(x.weight[], config.min_weight[], config.max_weight[])
  return x.weight[]
end

function ToggleEnable(x::ForwardConnectionConfig)
  x.enabled[] = !(x.enabled[])
  return x.enabled[]
end

function EnableGene(x::ForwardConnectionConfig)
  x.enabled[] = true
  return
end

function DisableGene(x::ForwardConnectionConfig)
  x.enabled[] = false
  return
end


