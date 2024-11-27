#=
# Refer src/NEAT/Network/Genes/Nodes/HiddenNodes/RecurrentHiddenNodes/RecurrentHiddenNodeConfig.jl for RecurrentHiddenNodeConfig, CheckConfig
# Refer src/Reference.jl for Reference
# Refer src/AbstractTypes.jl for RecurrentHiddenNodes, Connections, Configs
# Refer src/ActivationFunctions.jl for Sigmoid
# Refer src/RNG.jl for rng
=#

export RecurrentHiddenNode, Init, Run, ShiftBias, ChangeBias, ChangeWeight, ShiftWeight, ToggleEnable, EnableGene, DisableGene, ResetIO, ChangeActivationFunction

@kwdef mutable struct RecurrentHiddenNode <: RecurrentHiddenNodes
  GIN::Unsigned = 0 # Global Innovation Number
  in_connections::Vector{Connections} = Connections[] # list of all input connections to this node
  out_connections::Vector{Connections} = Connections[] # list of all output connections from this node
  input::Vector{Reference{Real}} = Reference{Real}[] # list of all inputs to this node
  output::Reference{Real} = Reference{Real}() # the output value of this node
  activation_function::Ref{Function} = Ref{Function}(Sigmoid)
  bias::Ref{Real} = Ref{Real}(0)
  weight::Ref{Real} = Ref{Real}(1)
  enabled::Ref{Bool} = Ref{Bool}(true)
  super::Union{Missing, Network} = missing # points to the network this node is present in
end

function Base.show(io::IO, x::RecurrentHiddenNode)
  println(io, summary(x))
  print(io, " GIN : $(x.GIN)
 in_connections : $(join(Int128[i.GIN for i in x.in_connections], ", "))
 out_connections : $(join(Int128[i.GIN for i in x.out_connections], ", "))
 input : $(join(Union{Missing, Real}[i[] for i in x.input], ", "))
 output : $(x.bias[])
 activation_function : $(x.activation_function[])
 bias : $(x.bias[])
 weight : $(x.weight[])
 enabled : $(x.enabled[])
 super : $(ismissing(x.super) ? missing : x.super.idx)
")
  return
end

function Init(x::RecurrentHiddenNode, config::RecurrentHiddenNodeConfig)
  CheckConfig(config)

  if ismissing(config.initial_bias[]) || isnan(config.initial_bias[])
    x.bias[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  else
    x.bias[] = config.initial_bias[]
  end

  if ismissing(config.initial_weight[]) || isnan(config.initial_weight[])
    x.weight[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
  else
    x.weight[] = config.initial_weight[]
  end

  if ismissing(config.initial_activation_function[])
    x.activation_function[] = rand(rng(), config.activation_functions[])
  else
    x.activation_function[] = config.initial_activation_function[]
  end

  return
end

function Run(x::RecurrentHiddenNode)
  if x.enabled[]
    x.output[] = x.output[] * x.weight[]
    for i in x.input
      if !ismissing(i[])
        if ismissing(x.output[])
          x.output[] = i[]
        else
          x.output[] += i[]
        end
      end
    end
    ismissing(x.output[]) || ( x.output[] = x.activation_function[](x.output[] + x.bias[]) )
  else
    x.output[] = missing
  end
  return x.output[]
end

function ResetIO(x::RecurrentHiddenNode)
  for i in x.input
    i[] = missing
  end
  x.output[] = missing
  return
end

function ChangeBias(x::RecurrentHiddenNode, config::RecurrentHiddenNodeConfig)
  CheckConfig(config)

  x.bias[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  return x.bias[]
end

function ShiftBias(x::RecurrentHiddenNode, config::RecurrentHiddenNodeConfig)
  CheckConfig(config)

  method = (config.shift_bias_method[] == "Random") ? rand(rng(), ["Uniform", "Gaussian"]) : config.shift_bias_method[]

  if method == "Uniform"
    x.bias[] += rand(rng()) * config.shift_bias[] * rand(rng(), [1, -1])
  elseif method == "Gaussian"
    x.bias[] += randn(rng()) * config.std_bias[]
  end
  x.bias[] = clamp(x.bias[], config.min_bias[], config.max_bias[])
  return x.bias[]
end

function ChangeWeight(x::RecurrentHiddenNode, config::RecurrentHiddenNodeConfig)
  CheckConfig(config)

  x.weight[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
  return x.weight[]
end

function ShiftWeight(x::RecurrentHiddenNode, config::RecurrentHiddenNodeConfig)
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

function ChangeActivationFunction(x::RecurrentHiddenNode, config::RecurrentHiddenNodeConfig)
  CheckConfig(config)

  x.activation_function[] = rand(rng(), config.activation_functions[])
  return x.activation_function[]
end

function ToggleEnable(x::RecurrentHiddenNode)
  x.enabled[] = !(x.enabled[])
  return x.enabled[]
end

function EnableGene(x::RecurrentHiddenNode)
  x.enabled[] = true
  return
end

function DisableGene(x::RecurrentHiddenNode)
  x.enabled[] = false
  return
end




