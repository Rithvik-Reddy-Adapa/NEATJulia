#=
# Refer src/NEAT/Network/Genes/Nodes/HiddenNodes/HiddenNodeConfig.jl for HiddenNodeConfig, CheckConfig
# Refer src/Reference.jl for Reference
# Refer src/AbstractTypes.jl for HiddenNodes, Connections, Configs
# Refer src/ActivationFunctions.jl for Sigmoid
# Refer src/RNG.jl for rng
=#

export HiddenNode, Run, ResetIO, ChangeBias, ShiftBias, ChangeActivationFunction, ToggleEnable, EnableGene, DisableGene, Init

@kwdef mutable struct HiddenNode <: HiddenNodes
  GIN::Unsigned = 0 # Global Innovation Number
  in_connections::Vector{Connections} = Connections[] # list of all input connections to this node
  out_connections::Vector{Connections} = Connections[] # list of all output connections from this node
  input::Vector{Reference{Real}} = Reference{Real}[] # list of all inputs to this node
  output::Reference{Real} = Reference{Real}() # the output value of this node
  activation_function::Ref{Function} = Ref{Function}(Sigmoid)
  bias::Ref{Real} = Ref{Real}(0)
  enabled::Ref{Bool} = Ref{Bool}(true)
  super::Union{Missing, Network} = missing # points to the network this node is present in
end

function Base.show(io::IO, x::HiddenNode)
  println(io, summary(x))
  print(io, " GIN : $(x.GIN)
 in_connections : $(join(Int128[i.GIN for i in x.in_connections], ", "))
 out_connections : $(join(Int128[i.GIN for i in x.out_connections], ", "))
 input : $(join(Union{Missing, Real}[i[] for i in x.input], ", "))
 output : $(x.bias[])
 activation_function : $(x.activation_function[])
 bias : $(x.bias[])
 enabled : $(x.enabled[])
 super : $(ismissing(x.super) ? missing : x.super.idx)
")
  return
end

function Init(x::HiddenNode, config::HiddenNodeConfig)
  CheckConfig(config)

  if ismissing(config.initial_bias[]) || isnan(config.initial_bias[])
    x.bias[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  else
    x.bias[] = config.initial_bias[]
  end

  if ismissing(config.initial_activation_function[])
    x.activation_function[] = rand(rng(), config.activation_functions[])
  else
    x.activation_function[] = config.initial_activation_function[]
  end

  return
end

function Run(x::HiddenNode)
  if x.enabled[]
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
    ismissing(x.output[]) || ( x.output[] = x.activation_function[](x.output[] + x.bias[]) )
  else
    x.output[] = missing
  end
  return x.output[]
end

function ResetIO(x::HiddenNode)
  for i in x.input
    i[] = missing
  end
  x.output[] = missing
  return
end

function ChangeBias(x::HiddenNode, config::HiddenNodeConfig)
  CheckConfig(config)

  x.bias[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  return x.bias[]
end

function ShiftBias(x::HiddenNode, config::HiddenNodeConfig)
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

function ChangeActivationFunction(x::HiddenNode, config::HiddenNodeConfig)
  CheckConfig(config)

  x.activation_function[] = rand(rng(), config.activation_functions[])
  return x.activation_function[]
end

function ToggleEnable(x::HiddenNode)
  x.enabled[] = !(x.enabled[])
  return x.enabled[]
end

function EnableGene(x::HiddenNode)
  x.enabled[] = true
  return
end

function DisableGene(x::HiddenNode)
  x.enabled[] = false
  return
end


