#=
# Refer src/NEAT/Network/Genes/Nodes/InputNodeConfig.jl for InputNodeConfig, CheckConfig
# Refer src/Reference.jl for Reference
# Refer src/AbstractTypes.jl for Nodes, Connections, Configs
# Refer src/ActivationFunctions.jl for Sigmoid
# Refer src/RNG.jl for rng
=#

export InputNode, Run, ResetIO, ChangeBias, ShiftBias, ChangeActivationFunction, Init

@kwdef mutable struct InputNode <: Nodes
  GIN::Unsigned = 0 # Global Innovation Number
  serial_number::Unsigned = 0 # index of this input node in the input set
  in_connections::Vector{Connections} = Connections[] # list of all input connections to this input node
  out_connections::Vector{Connections} = Connections[] # list of all output connections from this input node
  input::Vector{Reference{Real}} = Reference{Real}[] # list of all inputs to this input node
  output::Reference{Real} = Reference{Real}() # the output value of this input node
  activation_function::Ref{Function} = Ref{Function}(Sigmoid)
  bias::Ref{Real} = Ref{Real}(0.0)
  const enabled::Ref{Bool} = Ref{Bool}(true)
  super::Union{Missing, Network} = missing # points to the network this node is present in
end

function Base.show(io::IO, x::InputNode)
  println(io, summary(x))
  print(io, " GIN : $(x.GIN)
 serial_number : $(x.serial_number)
 in_connections : $(join(Int128[i.GIN for i in x.in_connections], ", "))
 out_connections : $(join(Int128[i.GIN for i in x.out_connections], ", "))
 input : $(join(Union{Missing, Real}[i[] for i in x.input], ", "))
 output : $(x.output[])
 activation_function : $(x.activation_function[])
 bias : $(x.bias[])
 enabled : $(x.enabled[])
 super : $(ismissing(x.super) ? missing : x.super.idx)
")
  return
end

function Init(x::InputNode, config::InputNodeConfig)
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

function Run(x::InputNode)
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
  return x.output[]
end

function ResetIO(x::InputNode)
  for i in x.input
    i[] = missing
  end
  x.output[] = missing
  return
end

function ChangeBias(x::InputNode, config::InputNodeConfig)
  CheckConfig(config)

  x.bias[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  return x.bias[]
end

function ShiftBias(x::InputNode, config::InputNodeConfig)
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

function ChangeActivationFunction(x::InputNode, config::InputNodeConfig)
  CheckConfig(config)

  x.activation_function[] = rand(rng(), config.activation_functions[])
  return x.activation_function[]
end



