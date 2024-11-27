# Refer https://colah.github.io/posts/2015-08-Understanding-LSTMs/
#=
# Refer src/NEAT/Network/Genes/Nodes/HiddenNodes/RecurrentHiddenNodes/LSTMNodeConfig.jl for LSTMNodeConfig, CheckConfig
# Refer src/Reference.jl for Reference
# Refer src/AbstractTypes.jl for RecurrentHiddenNodes, Connections, Configs
# Refer src/ActivationFunctions.jl for Sigmoid, Tanh
# Refer src/RNG.jl for rng
=#

export LSTMNode, Init, Run, ShiftBias, ChangeBias, ChangeWeight, ShiftWeight, ToggleEnable, EnableGene, DisableGene, ResetIO

@kwdef mutable struct LSTMNode <: RecurrentHiddenNodes
  GIN::Unsigned = 0 # Global Innovation Number
  in_connections::Vector{Connections} = Connections[] # list of all input connections to this node
  out_connections::Vector{Connections} = Connections[] # list of all output connections from this node
  input::Vector{Reference{Real}} = Reference{Real}[] # list of all inputs to this node
  output::Reference{Real} = Reference{Real}() # the output value of this node, this is also the hidden state of LSTM node
  cell_state::Real = Real(0)

  weight_f::Vector{Ref{Real}} = Ref{Real}[] # weights for forget gate
  weight_i::Vector{Ref{Real}} = Ref{Real}[] # weights for input gate
  weight_c::Vector{Ref{Real}} = Ref{Real}[]
  weight_o::Vector{Ref{Real}} = Ref{Real}[] # weights for output gate
  bias_f::Ref{Real} = Ref{Real}(0.0) # bias for forget gate
  bias_i::Ref{Real} = Ref{Real}(0.0) # bias for input gate
  bias_c::Ref{Real} = Ref{Real}(0.0)
  bias_o::Ref{Real} = Ref{Real}(0.0) # bias for output gate

  enabled::Ref{Bool} = Ref{Bool}(true)
  super::Union{Missing, Network} = missing # points to the network this node is present in
end

function Base.show(io::IO, x::LSTMNode)
  println(io, summary(x))
  print(io, " GIN : $(x.GIN)
 in_connections : $(join(Int128[i.GIN for i in x.in_connections], ", "))
 out_connections : $(join(Int128[i.GIN for i in x.out_connections], ", "))
 input : $(join(Union{Missing, Real}[i[] for i in x.input], ", "))
 output : $(x.output[])
 weight_f : $(join(getindex.(x.weight_f), ", "))
 weight_i : $(join(getindex.(x.weight_i), ", "))
 weight_c : $(join(getindex.(x.weight_c), ", "))
 weight_o : $(join(getindex.(x.weight_o), ", "))
 bias_f : $(x.bias_f[])
 bias_i : $(x.bias_i[])
 bias_c : $(x.bias_c[])
 bias_o : $(x.bias_o[])
 enabled : $(x.enabled[])
 super : $(ismissing(x.super) ? missing : x.super.idx)
")
end

function Init(x::LSTMNode, config::LSTMNodeConfig)
  CheckConfig(config)

  if ismissing(config.initial_bias[]) || isnan(config.initial_bias[])
    x.bias_f[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
    x.bias_i[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
    x.bias_c[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
    x.bias_o[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  else
    x.bias_f[] = config.initial_bias[]
    x.bias_i[] = config.initial_bias[]
    x.bias_c[] = config.initial_bias[]
    x.bias_o[] = config.initial_bias[]
  end

  # add weights for hidden state
  push!(x.weight_f, Ref{Real}())
  push!(x.weight_i, Ref{Real}())
  push!(x.weight_c, Ref{Real}())
  push!(x.weight_o, Ref{Real}())
  if ismissing(config.initial_weight[]) || isnan(config.initial_weight[])
    x.weight_f[1][] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    x.weight_i[1][] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    x.weight_c[1][] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    x.weight_o[1][] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
  else
    x.weight_f[1][] = config.initial_weight[]
    x.weight_i[1][] = config.initial_weight[]
    x.weight_c[1][] = config.initial_weight[]
    x.weight_o[1][] = config.initial_weight[]
  end

  return
end

function Run(x::LSTMNode)
  if x.enabled[]
    all(ismissing.(getindex.(x.input))) && (return missing) # return missing if all inputs are missing
    hi::Vector{Union{Missing, Real}} = [x.output[]; getindex.(x.input)] # concat hidden state (output) and inputs
    idx::BitVector = .!ismissing.(hi) # select indices that are not missing

    # forget gate
    x.cell_state *= Sigmoid(transpose(getindex.(x.weight_f)[idx]) * hi[idx] + x.bias_f[])
    # input gate
    x.cell_state += Sigmoid(transpose(getindex.(x.weight_i)[idx]) * hi[idx] + x.bias_i[])   *   Tanh(transpose(getindex.(x.weight_c)[idx]) * hi[idx] + x.bias_c[])
    # output gate
    x.output[] = Sigmoid(transpose(getindex.(x.weight_o)[idx]) * hi[idx] + x.bias_o[])   *   Tanh(x.cell_state)
  else
    x.output[] = missing
  end
  return x.output[]
end

function ResetIO(x::LSTMNode) # for LSTM node ResetIO resets both I/O and cell state, i.e. cell state is set to zero
  for i in x.input
    i[] = missing
  end
  x.output[] = missing
  x.cell_state = 0.0
  return
end

function ChangeBias(x::LSTMNode, config::LSTMNodeConfig)
  CheckConfig(config)

  x.bias_f[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  x.bias_i[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  x.bias_c[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  x.bias_o[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())

  return x.bias_f[], x.bias_i[], x.bias_c[], x.bias_o[]
end

function ShiftBias(x::LSTMNode, config::LSTMNodeConfig)
  CheckConfig(config)

  method = (config.shift_bias_method[] == "Random") ? rand(rng(), ["Uniform", "Gaussian"]) : config.shift_bias_method[]

  if method == "Uniform"
    x.bias_f[] += rand(rng()) * config.shift_bias[] * rand(rng(), [1, -1])
    x.bias_i[] += rand(rng()) * config.shift_bias[] * rand(rng(), [1, -1])
    x.bias_c[] += rand(rng()) * config.shift_bias[] * rand(rng(), [1, -1])
    x.bias_o[] += rand(rng()) * config.shift_bias[] * rand(rng(), [1, -1])
  elseif method == "Gaussian"
    x.bias_f[] += randn(rng()) * config.std_bias[]
    x.bias_i[] += randn(rng()) * config.std_bias[]
    x.bias_c[] += randn(rng()) * config.std_bias[]
    x.bias_o[] += randn(rng()) * config.std_bias[]
  end
  x.bias_f[] = clamp(x.bias_f[], config.min_bias[], config.max_bias[])
  x.bias_i[] = clamp(x.bias_i[], config.min_bias[], config.max_bias[])
  x.bias_c[] = clamp(x.bias_c[], config.min_bias[], config.max_bias[])
  x.bias_o[] = clamp(x.bias_o[], config.min_bias[], config.max_bias[])

  return x.bias_f[], x.bias_i[], x.bias_c[], x.bias_o[]
end

function ChangeWeight(x::LSTMNode, config::LSTMNodeConfig)
  CheckConfig(config)

  for (i,j,k,l) in zip(x.weight_f, x.weight_i, x.weight_c, x.weight_o)
    i[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    j[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    k[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    l[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
  end

  return getindex.(x.weight_f), getindex.(x.weight_i), getindex.(x.weight_c), getindex.(x.weight_o)
end

function ShiftWeight(x::LSTMNode, config::LSTMNodeConfig)
  CheckConfig(config)

  method = (config.shift_weight_method[] == "Random") ? rand(rng(), ["Uniform", "Gaussian"]) : config.shift_weight_method[]

  if method == "Uniform"
    for (i,j,k,l) in zip(x.weight_f, x.weight_i, x.weight_c, x.weight_o)
      i[] += rand(rng()) * config.shift_weight[] * rand(rng(), [1, -1])
      j[] += rand(rng()) * config.shift_weight[] * rand(rng(), [1, -1])
      k[] += rand(rng()) * config.shift_weight[] * rand(rng(), [1, -1])
      l[] += rand(rng()) * config.shift_weight[] * rand(rng(), [1, -1])

      i[] = clamp(i[], config.min_weight[], config.max_weight[])
      j[] = clamp(j[], config.min_weight[], config.max_weight[])
      k[] = clamp(k[], config.min_weight[], config.max_weight[])
      l[] = clamp(l[], config.min_weight[], config.max_weight[])
    end
  elseif method == "Gaussian"
    for (i,j,k,l) in zip(x.weight_f, x.weight_i, x.weight_c, x.weight_o)
      i[] += randn(rng()) * config.std_weight[]
      j[] += randn(rng()) * config.std_weight[]
      k[] += randn(rng()) * config.std_weight[]
      l[] += randn(rng()) * config.std_weight[]

      i[] = clamp(i[], config.min_weight[], config.max_weight[])
      j[] = clamp(j[], config.min_weight[], config.max_weight[])
      k[] = clamp(k[], config.min_weight[], config.max_weight[])
      l[] = clamp(l[], config.min_weight[], config.max_weight[])
    end
  end

  return getindex.(x.weight_f), getindex.(x.weight_i), getindex.(x.weight_c), getindex.(x.weight_o)
end

function ToggleEnable(x::LSTMNode)
  x.enabled[] = !(x.enabled[])
  return x.enabled[]
end

function EnableGene(x::LSTMNode)
  x.enabled[] = true
  return
end

function DisableGene(x::LSTMNode)
  x.enabled[] = false
  return
end


