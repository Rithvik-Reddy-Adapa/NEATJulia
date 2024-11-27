# Refer https://colah.github.io/posts/2015-08-Understanding-LSTMs/
#=
# Refer src/NEAT/Network/Genes/Nodes/HiddenNodes/RecurrentHiddenNodes/GRUNodeConfig.jl for GRUNodeConfig, CheckConfig
# Refer src/Reference.jl for Reference
# Refer src/AbstractTypes.jl for RecurrentHiddenNodes, Connections, Configs
# Refer src/ActivationFunctions.jl for Sigmoid, Tanh
# Refer src/RNG.jl for rng
=#

export GRUNode, Init, Run, ShiftBias, ChangeBias, ChangeWeight, ShiftWeight, ToggleEnable, EnableGene, DisableGene, ResetIO

@kwdef mutable struct GRUNode <: RecurrentHiddenNodes
  GIN::Unsigned = 0 # Global Innovation Number
  in_connections::Vector{Connections} = Connections[] # list of all input connections to this node
  out_connections::Vector{Connections} = Connections[] # list of all output connections from this node
  input::Vector{Reference{Real}} = Reference{Real}[] # list of all inputs to this node
  output::Reference{Real} = Reference{Real}() # the output value of this node, this is also the hidden state of GRU node

  weight_r::Vector{Ref{Real}} = Ref{Real}[] # weights for reset gate
  weight_u::Vector{Ref{Real}} = Ref{Real}[] # weights for update gate
  weight_c::Vector{Ref{Real}} = Ref{Real}[] # weights for candidate hidden state
  bias_r::Ref{Real} = Ref{Real}(0.0) # bias for reset gate
  bias_u::Ref{Real} = Ref{Real}(0.0) # bias for update gate
  bias_c::Ref{Real} = Ref{Real}(0.0) # bias for candidate hidden state

  enabled::Ref{Bool} = Ref{Bool}(true)
  super::Union{Missing, Network} = missing # points to the network this node is present in
end

function Base.show(io::IO, x::GRUNode)
  println(io, summary(x))
  print(io, " GIN : $(x.GIN)
 in_connections : $(join(Int128[i.GIN for i in x.in_connections], ", "))
 out_connections : $(join(Int128[i.GIN for i in x.out_connections], ", "))
 input : $(join(Union{Missing, Real}[i[] for i in x.input], ", "))
 output : $(x.output[])
 weight_r : $(join(getindex.(x.weight_r), ", "))
 weight_u : $(join(getindex.(x.weight_u), ", "))
 weight_c : $(join(getindex.(x.weight_c), ", "))
 bias_r : $(x.bias_r[])
 bias_u : $(x.bias_u[])
 bias_c : $(x.bias_c[])
 enabled : $(x.enabled[])
 super : $(ismissing(x.super) ? missing : x.super.idx)
")
end

function Init(x::GRUNode, config::GRUNodeConfig)
  CheckConfig(config)

  if ismissing(config.initial_bias[]) || isnan(config.initial_bias[])
    x.bias_r[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
    x.bias_u[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
    x.bias_c[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  else
    x.bias_r[] = config.initial_bias[]
    x.bias_u[] = config.initial_bias[]
    x.bias_c[] = config.initial_bias[]
  end

  # add weights for hidden state
  push!(x.weight_r, Ref{Real}())
  push!(x.weight_u, Ref{Real}())
  push!(x.weight_c, Ref{Real}())
  if ismissing(config.initial_weight[]) || isnan(config.initial_weight[])
    x.weight_r[1][] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    x.weight_u[1][] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    x.weight_c[1][] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
  else
    x.weight_r[1][] = config.initial_weight[]
    x.weight_u[1][] = config.initial_weight[]
    x.weight_c[1][] = config.initial_weight[]
  end

  return
end

function Run(x::GRUNode)
  if x.enabled[]
    all(ismissing.(getindex.(x.input))) && (return missing) # return missing if all inputs are missing
    hi::Vector{Union{Missing, Real}} = [x.output[]; getindex.(x.input)] # concat hidden state (output) and inputs
    idx::BitVector = .!ismissing.(hi) # select indices that are not missing

    r = Sigmoid(transpose(getindex.(x.weight_r)[idx]) * hi[idx] + x.bias_r[])
    u = Sigmoid(transpose(getindex.(x.weight_u)[idx]) * hi[idx] + x.bias_u[])
    output_tldr = Tanh(transpose(getindex.(x.weight_c)[idx]) * [hi[1]*r; hi[2:end]][idx] + x.bias_c[])
    if ismissing(x.output[])
      x.output[] = u*output_tldr
    else
      x.output[] = ((1 - u)*x.output[]) + (u*output_tldr)
    end
  else
    x.output[] = missing
  end
  return x.output[]
end

function ResetIO(x::GRUNode)
  for i in x.input
    i[] = missing
  end
  x.output[] = missing
  return
end

function ChangeBias(x::GRUNode, config::GRUNodeConfig)
  CheckConfig(config)

  x.bias_r[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  x.bias_u[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())
  x.bias_c[] = config.min_bias[] + (config.max_bias[] - config.min_bias[]) * rand(rng())

  return x.bias_r[], x.bias_u[], x.bias_c[]
end

function ShiftBias(x::GRUNode, config::GRUNodeConfig)
  CheckConfig(config)

  method = (config.shift_bias_method[] == "Random") ? rand(rng(), ["Uniform", "Gaussian"]) : config.shift_bias_method[]

  if method == "Uniform"
    x.bias_r[] += rand(rng()) * config.shift_bias[] * rand(rng(), [1, -1])
    x.bias_u[] += rand(rng()) * config.shift_bias[] * rand(rng(), [1, -1])
    x.bias_c[] += rand(rng()) * config.shift_bias[] * rand(rng(), [1, -1])
  elseif method == "Gaussian"
    x.bias_r[] += randn(rng()) * config.std_bias[]
    x.bias_u[] += randn(rng()) * config.std_bias[]
    x.bias_c[] += randn(rng()) * config.std_bias[]
  end
  x.bias_r[] = clamp(x.bias_f[], config.min_bias[], config.max_bias[])
  x.bias_u[] = clamp(x.bias_i[], config.min_bias[], config.max_bias[])
  x.bias_c[] = clamp(x.bias_c[], config.min_bias[], config.max_bias[])

  return x.bias_r[], x.bias_u[], x.bias_c[]
end

function ChangeWeight(x::GRUNode, config::GRUNodeConfig)
  CheckConfig(config)

  for (i,j,k) in zip(x.weight_r, x.weight_u, x.weight_c)
    i[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    j[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
    k[] = config.min_weight[] + (config.max_weight[] - config.min_weight[]) * rand(rng())
  end

  return getindex.(x.weight_r), getindex.(x.weight_u), getindex.(x.weight_c)
end

function ShiftWeight(x::GRUNode, config::GRUNodeConfig)
  CheckConfig(config)

  method = (config.shift_weight_method[] == "Random") ? rand(rng(), ["Uniform", "Gaussian"]) : config.shift_weight_method[]

  if method == "Uniform"
    for (i,j,k) in zip(x.weight_r, x.weight_u, x.weight_c)
      i[] += rand(rng()) * config.shift_weight[] * rand(rng(), [1, -1])
      j[] += rand(rng()) * config.shift_weight[] * rand(rng(), [1, -1])
      k[] += rand(rng()) * config.shift_weight[] * rand(rng(), [1, -1])

      i[] = clamp(i[], config.min_weight[], config.max_weight[])
      j[] = clamp(j[], config.min_weight[], config.max_weight[])
      k[] = clamp(k[], config.min_weight[], config.max_weight[])
    end
  elseif method == "Gaussian"
    for (i,j,k) in zip(x.weight_r, x.weight_u, x.weight_c)
      i[] += randn(rng()) * config.std_weight[]
      j[] += randn(rng()) * config.std_weight[]
      k[] += randn(rng()) * config.std_weight[]

      i[] = clamp(i[], config.min_weight[], config.max_weight[])
      j[] = clamp(j[], config.min_weight[], config.max_weight[])
      k[] = clamp(k[], config.min_weight[], config.max_weight[])
    end
  end

  return getindex.(x.weight_r), getindex.(x.weight_u), getindex.(x.weight_c)
end

function ToggleEnable(x::GRUNode)
  x.enabled[] = !(x.enabled[])
  return x.enabled[]
end

function EnableGene(x::GRUNode)
  x.enabled[] = true
  return
end

function DisableGene(x::GRUNode)
  x.enabled[] = false
  return
end


