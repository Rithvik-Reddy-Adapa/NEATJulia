#=
# Refer src/Reference.jl for Reference
# Refer src/AbstractTypes.jl for HiddenNodes, Connections, Configs
# Refer src/ActivationFunctions.jl for Sigmoid
# Refer src/RNG.jl for rng
=#

export HiddenNodeConfig, CheckConfig

@kwdef mutable struct HiddenNodeConfig <: Configs
  min_bias::Ref{Real} = Ref{Real}(-5)
  max_bias::Ref{Real} = Ref{Real}(5)
  shift_bias::Ref{Real} = Ref{Real}(0.01)
  std_bias::Ref{Real} = Ref{Real}(0.01)
  shift_bias_method::Ref{String} = Ref{String}("Gaussian")
  activation_functions::Ref{Vector{Function}} = Ref{Vector{Function}}([Sigmoid])
  initial_bias::Reference{Real} = Reference{Real}(0.0)
  initial_activation_function::Reference{Function} = Reference{Function}(Sigmoid)
end

function CheckConfig(x::HiddenNodeConfig)
  (isfinite(x.min_bias[])) || (error("HiddenNodeConfig : min_bias should be a finite Real number"))
  (isfinite(x.max_bias[])) || (error("HiddenNodeConfig : max_bias should be a finite Real number"))
  (x.max_bias[] < x.min_bias[]) && (error("HiddenNodeConfig : got max_bias < min_bias"))
  (isfinite(x.shift_bias[])) || (error("HiddenNodeConfig : shift_bias should be a finite Real number"))
  (isfinite(x.std_bias[])) || (error("HiddenNodeConfig : std_bias should be a finite Real number"))
  (x.shift_bias_method[] in ["Uniform", "Gaussian", "Random"]) || (error("HiddenNodeConfig : shift_bias_method should be one of \"Uniform\", \"Gaussian\", \"Random\""))
  (isempty(x.activation_functions[])) && (error("HiddenNodeConfig : got empty activation_functions"))
  return
end

function Base.show(io::IO, x::HiddenNodeConfig)
  println(io, summary(x))
  print(io, " min_bias : $(x.min_bias[])
 max_bias : $(x.max_bias[])
 shift_bias : $(x.shift_bias[])
 std_bias : $(x.std_bias[])
 shift_bias_method : $(x.shift_bias_method[])
 activation_functions : $(ismissing(x.activation_functions[]) ? missing : join(x.activation_functions[], ", "))
 initial_bias : $(x.initial_bias[])
 initial_activation_function : $(x.initial_activation_function[])
")
  return
end


