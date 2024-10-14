#=
# Refer src/Reference.jl for Reference
# Refer src/AbstractTypes.jl for RecurrentHiddenNodes, Connections, Configs
# Refer src/ActivationFunctions.jl for Sigmoid
# Refer src/RNG.jl for rng
=#

export RecurrentHiddenNodeConfig, CheckConfig

@kwdef mutable struct RecurrentHiddenNodeConfig <: Configs
  min_bias::Ref{Real} = Ref{Real}(-5)
  max_bias::Ref{Real} = Ref{Real}(5)
  shift_bias::Ref{Real} = Ref{Real}(0.01)
  std_bias::Ref{Real} = Ref{Real}(0.01)
  shift_bias_method::Ref{String} = Ref{String}("Gaussian")
  min_weight::Ref{Real} = Ref{Real}(-2)
  max_weight::Ref{Real} = Ref{Real}(2)
  shift_weight::Ref{Real} = Ref{Real}(0.01)
  std_weight::Ref{Real} = Ref{Real}(0.01)
  shift_weight_method::Ref{String} = Ref{String}("Gaussian")
  activation_functions::Ref{Vector{Function}} = Ref{Vector{Function}}([Sigmoid])
  initial_bias::Reference{Real} = Reference{Real}(0.0)
  initial_weight::Reference{Real} = Reference{Real}(1.0)
  initial_activation_function::Reference{Function} = Reference{Function}(Sigmoid)
end

function CheckConfig(x::RecurrentHiddenNodeConfig)
  (isfinite(x.min_bias[])) || (error("RecurrentHiddenNodeConfig : min_bias should be a finite Real number"))
  (isfinite(x.max_bias[])) || (error("RecurrentHiddenNodeConfig : max_bias should be a finite Real number"))
  (x.max_bias[] < x.min_bias[]) && (error("RecurrentHiddenNodeConfig : got max_bias < min_bias"))
  (isfinite(x.shift_bias[])) || (error("RecurrentHiddenNodeConfig : shift_bias should be a finite Real number"))
  (isfinite(x.std_bias[])) || (error("RecurrentHiddenNodeConfig : std_bias should be a finite Real number"))
  (x.shift_bias_method[] in ["Uniform", "Gaussian", "Random"]) || (error("RecurrentHiddenNodeConfig : shift_bias_method should be one of \"Uniform\", \"Gaussian\", \"Random\""))

  (isfinite(x.min_weight[])) || (error("RecurrentHiddenNodeConfig : min_weight should be a finite Real number"))
  (isfinite(x.max_weight[])) || (error("RecurrentHiddenNodeConfig : max_weight should be a finite Real number"))
  (x.max_weight[] < x.min_weight[]) && (error("RecurrentHiddenNodeConfig : got max_weight < min_weight"))
  (isfinite(x.shift_weight[])) || (error("RecurrentHiddenNodeConfig : shift_weight should be a finite Real number"))
  (isfinite(x.std_weight[])) || (error("RecurrentHiddenNodeConfig : std_weight should be a finite Real number"))
  (x.shift_weight_method[] in ["Uniform", "Gaussian", "Random"]) || (error("RecurrentHiddenNodeConfig : shift_weight_method should be one of \"Uniform\", \"Gaussian\", \"Random\""))

  (isempty(x.activation_functions[])) && (error("RecurrentHiddenNodeConfig : got empty activation_functions"))
  return
end

function Base.show(io::IO, x::RecurrentHiddenNodeConfig)
  println(io, summary(x))
  print(io, " min_bias : $(x.min_bias[])
 max_bias : $(x.max_bias[])
 shift_bias : $(x.shift_bias[])
 std_bias : $(x.std_bias[])
 shift_bias_method : $(x.shift_bias_method[])
 min_weight : $(x.min_weight[])
 max_weight : $(x.max_weight[])
 shift_weight : $(x.shift_weight[])
 std_weight : $(x.std_weight[])
 shift_weight_method : $(x.shift_weight_method[])
 activation_functions : $(ismissing(x.activation_functions[]) ? missing : join(x.activation_functions[], ", "))
 initial_bias : $(x.initial_bias[])
 initial_weight : $(x.initial_weight[])
 initial_activation_function : $(x.initial_activation_function[])
")
  return
end


