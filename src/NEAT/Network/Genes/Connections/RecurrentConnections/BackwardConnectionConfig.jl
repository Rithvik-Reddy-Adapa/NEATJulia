#=
# Refer src/Reference.jl for Reference
# Refer src/AbstractTypes.jl for Nodes, Connections, HiddenNodes, Configs
# Refer src/NEAT/Network/Gene/Node/InputNode.jl for InputNode
# Refer src/NEAT/Network/Gene/Node/OutputNode.jl for OutputNode
# Refer src/ActivationFunctions.jl for Sigmoid
# Refer src/RNG.jl for rng
=#

export BackwardConnectionConfig, CheckConfig

@kwdef mutable struct BackwardConnectionConfig <: Configs
  min_weight::Ref{Real} = Ref{Real}(-2)
  max_weight::Ref{Real} = Ref{Real}(2)
  shift_weight::Ref{Real} = Ref{Real}(0.01)
  std_weight::Ref{Real} = Ref{Real}(0.01)
  shift_weight_method::Ref{String} = Ref{String}("Gaussian")
  initial_weight::Reference{Real} = Reference{Real}(1.0)
end

function CheckConfig(x::BackwardConnectionConfig)
  (isfinite(x.min_weight[])) || (error("BackwardConnectionConfig : min_weight should be a finite Real number"))
  (isfinite(x.max_weight[])) || (error("BackwardConnectionConfig : max_weight should be a finite Real number"))
  (x.max_weight[] < x.min_weight[]) && (error("BackwardConnectionConfig : got max_weight < min_weight"))
  (isfinite(x.shift_weight[])) || (error("BackwardConnectionConfig : shift_weight should be a finite Real number"))
  (isfinite(x.std_weight[])) || (error("BackwardConnectionConfig : std_weight should be a finite Real number"))
  (x.shift_weight_method[] in ["Uniform", "Gaussian", "Random"]) || (error("BackwardConnectionConfig : shift_weight_method should be one of \"Uniform\", \"Gaussian\", \"Random\""))
  return
end

function Base.show(io::IO, x::BackwardConnectionConfig)
  println(io, summary(x))
  print(io, " min_weight : $(x.min_weight[])
 max_weight : $(x.max_weight[])
 shift_weight : $(x.shift_weight[])
 std_weight : $(x.std_weight[])
 shift_weight_method : $(x.shift_weight_method[])
 initial_weight : $(x.initial_weight[])
")
  return
end

