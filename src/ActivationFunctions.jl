export Identity, Relu, Sigmoid, Tanh, Sin

"""
Identity

Identity(x::Real) = x
"""
Identity(x::Real) = x

"""
Relu

Relu(x::Real) = max(x, 0.0)
"""
Relu(x::Real) = max(x, 0.0)

"""
Sigmoid

Sigmoid(x::Real) = 1/(1+exp(-x))
"""
Sigmoid(x::Real) = 1/(1+exp(-x))

"""
Tanh

Tanh(x::Real) = tanh(x)
"""
Tanh(x::Real) = tanh(x)

"""
Sin

function Sin(x::Real)
  if x == Inf
    return 1.0
  elseif x == -Inf
    return -1.0
  else
    return sin(x)
  end
end
"""
function Sin(x::Real)
  if x == Inf
    return 1.0
  elseif x == -Inf
    return -1.0
  elseif isnan(x)
    return 0
  else
    return sin(x)
  end
end

;
