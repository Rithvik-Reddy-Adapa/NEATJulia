export Identity, Relu, Sigmoid, Tanh, Sin

"""
```julia
Identity(x::Real) = x
```
Returns the input `x`.

One among `NEATJulia`'s default activation functions.
"""
Identity(x::Real) = x

"""
```julia
Relu(x::Real) = max(x, 0.0)
```
Returns `0` for negative input and returns `x` for positive input.

One among `NEATJulia`'s default activation functions.
"""
Relu(x::Real) = max(x, 0.0)

"""
```julia
Sigmoid(x::Real) = 1/(1+exp(-x))
```
Returns Sigmoid of `x`.

One among `NEATJulia`'s default activation functions.
"""
Sigmoid(x::Real) = 1/(1+exp(-x))

"""
```julia
Tanh(x::Real) = tanh(x)
```
Returns Tanh of `x`.

One among `NEATJulia`'s default activation functions.
"""
Tanh(x::Real) = tanh(x)

"""
```julia
function Sin(x::Real)
  if x == Inf
    return 1.0
  elseif x == -Inf
    return -1.0
  else
    return sin(x)
  end
end
```
Returns Sin of `x`.

One among `NEATJulia`'s default activation functions.
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
