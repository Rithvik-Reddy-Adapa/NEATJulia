using Random

export rng

"""
```julia
function rng()
  rand([ Random.seed!(time_ns()), MersenneTwister(time_ns()), Xoshiro(time_ns()) ])
end
```
"""
function rng()
  rand([ Random.seed!(time_ns()), MersenneTwister(time_ns()), Xoshiro(time_ns()) ])
end
