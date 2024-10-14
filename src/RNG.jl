using Random

export rng

function rng()
  rand([ Random.seed!(time_ns()), MersenneTwister(time_ns()), Xoshiro(time_ns()) ])
end
