using Debugger, DataStructures
include("src/NEATJulia.jl")
using .NEATJulia

function fitness_function(dict::Dict{String, Any}, network::Network, neat::NEAT)
  ret = 0.0
  for (amp_r, freq_r, phi_r) in zip(dict["amp"], dict["freq"], dict["phi"])
    T = 0:dict["dt"]:dict["max_time"]
    V = amp_r .* sin.(2 .*pi.*freq_r.*T .+ phi_r)
    buffer_t = CircularBuffer{Real}(dict["buffer_size"])
    buffer_v = CircularBuffer{Real}(dict["buffer_size"])
    ResetIO(network)
    for (t,v) in zip(T, V)
      push!(buffer_t, t)
      push!(buffer_v, v)
      SetInput(network, [t,v])
      amp, freq, phi = Run(network)
      (ismissing(amp) || !isfinite(amp) || ismissing(freq) || !isfinite(freq) || ismissing(phi) || !isfinite(phi)) && (return -Inf)
      v_out = amp .* sin.(2 .*pi.*freq.*buffer_t .+ phi)
      diff = v_out .- buffer_v
      diff = sum( abs.(diff) )
      isfinite(diff) || (return -Inf)
      ret -= diff
    end
  end
  return ret
end

fitness_test_dict = Dict("amp" => [12, 10, 3],
                         "freq" => [50, 55, 60],
                         "phi" => [0, pi/4, pi/3],
                         "max_time" => 0.2,
                         "buffer_size" => 256,
                         "dt" => 0.001,
                         "fitness_function" => fitness_function,
                        )

neat_config = NEATConfig(
                         n_inputs = 2,
                         n_outputs = 3,
                         population_size = 100,
                         max_generation = 1000,
                         fitness_test_dict = fitness_test_dict,
                        )

neat = NEAT(neat_config = neat_config)
Init(neat)


