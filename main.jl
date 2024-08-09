include("./NEATJulia.jl")
using .NEATJulia
using Debugger, DataFrames

function main()
  if false
    # gloabl neat = NEAT(3, 3)
    # Init(neat)
  end

  if false
    global neat = NEAT(2,1, max_generation=10_000, n_genomes_to_pass = 1, n_generations_to_pass = 20)
    Init(neat)

    while true
      rand_input = rand([0,1], 2)
      SetInput!(neat, rand_input)
      SetExpectedOutput!(neat, xor(rand_input...))
      ret = Run(neat, generation = true)
      # Save(neat)
      if typeof(ret) == Bool
        break
      end
    end
  end

  if true
    global neat = NEAT(1,1, max_generation=1_000_000, n_genomes_to_pass = 1, n_generations_to_pass = 90, threshold_fitness = -0.01)
    Init(neat)
    # mutation_probability = GetMutationProbability(neat)[1,:]|>DataFrame
    # mutation_probability[:,:change_weight] .= 6
    # mutation_probability[:,:change_bias] .= 6
    mutation_probability = Dict(:no_mutation => 0)
    SetMutationProbability!(neat, mutation_probability)

    global t = []
    push!(t, time())
    global input_list = [0]
    idx = 0
    while true
      idx += 1
      if idx > length(input_list)
        idx = 1
      end
      rand_input = input_list[idx]
      # rand_input = rand(-180:180)
      SetInput!(neat, rand_input)
      SetExpectedOutput!(neat, sind(rand_input))
      ret = Run(neat, generation = true)
      # if typeof(ret) == Bool
      #   break
      # end
      if ret == true
        push!(t, time())
        push!(input_list, input_list[end]+1)
        Save(neat)
        neat.generations_passed = 0x0
      elseif ret == false
        break
      end
    end
  end
end

main()
;
