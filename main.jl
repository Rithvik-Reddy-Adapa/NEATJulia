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
    global neat = NEAT(1,1, max_generation=10_000, n_genomes_to_pass = 1, n_generations_to_pass = 5)
    Init(neat)
    mutation_probability = GetMutationProbability(neat)[1,:]|>DataFrame
    mutation_probability[:,:disable_connection] .= 0
    mutation_probability[:,:disable_node] .= 0
    SetMutationProbability!(neat, mutation_probability)

    while true
      rand_input = rand(-180:180)
      SetInput!(neat, rand_input)
      SetExpectedOutput!(neat, sind(rand_input))
      ret = Run(neat, generation = true)
      # display(GetOutput(neat))
      # Save(neat)
      if typeof(ret) == Bool
        break
      end
    end
  end
end

main()
;
