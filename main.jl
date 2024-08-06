include("./NEATJulia.jl")
using .NEATJulia
using Debugger

function main()
  if false
    # gloabl neat = NEAT(3, 3)
    # Init(neat)
  end

  if true
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
end

main()
;
