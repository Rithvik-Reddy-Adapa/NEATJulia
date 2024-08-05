include("./NEATJulia.jl")
using .NEATJulia
using Debugger

neat = NEAT(3, 3)
Init(neat)

# neat = NEAT(1,1, max_generation=10_000)
# Init(neat)
#
# while true
#   rand_input = rand(-180:180)
#   SetInput!(neat, rand_input)
#   SetExpectedOutput!(neat, sind(rand_input))
#   ret = Run(neat, generation = true)
#   if typeof(ret) == Bool
#     break
#   end
# end
;
