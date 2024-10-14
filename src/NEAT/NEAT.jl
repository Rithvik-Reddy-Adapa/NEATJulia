
export NEAT

@kwdef mutable struct NEAT{Network} <: AllNEATTypes where Network <: AllNEATTypes
  generation::Unsigned = 0
  GIN::NamedTuple = (GIN = Unsigned[],
                     type = Type[],
                     start_node = Unsigned[],
                     stop_node = Unsigned[],
                    )
  population::Vector{Network} = Network[]
  species::Dict{Unsigned, Vector{Network}} = Dict{Unsigned, Vector{Network}}()
  specie_info::NamedTuple = (specie = Unsigned[],
                             alive = Bool[],
                             birth_generation = Unsigned[],
                             death_generation = Unsigned[],
                             minimum_fitness = Real[],
                             maximum_fitness = Real[],
                             mean_fitness = Real[],
                             last_topped_generation = Unsigned[],
                             last_improved_generation = Unsigned[],
                             last_highest_maximum_fitness = Real[],
                            )
  fitness::Vector{Real} = Real[]
  winners::Vector{Network} = Network[]
  n_networks_passed::Unsigned = 0
  n_generations_passed::Unsigned = 0

  neat_config::NEATConfig

  comments::Any = nothing
end

function Base.show(io::IO, x::NEAT)
  println(io, summary(x))
  print(io, " generation : $(x.generation)
 max_GIN : $(isempty(x.GIN.GIN) ? "0, Any, 0, 0" : join([x.GIN.GIN[end], x.GIN.type[end], x.GIN.start_node[end], x.GIN.stop_node[end]], ", "))
 species : $(join(x.species|>keys|>collect|>sort, ", "))
 winners : $(join(getfield.(x.winners, :idx), ", "))
 n_networks_passed : $(x.n_networks_passed)
 n_generations_passed : $(x.n_generations_passed)
 ")
  return
end

function Base.show(io::IO, x::@NamedTuple{GIN::Vector{Unsigned}, type::Vector{Type}, start_node::Vector{Unsigned}, stop_node::Vector{Unsigned}})
  println(io, "$(length(x.GIN))-row(s) $(summary(x))")
  println(io, " GIN\t\ttype\t\tstart_node\t\tstop_node")
  for i in 1:length(x.GIN)
    println(io, " $(x.GIN[i])\t\t$(x.type[i])\t\t$(x.start_node[i])\t\t$(x.stop_node[i])")
  end
  println()
  return
end

function Base.show(io::IO, x::@NamedTuple{specie::Vector{Unsigned}, alive::Vector{Bool}, birth_generation::Vector{Unsigned}, death_generation::Vector{Unsigned}, minimum_fitness::Vector{Real}, maximum_fitness::Vector{Real}, mean_fitness::Vector{Real}, last_topped_generation::Vector{Unsigned}, last_improved_generation::Vector{Unsigned}, last_highest_maximum_fitness::Vector{Real}})
  println(io, "$(length(x.specie))-row(s) $(summary(x))")
  println(io, " specie\talive\tbirth_generation\tdeath_generation\tminimum_fitness\tmaximum_fitness\tmean_fitness\tlast_topped_generation\tlast_improved_generation\tlast_highest_maximum_fitness")
  for i in 1:length(x.specie)
    println(io, " $(x.specie[i]), $(x.alive[i]), $(x.birth_generation[i]), $(x.death_generation[i]), $(x.minimum_fitness[i]), $(x.maximum_fitness[i]), $(x.mean_fitness[i]), $(x.last_topped_generation[i]), $(x.last_improved_generation[i]), $(x.last_highest_maximum_fitness[i])")
  end
  println()
  return
end


