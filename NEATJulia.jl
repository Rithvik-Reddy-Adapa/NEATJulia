
module NEATJulia
  using DataFrames, StatsBase, JLD2
  include("Reference.jl")
  export Reference, getindex, setindex!, Node, InputNode, HiddenNode, OutputNode, Connection, Genome, NEAT, Init, Run, Relu, SetInput!, GetInput, GetOutput, GetFitness, SetExpectedOutput!, GetExpectedOutput, RunFitness, Sum_Abs_Diferrence, GetFitnessFunction, SetFitnessFunction!, GetMutationProbability, SetMutationProbability!, GetLayers, GetGenomeInfo, GetSpecieInfo, Show, SetCrossoverProbability!, GetCrossoverProbability, GetGIN, GetGenomeDistance, Load, Save

  abstract type Node end

  mutable struct InputNode{Connection, Genome} <: Node
    GIN::Unsigned # Global Innovation Number
    input_number::Unsigned
    input::Reference{Real}
    output::Reference{Real}
    out_connections::Vector{Connection}
    const enabled::Bool
    processed::Bool
    super::Union{Nothing, Genome}
  end

  mutable struct HiddenNode{Connection, Genome} <: Node
    GIN::Unsigned # Global Innovation Number
    in_connections::Vector{Connection}
    input::Vector{Reference{Real}}
    output::Reference{Real}
    out_connections::Vector{Connection}
    activation::Function
    bias::Real
    enabled::Bool
    processed::Bool
    super::Union{Nothing, Genome}
  end

  mutable struct OutputNode{Connection, Genome} <: Node
    GIN::Unsigned # Global Innovation Number
    in_connections::Vector{Connection}
    input::Vector{Reference{Real}}
    output::Reference{Real}
    output_number::Unsigned
    activation::Function
    bias::Real
    const enabled::Bool
    processed::Bool
    super::Union{Nothing, Genome}
  end

  mutable struct Connection{Genome}
    in_node::Node
    out_node::Node
    GIN::Unsigned # Global Innovation Number
    weight::Real
    input::Reference{Real}
    output::Reference{Real}
    enabled::Bool
    processed::Bool
    super::Union{Nothing, Genome}
  end

  mutable struct Genome{NEAT}
    n_inputs::Unsigned
    n_outputs::Unsigned
    super::Union{Nothing, NEAT}
    const _n_mutations::Unsigned

    ID::Unsigned
    specie::Unsigned
    genome::Dict{Unsigned, Union{Node, Connection}} # Global Innovation Number => Node / Connection
    layers::Vector{Vector{Node}}
    input::Vector{Reference{Real}}
    output::Vector{Reference{Real}}
    expected_output::Vector{Reference{Real}}
    fitness::Reference{Real}
    fitness_function::Reference{Union{Nothing, Function}}
    mutation_probability::Union{Nothing, DataFrame} # Probobility of different types of mutations 1: update weight, 2: update bias, 3: add/enable connection, 4: add/enable node, 5: disable connection, 6: disable node
  end

  mutable struct NEAT
    n_inputs::Unsigned
    n_outputs::Unsigned
    population_size::Unsigned
    max_generation::Unsigned
    RNN_enabled::Bool
    threshold_fitness::Real
    n_genomes_to_pass::Unsigned # number of geneomes to pass fitness test for NEAT to pass
    fitness_function::Reference{Union{Nothing, Function}}
    max_weight::Real
    min_weight::Real
    max_bias::Real
    min_bias::Real
    n_individuals_considered_best::Real # number of individuals considered best in a specie in a generation. Takes real values >= 0. Number less than 1 is considered as ratio over total specie population, number >= 1 is considered as number of individuals.
    n_individuals_to_retain::Real # number of individuals to retain unchanged for next generation of a specie. Takes real values >= 0. Number less than 1 is considered as ratio over total specie population, number >= 1 is considered as number of individuals.
    crossover_probability::Vector{Real} # [intraspecie good and good, intraspecie good and bad, intraspecie bad and bad, interspecie good and good, interspecie good and bad, interspecie bad and bad]
    max_specie_stagnation::Unsigned
    distance_parameters::Vector{Real}
    threshold_distance::Real
    const _n_mutations::Unsigned

    population::Vector{Genome}
    generation::Unsigned
    n_species::Unsigned
    GIN::DataFrame # list of all Global Innovation Number
    input::Vector{Reference{Real}}
    best_genome::Union{Nothing, Genome}
    output::Matrix{Reference{Real}}
    expected_output::Vector{Reference{Real}}
    fitness::Vector{Reference{Real}}
    mutation_probability::DataFrame
    species::Dict{Unsigned, Vector{Genome}}
    specie_info::DataFrame # n_rows = n_species, columns = {specie number, minimum fitness, maximum fitness, mean fitness, last topped generation}
  end

  function InputNode(; GIN = 0, input_number = 0, input = Reference{Real}(), output = Reference{Real}(), out_connections = Connection[], processed = false, super = nothing)
    enabled = true
    InputNode{Connection, Genome}(GIN, input_number, input, output, out_connections, enabled, processed, super)
  end
  function HiddenNode(; GIN = 0, in_connections = Connection[], input = Reference{Real}[], output = Reference{Real}(), out_connections = Connection[], activation = Relu, bias = 0.0, processed = false, super = nothing)
    enabled = true
    HiddenNode{Connection, Genome}(GIN, in_connections, input, output, out_connections, activation, bias, enabled, processed, super)
  end
  function OutputNode(; GIN = 0, in_connections = Connection[], input = Reference{Real}[], output = Reference{Real}(), output_number = 0, activation = Relu, bias = 0.0, processed = false, super = nothing)
    enabled = true
    OutputNode{Connection, Genome}(GIN, in_connections, input, output, output_number, activation, bias, enabled, processed, super)
  end
  function Connection(in_node, out_node; GIN = 0, weight = rand(), enabled = true, processed = false, super = nothing)

    input = in_node.output
    push!(out_node.input, Reference{Real}())
    output = out_node.input[end]

    connection = Connection{Genome}(in_node, out_node, GIN, weight, input, output, enabled, processed, super)
    push!(in_node.out_connections, connection)
    push!(out_node.in_connections, connection)

    return connection
  end
  function Genome(n_inputs, n_outputs; ID = 0, specie = 1, genome = Dict{Unsigned, Union{Node, Connection}}(), input = Reference{Real}[], output = Reference{Real}[], expected_output = Reference{Real}[], fitness = Reference{Real}(), fitness_function = Reference{Union{Nothing, Function}}(Sum_Abs_Diferrence), super = nothing, mutation_probability::Union{Nothing, DataFrame, Dict{Symbol, T}, Dict{Symbol, Reference{Real}}} = nothing) where T<:Real

    (n_inputs > 0) || throw(ArgumentError("Invalid n_inputs $(n_inputs), should be > 0"))
    (n_outputs > 0) || throw(ArgumentError("Invalid n_outputs $(n_outputs), should be > 0"))

    (isempty(input)) && (input = [Reference{Real}() for i in 1:n_inputs])
    (isempty(output)) && (output = [Reference{Real}() for i in 1:n_outputs])

    layers = Vector{Node}[]

    types_of_mutation_probability = [:no_mutation, :change_weight, :change_bias, :add_connection, :add_node, :disable_connection, :disable_node, :enable_connection, :enable_node]
    _n_mutations = length(types_of_mutation_probability)|>Unsigned
    if isnothing(mutation_probability) # do nothing
      mutation_probability = DataFrame(types_of_mutation_probability[1] => Reference{Real}[Reference{Real}(3)],
                                       types_of_mutation_probability[2] => Reference{Real}[Reference{Real}(1)],
                                       types_of_mutation_probability[3] => Reference{Real}[Reference{Real}(1)],
                                       types_of_mutation_probability[4] => Reference{Real}[Reference{Real}(0.25)],
                                       types_of_mutation_probability[5] => Reference{Real}[Reference{Real}(0.25)],
                                       types_of_mutation_probability[6] => Reference{Real}[Reference{Real}(0.1)],
                                       types_of_mutation_probability[7] => Reference{Real}[Reference{Real}(0.1)],
                                       types_of_mutation_probability[8] => Reference{Real}[Reference{Real}(0.1)],
                                       types_of_mutation_probability[9] => Reference{Real}[Reference{Real}(0.1)],
                                      )
    elseif typeof(mutation_probability) <: DataFrame
      dataframe_columns = Symbol.(names(mutation_probability))
      all([i in types_of_mutation_probability for i in dataframe_columns]) || throw(ArgumentError("Got invalid column names in the DataFrame mutation_probability, valid column names are $(types_of_mutation_probability)"))
      all([i in dataframe_columns for i in types_of_mutation_probability]) || throw(ArgumentError("All types of mutation probability probabilities i.e. $(types_of_mutation_probability) are not mentioned in the input mutation_probability DataFrame"))
      size(mutation_probability)[1] == 1 || throw(ArgumentError("mutation_probability DataFrames should be of only 1 row got $(sizeof(mutation_probability)[1]) rows"))
      if all(typeof.(collect(mutation_probability[1,:])) .<: Reference{Real}) # do nothing
      elseif all(typeof.(collect(mutation_probability[1,:])) .<: Real)
        table = DataFrame()
        for i in types_of_mutation_probability
          table[:,i] = [Reference{Real}(mutation_probability[1,i])]
        end
        mutation_probability = table
      else
        throw(ArgumentError("Invalid values in mutation_probability DataFrame, should be either Reference{Real} or <:Real"))
      end
    elseif typeof(mutation_probability) <: Dict
      dict_keys = keys(mutation_probability)
      all([i in types_of_mutation_probability for i in dict_keys]) || throw(ArgumentError("Got invalid keys in the Dict mutation_probability, valid keys are $(types_of_mutation_probability)"))
      all([i in dict_keys for i in types_of_mutation_probability]) || throw(ArgumentError("All types of mutation probability probabilities i.e. $(types_of_mutation_probability) are not mentioned in the input mutation_probability Dict"))
      if typeof(mutation_probability[:no_mutation]) <: Reference
        table = DataFrame()
        for i in types_of_mutation_probability
          table[:,i] = [mutation_probability[i]]
        end
        mutation_probability = table
      else # Real
        table = DataFrame()
        for i in types_of_mutation_probability
          table[:,i] = [Reference{Real}(mutation_probability[i])]
        end
        mutation_probability = table
      end
    else
      throw(ArgumentError("Got invalid datatype i.e. $(typeof(mutation_probability)) for mutation_probability, mutation_probability should be a DataFrame of 1 row and $(types_of_mutation_probability) columns or a Dict of keys as column names."))
    end

    Genome{NEAT}(n_inputs, n_outputs, super, _n_mutations, ID, specie, genome, layers, input, output, expected_output, fitness, fitness_function, mutation_probability)
  end
  function NEAT(n_inputs, n_outputs; population_size = 20, max_generation = 50, RNN_enabled = false, threshold_fitness = -0.001, n_genomes_to_pass = 1, fitness_function = Reference{Union{Nothing, Function}}(Sum_Abs_Diferrence), max_weight = 10, min_weight = -10, max_bias = 5, min_bias = -5, n_individuals_considered_best = 0.25, n_individuals_to_retain = 1, crossover_probability = [0.5, 1, 0.1, 0, 0, 0], max_specie_stagnation = 0x20, distance_parameters = [1, 1, 1], threshold_distance = 5,

      population = Genome[], generation = 0, n_species = 1, best_genome = nothing, expected_output = Reference{Real}[], mutation_probability::Union{Nothing, DataFrame, Dict{Symbol, T}, Dict{Symbol, Vector{T}}, Dict{Symbol, Vector{Reference{Real}}}} = nothing, specie_info = nothing) where T<:Real

    (n_inputs > 0) || throw(ArgumentError("Invalid n_inputs $(n_inputs), should be > 0"))
    (n_outputs > 0) || throw(ArgumentError("Invalid n_outputs $(n_outputs), should be > 0"))
    (population_size > 0) || throw(ArgumentError("Invalid population_size $(population_size), should be > 0"))
    (isempty(population)) && (population = Vector{Genome}(undef, population_size))

    types_of_mutation_probability = [:no_mutation, :change_weight, :change_bias, :add_connection, :add_node, :disable_connection, :disable_node, :enable_connection, :enable_node]
    _n_mutations = length(types_of_mutation_probability)|>Unsigned
    if isnothing(mutation_probability)
      mutation_probability = DataFrame(types_of_mutation_probability[1] => [Reference{Real}(10.0) for i in 1:population_size],
                                       types_of_mutation_probability[2] => [Reference{Real}(2.0) for i in 1:population_size],
                                       types_of_mutation_probability[3] => [Reference{Real}(2.0) for i in 1:population_size],
                                       types_of_mutation_probability[4] => [Reference{Real}(-floatmax(Float32)) for i in 1:population_size],
                                       types_of_mutation_probability[5] => [Reference{Real}(-floatmax(Float32)) for i in 1:population_size],
                                       types_of_mutation_probability[6] => [Reference{Real}(0.5) for i in 1:population_size],
                                       types_of_mutation_probability[7] => [Reference{Real}(0.5) for i in 1:population_size],
                                       types_of_mutation_probability[8] => [Reference{Real}(0.5) for i in 1:population_size],
                                       types_of_mutation_probability[9] => [Reference{Real}(0.5) for i in 1:population_size],
                                      )
    elseif typeof(mutation_probability) <: DataFrame
      dataframe_columns = Symbol.(names(mutation_probability))
      all([i in types_of_mutation_probability for i in dataframe_columns]) || throw(ArgumentError("Got invalid column names in the DataFrame mutation_probability, valid column names are $(types_of_mutation_probability)"))
      all([i in dataframe_columns for i in types_of_mutation_probability]) || throw(ArgumentError("All types of mutation probability probabilities i.e. $(types_of_mutation_probability) are not mentioned in the input mutation_probability DataFrame"))
      if size(mutation_probability)[1] == 1
        if all(typeof.(collect(mutation_probability[1,:])) .<: Real)
          table = DataFrame()
          for i in types_of_mutation_probability
            table[:,i] = [Reference{Real}(mutation_probability[1,i]) for j in 1:population_size]
          end
          mutation_probability = table
        else
          throw(ArgumentError("Type of values of mutation_probability DataFrame of 1 row should be <:Real, instead got $(typeof(mutation_probability[1,1]))"))
        end
      elseif size(mutation_probability)[1] == population_size
        if all(typeof.(collect(mutation_probability[1,:])) .<: Real)
          table = DataFrame()
          for i in types_of_mutation_probability
            column = Reference{Real}[]
            for j in mutation_probability[:,i]
              push!(column, Reference{Real}(j))
            end
            table[:,i] = column
          end
          mutation_probability = table
        elseif all(typeof.(collect(mutation_probability[1,:])) .<: Reference{Real}) # do nothing
        else
          throw(ArgumentError("Type of values of mutation_probability DataFrame of population_size($(population_size)) rows should be either <:Real or Reference{Real}, instead got $(typeof(mutation_probability[1,1]))"))
        end
      else
        throw(ArgumentError("Number of rows in mutation_probability DataFrame should be either 1 or population_size $(population_size), instead got $(size(mutation_probability)[1]) rows"))
      end
    elseif typeof(mutation_probability) <: Dict
      dict_keys = keys(mutation_probability)
      all([i in types_of_mutation_probability for i in dict_keys]) || throw(ArgumentError("Got invalid keys in the Dict mutation_probability, valid keys are $(types_of_mutation_probability)"))
      all([i in dict_keys for i in types_of_mutation_probability]) || throw(ArgumentError("All types of mutation probability probabilities i.e. $(types_of_mutation_probability) are not mentioned in the input mutation_probability Dict"))
      if typeof(mutation_probability[:no_mutation]) <: Real
        table = DataFrame()
        for i in types_of_mutation_probability
          table[:,i] = [Reference{Real}(mutation_probability[i]) for j in 1:population_size]
        end
        mutation_probability = table
      elseif typeof(mutation_probability[:no_mutation]) == Vector{Reference{Real}}
        table = DataFrame()
        for i in types_of_mutation_probability
          table[:,i] = mutation_probability[i]
        end
        mutation_probability = table
      else # Vector{<:Real}
        table = DataFrame()
        for i in types_of_mutation_probability
          column = Reference{Real}[]
          for j in mutation_probability[i]
            push!(column, Reference{Real}(j))
          end
          table[:,i] = column
        end
        mutation_probability = table
      end
    else
      throw(ArgumentError("Got invalid datatype i.e. $(typeof(mutation_probability)) for mutation_probability, mutation_probability should be a DataFrame of 1 row or population_size($(population_size)) rows and $(types_of_mutation_probability) columns or a Dict of keys as column names."))
    end
    isnothing(specie_info) && (specie_info = DataFrame(specie = Unsigned[],
                                                       alive = Bool[],
                                                       birth_generation = Unsigned[],
                                                       death_generation = Unsigned[],
                                                       minimum_fitness = Real[],
                                                       maximum_fitness = Real[],
                                                       mean_fitness = Real[],
                                                       last_topped_generation = Unsigned[],
                                                       last_improved_generation = Unsigned[],
                                                       last_highest_fitness = Real[],))

    input = [Reference{Real}() for i = 1:n_inputs]
    output = Matrix{Reference{Real}}(undef, population_size, n_outputs)
    for i = 1:length(output)
      output[i] = Reference{Real}()
    end
    fitness = [Reference{Real}(-Inf) for i = 1:population_size]
    species = Dict{Unsigned, Vector{Genome}}()
    GIN = DataFrame(GIN = Unsigned[],
                      type = Type[],
                      in_node = Vector{Union{Nothing, Unsigned}}(),
                      out_node = Vector{Union{Nothing, Unsigned}}(),
                     )

    NEAT(n_inputs, n_outputs, population_size, max_generation, RNN_enabled, threshold_fitness, n_genomes_to_pass, fitness_function, max_weight, min_weight, max_bias, min_bias, n_individuals_considered_best, n_individuals_to_retain, crossover_probability, max_specie_stagnation, distance_parameters, threshold_distance, _n_mutations,

         population, generation, n_species, GIN, input, best_genome, output, expected_output, fitness, mutation_probability, species, specie_info)
  end

  function Init(x::Genome)
    temp_vec = Node[]
    for i = 1:x.n_inputs
      x.genome[i] = InputNode(GIN = i, input_number = i, super = x, input = x.input[i])
      push!(temp_vec, x.genome[i])
    end
    push!(x.layers, temp_vec)

    temp_vec = Node[]
    for i = 1:x.n_outputs
      x.genome[x.n_inputs + i] = OutputNode(GIN = x.n_inputs + i, output_number = i, super = x, output = x.output[i])
      push!(temp_vec, x.genome[x.n_inputs + i])
    end
    push!(x.layers, temp_vec)
    return
  end
  function Init(x::NEAT)
    for i = 1:x.population_size
      x.population[i] = Genome(x.n_inputs, x.n_outputs, ID = i, super = x, fitness_function = x.fitness_function, input = x.input, output = x.output[i,:], expected_output = x.expected_output, fitness = x.fitness[i], mutation_probability = DataFrame(x.mutation_probability[i,:]))
      Init(x.population[i])
    end
    for i = 1:x.n_inputs
      push!(x.GIN, [Unsigned(i), InputNode, nothing, nothing])
    end
    for i = 1:x.n_outputs
      push!(x.GIN, [Unsigned(x.n_inputs+i), OutputNode, nothing, nothing])
    end
    x.species[0x1] = [i for i in x.population]
    push!(x.specie_info, [0x1, true, 0x1, 0x0, -Inf, -Inf, -Inf, 0x1, 0x0, -Inf])
    return
  end

  function Crossover(x::Genome, y::Genome)
    parent1 = x
    parent2 = y
    if x.fitness[] < y.fitness[]
      parent1 = y
      parent2 = x
    end
    super = parent1.super
    parent1.super = nothing
    ret = deepcopy(parent1)
    ret.super = super
    parent1.super = super
    for i in keys(ret.genome)
      if haskey(parent2.genome, i) && (typeof(parent2.genome[i]) <: Connection) && rand([true, false])
        ret.genome[i].weight = parent2.genome[i].weight
      elseif haskey(parent2.genome, i) && (typeof(parent2.genome[i]) <: HiddenNode) && rand([true, false])
        ret.genome[i].bias = parent2.genome[i].bias
      elseif haskey(parent2.genome, i) && (typeof(parent2.genome[i]) <: OutputNode) && rand([true, false])
        ret.genome[i].bias = parent2.genome[i].bias
      end
    end
    ret.ID = 0
    ret.fitness[] = -Inf
    ret.input = parent1.input
    for i = 1:length(ret.input)
      ret.genome[i].input = ret.input[i]
    end
    ret.specie = 0x0
    for i = 1:size(ret.mutation_probability)[2]
      ret.mutation_probability[1,i][] = (parent1.mutation_probability[1,i][] + parent2.mutation_probability[1,i][])/2
    end
    for i in ret.output
      i[] = 0.0
    end

    return ret
  end

  function _custom_weights(x::Vector{T}) where T<:Real
    x[x .== Inf] .= floatmax(Float32)
    x[x .== -Inf] .= -floatmax(Float32)
    return Weights(Float64.(x))
  end

  function Mutation(x::Genome)
    mutation = sample(collect(1:x._n_mutations), _custom_weights(abs.(collect(GetMutationProbability(x)[1,:]))))
    if x.mutation_probability[1, :add_connection][] < 0 && mutation != 4
      x.mutation_probability[1, :add_connection][] *= 1.5
    end
    if x.mutation_probability[1, :add_node][] < 0 && mutation != 5
      x.mutation_probability[1, :add_node][] *= 1.1
    end

    if mutation == 1 # no mutation
      return 1
    elseif mutation == 2 # change weight
      genome_info = GetGenomeInfo(x, simple = true)
      connections = genome_info[(genome_info[:,:type].<:Connection),:GIN]
      if isempty(connections)
        return 2
      end
      connections = [x.genome[i] for i in connections]
      random_connection = rand(connections)
      random_connection.weight = x.super.min_weight + (x.super.max_weight - x.super.min_weight)*rand()
      return 2, random_connection.GIN
    elseif mutation == 3 # change bias
      genome_info = GetGenomeInfo(x, simple = true)
      nodes = genome_info[(genome_info[:,:type].<:HiddenNode).||(genome_info[:,:type].<:OutputNode),:GIN]
      nodes = [x.genome[i] for i in nodes]
      random_node = rand(nodes)
      random_node.bias = x.super.min_bias + (x.super.max_bias - x.super.min_bias)*rand()
      return 3, random_node.GIN
    elseif mutation == 4 # add connection
      for itr = 1:3
        start_layer = rand(1:length(x.layers)-1)
        start_node = rand(x.layers[start_layer])
        end_layer = rand(start_layer+1:length(x.layers))
        end_node = rand(x.layers[end_layer])
        next_nodes_of_start_node = [i.out_node for i in start_node.out_connections]
        if !(end_node in next_nodes_of_start_node)
          idx = findfirst(x.super.GIN[:,2].<:Connection .&& x.super.GIN[:,3].==start_node.GIN .&& x.super.GIN[:,4].==end_node.GIN)
          if x.mutation_probability[1,:add_connection][] < 0
            genome_info = GetGenomeInfo(x, simple = false)
            n_connections = sum( genome_info[:,:type] .<: Connection )
            x.mutation_probability[1,:add_connection][] = -1/(n_connections+1)
          end
          if isnothing(idx)
            GIN = x.super.GIN[end,1]+0x1
            x.genome[GIN] = Connection(start_node, end_node, GIN = GIN, super = x)
            push!(x.super.GIN, [GIN, Connection, start_node.GIN, end_node.GIN])
            return 4, GIN, "new"
          else
            GIN = x.super.GIN[idx,1]
            x.genome[GIN] = Connection(start_node, end_node, GIN = GIN, super = x)
            return 4, GIN, "old"
          end
        end
      end
      return 4
    elseif mutation == 5 # add node
      genome_info = GetGenomeInfo(x, simple = true)
      connections = genome_info[(genome_info[:,:type].<:Connection),:GIN]
      if isempty(connections)
        return 5
      end
      connections = [x.genome[i] for i in connections]
      # get a random connection
      random_connection = rand(connections)
      while !(random_connection.enabled)
        random_connection = rand(connections)
      end

      # find the layer number of the random connection's in node
      start_layer = 1
      for i = 1:length(x.layers)-1
        break_loop = false
        for j in x.layers[i]
          if j == random_connection.in_node
            start_layer = i
            break_loop = true
            break
          end
        end
        if break_loop break end
      end

      # find the layer number of the random connection's out node
      end_layer = 1
      for i = start_layer+1:length(x.layers)
        break_loop = false
        for j in x.layers[i]
          if j == random_connection.out_node
            end_layer = i
            break_loop = true
            break
          end
        end
        if break_loop break end
      end

      GIN = x.super.GIN[end,1]+0x1
      new_node = HiddenNode(GIN = GIN, super = x)
      x.genome[GIN] = new_node
      push!(x.super.GIN, [GIN, HiddenNode, nothing, nothing])

      GIN = GIN+0x1
      new_node_in_connection = Connection(random_connection.in_node, new_node, GIN = GIN, super = x, weight = 1)
      x.genome[GIN] = new_node_in_connection
      push!(x.super.GIN, [GIN, Connection, new_node_in_connection.in_node.GIN, new_node_in_connection.out_node.GIN])

      GIN = GIN+0x1
      new_node_out_connection = Connection(new_node, random_connection.out_node, GIN = GIN, super = x, weight = random_connection.weight)
      x.genome[GIN] = new_node_out_connection
      push!(x.super.GIN, [GIN, Connection, new_node_out_connection.in_node.GIN, new_node_out_connection.out_node.GIN])

      random_connection.enabled = false

      # get a random layer for new node
      new_node_layer = rand(start_layer+0.5:0.5:end_layer-0.5)
      if new_node_layer == floor(new_node_layer) # add into existing layer
        push!(x.layers[Unsigned(new_node_layer)], new_node)
      else
        insert!(x.layers, Unsigned(ceil(new_node_layer)), [new_node])
      end
      if x.mutation_probability[1,:add_node][] < 0
        genome_info = GetGenomeInfo(x, simple = false)
        n_nodes = sum( genome_info[:,:type] .<: HiddenNode )
        x.mutation_probability[1,:add_node][] = -1/(n_nodes)
      end

      return 5, new_node.GIN, new_node_in_connection.GIN, new_node_out_connection.GIN
    elseif mutation == 6 # disable connection
      genome_info = GetGenomeInfo(x, simple = true)
      connections = genome_info[(genome_info[:,:type].<:Connection),:GIN]
      if isempty(connections)
        return 6
      end
      connections = [x.genome[i] for i in connections]
      random_connection = rand(connections)
      random_connection.enabled = false
      return 6, random_connection.GIN
    elseif mutation == 7 # disable node
      genome_info = GetGenomeInfo(x, simple = true)
      hidden_nodes = genome_info[(genome_info[:,:type].<:HiddenNode),:GIN]
      if isempty(hidden_nodes)
        return 7
      end
      hidden_nodes = [x.genome[i] for i in hidden_nodes]
      random_hidden_node = rand(hidden_nodes)
      random_hidden_node.enabled = false
      return 7, random_hidden_node.GIN
    elseif mutation == 8 # enable connection
      genome_info = GetGenomeInfo(x)
      connections = genome_info[(genome_info[:,:type].<:Connection).&&(genome_info[:,:enabled].==false),:GIN]
      if isempty(connections)
        return 8
      end
      connections = [x.genome[i] for i in connections]
      random_connection = rand(connections)
      random_connection.enabled = true
      return 8, random_connection.GIN
    elseif mutation == 9 # enable node
      genome_info = GetGenomeInfo(x)
      hidden_nodes = genome_info[(genome_info[:,:type].<:HiddenNode).&&(genome_info[:,:enabled].==false),:GIN]
      if isempty(hidden_nodes)
        return 9
      end
      hidden_nodes = [x.genome[i] for i in hidden_nodes]
      random_hidden_node = rand(hidden_nodes)
      random_hidden_node.enabled = true
      return 9, random_hidden_node.GIN
    end
  end

  function Speciation(x::NEAT)
    new_species = Dict{Unsigned, Vector{Genome}}()

    children = Genome[]
    for i in x.population
      if i.specie > 0
        if haskey(new_species, i.specie)
          push!(new_species[i.specie], i)
        else
          new_species[i.specie] = Genome[i]
        end
      else
        push!(children, i)
      end
    end

    for i in children
      selected_specie::Unsigned = 0
      best_distance::Real = Inf
      for j in keys(x.species)
        distance = GetGenomeDistance(i, x.species[j][1], x.distance_parameters)
        if distance <= x.threshold_distance && distance < best_distance
          best_distance = distance
          selected_specie = j
        end
      end
      if selected_specie == 0
        for j in keys(new_species)
          distance::Real = GetGenomeDistance(i, new_species[j][1], x.distance_parameters)
          if distance <= x.threshold_distance && distance < best_distance
            best_distance = distance
            selected_specie = j
          end
        end
      end

      if selected_specie != 0
        if haskey(new_species, selected_specie)
          push!(new_species[selected_specie], i)
        else
          new_species[selected_specie] = Genome[i]
          x.specie_info[selected_specie, :last_improved_generation] = x.generation
        end
        x.specie_info[selected_specie, :alive] = true
      else
        new_specie::Unsigned = x.specie_info[end,:specie] + 0x1
        push!(x.specie_info, [new_specie, true, x.generation+0x1, 0x0, -Inf, -Inf, -Inf, 0x0, x.generation, -Inf])
        new_species[new_specie] = Genome[i]
      end
    end
    
    x.species = new_species

    return nothing
  end

  function Run(x::InputNode)
    x.output[] = x.input[]
    x.processed = true
    for i in x.out_connections
      Run(i)
    end
    return x.output[]
  end
  function Run(x::HiddenNode)
    if !(x.enabled)
      x.processed = true
      return
    end
    output = 0.0
    for i in x.in_connections
      if i.enabled
        if !(i.processed)
          return
        end
        output += i.output[]
      end
    end
    x.output[] = x.activation(output + x.bias)
    x.processed = true
    for i in x.out_connections
      Run(i)
    end
    return x.output
  end
  function Run(x::OutputNode)
    output = 0.0
    for i in x.in_connections
      if i.enabled
        if !(i.processed)
          return
        end
        output += i.output[]
      end
    end
    x.output[] = x.activation(output + x.bias)
    x.processed = true
    return x.output
  end
  function Run(x::Connection)
    if !(x.enabled) || !(x.in_node.enabled) || !(x.out_node.enabled)
      x.processed = true
      return
    end
    if !(x.in_node.processed)
      return
    end
    x.output[] = x.weight * x.input[]
    x.processed = true
    Run(x.out_node)
    return x.output
  end
  function Run(x::Genome, evaluate::Bool = false)
    for i in x.layers
      for j in i
        if !(j.processed)
          Run(j)
        end
      end
    end
    evaluate && (RunFitness(x))
    for i in values(x.genome)
      i.processed = false
    end
  end
  function Run(x::NEAT; evaluate::Bool = false, crossover::Bool = false, mutate::Bool = false, generation::Bool = false)
    passed_genomes = getindex.(getproperty.(x.population, :fitness)) .>= x.threshold_fitness
    n_passed = sum(passed_genomes)
    if n_passed >= x.n_genomes_to_pass
      @info "!! Congratulations, NEAT terminated\n!! $(n_passed) genomes have reached the threshold fitness\n!! Passed genomes = $(findall(passed_genomes))"
      return true
    elseif x.generation >= x.max_generation
      @info "NEAT terminated: reached maximum generation"
      return false
    end

    mutate = generation ? true : mutate
    crossover = mutate ? true : crossover
    evaluate = crossover ? true : evaluate
    for i in x.population
      Run(i, evaluate)
    end

    passed_genomes = getindex.(getproperty.(x.population, :fitness)) .>= x.threshold_fitness
    n_passed = sum(passed_genomes)
    if n_passed >= x.n_genomes_to_pass
      @info "!! Congratulations, NEAT terminated\n!! $(n_passed) genomes have reached the threshold fitness\n!! Passesd genomes = $(findall(passed_genomes))"
      return true
    elseif x.generation >= x.max_generation
      @info "NEAT terminated: reached maximum generation"
      return false
    end

    if crossover
      new_population = Genome[]
      good_individuals = Dict{Unsigned, Vector{Genome}}()
      bad_individuals = Dict{Unsigned, Vector{Genome}}()
      for i in keys(x.species)
        idx = sortperm(GetFitness.(x.species[i]), rev = true)
        x.species[i] = x.species[i][idx]

        x.specie_info[i,:minimum_fitness] = x.species[i][end].fitness[]
        x.specie_info[i,:maximum_fitness] = x.species[i][1].fitness[]
        x.specie_info[i,:mean_fitness] = mean(GetFitness.(x.species[i]))
        if (x.specie_info[i,:last_highest_fitness] < x.specie_info[i,:mean_fitness])
          x.specie_info[i,:last_highest_fitness] = x.specie_info[i,:mean_fitness]
          x.specie_info[i,:last_improved_generation] = x.generation
        end

        if (x.generation-x.specie_info[i,:last_improved_generation]) > x.max_specie_stagnation
          x.specie_info[i,:alive] = false
          x.specie_info[i,:death_generation] = x.generation
          delete!(x.species, i)
          continue
        end

        if x.n_individuals_to_retain >= 1
          x.n_individuals_to_retain = round(x.n_individuals_to_retain)
          n_individuals_to_retain = min(x.n_individuals_to_retain, length(x.species[i]))
        elseif x.n_individuals_to_retain >= 0 && x.n_individuals_to_retain < 1
          n_individuals_to_retain = ceil( x.n_individuals_to_retain * length(x.species[i]) )
        else
          throw(ArgumentError("Invalid value for n_individuals_to_retain, got $(x.n_individuals_to_retain), should be >= 0"))
        end
        n_individuals_to_retain = Unsigned(n_individuals_to_retain)
        append!(new_population, x.species[i][1:n_individuals_to_retain])

        if x.n_individuals_considered_best >= 1
          x.n_individuals_considered_best = round(x.n_individuals_considered_best)
          n_individuals_considered_best = min(x.n_individuals_considered_best, length(x.species[i]))
        elseif x.n_individuals_considered_best >= 0 && x.n_individuals_considered_best < 1
          n_individuals_considered_best = ceil( x.n_individuals_considered_best * length(x.species[i]) )
        else
          throw(ArgumentError("Invalid value for n_individuals_considered_best, got $(x.n_individuals_considered_best), should be >= 0"))
        end
        n_individuals_considered_best = Unsigned(n_individuals_considered_best)
        good_individuals[i] = x.species[i][1:n_individuals_considered_best]
        bad_individuals[i] = x.species[i][n_individuals_considered_best+1:end]
      end
      best_fitness_specie = argmax(x.specie_info[:,:mean_fitness])
      x.specie_info[best_fitness_specie,:last_topped_generation] = x.generation

      for i = 1:length(new_population)
        new_population[i].ID = i
      end
      for i = length(new_population)+1:x.population_size
        crossover_probability = append!(x.crossover_probability[1:3], size(GetSpecieInfo(x, simple=true))[1]>1 ? x.crossover_probability[4:6] : [0,0,0])
        crossover = sample([1,2,3,4,5,6], _custom_weights(crossover_probability))
        if crossover == 1 # intraspecie good and good
          specie1 = sample(GetSpecieInfo(x, simple=true)[:,:specie], _custom_weights(GetSpecieInfo(x, simple=true)[:,:mean_fitness]))
          specie2 = specie1

          parent1 = rand(good_individuals[specie1])
          parent2 = rand(good_individuals[specie2])

          child = Crossover(parent1, parent2)
        elseif crossover == 2 # intraspecie good and bad
          specie1 = sample(GetSpecieInfo(x, simple=true)[:,:specie], _custom_weights(GetSpecieInfo(x, simple=true)[:,:mean_fitness]))
          specie2 = specie1

          parent1 = rand(good_individuals[specie1])
          parent2 = isempty(bad_individuals[specie2]) ? rand(good_individuals[specie2]) : rand(bad_individuals[specie2])

          child = Crossover(parent1, parent2)
        elseif crossover == 3 # intraspecie bad and bad
          specie1 = sample(GetSpecieInfo(x, simple=true)[:,:specie], _custom_weights(GetSpecieInfo(x, simple=true)[:,:mean_fitness]))
          specie2 = specie1

          parent1 = isempty(bad_individuals[specie1]) ? rand(good_individuals[specie1]) : rand(bad_individuals[specie1])
          parent2 = isempty(bad_individuals[specie2]) ? rand(good_individuals[specie2]) : rand(bad_individuals[specie2])

          child = Crossover(parent1, parent2)
        elseif crossover == 4 # interspecie good and good
          specie1 = sample(GetSpecieInfo(x, simple=true)[:,:specie], _custom_weights(GetSpecieInfo(x, simple=true)[:,:mean_fitness]))
          specie2 = sample(GetSpecieInfo(x, simple=true)[:,:specie], _custom_weights(GetSpecieInfo(x, simple=true)[:,:mean_fitness]))

          parent1 = rand(good_individuals[specie1])
          parent2 = rand(good_individuals[specie2])

          child = Crossover(parent1, parent2)
        elseif crossover == 5 # interspecie good and bad
          specie1 = sample(GetSpecieInfo(x, simple=true)[:,:specie], _custom_weights(GetSpecieInfo(x, simple=true)[:,:mean_fitness]))
          specie2 = sample(GetSpecieInfo(x, simple=true)[:,:specie], _custom_weights(GetSpecieInfo(x, simple=true)[:,:mean_fitness]))

          parent1 = rand(good_individuals[specie1])
          parent2 = isempty(bad_individuals[specie2]) ? rand(good_individuals[specie2]) : rand(bad_individuals[specie2])

          child = Crossover(parent1, parent2)
        else # interspecie bad and bad
          specie1 = sample(GetSpecieInfo(x, simple=true)[:,:specie], _custom_weights(GetSpecieInfo(x, simple=true)[:,:mean_fitness]))
          specie2 = sample(GetSpecieInfo(x, simple=true)[:,:specie], _custom_weights(GetSpecieInfo(x, simple=true)[:,:mean_fitness]))

          parent1 = isempty(bad_individuals[specie1]) ? rand(good_individuals[specie1]) : rand(bad_individuals[specie1])
          parent2 = isempty(bad_individuals[specie2]) ? rand(good_individuals[specie2]) : rand(bad_individuals[specie2])

          child = Crossover(parent1, parent2)
        end
        child.ID = i
        
        if mutate
          Mutation(child)
        end
        push!(new_population, child)
      end
      x.population = new_population

      x.mutation_probability = DataFrame()
      x.fitness = Reference{Real}[]
      x.output = Matrix{Reference{Real}}(undef, x.population_size, x.n_outputs)
      for (i,j) in zip(x.population, 1:x.population_size)
        append!(x.mutation_probability, i.mutation_probability)
        push!(x.fitness, i.fitness)
        x.output[j,:] .= i.output
      end
    end

    if generation
      Speciation(x)
      x.generation += 0x1
    end

  end

  function Show(x::Genome; directed::Bool = true, rankdir_LR::Bool = true, connection_label::Bool = true, pen_width::Real = 1.0, export_type::String = "svg", simple::Bool = true)
    graphviz_code = ""

    if rankdir_LR
      graphviz_code *= "\trankdir=LR;\n"
    end
    graphviz_code *= "\tnode [shape=circle];\n"
    for l in x.layers
      temp = "\t{rank=same; "
      for n in l
        if simple && !n.enabled
          continue
        end
        temp *= "$(n.GIN), "
        if !(typeof(n) <: OutputNode)
          for c in n.out_connections
            if simple && !c.enabled
              continue
            end
            if directed
              graphviz_code *= "\t$(n.GIN)->$(c.out_node.GIN)"
            else
              graphviz_code *= "\t$(n.GIN)--$(c.out_node.GIN)"
            end
            graphviz_code *= "["
            if c.enabled
              graphviz_code *= "color=green"
            else
              graphviz_code *= "color=red"
            end
            graphviz_code *= ","
            if connection_label
              graphviz_code *= "label=\"$(c.GIN),  $(round(c.weight,digits=3))\""
            end
            graphviz_code *= ","
            graphviz_code *= "penwidth=$(pen_width)"
            graphviz_code *= "];\n"
          end
        end
        graphviz_code *= n.enabled ? "\t$(n.GIN) [color=green]\n" : "\t$(n.GIN) [color=red]\n"
      end
      temp = temp[1:end-2] * "}\n"
      graphviz_code *= temp
    end

    if directed
      graphviz_code = "digraph {\n" * graphviz_code * "}"
    else
      graphviz_code = "graph {\n" * graphviz_code * "}"
    end

    dot_filename = "neat_$(x.super.generation)_$(x.ID).dot"
    open(dot_filename, "w") do file
      write(file, graphviz_code)
    end

    export_types = ["svg", "png", "jpg", "jpeg", "none"]
    if (export_type in export_types) && (export_type != "none")
      export_filename = dot_filename[1:end-3] * export_type
      run(`dot -T$(export_type) $(dot_filename) -o $(export_filename)`)
    elseif export_type == "none"
    else
      throw(ArgumentError("Invalid export_type, got $(export_type), accepted values = $(export_types)"))
    end

    return graphviz_code
  end
  function Show(x::NEAT, idx::Union{Integer, Vector{Integer}, OrdinalRange{Integer, Integer}}; directed::Bool = true, rankdir_LR::Bool = true, connection_label::Bool = true, pen_width::Real = 1.0, export_type::String = "svg", simple::Bool = true)
    for i in idx
      i = Unsigned(i)
      Show(x.population[i], directed = directed, rankdir_LR = rankdir_LR, connection_label = connection_label, pen_width = pen_width, export_type = export_type, simple = simple)
    end
  end

  function SetInput!(x::Union{NEAT, Genome}, args::Real...)
    (length(args) == x.n_inputs) || throw(ArgumentError("Invalid number of inputs, expecting $(x.n_inputs) inputs, got $(length(args))"))
    for i = 1:x.n_inputs
      x.input[i][] = args[i]
    end
  end
  function SetInput!(x::Union{NEAT, Genome}, y::Vector{Real})
    SetInput!(x, y...)
  end

  function GetInput(x::Union{NEAT, Genome})
    return [i[] for i in x.input]
  end

  function GetOutput(x::Genome)
    return [i[] for i in x.output]
  end
  function GetOutput(x::NEAT, idx::Integer)
    idx = Unsigned(idx)
    GetOutput(x.population[idx])
  end
  function GetOutput(x::NEAT, idx::Union{Vector{T}, OrdinalRange{T,T}}) where T<:Integer
    ret = Matrix{Union{Real, Nothing}}(undef, 0,x.n_outputs)
    for i in idx
      i = Unsigned(i)
      ret = vcat(ret, GetOutput(x.population[i]))
    end
    return ret
  end
  function GetOutput(x::NEAT)
    ret = Matrix{Union{Real, Nothing}}(undef, x.population_size,x.n_outputs)
    for i = 1:length(x.output)
      ret[i] = x.output[i][]
    end
    return ret
  end

  function SetExpectedOutput!(x::Union{NEAT, Genome}, args::Real...)
    (length(args) == x.n_outputs) || throw(ArgumentError("Invalid number of expected_output, expecting $(x.n_outputs) outputs, got $(length(args))"))
    if isempty(x.expected_output)
      for i = args
        push!(x.expected_output, Reference{Real}(i))
      end
    else
      for i = 1:x.n_outputs
        x.expected_output[i][] = args[i]
      end
    end
  end
  function SetExpectedOutput!(x::Union{NEAT, Genome}, y::Vector{Real})
    SetExpectedOutput!(x, y...)
  end

  function GetExpectedOutput(x::Union{NEAT, Genome})
    return [i[] for i in x.expected_output]
  end

  function GetMutationProbability(x::Genome)
    ret = DataFrame()
    for i in Symbol.(names(x.mutation_probability))
      ret[:,i] = Real[x.mutation_probability[1,i][]]
    end
    return ret
  end
  function GetMutationProbability(x::NEAT, idx::Union{T, Vector{T}, OrdinalRange{T,T}}) where T<:Integer
    ret = DataFrame()
    for i in Symbol.(names(x.mutation_probability))
      ret[:,i] = Real[]
    end
    for i in idx
      i = Unsigned(i)
      append!(ret, GetMutationProbability(x.population[i]))
    end
    return ret
  end
  function GetMutationProbability(x::NEAT)
    return GetMutationProbability(x, 1:x.population_size)
  end

  function SetMutationProbability!(x::Genome, y::Dict{Symbol, T}) where T<:Real
    types_of_mutation_probability = Symbol.(names(x.mutation_probability))
    all([i in types_of_mutation_probability for i in keys(y)]) || throw(ArgumentError("All keys in input Dict should be from $(types_of_mutation_probability)"))

    for i in keys(y)
      x.mutation_probability[1,i][] = y[i]
    end
  end
  function SetMutationProbability!(x::Genome, y::DataFrame)
    types_of_mutation_probability = Symbol.(names(x.mutation_probability))
    all([i in types_of_mutation_probability for i in Symbol.(names(y))]) || throw(ArgumentError("All columns in input DataFrame should be from $(types_of_mutation_probability)"))
    all(typeof.(collect(y[1,:])) .<: Real) || throw(ArgumentError("All values in the input DataFrame should be <: Real"))

    for i in Symbol.(names(y))
      x.mutation_probability[1,i][] = y[1,i]
    end
  end
  function SetMutationProbability!(x::NEAT, idx::Union{T, Vector{T}, OrdinalRange{T,T}}, y::Dict{Symbol, U}) where {T<:Integer, U<:Real}
    for i in idx
      i = Unsigned.(i)
      SetMutationProbability!(x.population[i], y)
    end
  end
  function SetMutationProbability!(x::NEAT, idx::Union{T, Vector{T}, OrdinalRange{T,T}}, y::DataFrame) where T<:Integer
    for i in idx
      i = Unsigned.(i)
      SetMutationProbability!(x.population[i], y)
    end
  end
  function SetMutationProbability!(x::NEAT, y::Dict{Symbol, T}) where T<:Real
    SetMutationProbability!(x, 1:x.population_size, y)
  end
  function SetMutationProbability!(x::NEAT, y::DataFrame)
    SetMutationProbability!(x, 1:x.population_size, y)
  end

  function GetLayers(x::Genome; simple::Bool = true)
    ret = Vector{Unsigned}[]
    for i in x.layers
      temp = [j.GIN for j in i if (!simple || j.enabled)]
      push!(ret, temp)
    end

    return ret
  end
  function GetLayers(x::NEAT, idx::Integer; simple::Bool = true)
    idx = Unsigned(idx)
    GetLayers(x.population[idx], simple = simple)
  end
  function GetLayers(x::NEAT, idx::Union{Vector{T}, OrdinalRange{T,T}}; simple::Bool = true) where T<:Integer
    ret = Vector{Vector{Unsigned}}[]
    for i in idx
      i = Unsigned(i)
      push!(ret, GetLayers(x.population[i], simple = simple))
    end
    return ret
  end
  function GetLayers(x::NEAT; simple::Bool = true)
    ret = Vector{Vector{Unsigned}}[]
    for i in x.population
      push!(ret, GetLayers(i, simple = simple))
    end

    return ret
  end

  function RunFitness(x::Genome)
    x.fitness[] = x.fitness_function[](GetOutput(x), GetExpectedOutput(x))
  end
  function RunFitness(x::NEAT, idx::Integer)
    idx = Unsigned(idx)
    RunFitness(x.population[idx])
  end
  function RunFitness(x::NEAT, idx::Union{Vector{T}, OrdinalRange{T,T}}) where T<:Integer
    for i in idx
      i = Unsigned(i)
      RunFitness(x.population[i])
    end
  end
  function RunFitness(x::NEAT)
    for i in x.population
      RunFitness(i)
    end
  end

  function GetFitness(x::Genome)
    return x.fitness[]
  end
  function GetFitness(x::NEAT, idx::Integer)
    idx = Unsigned(idx)
    GetFitness(x.population[idx])
  end
  function GetFitness(x::NEAT, idx::Union{Vector{T}, OrdinalRange{T,T}}) where T<:Integer
    ret = Real[]
    for i in idx
      i = Unsigned(i)
      push!(ret, GetFitness(x, i))
    end
    return ret
  end
  function GetFitness(x::NEAT)
    return [GetFitness(i) for i in x.population]
  end

  function GetFitnessFunction(x::Union{Genome, NEAT})
    return x.fitness_function[]
  end

  function SetFitnessFunction!(x::Union{Genome, NEAT}, func::Function)
    x.fitness_function[] = func
  end

  function GetGenomeInfo(x::Genome; simple::Bool = false)
    ret = DataFrame(GIN = Unsigned[],
                    type = Type[],
                    in_node = Vector{Union{Nothing, Unsigned}}(),
                    out_node = Vector{Union{Nothing, Unsigned}}(),
                    enabled = Vector{Union{Nothing, Bool}}(),)
    for i in sort(collect(keys(x.genome)))
      if (simple && !x.genome[i].enabled)
        continue
      end
      temp = []
      push!(temp, x.genome[i].GIN)
      if typeof(x.genome[i])<:Node
        push!(temp, typeof(x.genome[i]))
        push!(temp, nothing)
        push!(temp, nothing)
        push!(temp, x.genome[i].enabled)
      else
        push!(temp, typeof(x.genome[i]))
        push!(temp, x.genome[i].in_node.GIN)
        push!(temp, x.genome[i].out_node.GIN)
        push!(temp, x.genome[i].enabled)
      end
      push!(ret, temp)
    end

    # if simple
    #   ret = ret[ret[:,5].!=false,:]
    # end

    return ret
  end
  function GetGenomeInfo(x::NEAT, idx::Integer; simple::Bool = false)
    idx = Unsigned(idx)
    GetGenomeInfo(x.population[idx], simple = simple)
  end
  function GetGenomeInfo(x::NEAT, idx::Union{Vector{T}, OrdinalRange{T,T}}; simple::Bool = false) where T<:Integer
    ret = DataFrame[]
    for i in idx
      i = Unsigned(i)
      push!(ret, GetGenomeInfo(x.population[i], simple = simple))
    end
    return ret
  end
  function GetGenomeInfo(x::NEAT; simple = false)
    GetGenomeInfo(x, 1:x.population_size, simple = simple)
  end

  function GetSpecieInfo(x::NEAT; simple::Bool = true)
    if simple
      ret = copy(x.specie_info[x.specie_info[:,:alive].==true,:])
    else
      ret = copy(x.specie_info)
    end
    
    return ret
  end
  function GetSpecieInfo(x::NEAT, y::Unsigned)
    idx = findfirst(x.specie_info[:,:specie].==y)
    if isnothing(idx)
      throw("Got invalid specie number $(y)")
      return
    end

    return GetSpecieInfo(x)[idx,:]
  end

  function SetCrossoverProbability!(x::NEAT, y::Vector{Union{Nothing, T}}) where T<:Real
    (length(y)==6) || throw(ArgumentError("Got vector of length != 6"))
    x.crossover_probability[.!isnothing.(y)] .= y[.!isnothing.(y)]
  end

  function GetCrossoverProbability(x::NEAT)
    return copy(x.crossover_probability)
  end

  function GetGIN(x::NEAT)
    return copy(x.GIN)
  end

  function GetGenomeDistance(x::Genome, y::Genome, z::Vector{T}) where T<:Real
    (length(z)==3) || throw(ArgumentError("Input vector should of length 3, instead got of length $(length(z))"))
    x_genome_info = GetGenomeInfo(x)
    y_genome_info = GetGenomeInfo(y)

    max_GIN = max(x_genome_info[end,:GIN], y_genome_info[end,:GIN])
    x_GIN = [i in x_genome_info[:,:GIN] ? true : false for i = 1:max_GIN]
    y_GIN = [i in y_genome_info[:,:GIN] ? true : false for i = 1:max_GIN]
    max_common_GIN_index = findlast(x_GIN .&& y_GIN)

    n_disjoint = sum(xor.(x_GIN[1:max_common_GIN_index], y_GIN[1:max_common_GIN_index]))
    n_excess = sum(xor.(x_GIN[max_common_GIN_index:end], y_GIN[max_common_GIN_index:end]))

    weight_difference = Real[]
    for i = 1:max_common_GIN_index
      if x_GIN[i] && y_GIN[i] && typeof(x.genome[i])<:Connection
        push!(weight_difference, abs(x.genome[i].weight-y.genome[i].weight))
      end
    end
    mean_weight_difference = mean(weight_difference)
    if isnan(mean_weight_difference)
      mean_weight_difference = 0
    end

    N = max(size(x_genome_info,1), size(x_genome_info,1))
    ret = sum( [n_disjoint, n_excess, mean_weight_difference] .* z ./ [N, N, 1] )

    return ret
  end

  function Relu(x::Real)
    return max(x, 0.0)
  end
   
  function Sum_Abs_Diferrence(output::Vector{<:Real}, expected_output::Vector{<:Real})
    fitness = -sum(abs.(output .- expected_output))
  end

  function Save(x::NEAT, filename::Union{Nothing, String} = nothing)
    isnothing(filename) && (filename = "NEAT_$(x.generation).jld2")
    jldsave(filename; x)
  end
  function Save(x::NEAT, idx::Integer, filename::Union{Nothing, String} = nothing)
    idx = Unsigned(idx)
    Save(x.population[idx], filename)
  end
  function Save(x::Genome, filename::Union{Nothing, String} = nothing)
    isnothing(filename) && (filename = "Genome_$(x.super.generation)_$(x.ID).jld2")
    jldsave(filename; x)
  end

  function Load(filename::String)
    ret = IdDict()
    jldopen(filename, "r") do f
      for i in keys(f)
        ret[i] = f[i]
      end
    end
    return ret
  end
end
;
