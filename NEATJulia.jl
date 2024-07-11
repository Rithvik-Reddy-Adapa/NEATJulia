
module NEATJulia
  include("Reference.jl")
  export Reference, getindex, setindex!, Node, InputNode, HiddenNode, OutputNode, Connection, Genome, NEAT, Init, Run, Relu, SetInput!, GetInput, GetOutput, GetFitness, SetExpectedOutput!, GetExpectedOutput, RunFitness, Sum_Abs_Diferrence, GetFitnessFunction, SetFitnessFunction!, GetMutationProbability, SetMutationProbability, GetLayers, GetGenomeInfo, GetSpecieFitness, Show

  abstract type Node end

  mutable struct InputNode{Connection, Genome} <: Node
    GIN::Unsigned # Global Innovation Number
    input_number::Unsigned
    input::Reference{Real}
    output::Reference{Real}
    out_connections::Vector{Connection}
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
    ID::Unsigned
    specie::Unsigned
    genome::Dict{Unsigned, Union{Node, Connection}} # Global Innovation Number => Node / Connection
    layers::Vector{Vector{Node}}
    input::Vector{Reference{Real}}
    output::Vector{Reference{Real}}
    expected_output::Vector{Reference{Real}}
    fitness::Reference{Real}
    fitness_function::Reference{Union{Nothing, Function}}
    super::Union{Nothing, NEAT}
    mutation_probability::Vector{Reference{Real}} # Probobility of different types of mutations 1: update weight, 2: update bias, 3: add connection, 4: add node
  end

  mutable struct NEAT
    n_inputs::Unsigned
    n_outputs::Unsigned
    population_size::Unsigned
    max_generation::Unsigned
    max_species::Unsigned
    RNN_enabled::Bool
    threshold_fitness::Real
    n_genomes_to_pass::Unsigned # number of geneomes to pass fitness test for NEAT to pass
    fitness_function::Reference{Union{Nothing, Function}}
    max_weight::Real
    min_weight::Real
    max_bias::Real
    min_bias::Real

    population::Vector{Genome}
    generation::Unsigned
    n_species::Unsigned
    GIN::Matrix{Union{Unsigned, String, Nothing}} # list of all Global Innovation Number
    input::Vector{Reference{Real}}
    best_genome::Union{Nothing, Genome}
    output::Matrix{Reference{Real}}
    expected_output::Vector{Reference{Real}}
    fitness::Vector{Reference{Real}}
    mutation_probability::Matrix{Reference{Real}}
    specie_fitness::Matrix{Union{Nothing, Real}} # n_rows = n_species, columns = {specie number, minimum fiteness, maximum fitness, mean fitness, last topped generation}
  end

  function InputNode(; GIN = 0, input_number = 0, input = Reference{Real}(), output = Reference{Real}(), out_connections = Connection[], processed = false, super = nothing)
    InputNode{Connection, Genome}(GIN, input_number, input, output, out_connections, processed, super)
  end
  function HiddenNode(; GIN = 0, in_connections = Connection[], input = Reference{Real}[], output = Reference{Real}(), out_connections = Connection[], activation = Relu, bias = 0.0, processed = false, super = nothing)
    HiddenNode{Connection, Genome}(GIN, in_connections, input, output, out_connections, activation, bias, processed, super)
  end
  function OutputNode(; GIN = 0, in_connections = Connection[], input = Reference{Real}[], output = Reference{Real}(), output_number = 0, activation = Relu, bias = 0.0, processed = false, super = nothing)
    OutputNode{Connection, Genome}(GIN, in_connections, input, output, output_number, activation, bias, processed, super)
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
  function Genome(n_inputs, n_outputs; ID = 0, specie = 1, genome = Dict{Unsigned, Union{Node, Connection}}(), input = Reference{Real}[], output = Reference{Real}[], expected_output = Reference{Real}[], fitness = Reference{Real}(), fitness_function = Reference{Union{Nothing, Function}}(Sum_Abs_Diferrence), super = nothing, mutation_probability = Reference{Real}.([1, 1, 0.25, 0.25]))

    (n_inputs > 0) || throw(ArgumentError("Invalid n_inputs $(n_inputs), should be > 0"))
    (n_outputs > 0) || throw(ArgumentError("Invalid n_outputs $(n_outputs), should be > 0"))

    (isempty(input)) && (input = fill(Reference{Real}(), n_inputs))
    (isempty(output)) && (output = fill(Reference{Real}(), n_outputs))

    layers = Vector{Node}[]

    Genome{NEAT}(n_inputs, n_outputs, ID, specie, genome, layers, input, output, expected_output, fitness, fitness_function, super, mutation_probability)
  end
  function NEAT(n_inputs, n_outputs; population_size = 20, max_generation = 50, max_species = typemax(UInt64), RNN_enabled = false, threshold_fitness = 1.0, n_genomes_to_pass = 1, fitness_function = Reference{Union{Nothing, Function}}(Sum_Abs_Diferrence), max_weight = 10, min_weight = -10, max_bias = 5, min_bias = -5, population = Genome[], generation = 0, n_species = 1, GIN = Matrix{Union{Unsigned, String, Nothing}}(undef, 0,4), best_genome = nothing, expected_output = Reference{Real}[], mutation_probability = Matrix{Reference{Real}}(undef, 0,0), specie_fitness = [1 -Inf -Inf -Inf 1])

    (n_inputs > 0) || throw(ArgumentError("Invalid n_inputs $(n_inputs), should be > 0"))
    (n_outputs > 0) || throw(ArgumentError("Invalid n_outputs $(n_outputs), should be > 0"))
    (population_size > 0) || throw(ArgumentError("Invalid population_size $(population_size), should be > 0"))
    (isempty(population)) && (population = Vector{Genome}(undef, population_size))
    if isempty(mutation_probability)
      mutation_probability = Matrix{Reference{Real}}(undef, population_size,4)
      for i = eachrow(mutation_probability)
        i[1] = Reference{Real}(1.0)
        i[2] = Reference{Real}(1.0)
        i[3] = Reference{Real}(0.25)
        i[4] = Reference{Real}(0.25)
      end
    elseif size(mutation_probability) == (1,4) || size(mutation_probability) == (4,)
      temp_mutation_probability = Matrix{Reference{Real}}(undef, population_size,4)
      for i = eachrow(temp_mutation_probability)
        i[1] = Reference{Real}(mutation_probability[1][])
        i[2] = Reference{Real}(mutation_probability[2][])
        i[3] = Reference{Real}(mutation_probability[3][])
        i[4] = Reference{Real}(mutation_probability[4][])
      end
      mutation_probability = temp_mutation_probability
    else
      throw(ArgumentError("mutation_probability should be of type Matrix{Reference{Real}} and of size ($(population_size), 4) or (1, 4) or (4,)"))
    end

    input = [Reference{Real}() for i = 1:n_inputs]
    output = Matrix{Reference{Real}}(undef, population_size, n_outputs)
    for i = 1:length(output)
      output[i] = Reference{Real}()
    end
    fitness = [Reference{Real}(-Inf) for i = 1:population_size]

    NEAT(n_inputs, n_outputs, population_size, max_generation, max_species, RNN_enabled, threshold_fitness, n_genomes_to_pass, fitness_function, max_weight, min_weight, max_bias, min_bias, population, generation, n_species, GIN, input, best_genome, output, expected_output, fitness, mutation_probability, specie_fitness)
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
      x.population[i] = Genome(x.n_inputs, x.n_outputs, ID = i, super = x, fitness_function = x.fitness_function, input = x.input, output = x.output[i,:], expected_output = x.expected_output, fitness = x.fitness[i], mutation_probability = x.mutation_probability[i,:])
      Init(x.population[i])
    end
    for i = 1:x.n_inputs+x.n_outputs
      x.GIN = vcat(x.GIN, [Unsigned(i) "Node" nothing nothing])
    end
    return
  end

  function Crossover(x::Genome, y::Genome)
    if x.fitness[] > y.fitness[]
      ret = deepcopy(x)
      for i in keys(ret.genome)
        if haskey(y.genome, i) && (typeof(y.genome[i]) <: Connection) && rand([true, false])
          ret.genome[i].weight = y.genome[i].weight
        elseif haskey(y.genome, i) && (typeof(y.genome[i]) <: HiddenNode) && rand([true, false])
          ret.genome[i].bias = y.genome[i].bias
        elseif haskey(y.genome, i) && (typeof(y.genome[i]) <: OutputNode) && rand([true, false])
          ret.genome[i].bias = y.genome[i].bias
        end
      end
    else
      ret = deepcopy(y)
      for i in keys(ret.genome)
        if haskey(x.genome, i) && (typeof(x.genome[i]) <: Connection) && rand([true, false])
          ret.genome[i].weight = x.genome[i].weight
        elseif haskey(x.genome, i) && (typeof(x.genome[i]) <: HiddenNode) && rand([true, false])
          ret.genome[i].bias = x.genome[i].bias
        elseif haskey(x.genome, i) && (typeof(x.genome[i]) <: OutputNode) && rand([true, false])
          ret.genome[i].bias = x.genome[i].bias
        end
      end
    end
    ret.ID = 0
    ret.fitness[] = -Inf
    ret.input = x.input
    for i = 1:length(ret.mutation_probability)
      ret.mutation_probability[i][] = (x.mutation_probability[i][] + y.mutation_probability[i][])/2
    end
    # for i in ret.output
    #   i[] = 0.0
    # end

    return ret
  end

  function Mutation(x::Genome)
    cum_mutation_probability = cumsum(GetMutationProbability(x))
    random_value = rand()*cum_mutation_probability[end]

    if random_value <= cum_mutation_probability[1] # update weight
      connections = [i for i in values(x.genome) if typeof(i)<:Connection]
      if !(isempty(connections))
        random_connection = rand(connections)
        while !(random_connection.enabled)
          random_connection = rand(connections)
        end
        random_connection.weight = x.super.min_weight + (x.super.max_weight - x.super.min_weight)*rand()
        return 1, random_connection.GIN
      end
    elseif random_value <= cum_mutation_probability[2] # update bias
      nodes = [i for i in values(x.genome) if ((typeof(i)<:HiddenNode) || (typeof(i)<:OutputNode))]
      random_node = rand(nodes)
      random_node.bias = x.super.min_bias + (x.super.max_bias - x.super.min_bias)*rand()
      return 2, random_node.GIN
    elseif random_value <= cum_mutation_probability[3] # add connection
      start_layer = rand(1:length(x.layers)-1)
      start_node = rand(x.layers[start_layer])
      end_layer = rand(start_layer+1:length(x.layers))
      end_node = rand(x.layers[end_layer])
      next_nodes_of_start_node = [i.out_node for i in start_node.out_connections]
      if !(end_node in next_nodes_of_start_node)
        idx = findfirst(x.super.GIN[:,2].=="Connection" .&& x.super.GIN[:,3].==start_node.GIN .&& x.super.GIN[:,4].==end_node.GIN)
        if isnothing(idx)
          GIN = x.super.GIN[end,1]+0x1
          x.genome[GIN] = Connection(start_node, end_node, GIN = GIN, super = x)
          x.super.GIN = vcat(x.super.GIN, [GIN "Connection" start_node.GIN end_node.GIN])
          return 3, GIN, "new"
        else
          GIN = x.super.GIN[idx,1]
          x.genome[GIN] = Connection(start_node, end_node, GIN = GIN, super = x)
          return 3, GIN, "old"
        end
      end
    elseif random_value <= cum_mutation_probability[4] # add node
      connections = [i for i in values(x.genome) if typeof(i)<:Connection]
      if !(isempty(connections))
        # get a random connection
        random_connection = rand(connections)
        while !(random_connection.enabled)
          random_connection = rand(connections)
        end

        # find the layer number of the random connection's in node
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
        x.super.GIN = vcat(x.super.GIN, [GIN "Node" nothing nothing])

        GIN = GIN+0x1
        new_node_in_connection = Connection(random_connection.in_node, new_node, GIN = GIN, super = x, weight = 1)
        x.genome[GIN] = new_node_in_connection
        x.super.GIN = vcat(x.super.GIN, [GIN "Connection" new_node_in_connection.in_node.GIN new_node_in_connection.out_node.GIN])

        GIN = GIN+0x1
        new_node_out_connection = Connection(new_node, random_connection.out_node, GIN = GIN, super = x, weight = random_connection.weight)
        x.genome[GIN] = new_node_out_connection
        x.super.GIN = vcat(x.super.GIN, [GIN "Connection" new_node_out_connection.in_node.GIN new_node_out_connection.out_node.GIN])

        random_connection.enabled = false

        # get a random layer for new node
        new_node_layer = rand(start_layer+0.5:0.5:end_layer-0.5)
        if new_node_layer == floor(new_node_layer) # add into existing layer
          push!(x.layers[Unsigned(new_node_layer)], new_node)
        else
          insert!(x.layers, Unsigned(ceil(new_node_layer)), [new_node])
        end

        return 4, new_node.GIN, new_node_in_connection.GIN, new_node_out_connection.GIN
      end
    end
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
    if !(x.enabled)
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
  function Run(x::NEAT, evaluate::Bool = false)
    for i in x.population
      Run(i, evaluate)
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

  function SetInput!(x::Union{NEAT, Genome}, args::Real...)
    (length(args) == x.n_inputs) || throw(ArgumentError("Invalid number of inputs, expecting $(x.n_inputs) inputs, got $(length(args))"))
    for i = 1:x.n_inputs
      x.input[i][] = args[i]
    end
  end
  function GetInput(x::Union{NEAT, Genome})
    return [i[] for i in x.input]
  end

  function GetOutput(x::Genome)
    return [i[] for i in x.output]
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

  function GetExpectedOutput(x::Union{NEAT, Genome})
    return [i[] for i in x.expected_output]
  end

  function GetMutationProbability(x::Genome)
    return [i[] for i in x.mutation_probability]
  end
  function GetMutationProbability(x::NEAT)
    ret = Matrix{Union{Nothing, Real}}(undef, x.population_size,4)
    for i = 1:length(x.mutation_probability)
      ret[i] = x.mutation_probability[i][]
    end

    return ret
  end

  function SetMutationProbability(x::Genome, args::Real...)
    if length(args) != 4
      throw(ArgumentError("Invalid number of probabilities expected 4, instead got $(length(args))"))
    end
    if any(args.<0.0)
      throw(ArgumentError("Probabilities should be greater >= 0.0, instead got a negative number"))
    end

    for (i,j) in zip(x.mutation_probability, args)
      i[] = j
    end
  end
  function SetMutationProbability(x::NEAT, idx::Union{Unsigned, Vector{Unsigned}, OrdinalRange{Unsigned, Unsigned}}, args::Real...)
    for i in idx
      SetMutationProbability(x.population[i], args...)
    end
  end

  function GetLayers(x::Genome)
    ret = Vector{Unsigned}[]
    for i in x.layers
      temp = [j.GIN for j in i]
      push!(ret, temp)
    end

    return ret
  end
  function GetLayers(x::NEAT)
    ret = Vector{Vector{Unsigned}}[]
    for i in x.population
      push!(ret, GetLayers(i))
    end

    return ret
  end

  function RunFitness(x::Genome)
    x.fitness[] = x.fitness_function[](GetOutput(x), GetExpectedOutput(x))
  end
  function RunFitness(x::NEAT)
    for i in x.population
      RunFitness(i)
    end
  end

  function GetFitness(x::Genome)
    return x.fitness[]
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

  function GetGenomeInfo(x::Genome)
    ret = Matrix{Union{Unsigned, String, Bool, Nothing}}(undef, length(x.genome),5)
    for (i,j) in zip(sort(collect(keys(x.genome))), 1:length(x.genome))
      ret[j,1] = x.genome[i].GIN
      if typeof(x.genome[i])<:Node
        ret[j,2] = "Node"
        ret[j,3] = nothing
        ret[j,4] = nothing
        ret[j,5] = nothing
      else
        ret[j,2] = "Connection"
        ret[j,3] = x.genome[i].in_node.GIN
        ret[j,4] = x.genome[i].out_node.GIN
        ret[j,5] = x.genome[i].enabled
      end
    end

    return ret
  end

  function GetSpecieFitness(x::NEAT)
    return x.specie_fitness[.!isnothing.(x.specie_fitness[:,1]),:]
  end
  function GetSpecieFitness(x::NEAT, y::Unsigned)
    idx = findfirst(x.specie_fitness[:,1].==y)
    if isnothing(idx)
      throw("Given invalid specie number $(y)")
      return
    end

    return x.specie_fitness[idx,:]
  end

  function Relu(x::Real)
    return max(x, 0.0)
  end
   
  function Sum_Abs_Diferrence(output::Vector{<:Real}, expected_output::Vector{<:Real})
    fitness = -sum(abs.(output .- expected_output))
  end
end
;
