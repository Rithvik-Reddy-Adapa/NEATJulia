
module NEATJulia
  include("Reference.jl")
  export Reference, getindex, setindex!, Node, InputNode, HiddenNode, OutputNode, Connection, Genome, NEAT, Init, Run, Relu, SetInput!, GetInput, GetOutput, GetFitness, SetExpectedOutput!, GetExpectedOutput, RunFitness, Sum_Abs_Diferrence, GetFitnessFunction, SetFitnessFunction!, Crossover

  abstract type Node end

  mutable struct InputNode{Connection, Genome} <: Node
    GIN::UInt64 # Global Innovation Number
    input_number::UInt64
    input::Reference{Real}
    output::Reference{Real}
    out_connections::Vector{Connection}
    processed::Bool
    super::Union{Nothing, Genome}
  end

  mutable struct HiddenNode{Connection, Genome} <: Node
    GIN::UInt64 # Global Innovation Number
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
    GIN::UInt64 # Global Innovation Number
    in_connections::Vector{Connection}
    input::Vector{Reference{Real}}
    output::Reference{Real}
    output_number::UInt64
    activation::Function
    bias::Real
    processed::Bool
    super::Union{Nothing, Genome}
  end

  mutable struct Connection{Genome}
    in_node::Node
    out_node::Node
    GIN::UInt64 # Global Innovation Number
    weight::Real
    input::Reference{Real}
    output::Reference{Real}
    enabled::Bool
    processed::Bool
    super::Union{Nothing, Genome}
  end

  mutable struct Genome{NEAT}
    n_inputs::UInt64
    n_outputs::UInt64
    ID::UInt64
    specie::UInt64
    genome::Dict{UInt64, Union{Node, Connection}} # Global Innovation Number => Node / Connection
    input::Vector{Reference{Real}}
    output::Vector{Reference{Real}}
    expected_output::Vector{Reference{Real}}
    fitness::Reference{Real}
    fitness_function::Reference{Union{Nothing, Function}}
    super::Union{Nothing, NEAT}
    mutation_probability::Vector{Real} # Probobility of different types of mutations 1: update weight, 2: update bias, 3: add connection, 4: add node
  end

  mutable struct NEAT
    n_inputs::UInt64
    n_outputs::UInt64
    population_size::UInt64
    max_generation::UInt64
    max_species::UInt64
    RNN_enabled::Bool
    threshold_fitness::Real
    n_genomes_to_pass::UInt64 # number of geneomes to pass fitness test for NEAT to pass
    fitness_function::Reference{Union{Nothing, Function}}
    max_weight::Real
    min_weight::Real
    max_bias::Real
    min_bias::Real

    population::Vector{Genome}
    generation::UInt64
    species::UInt64
    GIN::UInt64 # current Global Innovation Number
    input::Vector{Reference{Real}}
    best_genome::Union{Nothing, Genome}
    output::Matrix{Reference{Real}}
    expected_output::Vector{Reference{Real}}
    fitness::Vector{Reference{Real}}
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
  function Genome(n_inputs, n_outputs; ID = 0, specie = 0, genome = Dict{UInt64, Union{Node, Connection}}(), input = Reference{Real}[], output = Reference{Real}[], expected_output = Reference{Real}[], fitness = Reference{Real}(), fitness_function = Reference{Union{Nothing, Function}}(Sum_Abs_Diferrence), super = nothing, mutation_probability = [1, 1, 0.25, 0.25])

    (n_inputs > 0) || throw(ArgumentError("Invalid n_inputs $(n_inputs), should be > 0"))
    (n_outputs > 0) || throw(ArgumentError("Invalid n_outputs $(n_outputs), should be > 0"))

    (isempty(input)) && (input = fill(Reference{Real}(), n_inputs))
    (isempty(output)) && (output = fill(Reference{Real}(), n_outputs))

    Genome{NEAT}(n_inputs, n_outputs, ID, specie, genome, input, output, expected_output, fitness, fitness_function, super, mutation_probability)
  end
  function NEAT(n_inputs, n_outputs; population_size = 20, max_generation = 50, max_species = 2, RNN_enabled = false, threshold_fitness = 1.0, n_genomes_to_pass = 1, fitness_function = Reference{Union{Nothing, Function}}(Sum_Abs_Diferrence), max_weight = 10, min_weight = -10, max_bias = 5, min_bias = -5, population = Genome[], generation = 0, species = 0, GIN = 0, best_genome = nothing, expected_output = Reference{Real}[])

    (n_inputs > 0) || throw(ArgumentError("Invalid n_inputs $(n_inputs), should be > 0"))
    (n_outputs > 0) || throw(ArgumentError("Invalid n_outputs $(n_outputs), should be > 0"))
    (population_size > 0) || throw(ArgumentError("Invalid population_size $(population_size), should be > 0"))
    (isempty(population)) && (population = Vector{Genome}(undef, population_size))

    input = [Reference{Real}() for i = 1:n_inputs]
    output = Matrix{Reference{Real}}(undef, population_size, n_outputs)
    for i = 1:length(output)
      output[i] = Reference{Real}()
    end
    fitness = [Reference{Real}(-Inf) for i = 1:population_size]

    NEAT(n_inputs, n_outputs, population_size, max_generation, max_species, RNN_enabled, threshold_fitness, n_genomes_to_pass, fitness_function, max_weight, min_weight, max_bias, min_bias, population, generation, species, GIN, input, best_genome, output, expected_output, fitness)
  end

  function Init(x::Genome)
    for i = 1:x.n_inputs
      x.genome[i] = InputNode(GIN = i, input_number = i, super = x, input = x.input[i])
    end
    for i = 1:x.n_outputs
      x.genome[x.n_inputs + i] = OutputNode(GIN = x.n_inputs + i, output_number = i, super = x, output = x.output[i])
    end
    return
  end
  function Init(x::NEAT)
    for i = 1:x.population_size
      x.population[i] = Genome(x.n_inputs, x.n_outputs, ID = i, super = x, fitness_function = x.fitness_function, input = x.input, output = x.output[i,:], expected_output = x.expected_output, fitness = x.fitness[i])
      Init(x.population[i])
    end
    x.GIN = x.n_inputs + x.n_outputs
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
    ret.mutation_probability = (x.mutation_probability .+ y.mutation_probability)./2
    # for i in ret.output
    #   i[] = 0.0
    # end

    return ret
  end

  function Mutation(x::Genome)
    cum_mutation_probability = cumsum(x.mutation_probability)
    random_value = rand()*cum_mutation_probability[end]

    if random_value <= cum_mutation_probability[1] # update weight
      connections = [i for i in x.genome if typeof(i)<:Connection]
      if !(isempty(connections))
        random_connection = rand(connections)
        while !(random_connection.enabled)
          random_connection = rand(connections)
        end
        random_connection.weight = x.super.min_weight + (x.super.max_weight - x.super.min_weight)*rand() 
      end
    elseif random_value <= cum_mutation_probability[2] # update bias
      nodes = [i for i in x.genome if (typeof(i)<:HiddenNode) || (typeof(i)<:OutputNode)]
      random_node = rand(nodes)
      random_node.bias = x.super.min_bias + (x.super.max_bias - x.super.min_bias)*rand() 
    elseif random_value <= cum_mutation_probability[3] # add connection
    elseif random_value <= cum_mutation_probability[4] # add node
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
    for i in values(x.genome)
      if !(i.processed)
        Run(i)
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

  function Relu(x::Real)
    return max(x, 0.0)
  end
   
  function Sum_Abs_Diferrence(output::Vector{<:Real}, expected_output::Vector{<:Real})
    fitness = -sum(abs.(output .- expected_output))
  end
end
;
