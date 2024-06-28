
module NEATJulia
  export Node, InputNode, HiddenNode, OutputNode, Connection, Genome, NEAT, Init, Run, Relu, SetInput

  abstract type Node end

  mutable struct InputNode{Connection, Genome} <: Node
    GIN::UInt64 # Global Innovation Number
    input_number::UInt64
    input::Ref{Real}
    output::Ref{Real}
    out_connections::Vector{Connection}
    processed::Bool
    super::Union{Nothing, Genome}
  end

  mutable struct HiddenNode{Connection, Genome} <: Node
    GIN::UInt64 # Global Innovation Number
    in_connections::Vector{Connection}
    input::Vector{Ref{Real}}
    output::Ref{Real}
    out_connections::Vector{Connection}
    activation::Function
    bias::Real
    processed::Bool
    super::Union{Nothing, Genome}
  end

  mutable struct OutputNode{Connection, Genome} <: Node
    GIN::UInt64 # Global Innovation Number
    in_connections::Vector{Connection}
    input::Vector{Ref{Real}}
    output::Ref{Real}
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
    input::Ref{Real}
    output::Ref{Real}
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
    input::Vector{Ref{Real}}
    output::Vector{Ref{Real}}
    fitness::Ref{Real}
    fitness_function::Ref{Union{Nothing, Function}}
    super::Union{Nothing, NEAT}
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
    fitness_function::Ref{Union{Nothing, Function}}

    population::Vector{Genome}
    generation::UInt64
    species::UInt64
    GIN::UInt64 # current Global Innovation Number
    input::Vector{Ref{Real}}
    best_genome::Union{Nothing, Genome}
    output::Vector{Vector{Ref{Real}}}
    fitness::Vector{Ref{Real}}
  end

  function InputNode(; GIN = 0, input_number = 0, input = Ref{Real}(), output = Ref{Real}(), out_connections = Connection[], processed = false, super = nothing)
    InputNode{Connection, Genome}(GIN, input_number, input, output, out_connections, processed, super)
  end
  function HiddenNode(; GIN = 0, in_connections = Connection[], input = Ref{Real}[], output = Ref{Real}(), out_connections = Connection[], activation = Relu, bias = 0.0, processed = false, super = nothing)
    HiddenNode{Connection, Genome}(GIN, in_connections, input, output, out_connections, activation, bias, processed, super)
  end
  function OutputNode(; GIN = 0, in_connections = Connection[], input = Ref{Real}[], output = Ref{Real}(), output_number = 0, activation = Relu, bias = rand(), processed = false, super = nothing)
    OutputNode{Connection, Genome}(GIN, in_connections, input, output, output_number, activation, bias, processed, super)
  end
  function Connection(; in_node, out_node, GIN = 0, weight = rand(), enabled = true, processed = false, super = nothing)

    input = in_node.output
    output = out_node.input[end]

    Connection{Genome}(in_node, out_node, GIN, weight, input, output, enabled, processed, super)
  end
  function Genome(n_inputs, n_outputs; ID = 0, specie = 0, genome = Dict{UInt64, Union{Node, Connection}}(), input = Ref{Real}[], output = Ref{Real}[], fitness = Ref{Real}(), fitness_function = Ref{Union{Nothing, Function}}(nothing), super = nothing)

    (n_inputs > 0) || throw(ArgumentError("Invalid n_inputs $(n_inputs), should be > 0"))
    (n_outputs > 0) || throw(ArgumentError("Invalid n_outputs $(n_outputs), should be > 0"))

    (isempty(input)) && (input = fill(Ref{Real}(0.0), n_inputs))
    (isempty(output)) && (output = fill(Ref{Real}(0.0), n_outputs))

    Genome{NEAT}(n_inputs, n_outputs, ID, specie, genome, input, output, fitness, fitness_function, super)
  end
  function NEAT(n_inputs, n_outputs; population_size = 20, max_generation = 50, max_species = 2, RNN_enabled = false, threshold_fitness = 1.0, n_genomes_to_pass = 1, fitness_function = Ref{Union{Nothing, Function}}(nothing), population = Genome[], generation = 0, species = 0, GIN = 0, best_genome = nothing)

    (n_inputs > 0) || throw(ArgumentError("Invalid n_inputs $(n_inputs), should be > 0"))
    (n_outputs > 0) || throw(ArgumentError("Invalid n_outputs $(n_outputs), should be > 0"))
    (population_size > 0) || throw(ArgumentError("Invalid population_size $(population_size), should be > 0"))
    (isempty(population)) && (population = Vector{Genome}(undef, population_size))

    input = [Ref{Real}(0.0) for i = 1:n_inputs]
    output = [[Ref{Real}(0.0) for j = 1:n_outputs] for i = 1:population_size]
    fitness = [Ref{Real}(0.0) for i = 1:population_size]

    NEAT(n_inputs, n_outputs, population_size, max_generation, max_species, RNN_enabled, threshold_fitness, n_genomes_to_pass, fitness_function, population, generation, species, GIN, input, best_genome, output, fitness)
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
      x.population[i] = Genome(x.n_inputs, x.n_outputs, ID = i, super = x, fitness_function = x.fitness_function, input = x.input, output = x.output[i], fitness = x.fitness[i])
      Init(x.population[i])
    end
    x.GIN = x.n_inputs + x.n_outputs
    return
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
    evaluate && (x.fitness[] = x.fitness_function[](x.output))
    for i in values(x.genome)
      i.processed = false
    end
  end
  function Run(x::NEAT, evaluate::Bool = false)
    for i in x.population
      Run(i, evaluate)
    end
  end

  function SetInput(x::NEAT, args::Real...)
    (length(args) == x.n_inputs) || throw(ArgumentError("Invalid number of inputs, expecting $(x.n_inputs) inputs, got $(length(args))"))
    for i = 1:x.n_inputs
      x.input[i][] = args[i]
    end
  end

  function Relu(x::Real)
    return max(x, 0.0)
  end
   
end
;
