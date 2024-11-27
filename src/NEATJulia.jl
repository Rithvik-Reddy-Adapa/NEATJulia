# Disable precompilation, reason: Method overwriting is not permitted during Module precompilation., related to: src/NEAT/NEAT.jl:4 & src/NEAT/NEATUtils.jl
__precompile__(false)

module NEATJulia

  # Included all files in specific order to not have dependency issues
  include("Reference.jl")
  include("AbstractTypes.jl")
  include("RNG.jl")
  include("ActivationFunctions.jl")
  include("Probabilities.jl")

  include("NEAT/Network/Genes/Nodes/InputNodeConfig.jl")
  include("NEAT/Network/Genes/Nodes/OutputNodeConfig.jl")
  include("NEAT/Network/Genes/Nodes/HiddenNodes/HiddenNodeConfig.jl")
  include("NEAT/Network/Genes/Connections/ForwardConnectionConfig.jl")
  include("NEAT/Network/Genes/Connections/RecurrentConnections/BackwardConnectionConfig.jl")
  include("NEAT/Network/Genes/Nodes/HiddenNodes/RecurrentHiddenNodes/RecurrentHiddenNodeConfig.jl")
  include("NEAT/Network/Genes/Nodes/HiddenNodes/RecurrentHiddenNodes/LSTMNodeConfig.jl")
  include("NEAT/Network/Genes/Nodes/HiddenNodes/RecurrentHiddenNodes/GRUNodeConfig.jl")

  include("NEAT/Network/NetworkConfig.jl")

  include("NEAT/NEATConfig.jl")

  include("NEAT/NEAT.jl")

  include("NEAT/Network/Network.jl")

  include("NEAT/Network/Genes/Nodes/InputNode.jl")
  include("NEAT/Network/Genes/Nodes/OutputNode.jl")
  include("NEAT/Network/Genes/Nodes/HiddenNodes/HiddenNode.jl")
  include("NEAT/Network/Genes/Connections/ForwardConnection.jl")
  include("NEAT/Network/Genes/Connections/RecurrentConnections/BackwardConnection.jl")
  include("NEAT/Network/Genes/Nodes/HiddenNodes/RecurrentHiddenNodes/RecurrentHiddenNode.jl")
  include("NEAT/Network/Genes/Nodes/HiddenNodes/RecurrentHiddenNodes/LSTMNode.jl")
  include("NEAT/Network/Genes/Nodes/HiddenNodes/RecurrentHiddenNodes/GRUNode.jl")

  include("NEAT/Network/NetworkUtils.jl")

  include("NEAT/NEATUtils.jl")

end

;
