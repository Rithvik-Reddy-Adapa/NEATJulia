export AllNEATTypes, Genes, Nodes, HiddenNodes, RecurrentHiddenNodes, Connections, RecurrentConnections, Configs, Probabilities

"""
AllNEATTypes is the super type of types in this NEAT module
"""
abstract type AllNEATTypes end

"""
Genes is super type of all Node and Connection genes
"""
abstract type Genes <: AllNEATTypes end

"""
Nodes is super type of all Node genes i.e. InputNode, HiddenNode and OutputNode
"""
abstract type Nodes <: Genes end

"""
HiddenNodes is super type of all Hidden Node genes i.e. HiddenNode, RecurrentNode, LSTMNode, etc.
"""
abstract type HiddenNodes <: Nodes end

"""
RecurrentHiddenNodes is super type of all Recurrent Hidden Node genes
"""
abstract type RecurrentHiddenNodes <: HiddenNodes end

"""
Connections is super type of all Connection genes
"""
abstract type Connections <: Genes end

"""
RecurrentConnections is super type of all Recurrent Connection genes
"""
abstract type RecurrentConnections <: Connections end


"""
Configs is super type of all configs
"""
abstract type Configs <: AllNEATTypes end

"""
Probabilities is super type of all probabilities
"""
abstract type Probabilities <: AllNEATTypes end



function Base.getindex(x::T, idx::Integer) where T <: AllNEATTypes
  return Base.getfield(x, idx)
end

function Base.getindex(x::U, idx::Union{OrdinalRange{T, T}, Vector{T}}) where {T <: Integer, U <: AllNEATTypes}
  return [Base.getfield(x, i) for i in idx]
end

function Base.collect(x::T) where T <: AllNEATTypes
  return [(i => Base.getfield(x, i)) for i in Base.fieldnames(typeof(x))]
end

function Base.Dict(x::T) where T <: AllNEATTypes
  return Base.Dict(Base.collect(x))
end

Base.firstindex(x::T) where T <: AllNEATTypes = 1
Base.lastindex(x::T) where T <: AllNEATTypes = length(Base.fieldnames(typeof(x)))
Base.length(x::T) where T <: AllNEATTypes = length(Base.fieldnames(typeof(x)))

function Base.getindex(x::T, ::Colon) where T <: AllNEATTypes
  return [Base.getfield(x, i) for i in Base.fieldnames(typeof(x))]
end

function Base.setindex!(x::T, value, idx::Integer) where T <: AllNEATTypes
  Base.setfield!(x, idx, value)
  return nothing
end
