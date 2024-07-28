import Base:getindex
import Base:setindex!

mutable struct Reference{T}
  v::Union{Nothing, T}

  function Reference{T}(v::Union{Nothing, T} = nothing) where T
    new(v)
  end
  function Reference(v::Union{Nothing, T} = nothing) where T
    type = v == nothing ? Any : typeof(v)
    new{type}(v)
  end
end

function getindex(x::Reference)
  x.v
end

function setindex!(x::Reference, v )
  x.v = v
end
