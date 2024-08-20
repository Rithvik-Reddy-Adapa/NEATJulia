import Base:getindex
import Base:setindex!

mutable struct Reference{T}
  v::Union{Missing, T}

  function Reference{T}(v::Union{Missing, T} = missing) where T
    new(v)
  end
  function Reference(v::Union{Missing, T} = missing) where T
    type = v == missing ? Any : typeof(v)
    new{type}(v)
  end
end

function getindex(x::Reference)
  x.v
end

function setindex!(x::Reference, v)
  x.v = v
end
