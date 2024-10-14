export Reference

"""
Reference{T} = Ref{Union{Missing, T}}

Reference(x) = Reference{typeof(x)}(x)

Reference{T}() where T = Reference{T}(missing)
"""
Reference{T} = Ref{Union{Missing, T}}
Reference(x) = Reference{typeof(x)}(x)
Reference{T}() where T = Reference{T}(missing)
