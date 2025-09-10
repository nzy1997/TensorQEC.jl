struct NumberedMeasure{BT<:Measure,D} <: AbstractContainer{BT,D}
    m::BT
    num::Int
    function NumberedMeasure(m::BT, num::Int) where BT <: Measure
        new{BT,nlevel(m)}(m, num)
    end
end

Yao.nqudits(nm::NumberedMeasure) = nqudits(nm.m)
Yao.print_block(io::IO, m::NumberedMeasure) = print(io, "M[$(m.num)]")
Yao.content(m::NumberedMeasure) = m.m
YaoBlocks.Optimise.to_basictypes(m::NumberedMeasure) = m
Yao.chsubblocks(c::NumberedMeasure, block) = NumberedMeasure(block..., c.num)

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::NumberedMeasure, address, controls)
    @assert length(controls) == 0
    f_vec = copy(c.frontier)
    f_vec[address[1]] += 1
    YaoPlots._texttop!(YaoPlots.CircuitGrid(f_vec, c.w_depth, c.w_line,c.gatestyles), address[1], CircuitStyles.boxsize(c.gatestyles.g)..., 0.25, "rec[$(p.num)]")
    YaoPlots.draw!(c, p.m, address, controls)
end
Yao.apply!(reg::AbstractRegister, m::NumberedMeasure) = apply!(reg, m.m)

struct ConditionBlock{N, BTT<:AbstractBlock{N}, BTF<:AbstractBlock{N}} <: CompositeBlock{N}
    m::NumberedMeasure
    block_true::BTT
    block_false::BTF
    function ConditionBlock(m::NumberedMeasure, block_true::BTT, block_false::BTF) where {N, BTT<:AbstractBlock{N}, BTF<:AbstractBlock{N}}
        new{N, BTT, BTF}(m, block_true, block_false)
    end
end

Yao.subblocks(c::ConditionBlock) = (c.m, c.block_true, c.block_false)
Yao.chsubblocks(c::ConditionBlock, blocks) = ConditionBlock(blocks...)

function Yao.apply!(reg::AbstractRegister{B}, c::ConditionBlock) where B
    @show c.m.m
    if !isdefined(c.m.m, :results)
        println("Conditioned on a measurement that has not been performed.")
        throw(UndefRefError())
    end
    reg |> (c.m.m.results == 0 ? c.block_false : c.block_true)
    reg
end

condition(m, a::AbstractBlock{N}, b::Nothing) where N = ConditionBlock(m, a, IdentityGate{2}(N))
condition(m, a::Nothing, b::AbstractBlock{N}) where N = ConditionBlock(m, IdentityGate{2}(N), b)
Yao.mat(c::ConditionBlock) = throw(ArgumentError("ConditionBlock does not has matrix representation, try `mat(c.block_true)` or `mat(c.block_false)`"))

function Yao.print_block(io::IO, c::ConditionBlock)
    print(io, "if result(id = $(objectid(c.m)))")
end

Yao.nqudits(c::ConditionBlock) = Yao.nqudits(c.block_true)

YaoBlocks.Optimise.to_basictypes(c::ConditionBlock) = c


function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::ConditionBlock, address, controls)
    bts1 = length(controls)>=1 ? YaoPlots.get_cbrush_texts(c, p.block_true) : YaoPlots.get_brush_texts(c, p.block_true)
    bts2 = length(controls)>=1 ? YaoPlots.get_cbrush_texts(c, p.block_false) : YaoPlots.get_brush_texts(c, p.block_false)
    YaoPlots._draw!(c, [controls..., (getindex.(Ref(address), occupied_locs(p)),bts1[1], "$(bts1[2]) or $(bts2[2])")])
end

# abstract type AbstractDetectorBlock{D} <: TrivialGate{D} end
struct DetectorBlock{D} <: TrivialGate{D}
    vm::Vector{NumberedMeasure}
    num::Int
    detector_type::Int # 0: detector, 1: logical
end

Yao.nqudits(sr::DetectorBlock) = 1
Yao.print_block(io::IO, sr::DetectorBlock) = iszero(sr.detector_type) ? print(io, "D[$(sr.num)]: rec$(getfield.(sr.vm, :num))") : print(io, "L[$(sr.num)]: rec$(getfield.(sr.vm, :num))")

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::DetectorBlock, address, controls)
    @assert length(controls) == 0
    YaoPlots._draw!(c, [(getindex.(Ref(address), (1,)), c.gatestyles.g, iszero(p.detector_type) ? "D[$(p.num)]: rec$(getfield.(p.vm, :num))" : "L[$(p.num)]: rec$(getfield.(p.vm, :num))")])
end