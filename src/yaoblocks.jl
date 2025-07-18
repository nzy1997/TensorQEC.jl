struct ConditionBlock{N, BTT<:AbstractBlock{N}, BTF<:AbstractBlock{N}} <: CompositeBlock{N}
    m::Measure
    block_true::BTT
    block_false::BTF
    function ConditionBlock(m::Measure, block_true::BTT, block_false::BTF) where {N, BTT<:AbstractBlock{N}, BTF<:AbstractBlock{N}}
        new{N, BTT, BTF}(m, block_true, block_false)
    end
end

Yao.subblocks(c::ConditionBlock) = (c.m, c.block_true, c.block_false)
Yao.chsubblocks(c::ConditionBlock, blocks) = ConditionBlock(blocks...)

function _apply!(reg::AbstractRegister{B}, c::ConditionBlock) where B
    if !isdefined(c.m, :results)
        println("Conditioned on a measurement that has not been performed.")
        throw(UndefRefError())
    end
    for i = 1:B
        viewbatch(reg, i) |> (c.m.results[i] == 0 ? c.block_false : c.block_true)
    end
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


# DetectorBlock

struct DetectorBlock{D,K, OT, LT, PT, RNG} <: TrivialGate{D}
    vm::Vector{Measure{D,K, OT, LT, PT, RNG}}
end

Yao.nqudits(sr::DetectorBlock) = 1
Yao.print_block(io::IO, sr::DetectorBlock) = print(io, "DETECTOR($(length(sr.vm)))")

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::DetectorBlock, address, controls)
    @assert length(controls) == 0
    YaoPlots._draw!(c, [(getindex.(Ref(address), (1,)), c.gatestyles.g, "DETECTOR($(length(p.vm)))")])
end