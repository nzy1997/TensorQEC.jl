using Test, TensorQEC
using QECCore: code_n

@testset "DecodingProblem from code" begin
    @testset "CSS code with depolarizing error" begin
        code = SteaneCode()
        em = iid_error(0.01, 0.01, 0.01, code_n(code))
        problem = DecodingProblem(code, em)
        @test problem isa IndependentDepolarizingDecodingProblem
        @test problem.tanner isa CSSTannerGraph
        @test problem.tanner.stgx.nq == code_n(code)
        @test problem.tanner.stgz.nq == code_n(code)
    end

    @testset "CSS code with flip error" begin
        code = SteaneCode()
        em = iid_error(0.01, code_n(code))
        problem = DecodingProblem(code, em)
        @test problem isa ClassicalDecodingProblem
        @test problem.tanner isa SimpleTannerGraph
        @test problem.tanner.nq == code_n(code)
    end

    @testset "End-to-end decode with BPDecoder" begin
        code = SteaneCode()
        em = iid_error(0.01, 0.01, 0.01, code_n(code))
        problem = DecodingProblem(code, em)
        compiled = compile(BPDecoder(), problem)

        # Zero syndrome should decode successfully
        # SteaneCode has 3 X-stabilizers and 3 Z-stabilizers
        ns = 3
        syndrome = CSSSyndrome(Mod2.(zeros(Int, ns)), Mod2.(zeros(Int, ns)))
        result = decode(compiled, syndrome)
        @test result isa DecodingResult
        @test result.success_tag
    end

    @testset "Different codes" begin
        for code in [SteaneCode(), SurfaceCode(3, 3)]
            n = code_n(code)
            em = iid_error(0.01, 0.01, 0.01, n)
            problem = DecodingProblem(code, em)
            @test problem isa IndependentDepolarizingDecodingProblem
            @test problem.tanner.stgx.nq == n
        end
    end
end
