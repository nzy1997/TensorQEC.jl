using Test, TensorQEC
using Random
using QECCore: code_n
using TensorQEC: IndependentDepolarizingDecodingProblem, ClassicalDecodingProblem

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

@testset "Multi-decoder pipeline" begin
    @testset "BPDecoder with multiple CSS codes" begin
        for code in [SteaneCode(), SurfaceCode(3, 3), ShorCode(), ToricCode(3, 3)]
            Random.seed!(42)
            n = code_n(code)
            em = iid_error(0.01, 0.01, 0.01, n)
            problem = DecodingProblem(code, em)
            compiled = compile(BPDecoder(), problem)

            tanner = CSSTannerGraph(code)
            error = random_error_pattern(em)
            syndrome = syndrome_extraction(error, tanner)
            result = decode(compiled, syndrome)
            @test result isa DecodingResult
            @test result.error_pattern isa CSSErrorPattern
        end
    end

    @testset "IP Decoder with SurfaceCode" begin
        Random.seed!(42)
        code = SurfaceCode(3, 3)
        n = code_n(code)
        em = iid_error(0.01, 0.01, 0.01, n)
        problem = DecodingProblem(code, em)
        compiled = compile(IPDecoder(), problem)

        tanner = CSSTannerGraph(code)
        error = random_error_pattern(em)
        syndrome = syndrome_extraction(error, tanner)
        result = decode(compiled, syndrome)
        @test result isa DecodingResult
    end

    @testset "IP Decoder with SteaneCode" begin
        Random.seed!(42)
        code = SteaneCode()
        n = code_n(code)
        em = iid_error(0.01, 0.01, 0.01, n)
        problem = DecodingProblem(code, em)
        compiled = compile(IPDecoder(), problem)

        tanner = CSSTannerGraph(code)
        error = random_error_pattern(em)
        syndrome = syndrome_extraction(error, tanner)
        result = decode(compiled, syndrome)
        @test result isa DecodingResult
    end

    @testset "Matching Decoder with SurfaceCode" begin
        Random.seed!(42)
        code = SurfaceCode(3, 3)
        n = code_n(code)
        em = iid_error(0.01, 0.01, 0.01, n)
        problem = DecodingProblem(code, em)
        compiled = compile(MatchingDecoder(TensorQEC.IPMatchingSolver()), problem)

        tanner = CSSTannerGraph(code)
        error = random_error_pattern(em)
        syndrome = syndrome_extraction(error, tanner)
        result = decode(compiled, syndrome)
        @test result isa DecodingResult
        @test result.error_pattern isa CSSErrorPattern
    end

    @testset "Matching Decoder with ToricCode" begin
        Random.seed!(42)
        code = ToricCode(3, 3)
        n = code_n(code)
        em = iid_error(0.01, 0.01, 0.01, n)
        problem = DecodingProblem(code, em)
        compiled = compile(MatchingDecoder(TensorQEC.GreedyMatchingSolver()), problem)

        tanner = CSSTannerGraph(code)
        error = random_error_pattern(em)
        syndrome = syndrome_extraction(error, tanner)
        result = decode(compiled, syndrome)
        @test result isa DecodingResult
        @test result.error_pattern isa CSSErrorPattern
    end

    @testset "TN MAP Decoder with SteaneCode" begin
        Random.seed!(42)
        code = SteaneCode()
        n = code_n(code)
        em = iid_error(0.01, 0.01, 0.01, n)
        problem = DecodingProblem(code, em)
        compiled = compile(TNMAP(), problem)

        tanner = CSSTannerGraph(code)
        error = random_error_pattern(em)
        syndrome = syndrome_extraction(error, tanner)
        result = decode(compiled, syndrome)
        @test result isa DecodingResult
    end

    @testset "TNMMAP Decoder with SteaneCode" begin
        Random.seed!(42)
        code = SteaneCode()
        n = code_n(code)
        em = iid_error(0.01, 0.01, 0.01, n)
        problem = DecodingProblem(code, em)
        compiled = compile(TNMMAP(), problem)

        tanner = CSSTannerGraph(code)
        error = random_error_pattern(em)
        syndrome = syndrome_extraction(error, tanner)
        result = decode(compiled, syndrome)
        @test result isa DecodingResult
        @test result.error_pattern isa CSSErrorPattern
    end
end

@testset "Logical error checking" begin
    @testset "Zero syndrome decodes to no error" begin
        code = SteaneCode()
        em = iid_error(0.01, 0.01, 0.01, code_n(code))
        problem = DecodingProblem(code, em)
        compiled = compile(BPDecoder(), problem)

        # SteaneCode has 3 X-stabilizers and 3 Z-stabilizers
        zero_syndrome = CSSSyndrome(Mod2.(zeros(Int, 3)), Mod2.(zeros(Int, 3)))
        result = decode(compiled, zero_syndrome)
        @test result.success_tag
        @test all(x -> !x.x, result.error_pattern.xerror)
        @test all(x -> !x.x, result.error_pattern.zerror)
    end

    @testset "Zero syndrome with SurfaceCode" begin
        code = SurfaceCode(3, 3)
        em = iid_error(0.01, 0.01, 0.01, code_n(code))
        problem = DecodingProblem(code, em)
        compiled = compile(BPDecoder(), problem)

        tanner = CSSTannerGraph(code)
        nsx = TensorQEC.ns(tanner.stgx)
        nsz = TensorQEC.ns(tanner.stgz)
        zero_syndrome = CSSSyndrome(Mod2.(zeros(Int, nsx)), Mod2.(zeros(Int, nsz)))
        result = decode(compiled, zero_syndrome)
        @test result.success_tag
        @test all(x -> !x.x, result.error_pattern.xerror)
        @test all(x -> !x.x, result.error_pattern.zerror)
    end

    @testset "Error round-trip with low error rate" begin
        Random.seed!(123)
        code = SteaneCode()
        n = code_n(code)
        em = iid_error(0.01, 0.01, 0.01, n)
        problem = DecodingProblem(code, em)
        tanner = CSSTannerGraph(code)
        compiled = compile(TNMAP(), problem)

        # Run multiple trials - with low error rate, decoder should mostly succeed
        successes = 0
        trials = 20
        for _ in 1:trials
            error = random_error_pattern(em)
            syndrome = syndrome_extraction(error, tanner)
            result = decode(compiled, syndrome)
            if result.success_tag
                successes += 1
            end
        end
        # At 1% error rate on 7 qubits, decoding should succeed most of the time
        @test successes >= trials ÷ 2
    end
end

@testset "Classical decoding problem" begin
    @testset "BPDecoder with classical problem from SteaneCode" begin
        code = SteaneCode()
        n = code_n(code)
        em = iid_error(0.05, n)
        problem = DecodingProblem(code, em)
        @test problem isa ClassicalDecodingProblem
        @test problem.tanner isa SimpleTannerGraph

        compiled = compile(BPDecoder(), problem)
        # Generate a syndrome
        error = random_error_pattern(em)
        syndrome = syndrome_extraction(error, problem.tanner)
        result = decode(compiled, syndrome)
        @test result isa DecodingResult
    end

    @testset "BPDecoder with classical problem from SurfaceCode" begin
        code = SurfaceCode(3, 3)
        n = code_n(code)
        em = iid_error(0.01, n)
        problem = DecodingProblem(code, em)
        @test problem isa ClassicalDecodingProblem

        compiled = compile(BPDecoder(), problem)
        error = random_error_pattern(em)
        syndrome = syndrome_extraction(error, problem.tanner)
        result = decode(compiled, syndrome)
        @test result isa DecodingResult
    end

    @testset "Zero syndrome classical" begin
        code = SteaneCode()
        n = code_n(code)
        em = iid_error(0.01, n)
        problem = DecodingProblem(code, em)
        compiled = compile(BPDecoder(), problem)

        zero_syndrome = SimpleSyndrome(Mod2.(zeros(Int, TensorQEC.ns(problem.tanner))))
        result = decode(compiled, zero_syndrome)
        @test result.success_tag
        @test all(x -> !x.x, result.error_pattern)
    end
end

@testset "Multiple code types with BPDecoder round-trip" begin
    Random.seed!(42)
    for (name, code) in [
        ("SteaneCode", SteaneCode()),
        ("SurfaceCode(3,3)", SurfaceCode(3, 3)),
        ("ShorCode", ShorCode()),
        ("ToricCode(3,3)", ToricCode(3, 3)),
    ]
        @testset "$name" begin
            n = code_n(code)
            em = iid_error(0.01, 0.01, 0.01, n)
            problem = DecodingProblem(code, em)
            tanner = problem.tanner
            compiled = compile(BPDecoder(), problem)

            # Test with zero syndrome
            nsx = TensorQEC.ns(tanner.stgx)
            nsz = TensorQEC.ns(tanner.stgz)
            zero_syndrome = CSSSyndrome(Mod2.(zeros(Int, nsx)), Mod2.(zeros(Int, nsz)))
            result = decode(compiled, zero_syndrome)
            @test result.success_tag
            @test all(x -> !x.x, result.error_pattern.xerror)
            @test all(x -> !x.x, result.error_pattern.zerror)

            # Test with random error
            error = random_error_pattern(em)
            syndrome = syndrome_extraction(error, tanner)
            result = decode(compiled, syndrome)
            @test result isa DecodingResult
            @test result.error_pattern isa CSSErrorPattern
            @test length(result.error_pattern.xerror) == n
            @test length(result.error_pattern.zerror) == n
        end
    end
end
