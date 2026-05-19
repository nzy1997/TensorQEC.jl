const BUNDLE_MATCHING_FORMAT = "toric_builder_css_decode_bundle"
const BUNDLE_MATCHING_ENCODING = "dense_f2_row_major_txt"
const BUNDLE_MATCHING_EXPECTED_SHAPES = (
    check_matrix = (:c0_dim, :c1_dim),
    epsilon_c = (:anyon_dim, :c0_dim),
    d1_toric = (:t0_dim, :t1_dim),
    phi_0 = (:t0_dim, :c0_dim),
    psi_1 = (:c1_dim, :t1_dim),
    K_0 = (:c1_dim, :c0_dim),
)

struct BundleSideDecodingProblem <: AbstractDecodingProblem
    bundle_dir::String
    side::Symbol

    function BundleSideDecodingProblem(bundle_dir::String, side::Symbol)
        side in (:decode_x, :decode_z) || throw(ArgumentError("side must be :decode_x or :decode_z"))
        return new(bundle_dir, side)
    end
end

struct BundleCSSDecodingProblem <: AbstractDecodingProblem
    bundle_dir::String
end

Base.@kwdef struct BundleMatchingDecoder <: AbstractClassicalDecoder
    check_valid_syndrome::Bool = true
end

struct CompiledBundleSideMatching <: CompiledDecoder
    bundle_dir::String
    side::Symbol
    check_valid_syndrome::Bool
    c0_dim::Int
    c1_dim::Int
    t0_dim::Int
    t1_dim::Int
    anyon_dim::Int
    toric_layer_count::Int
    toric_layer_size::Int
    check_matrix::Matrix{Mod2}
    epsilon_c::Matrix{Mod2}
    d1_toric::Matrix{Mod2}
    phi_0::Matrix{Mod2}
    psi_1::Matrix{Mod2}
    K_0::Matrix{Mod2}
    matching_graph::MatchingGraph
    edge2qubit::Matrix{Int}
    toric_components::Vector{Int}
    events_buf::Vector{Int}
end

function _bundle_matching_key(dict::AbstractDict, key::AbstractString, context::AbstractString)
    haskey(dict, key) || throw(ArgumentError("$context is missing required key \"$key\""))
    return dict[key]
end

function _bundle_matching_dict(dict::AbstractDict, key::AbstractString, context::AbstractString)
    value = _bundle_matching_key(dict, key, context)
    value isa AbstractDict || throw(ArgumentError("$context.$key must be an object"))
    return value
end

function _bundle_matching_string(dict::AbstractDict, key::AbstractString, context::AbstractString)
    value = _bundle_matching_key(dict, key, context)
    value isa AbstractString || throw(ArgumentError("$context.$key must be a string"))
    return String(value)
end

function _bundle_matching_int(dict::AbstractDict, key::AbstractString, context::AbstractString)
    value = _bundle_matching_key(dict, key, context)
    value isa Integer || throw(ArgumentError("$context.$key must be an integer"))
    value >= 0 || throw(ArgumentError("$context.$key must be nonnegative"))
    return Int(value)
end

function _bundle_matching_shape(dict::AbstractDict, key::AbstractString, context::AbstractString)
    value = _bundle_matching_key(dict, key, context)
    value isa AbstractVector || throw(ArgumentError("$context.$key must be a 2-element shape vector"))
    length(value) == 2 || throw(ArgumentError("$context.$key must contain exactly 2 entries"))
    all(entry -> entry isa Integer, value) || throw(ArgumentError("$context.$key must contain integers"))
    return (Int(value[1]), Int(value[2]))
end

function _load_bundle_matching_manifest(bundle_dir::AbstractString)
    manifest_path = joinpath(bundle_dir, "manifest.json")
    isfile(manifest_path) || throw(ArgumentError("bundle manifest not found at $manifest_path"))
    manifest = try
        JSON.parsefile(manifest_path)
    catch err
        throw(ArgumentError("failed to parse bundle manifest at $manifest_path: $(sprint(showerror, err))"))
    end
    manifest isa AbstractDict || throw(ArgumentError("bundle manifest at $manifest_path must be a JSON object"))
    format = _bundle_matching_string(manifest, "format", "bundle manifest")
    format == BUNDLE_MATCHING_FORMAT || throw(
        ArgumentError("unsupported bundle format $format; expected $BUNDLE_MATCHING_FORMAT"),
    )
    return manifest
end

function _load_bundle_matching_dimensions(side_name::AbstractString, side_manifest::AbstractDict)
    dimensions = _bundle_matching_dict(side_manifest, "dimensions", side_name)
    return (
        c0_dim = _bundle_matching_int(dimensions, "c0_dim", "$side_name.dimensions"),
        c1_dim = _bundle_matching_int(dimensions, "c1_dim", "$side_name.dimensions"),
        t0_dim = _bundle_matching_int(dimensions, "t0_dim", "$side_name.dimensions"),
        t1_dim = _bundle_matching_int(dimensions, "t1_dim", "$side_name.dimensions"),
        anyon_dim = _bundle_matching_int(dimensions, "anyon_dim", "$side_name.dimensions"),
        toric_layer_count = _bundle_matching_int(dimensions, "toric_layer_count", "$side_name.dimensions"),
        toric_layer_size = _bundle_matching_int(dimensions, "toric_layer_size", "$side_name.dimensions"),
    )
end

function _bundle_matching_expected_shape(dims, matrix_key::Symbol)
    row_key, col_key = getproperty(BUNDLE_MATCHING_EXPECTED_SHAPES, matrix_key)
    return (getproperty(dims, row_key), getproperty(dims, col_key))
end

function _bundle_matching_within(root::AbstractString, path::AbstractString)
    relative = relpath(path, root)
    return relative != ".." && !startswith(relative, "..$(Base.Filesystem.path_separator)")
end

function _resolve_bundle_matching_path(
    bundle_dir::AbstractString,
    side::Symbol,
    side_name::AbstractString,
    matrix_name::AbstractString,
    file_manifest::AbstractDict,
)
    relative_path = _bundle_matching_string(
        file_manifest,
        "path",
        "$side_name.files.$matrix_name",
    )
    isabspath(relative_path) && throw(
        ArgumentError("$side_name.$matrix_name path must be relative, got $relative_path"),
    )
    resolved_path = normpath(joinpath(bundle_dir, relative_path))
    side_root = normpath(joinpath(bundle_dir, String(side)))
    _bundle_matching_within(side_root, resolved_path) || throw(
        ArgumentError("$side_name.$matrix_name path $relative_path must stay within $side_root"),
    )
    isfile(resolved_path) || throw(
        ArgumentError("$side_name.$matrix_name file not found at $relative_path"),
    )
    return resolved_path
end

function _matrix_to_mod2(raw_matrix::AbstractMatrix, label::AbstractString)
    matrix = Matrix{Mod2}(undef, size(raw_matrix)...)
    @inbounds for index in eachindex(raw_matrix)
        value = raw_matrix[index]
        (value == 0 || value == 1) || throw(
            ArgumentError("$label must contain only 0/1 entries, found $value"),
        )
        matrix[index] = Mod2(Int(value))
    end
    return matrix
end

function _load_bundle_matching_matrix(
    bundle_dir::AbstractString,
    side::Symbol,
    side_name::AbstractString,
    matrix_name::Symbol,
    files_manifest::AbstractDict,
    dims,
)
    entry = _bundle_matching_dict(files_manifest, String(matrix_name), "$side_name.files")
    encoding = _bundle_matching_string(entry, "encoding", "$side_name.files.$matrix_name")
    encoding == BUNDLE_MATCHING_ENCODING || throw(
        ArgumentError(
            "$side_name.$matrix_name uses unsupported encoding $encoding; expected $BUNDLE_MATCHING_ENCODING",
        ),
    )
    expected_shape = _bundle_matching_expected_shape(dims, matrix_name)
    manifest_shape = _bundle_matching_shape(entry, "shape", "$side_name.files.$matrix_name")
    manifest_shape == expected_shape || throw(
        ArgumentError(
            "$side_name.$matrix_name manifest shape $manifest_shape does not match expected shape $expected_shape",
        ),
    )
    matrix_path = _resolve_bundle_matching_path(
        bundle_dir,
        side,
        side_name,
        String(matrix_name),
        entry,
    )
    raw_matrix = try
        readdlm(matrix_path, Int)
    catch err
        throw(ArgumentError("failed to load $side_name.$matrix_name from $matrix_path: $(sprint(showerror, err))"))
    end
    raw_matrix isa AbstractMatrix || throw(
        ArgumentError("$side_name.$matrix_name at $matrix_path must be a dense matrix"),
    )
    size(raw_matrix) == expected_shape || throw(
        ArgumentError(
            "$side_name.$matrix_name has shape $(size(raw_matrix)) but manifest expects $expected_shape",
        ),
    )
    return _matrix_to_mod2(raw_matrix, "$side_name.$matrix_name")
end

function _validate_bundle_matching_runtime_dimensions(side_name::AbstractString, dims, matrices)
    size(matrices.check_matrix, 1) == dims.c0_dim || throw(
        ArgumentError("$side_name check_matrix row count must equal c0_dim"),
    )
    size(matrices.check_matrix, 2) == dims.c1_dim || throw(
        ArgumentError("$side_name check_matrix column count must equal c1_dim"),
    )
    size(matrices.epsilon_c, 1) == dims.anyon_dim || throw(
        ArgumentError("$side_name epsilon_c row count must equal anyon_dim"),
    )
    size(matrices.epsilon_c, 2) == size(matrices.check_matrix, 1) || throw(
        ArgumentError("$side_name epsilon_c column count must equal check_matrix row count"),
    )
    size(matrices.phi_0, 1) == dims.t0_dim || throw(
        ArgumentError("$side_name phi_0 row count must equal t0_dim"),
    )
    size(matrices.phi_0, 2) == size(matrices.check_matrix, 1) || throw(
        ArgumentError("$side_name phi_0 column count must equal check_matrix row count"),
    )
    size(matrices.psi_1, 1) == size(matrices.check_matrix, 2) || throw(
        ArgumentError("$side_name psi_1 row count must equal check_matrix column count"),
    )
    size(matrices.psi_1, 2) == dims.t1_dim || throw(
        ArgumentError("$side_name psi_1 column count must equal t1_dim"),
    )
    size(matrices.K_0) == reverse(size(matrices.check_matrix)) || throw(
        ArgumentError("$side_name K_0 shape must equal reverse(check_matrix shape)"),
    )
    size(matrices.d1_toric, 1) == size(matrices.phi_0, 1) || throw(
        ArgumentError("$side_name d1_toric row count must equal phi_0 row count"),
    )
    size(matrices.d1_toric, 2) == size(matrices.psi_1, 2) || throw(
        ArgumentError("$side_name d1_toric column count must equal psi_1 column count"),
    )
    return nothing
end

function _validate_bundle_matching_d1_toric(side_name::AbstractString, d1_toric::AbstractMatrix{Mod2})
    for column_index in axes(d1_toric, 2)
        column_weight = count(entry -> entry.x, view(d1_toric, :, column_index))
        column_weight == 2 || throw(
            ArgumentError(
                "$side_name d1_toric column $column_index must have Hamming weight 2, got $column_weight",
            ),
        )
    end
    return nothing
end

function _bundle_matching_components(d1_toric::AbstractMatrix{Mod2})
    graph = SimpleGraph(size(d1_toric, 1))
    for column_index in axes(d1_toric, 2)
        a, b = findall(entry -> entry.x, view(d1_toric, :, column_index))
        Graphs.add_edge!(graph, a, b)
    end
    labels = zeros(Int, nv(graph))
    for (component_index, component) in enumerate(connected_components(graph))
        for vertex in component
            labels[vertex] = component_index
        end
    end
    return labels
end

function _load_bundle_matching_side(bundle_dir::AbstractString, side::Symbol)
    manifest = _load_bundle_matching_manifest(bundle_dir)
    side_name = String(side)
    side_manifest = _bundle_matching_dict(manifest, side_name, "bundle manifest")
    dims = _load_bundle_matching_dimensions(side_name, side_manifest)
    files_manifest = _bundle_matching_dict(side_manifest, "files", side_name)
    matrices = (
        check_matrix = _load_bundle_matching_matrix(
            bundle_dir,
            side,
            side_name,
            :check_matrix,
            files_manifest,
            dims,
        ),
        epsilon_c = _load_bundle_matching_matrix(
            bundle_dir,
            side,
            side_name,
            :epsilon_c,
            files_manifest,
            dims,
        ),
        d1_toric = _load_bundle_matching_matrix(
            bundle_dir,
            side,
            side_name,
            :d1_toric,
            files_manifest,
            dims,
        ),
        phi_0 = _load_bundle_matching_matrix(
            bundle_dir,
            side,
            side_name,
            :phi_0,
            files_manifest,
            dims,
        ),
        psi_1 = _load_bundle_matching_matrix(
            bundle_dir,
            side,
            side_name,
            :psi_1,
            files_manifest,
            dims,
        ),
        K_0 = _load_bundle_matching_matrix(
            bundle_dir,
            side,
            side_name,
            :K_0,
            files_manifest,
            dims,
        ),
    )
    _validate_bundle_matching_runtime_dimensions(side_name, dims, matrices)
    _validate_bundle_matching_d1_toric(side_name, matrices.d1_toric)
    return dims, matrices
end

function compile(decoder::BundleMatchingDecoder, problem::BundleSideDecodingProblem)
    dims, matrices = _load_bundle_matching_side(problem.bundle_dir, problem.side)
    matching_graph, edge2qubit = parity_matrix2matching_graph(matrices.d1_toric)
    toric_components = _bundle_matching_components(matrices.d1_toric)
    events_buf = Int[]
    sizehint!(events_buf, dims.t0_dim)
    return CompiledBundleSideMatching(
        problem.bundle_dir,
        problem.side,
        decoder.check_valid_syndrome,
        dims.c0_dim,
        dims.c1_dim,
        dims.t0_dim,
        dims.t1_dim,
        dims.anyon_dim,
        dims.toric_layer_count,
        dims.toric_layer_size,
        matrices.check_matrix,
        matrices.epsilon_c,
        matrices.d1_toric,
        matrices.phi_0,
        matrices.psi_1,
        matrices.K_0,
        matching_graph,
        edge2qubit,
        toric_components,
        events_buf,
    )
end

function _has_nonzero_mod2(entries::AbstractArray{Mod2})
    return any(entry -> entry.x, entries)
end

function _bundle_matching_events_matchable(events::Vector{Int}, component_labels::Vector{Int})
    component_parity = falses(maximum(component_labels))
    @inbounds for event in events
        component_parity[component_labels[event]] ⊻= true
    end
    return !any(component_parity)
end

function decode(compiled::CompiledBundleSideMatching, syndrome::SimpleSyndrome)
    length(syndrome.s) == compiled.c0_dim || throw(
        ArgumentError("expected syndrome length $(compiled.c0_dim), got $(length(syndrome.s))"),
    )

    if compiled.check_valid_syndrome
        _has_nonzero_mod2(compiled.epsilon_c * syndrome.s) && throw(
            ArgumentError("syndrome violates epsilon_c consistency for $(compiled.side)"),
        )
    end

    s_t = compiled.phi_0 * syndrome.s
    events = compiled.events_buf
    fill_events_mod2!(events, s_t, 1, compiled.t0_dim)

    e_t = fill(Mod2(0), compiled.t1_dim)
    if _bundle_matching_events_matchable(events, compiled.toric_components)
        error_vec = SparseBlossom.decode_to_edges(compiled.matching_graph, events)
        @inbounds for j in 1:2:length(error_vec)
            qubit = compiled.edge2qubit[error_vec[j], error_vec[j + 1]]
            qubit != 0 && (e_t[qubit] += Mod2(1))
        end
    elseif compiled.check_valid_syndrome
        throw(ArgumentError("toric syndrome is not matchable for $(compiled.side)"))
    end

    e_c = compiled.psi_1 * e_t + compiled.K_0 * syndrome.s
    success = compiled.check_matrix * e_c == syndrome.s
    return DecodingResult(success, e_c)
end

function compile(decoder::BundleMatchingDecoder, problem::BundleCSSDecodingProblem)
    return CompiledClassicalDecoder(
        compile(decoder, BundleSideDecodingProblem(problem.bundle_dir, :decode_x)),
        compile(decoder, BundleSideDecodingProblem(problem.bundle_dir, :decode_z)),
    )
end
