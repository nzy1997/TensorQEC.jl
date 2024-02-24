using LuxorGraphPlot
using Luxor

@enum Direction K R D L U

function showgrid(m, n; unit=50, lw=1, color="black")
    grid = Matrix{Node}(undef, m, n)
    for i=1:m, j=1:n
        # show a dot
        grid[i, j] = dotnode((i*unit, j*unit))
    end
    @layer begin
        setcolor(color)
        setline(lw)
        for i=1:m, j=1:n
            # show edges
            i != m && LuxorGraphPlot.stroke(Connection(grid[i,j], grid[i+1, j]))
            j != n && LuxorGraphPlot.stroke(Connection(grid[i,j], grid[i, j+1]))
        end
    end
    return grid
end

function xznode(start, stop)
    circlenode(midpoint(start, stop), 10)
end
function xzpath(list::AbstractVector)
    nodes = [xznode(list[i], list[i+1]) for i=1:length(list)-1]
    path = polygonnode(list; close=list[1] == list[end])
    return nodes, path
end

# create a path on a 2D grid
function create_path(i::Int, j::Int, moves::Vector)
    path = [(i, j)]
    for m in moves
        if m == R
            i += 1
        elseif m == D
            j += 1
        elseif m == L
            i -= 1
        elseif m == U
            j -= 1
        end
        push!(path, (i, j))
    end
    return path
end
function create_xpath(grid, ij, moves::Vector)
    locs = create_path(ij..., moves)
    closed = mod1.(locs[1], size(grid) .- 1) == mod1.(locs[end], size(grid) .- 1)
    nodes, lines = xzpath([grid[i, j] for (i, j) in locs])
    showxpath(nodes, lines; closed)
    return nodes, lines
end
function create_zpath(grid, ij, moves::Vector)
    locs = create_path(ij..., moves)
    closed = mod1.(locs[1], size(grid) .- 1) == mod1.(locs[end], size(grid) .- 1)
    unit = grid[2, 2].loc - grid[1, 1].loc
    nodes, lines = xzpath([offset(grid[i, j], -unit/2) for (i, j) in locs])
    showzpath(nodes, lines; closed)
    return nodes, lines
end

function showxzpath(nodes, path; label, color, closed)
    @layer begin
        # stroke path
        setcolor(color)
        setline(6)
        setopacity(0.5)
        LuxorGraphPlot.stroke(path)
    end
    @layer begin
        showxznode.(nodes; label=label, color=color)
    end
    if !closed
        # show end points as dots
        @layer begin
            # dirty patch
            setcolor(color == "#CC6666" ? "red" : "blue")
            setline(1)
            for node in [path.relpath[1], path.relpath[end]]
                fill(circlenode(node, 5))
            end
        end
    end
end

function showxznode(node; label, color)
    setcolor(color)
    fill(node)
    setcolor("white")
    fontsize(16)
    text(label, node)
end
showzpath(nodes, path; closed) = showxzpath(nodes, path; label="Z", color="#CC6666", closed)
showxpath(nodes, path; closed) = showxzpath(nodes, path; label="X", color="#6666CC", closed)

function fig1()
    @svg begin
        origin(0, 0)
        background("white")
        # create a grid
        unit = 50
        grid = showgrid(7, 7; unit)

        # create two plaquettes
        zblock, zlines = xzpath([offset(grid[i, j], (-unit/2, -unit/2)) for (i, j) in [(3, 3), (3, 4), (4, 4), (4, 3), (3, 3)]])
        xblock, xlines = xzpath([grid[i, j] for (i, j) in [(5, 5), (5, 6), (6, 6), (6, 5), (5, 5)]])
        showzpath(zblock, zlines; closed=true)
        showxpath(xblock, xlines; closed=true)
    end 400 400 joinpath(@__DIR__, "grid.svg")
end

function fig2()
    @svg begin
        background("white")
        unit = 50
        m, n = 7, 7
        for k=1:4
            origin((k-1)*400, 0)
            # create a grid
            grid = showgrid(m, n; unit)

            # create two lines
            if k == 2 || k == 4
                showxpath(xzpath([grid[i, 3] for i in 1:m])...; closed=true)
            end
            if k == 3 || k == 4
                showxpath(xzpath([grid[3, i] for i in 1:m])...; closed=true)
            end
        end
    end 1600 400 joinpath(@__DIR__, "fourstates.svg")
end

function fig3()
    @svg begin
        background("white")
        unit = 50
        m, n = 7, 7
        for k=1:2
            origin((k-1)*400, 0)
            # create a grid
            grid = showgrid(m, n; unit)

            # create two plaquettes
            if k == 1
                showxpath(xzpath([grid[i, 3] for i in 1:m])...; closed=true)
                xblock, xlines = xzpath([grid[i, j] for (i, j) in [(5, 3), (5, 4), (6, 4), (6, 3), (5, 3)]])
                showxpath(xblock, xlines; closed=true)
                @layer begin
                    fontsize(22)
                    text("B", midpoint(xblock[1], xblock[3]))
                end
                showxznode(xblock[4]; label="X²", color="#6666CC")
            end
            if k == 2
                create_xpath(grid, (1, 3), [R, R, R, R, D, R, U, R])
            end
        end
        fontsize(60)
        setcolor("black")
        text("=", 0, 200; valign=:middle, halign=:center)
    end 800 400 joinpath(@__DIR__, "fig3.svg")
end

function fig4()
    @svg begin
        background("white")
        unit = 50
        m, n = 7, 7
        for k=1:2
            origin((k-1)*400, 0)
            # create a grid
            grid = showgrid(m, n; unit)

            # create two plaquettes
            if k == 1
                znodes, = create_zpath(grid, (2, 3), [R])
                create_xpath(grid, (2, 4), [R, R])
                @layer begin
                    fontsize(16)
                    text("e₁", offset(grid[2, 4], (-unit/4, -unit/4)))
                    text("m₁", offset(grid[2, 3], (-unit/2, -3unit/4)))
                    text("e₂", offset(grid[4, 4], (-unit/4, -unit/4)))
                    text("m₂", offset(grid[3, 3], (-unit/2, -3unit/4)))
                end
            end
            if k == 2
                znodes, = create_zpath(grid, (2, 3), [R, R, R, R])
                create_xpath(grid, (2, 4), [R, R, R, R, U, U, L, L, D, D])
                @layer begin
                    showxznode(znodes[3]; label="±", color="#CC66CC")
                    fontsize(16)
                    setcolor("black")
                    text("m₂'", offset(grid[6, 3], (-unit/2, -3unit/4)))
                end
            end
        end
        fontsize(30)
        setcolor("black")
        text("initial", -200, 370; valign=:middle, halign=:center)
        text("final", 200, 370; valign=:middle, halign=:center)
    end 800 400 joinpath(@__DIR__, "fig4.svg")
end

function fig5()
    @svg begin
        background("white")
        unit = 50
        m, n = 7, 7
        for k=1:2
            origin((k-1)*400, 0)
            # create a grid
            grid = showgrid(m, n; unit)

            # create two lines
            if k == 1
                showxpath(xzpath([grid[i, 3] for i in 1:m])...; closed=true)
                showzpath(xzpath([offset(grid[3, i], (-unit/2, -unit/2)) for i in 1:m])...; closed=true)
            end
            if k == 2
                showxpath(xzpath([grid[3, i] for i in 1:m])...; closed=true)
                #showxpath(xzpath([grid[3, i] for i in 1:m])...; closed=true)
                create_zpath(grid, (4, 7), [U, L, U, U,R,U, U, U])
            end
        end
    end 800 400 joinpath(@__DIR__, "fig5.svg")
end

