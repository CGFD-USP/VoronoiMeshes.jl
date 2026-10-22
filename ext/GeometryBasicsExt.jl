module GeometryBasicsExt

using VoronoiMeshes
using LinearAlgebra: normalize
using TensorsLite: Vec
using TensorsLiteGeometry, SmallCollections
using GeometryBasics: GeometryBasics, Polygon, Point2, Point2f, MultiPolygon

const PolType = Polygon{2, Float32}
const MultiPolType = MultiPolygon{2, Float32}

@inline _capacity(::Type{<:SmallVector{N}}) where {N} = N
@inline _capacity(::Type{<:FixedVector{N}}) where {N} = N

function create_polygons_periodic(vert_pos, polygon_pos, verticesOnPolygon, x_period, y_period)

    polygons = Vector{PolType}(undef, length(polygon_pos))
    NE = _capacity(eltype(verticesOnPolygon))
    @parallel for i in eachindex(polygon_pos)
        @inbounds begin
            ppos = polygon_pos[i]
            local_vertices = SmallVector{NE, Point2f}()
            for i_v in verticesOnPolygon[i]
                vpos = closest(ppos, vert_pos[i_v], x_period, y_period)
                local_vertices = @inbounds push(local_vertices, Point2f(vpos.x, vpos.y))
            end
            polygons[i] = Polygon(Array(local_vertices))
        end
    end

    return polygons
end

function VoronoiMeshes.create_cell_polygons(mesh::AbstractVoronoiMesh{false})
    return create_polygons_periodic(mesh.vertices.position, mesh.cells.position, mesh.cells.vertices, mesh.x_period, mesh.y_period)
end

function VoronoiMeshes.create_dual_triangles(mesh::AbstractVoronoiMesh{false})
    return create_polygons_periodic(mesh.cells.position, mesh.vertices.position, mesh.vertices.cells, mesh.x_period, mesh.y_period)
end

@inline function _closest_lonlat(c_lon, lon, lat)
    lon_aux = rad2deg(lon)
    lon_aux = lon_aux > 180 ? lon_aux - 360 : lon_aux
    return closest(c_lon, lon_aux), rad2deg(lat)
end

function subdivide_spherical_polygon_edges(lon, lat, base_lon)
    minlon = 2

    c_lon = rad2deg(base_lon)
    c_lon = c_lon > 180 ? c_lon - 360 : c_lon

    n = length(lon)
    local_vertices = Point2f[]

    for i in 1:n
        i_next = i == n ? 1 : i + 1

        lon1, lat1 = _closest_lonlat(c_lon, lon[i], lat[i])
        lon2, lat2 = _closest_lonlat(c_lon, lon[i_next], lat[i_next])

        push!(local_vertices, Point2f(lon1, lat1))

        difflon = abs(lon2 - lon1)

        if difflon > minlon
            nparts = div(difflon, minlon) + 2

            p1 = lonlat_to_position(1, deg2rad(lon1), deg2rad(lat1))
            p2 = lonlat_to_position(1, deg2rad(lon2), deg2rad(lat2))

            for l in 2:(nparts - 1)
                w = (nparts - l) / (nparts - 1)
                p = normalize(w * p1 + (1 - w) * p2)

                lon_p, lat_p = position_to_lonlat(p)
                lon_p, lat_p = _closest_lonlat(c_lon, lon_p, lat_p)

                push!(local_vertices, Point2f(lon_p, lat_p))
            end
        end
    end

    return polygon_to_multipoly(Polygon(local_vertices))
end

function create_polygons_sphere(vert_lon::Vector{T}, vert_lat,
                                base_lon,
                                edge_lon, edge_lat,
                                verticesOnElement, edgesOnElement) where {T <: Number}

    polygons = Vector{MultiPolType}(undef, length(base_lon))

    @parallel for i in eachindex(base_lon)
        @inbounds begin
            vertices = verticesOnElement[i]
            edges = edgesOnElement[i]

            lon = promote_type(eltype(vert_lon), eltype(edge_lon))[]
            lat = promote_type(eltype(vert_lat), eltype(edge_lat))[]

            push!(lon, vert_lon[vertices[end]])
            push!(lat, vert_lat[vertices[end]])

            for j in eachindex(edges)
                e = edges[j]
                v = vertices[j]

                push!(lon, edge_lon[e], vert_lon[v])
                push!(lat, edge_lat[e], vert_lat[v])
            end

            pop!(lon)
            pop!(lat)

            polygons[i] = subdivide_spherical_polygon_edges(lon, lat, base_lon[i])
        end
    end

    return polygons
end

function create_polygons_sphere(vert_lon::Vector{T}, vert_lat,
                                base_lon, verticesOnElement) where {T <: Number}

    polygons = Vector{MultiPolType}(undef, length(base_lon))

    @parallel for i in eachindex(base_lon)
        @inbounds begin
            vertices = verticesOnElement[i]

            lon = vert_lon[vertices]
            lat = vert_lat[vertices]

            polygons[i] = subdivide_spherical_polygon_edges(lon, lat, base_lon[i])
        end
    end

    return polygons
end

#From ChatGPT
function clip_polygon_x(vertices::AbstractVector{<:Point2}, boundary::Float32, side::Symbol)

    inside(x) = side === :le ? x <= boundary : x >= boundary

    result = Point2f[]

    n = length(vertices)

    for i in 1:n
        current = vertices[i]
        previous = vertices[mod1(i - 1, n)]

        ic = inside(current[1])
        ip = inside(previous[1])

        if ic
            if !ip
                # Entering the clipping region.
                t = (boundary - previous[1]) /
                    (current[1] - previous[1])

                y = muladd(t, current[2] - previous[2], previous[2])

                push!(result, Point2f(boundary, y))
            end

            push!(result, current)

        elseif ip
            # Leaving the clipping region.
            t = (boundary - previous[1]) /
                (current[1] - previous[1])

            y = muladd(t, current[2] - previous[2], previous[2])

            push!(result, Point2f(boundary, y))
        end
    end

    return result
end

#From ChatGPT
function polygon_to_multipoly(p::PolType)
    vertices = p.exterior

    # Determine whether the polygon extends beyond either boundary.
    minlon = minimum(p[1] for p in vertices)
    maxlon = maximum(p[1] for p in vertices)

    # No crossing: just return the polygon as a one-component MultiPolygon.
    if minlon >= -180f0 && maxlon <= 180f0
        return MultiPolType([p])
    end

    polygons = PolType[]

    if maxlon > 180f0
        # Part on the normal [-180, 180] domain.
        p1 = clip_polygon_x(vertices, 180f0, :le)
        !isempty(p1) && push!(polygons, Polygon(p1))

        # Part beyond +180, shifted by -360.
        shifted = Point2f[
            Point2f(v[1] - 360f0, v[2])
            for v in vertices
        ]

        p2 = clip_polygon_x(shifted, -180f0, :ge)
        !isempty(p2) && push!(polygons, Polygon(p2))

    elseif minlon < -180f0
        # Part on the normal [-180, 180] domain.
        p1 = clip_polygon_x(vertices, -180f0, :ge)
        !isempty(p1) && push!(polygons, Polygon(p1))

        # Part beyond -180, shifted by +360.
        shifted = Point2f[
            Point2f(v[1] + 360f0, v[2])
            for v in vertices
        ]

        p2 = clip_polygon_x(shifted, 180f0, :le)
        !isempty(p2) && push!(polygons, Polygon(p2))
    end

    return MultiPolType(polygons)
end

function VoronoiMeshes.create_cell_polygons(mesh::AbstractVoronoiMesh{true})
    return create_polygons_sphere(mesh.vertices.longitude, mesh.vertices.latitude,
                                  mesh.cells.longitude,
                                  mesh.edges.longitude, mesh.edges.latitude,
                                  mesh.cells.vertices, mesh.cells.edges)
end

function VoronoiMeshes.create_dual_triangles(mesh::AbstractVoronoiMesh{true})
    return create_polygons_sphere(mesh.cells.longitude, mesh.cells.latitude,
                                  mesh.vertices.longitude,
                                  mesh.edges.longitude, mesh.edges.latitude,
                                  mesh.vertices.cells, mesh.vertices.edges)
end

function create_edge_quadrilaterals_periodic(edge_pos, vert_pos, cell_pos, verticesOnEdge, cellsOnEdge, x_period, y_period)

    edge_quadrilaterals = Vector{PolType}(undef, length(edge_pos))

    @parallel for i in eachindex(edge_pos)
        @inbounds begin
            epos = edge_pos[i]
            iv1, iv2 = verticesOnEdge[i]
            ic1, ic2 = cellsOnEdge[i]

            cpos1 = closest(epos, cell_pos[ic1], x_period, y_period)

            vpos1 = closest(epos, vert_pos[iv1], x_period, y_period)

            cpos2 = closest(epos, cell_pos[ic2], x_period, y_period)

            vpos2 = closest(epos, vert_pos[iv2], x_period, y_period)

            edge_quadrilaterals[i] = Polygon([Point2f(cpos1.x, cpos1.y), Point2f(vpos1.x, vpos1.y), Point2f(cpos2.x, cpos2.y), Point2f(vpos2.x, vpos2.y)])
        end
    end

    return edge_quadrilaterals
end

function VoronoiMeshes.create_edge_quadrilaterals(mesh::AbstractVoronoiMesh{false})
    return create_edge_quadrilaterals_periodic(mesh.edges.position, mesh.vertices.position, mesh.cells.position, mesh.edges.vertices, mesh.edges.cells, mesh.x_period, mesh.y_period)
end

function create_edge_quadrilaterals_spherical(edge_lon,
                                              vert_lon, vert_lat,
                                              cell_lon, cell_lat,
                                              verticesOnEdge, cellsOnEdge)

    edge_quadrilaterals = Vector{MultiPolType}(undef, length(edge_lon))

    @parallel for e in eachindex(edge_lon)
        @inbounds begin
            iv1, iv2 = verticesOnEdge[e]
            ic1, ic2 = cellsOnEdge[e]

            lon = Vector{eltype(vert_lon)}(undef, 4)
            lat = Vector{eltype(vert_lat)}(undef, 4)

            lon[1] = cell_lon[ic1]
            lon[2] = vert_lon[iv1]
            lon[3] = cell_lon[ic2]
            lon[4] = vert_lon[iv2]

            lat[1] = cell_lat[ic1]
            lat[2] = vert_lat[iv1]
            lat[3] = cell_lat[ic2]
            lat[4] = vert_lat[iv2]


            edge_quadrilaterals[e] = subdivide_spherical_polygon_edges(lon, lat, edge_lon[e])
        end
    end

    return edge_quadrilaterals
end

function VoronoiMeshes.create_edge_quadrilaterals(mesh::AbstractVoronoiMesh{true})
    return create_edge_quadrilaterals_spherical(mesh.edges.longitude, mesh.vertices.longitude, mesh.vertices.latitude, mesh.cells.longitude, mesh.cells.latitude, mesh.edges.vertices, mesh.edges.cells)
end

"""
        create_polygon_linesegments_periodic(vert_pos, edge_pos, polygon_pos, edgesOnPolygon, verticesOnEdge, x_period, y_period)

Build line-segment coordinate lists for polygons while taking periodic
domain into account. This helper is used by the exported wrappers
`create_cell_linesegments` and `create_dual_triangles_linesegments`.

Arguments
- `vert_pos` : array-like of vertex positions (objects with `.x, .y` fields)
- `edge_pos` : array-like of edge midpoints/positions (with `.x, .y`)
- `polygon_pos` : array-like of polygon centers (used as reference to pick
    the closest periodic images)
- `edgesOnPolygon` : topology mapping polygon -> list of incident edges
- `verticesOnEdge` : topology mapping edge -> tuple/list of its two vertices
- `x_period, y_period` : periodic domain lengths (used by `closest` to
    compute wrapped images)

Returns
- A tuple `((x, y), (x_periodic, y_periodic))` where:
    - `(x, y)` are vectors of x/y coordinates for interior (non-wrapping)
        line-segment endpoints (interleaved pairs represent segment endpoints)
    - `(x_periodic, y_periodic)` are vectors of coordinates for segments
        that cross periodic boundaries (wrapped pieces)

Notes
- The function avoids duplicate segments by tracking processed edge
    positions via a `Set`. It uses `closest(...)` to select the image of
    vertices/edges nearest the polygon center so that segments are drawn
    consistently across periodic boundaries.
"""
function create_polygon_linesegments_periodic(vert_pos, edge_pos, polygon_pos, edgesOnPolygon, verticesOnEdge, x_period, y_period)
    x = eltype(edge_pos.x)[]
    y = eltype(edge_pos.y)[]
    nEdges = length(verticesOnEdge)
    sizehint!(x, 2 * nEdges)
    sizehint!(y, 2 * nEdges)
    x_periodic = eltype(edge_pos.x)[]
    y_periodic = eltype(edge_pos.y)[]
    sizehint!(x_periodic, nEdges ÷ 2)
    sizehint!(y_periodic, nEdges ÷ 2)

    touched_edges_pos = Set{eltype(edge_pos)}()

    @inbounds for i in eachindex(edgesOnPolygon)
        c_pos = polygon_pos[i]
        edges_ind = edgesOnPolygon[i]

        for j in edges_ind
            e_pos = edge_pos[j]
            closest_e_pos = closest(c_pos, e_pos, x_period, y_period)
            if !(closest_e_pos in touched_edges_pos)
                p = verticesOnEdge[j]

                is_interior = closest_e_pos == e_pos
                for k in p
                    v_pos = vert_pos[k]
                    closest_v_pos = closest(closest_e_pos, v_pos, x_period, y_period)
                    if (is_interior)
                        push!(x, closest_v_pos.x)
                        push!(y, closest_v_pos.y)
                    else
                        push!(x_periodic, closest_v_pos.x)
                        push!(y_periodic, closest_v_pos.y)
                    end
                end

                push!(touched_edges_pos, closest_e_pos)
            end

        end

    end

    return ((x, y), (x_periodic, y_periodic))
end

function VoronoiMeshes.create_cell_linesegments(mesh::AbstractVoronoiMesh{false})
    return create_polygon_linesegments_periodic(mesh.vertices.position, mesh.edges.position, mesh.cells.position, mesh.cells.edges, mesh.edges.vertices, mesh.x_period, mesh.y_period)
end

function VoronoiMeshes.create_dual_triangles_linesegments(mesh::AbstractVoronoiMesh{false})
    return create_polygon_linesegments_periodic(mesh.cells.position, mesh.edges.position, mesh.vertices.position, mesh.vertices.edges, mesh.edges.cells, mesh.x_period, mesh.y_period)
end

function create_diagram_linesegments_periodic(vert_pos, cell_pos, verticesOnCell, cellsOnVertex, x_period, y_period)
    x = eltype(vert_pos.x)[]
    y = eltype(vert_pos.y)[]
    nEdges = length(verticesOnCell) + length(cellsOnVertex)
    sizehint!(x, 2 * nEdges)
    sizehint!(y, 2 * nEdges)

    touched_edges_pos = Set{eltype(vert_pos)}()

    @inbounds for i in eachindex(verticesOnCell)
        c_pos = cell_pos[i]
        vertices_ind = verticesOnCell[i]

        L = length(vertices_ind)
        for j in Base.OneTo(L)
            j
            jp1 = j + 1
            j2 = ifelse(jp1 > L, jp1 - L, jp1)
            v1 = vertices_ind[j]
            v2 = vertices_ind[j2]
            v1_pos = closest(c_pos, vert_pos[v1], x_period, y_period)
            v2_pos = closest(c_pos, vert_pos[v2], x_period, y_period)
            e_pos = (v1_pos + v2_pos) / 2
            if !(e_pos in touched_edges_pos)
                push!(x, v1_pos.x)
                push!(x, v2_pos.x)
                push!(y, v1_pos.y)
                push!(y, v2_pos.y)
                push!(touched_edges_pos, e_pos)
            end
        end
    end
    return (x, y)
end

function VoronoiMeshes.create_diagram_linesegments(diagram::VoronoiDiagram{false})
    return create_diagram_linesegments_periodic(diagram.vertices, diagram.generators, diagram.verticesOnCell, diagram.cellsOnVertex, diagram.x_period, diagram.y_period)
end

end

