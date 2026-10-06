"""
    WriteVTKExt

Export VoronoiMeshes.jl meshes to VTU (VTK Unstructured Grid) format.
- Provides functions to export Voronoi and Delaunay meshes with periodic ghost vertices.
- Uses WriteVTK for file output; supports mesh metadata and ghost marking.
- Includes both specific and generic mesh export helpers for flexible usage.
"""
module WriteVTKExt

using VoronoiMeshes, TensorsLite, TensorsLiteGeometry, Zeros, SmallCollections, LinearAlgebra
import VoronoiMeshes: save_triangulation_to_vtu, save_voronoi_to_vtu

using PrecompileTools

using WriteVTK # For saving meshes in VTU format
using WriteVTK.VTKBase


"""
    save_voronoi_to_vtu(file_name::String, mesh::AbstractVoronoiMesh{false})

Export the Voronoi mesh to a VTU file, handling periodic ghost vertices.
"""
function save_voronoi_to_vtu(file_name::String, mesh::AbstractVoronoiMesh{false}, fields::NamedTuple = ())

    # Here the vertices are Voronoi cell vertices and the polygons are the Voronoi cells
    vertices_with_ghosts, verticesOnPolygon_with_ghosts, n_ghosts, ghost_dict, vertex_fields_indices =
        create_ghost_periodic_voronoi_vertices(mesh)

    n_polys = mesh.cells.n #number of Voronoi cells
    n_vertices = mesh.vertices.n #number of Voronoi cell vertices (matches number of triangles)

    points_mat = hcat((v[1:2] for v in vertices_with_ghosts)...)

    cells = Vector{MeshCell}(undef, n_polys)

    for i in 1:n_polys
        links = collect(verticesOnPolygon_with_ghosts[i])
        cells[i] = MeshCell(VTKCellTypes.VTK_POLYGON, links)
    end
    # Save ghost periodicity information

    # Vertices from 1:n_polys will save their index 1:n_polys
    # Ghost vertices will save the index of the original vertex they correspond to
    n_vertices_with_ghosts = length(vertices_with_ghosts)

    ghost_idx = collect(1:n_vertices_with_ghosts)

    for i in 1:n_vertices
        if haskey(ghost_dict, i)
            for j in ghost_dict[i]
                ghost_idx[j] = i
            end
        end
    end

    # Shift indices by -1 for storage convention (0-based in VTK)
    @inbounds ghost_idx .-= 1

    # add mesh metadata
    saved_file = vtk_grid(file_name, points_mat, cells) do vtk
        vtk["Index", VTKPointData()] = ghost_idx
        vtk["NumCells"] = mesh.cells.n
        vtk["NumVertices"] = mesh.vertices.n
        vtk["NumEdges"] = mesh.edges.n
        vtk["XPeriod"] = mesh.x_period
        vtk["YPeriod"] = mesh.y_period
        vtk["NumPeriodicGhosts"] = n_ghosts
        write_to_voronoi_vtu!(vtk, mesh, vertex_fields_indices, fields)
    end

    return saved_file
end

write_to_voronoi_vtu!(vtk, mesh::AbstractVoronoiMesh{false}, cIdx, fields::@NamedTuple{}) = nothing

function write_to_voronoi_vtu!(vtk, mesh::AbstractVoronoiMesh{false}, vIdx, fields::NamedTuple{nameT, Tuple{T}}) where {nameT, T<:AbstractVector}
    data = fields[1]
    name = string(nameT[1])
    ld = length(data)
    if ld == mesh.vertices.n
        vtk[name, VTKPointData()] = transform_data(data, vIdx)
    elseif ld == mesh.cells.n
        vtk[name, VTKCellData()] = transform_data(data)
    end
end

function write_to_voronoi_vtu!(vtk, mesh::AbstractVoronoiMesh{false}, cIdx, fields::NamedTuple)
    write_to_voronoi_vtu!(vtk, mesh, cIdx, Base.front(fields))
    write_to_voronoi_vtu!(vtk, mesh, cIdx, Base.tail(fields))
end

"""
    save_triangulation_to_vtu(file_name::String, mesh::AbstractVoronoiMesh{false})

Export the Delaunay triangulation (dual mesh) to a VTU file, handling periodic ghost vertices.
"""
function save_triangulation_to_vtu(file_name::String, mesh::AbstractVoronoiMesh{false}, fields::NamedTuple = NamedTuple())

    # Here the vertices are cell centers and the polygons are the triangles
    vertices_with_ghosts, verticesOnPolygon_with_ghosts, n_ghosts, ghost_dict, cell_field_indices =
        create_ghost_periodic_triangulation_vertices(mesh)

    n_polys = mesh.vertices.n #number of mesh triangles
    n_vertices = mesh.cells.n #number of triangle vertices

    points_mat = hcat((v[1:2] for v in vertices_with_ghosts)...)

    cells = Vector{MeshCell}(undef, n_polys)

    for i in 1:n_polys
        links = collect(verticesOnPolygon_with_ghosts[i])
        cells[i] = MeshCell(VTKCellTypes.VTK_TRIANGLE, links)
    end

    # Save ghost periodicity information

    # Vertices from 1:n_polys will save their index 1:n_polys
    # Ghost vertices will save the index of the original vertex they correspond to
    n_vertices_with_ghosts = length(vertices_with_ghosts)

    ghost_idx = collect(1:n_vertices_with_ghosts)

    for i in 1:n_vertices
        if haskey(ghost_dict, i)
            for j in ghost_dict[i]
                ghost_idx[j] = i
            end
        end
    end

    # Shift indices by -1 for storage convention (VTK uses 0-based indexing)
    ghost_idx .-= 1

    # add mesh metadata
    saved_file = vtk_grid(file_name, points_mat, cells) do vtk
        vtk["Index", VTKPointData()] = ghost_idx
        vtk["NumCells"] = mesh.cells.n
        vtk["NumVertices"] = mesh.vertices.n
        vtk["NumEdges"] = mesh.edges.n
        vtk["XPeriod"] = mesh.x_period
        vtk["YPeriod"] = mesh.y_period
        vtk["NumPeriodicGhosts"] = n_ghosts
        write_to_triangulation_vtu!(vtk, mesh, cell_field_indices, fields)
    end

    return saved_file
end

write_to_triangulation_vtu!(vtk, mesh::AbstractVoronoiMesh{false}, cIdx, fields::@NamedTuple{}) = nothing

function write_to_triangulation_vtu!(vtk, mesh::AbstractVoronoiMesh{false}, cIdx, fields::NamedTuple{nameT, Tuple{T}}) where {nameT, T<:AbstractVector}
    data = fields[1]
    ld = length(data)
    name = string(nameT[1])
    if ld == mesh.cells.n
        vtk[name, VTKPointData()] = transform_data(data, cIdx)
    elseif ld == mesh.vertices.n
        vtk[name, VTKCellData()] = transform_data(data)
    end
end

function write_to_triangulation_vtu!(vtk, mesh::AbstractVoronoiMesh{false}, cIdx, fields::NamedTuple)
    write_to_triangulation_vtu!(vtk, mesh, cIdx, Base.front(fields))
    write_to_triangulation_vtu!(vtk, mesh, cIdx, Base.tail(fields))
end

transform_data(data::AbstractVector) = data
transform_data(data::AbstractVector, Idx) = view(data, Idx)

transform_data(data::VecArray) = (data.x, data.y)
transform_data(data::VecArray, Idx) = (view(data.x, Idx), view(data.y, Idx))

end # module VTKExt
