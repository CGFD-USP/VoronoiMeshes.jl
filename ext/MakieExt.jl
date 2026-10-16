module MakieExt

using VoronoiMeshes
import VoronoiMeshes: plotmesh, plotmesh!, plotdiagram, plotdiagram!, plotdualmesh, plotdualmesh!
import Makie as plt

plt.@recipe(PlotMesh, mesh) do scene
    plt.Theme(
        color = :deepskyblue3,
        edgelinestyle = :solid,
        ghostedgelinestyle = :dash,
        linewidth = 1,
    )
end

const PeriodicMesh = PlotMesh{<:Tuple{<:AbstractVoronoiMesh{false}}}

function plt.plot!(plot::PeriodicMesh)
    graph = plot.attributes

    plt.map!(
        VoronoiMeshes.create_cell_linesegments,
        graph,
        :mesh,
        :both_edges,
    )

    plt.map!(graph, :both_edges, :x_edges) do edges
        edges[1][1]
    end

    plt.map!(graph, :both_edges, :y_edges) do edges
        edges[1][2]
    end

    plt.map!(graph, :both_edges, :x_ghost_edges) do edges
        edges[2][1]
    end

    plt.map!(graph, :both_edges, :y_ghost_edges) do edges
        edges[2][2]
    end

    plt.linesegments!(
        plot,
        plot.x_edges,
        plot.y_edges,
        color = plot.color,
        linestyle = plot.edgelinestyle,
        linewidth = plot.linewidth,
    )

    plt.linesegments!(
        plot,
        plot.x_ghost_edges,
        plot.y_ghost_edges,
        color = plot.color,
        linestyle = plot.ghostedgelinestyle,
        linewidth = plot.linewidth,
    )

    return plot
end

const SphericalMesh = PlotMesh{<:Tuple{<:AbstractVoronoiMesh{true}}}

function plt.plot!(plot::SphericalMesh)
    graph = plot.attributes

    plt.map!(
        VoronoiMeshes.create_cell_polygons,
        graph,
        :mesh,
        :cells,
    )

    plt.poly!(
        plot,
        plot.cells,
        color = :transparent,
        strokecolor = plot.color,
        linestyle = plot.edgelinestyle,
        strokewidth = plot.linewidth,
    )

    return plot
end

plt.@recipe(PlotDiagram, diagram) do scene
    plt.Theme(
        color = :deepskyblue3,
        linestyle = :solid,
        linewidth = 1,
    )
end

const PeriodicDiagram = PlotDiagram{<:Tuple{<:VoronoiDiagram{false}}}

function plt.plot!(plot::PeriodicDiagram)
    graph = plot.attributes

    plt.map!(
        VoronoiMeshes.create_diagram_linesegments,
        graph,
        :diagram,
        :edges,
    )

    plt.map!(graph, :edges, :x_edges) do edges
        edges[1]
    end

    plt.map!(graph, :edges, :y_edges) do edges
        edges[2]
    end

    plt.linesegments!(
        plot,
        plot.x_edges,
        plot.y_edges,
        color = plot.color,
        linestyle = plot.linestyle,
        linewidth = plot.linewidth,
    )

    return plot
end

const SphericalDiagram = PlotDiagram{<:Tuple{<:VoronoiDiagram{true}}}

function plt.plot!(plot::SphericalDiagram)
    graph = plot.attributes

    plt.map!(
        VoronoiMeshes.create_cell_polygons,
        graph,
        :mesh,
        :cells,
    )

    plt.poly!(
        plot,
        plot.cells,
        color = :transparent,
        strokecolor = plot.color,
        linestyle = plot.linestyle,
        strokewidth = plot.linewidth,
    )

    return plot
end

plt.@recipe(PlotDualMesh, mesh) do scene
    plt.Theme(
        color = :darkorange2,
        edgelinestyle = :solid,
        ghostedgelinestyle = :dash,
        linewidth = 1,
    )
end

const PeriodicDualMesh = PlotDualMesh{<:Tuple{<:AbstractVoronoiMesh{false}}}

function plt.plot!(plot::PeriodicDualMesh)
    graph = plot.attributes

    plt.map!(
        VoronoiMeshes.create_dual_triangles_linesegments,
        graph,
        :mesh,
        :both_edges,
    )

    plt.map!(graph, :both_edges, :x_edges) do edges
        edges[1][1]
    end

    plt.map!(graph, :both_edges, :y_edges) do edges
        edges[1][2]
    end

    plt.map!(graph, :both_edges, :x_ghost_edges) do edges
        edges[2][1]
    end

    plt.map!(graph, :both_edges, :y_ghost_edges) do edges
        edges[2][2]
    end

    plt.linesegments!(
        plot,
        plot.x_edges,
        plot.y_edges,
        color = plot.color,
        linestyle = plot.edgelinestyle,
        linewidth = plot.linewidth,
    )

    plt.linesegments!(
        plot,
        plot.x_ghost_edges,
        plot.y_ghost_edges,
        color = plot.color,
        linestyle = plot.ghostedgelinestyle,
        linewidth = plot.linewidth,
    )

    return plot
end

const SphericalDualMesh = PlotDualMesh{<:Tuple{<:AbstractVoronoiMesh{true}}}

function plt.plot!(plot::SphericalDualMesh)
    graph = plot.attributes

    plt.map!(
        VoronoiMeshes.create_dual_triangles,
        graph,
        :mesh,
        :triangles,
    )

    plt.poly!(
        plot,
        plot.triangles,
        color = :transparent,
        strokecolor = plot.color,
        linestyle = plot.edgelinestyle,
        strokewidth = plot.linewidth,
    )

    return plot
end

end
