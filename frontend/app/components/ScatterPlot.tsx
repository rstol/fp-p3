import * as d3 from 'd3';
import { Circle, GrabIcon, Info, Minus, Move, Plus, RefreshCcw, ZoomIn } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useLoaderData, useNavigation, useSearchParams } from 'react-router';
import type { clientLoader } from '~/routes/_index';
import type { Point } from '~/types/data';
import Filters from './Filters';
import { ScatterPlotSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { useDashboardStore } from '~/lib/stateStore';
import { getPlayId } from '~/lib/utils';

const defaultDimensions = { width: 500, height: 400 };

const getZoomMethod =
  (svgRef: React.RefObject<SVGSVGElement | null>, method: string) =>
  (...args: any[]) => {
    const svg = svgRef?.current;
    if (svg && typeof (svg as any)[method] === 'function') {
      (svg as any)[method](...args);
    }
  };

function Legend({
  clusters,
  color,
}: {
  clusters: string[];
  color: d3.ScaleOrdinal<string, string>;
}) {
  return (
    <div className="absolute top-2 right-2 z-10 flex p-1.5">
      <div className="space-x-1">
        {clusters.map((cluster, index) => (
          <Button
            key={index}
            size="sm"
            className="legendItem hover:bg-opacity-100 gap-1"
            variant="secondary"
            value={cluster}
            style={{ color: color(String(cluster)) }}
          >
            <Circle fill={color(String(cluster))} width={12} height={12} />
            Cluster {cluster}
          </Button>
        ))}
      </div>
    </div>
  );
}

function ZoomControls({ svgRef }: { svgRef: React.RefObject<SVGSVGElement | null> }) {
  return (
    <div className="absolute right-3 bottom-3 z-10 flex justify-center space-x-2">
      <Button
        size="sm"
        variant="secondary"
        className="rounded-xs bg-gray-200 px-1.5! hover:bg-gray-100"
        onClick={getZoomMethod(svgRef, 'zoomIn')}
      >
        <Plus className="size-3.5" />
      </Button>
      <Button
        size="sm"
        variant="secondary"
        className="rounded-xs bg-gray-200 px-1.5! hover:bg-gray-100"
        onClick={getZoomMethod(svgRef, 'zoomOut')}
      >
        <Minus className="size-3.5" />
      </Button>
      <Button
        size="sm"
        variant="secondary"
        className="rounded-xs bg-gray-200 px-1.5! hover:bg-gray-100"
        onClick={getZoomMethod(svgRef, 'zoomReset')}
      >
        <RefreshCcw className="size-3.5" />
      </Button>
    </div>
  );
}

function InfoBar() {
  return (
    <div className="flex items-center justify-between border border-t-0 border-r-0 border-gray-200 bg-gray-50 p-2">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-medium">Basketball play positions clustered by similarity</h3>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button className="rounded-full p-1 hover:bg-gray-200">
                <Info className="h-4 w-4 text-gray-500" />
              </button>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <p>Different colors represent different types of plays.</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Control hints */}
      <div className="flex items-center gap-3">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <GrabIcon className="h-3 w-3" />
                <span>Drag points</span>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>Drag points to reassign clusters</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <Move className="h-3 w-3" />
                <span>Pan</span>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>Drag the background to pan</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <ZoomIn className="h-3 w-3" />
                <span>Zoom</span>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>Use mouse wheel to zoom</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}

const ScatterPlot = ({ teamID }: { teamID: string }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const loaderData = useLoaderData<typeof clientLoader>();
  const data = loaderData?.scatterData ?? [];
  const selectedPlay = useDashboardStore((state) => state.selectedPlay);
  const updatePlay = useDashboardStore((state) => state.updatePlay);
  const tooltipRef = useRef<d3.Selection<HTMLDivElement, unknown, HTMLElement, undefined> | null>(
    null,
  );
  const color = d3.scaleOrdinal(d3.schemeCategory10);
  // Main rendering effect when data or dimensions change
  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const margin = 30;

    // Set up scales with original domains
    const xExtent = d3.extent(data, (d) => d.x) as [number, number];
    const yExtent = d3.extent(data, (d) => d.y) as [number, number];

    const xScale = d3
      .scaleLinear()
      .domain(xExtent)
      .range([margin, defaultDimensions.width - margin]);

    const yScale = d3
      .scaleLinear()
      .domain(yExtent)
      .range([defaultDimensions.height - margin, margin]);

    // Clear the SVG
    svg.selectAll('*').remove();

    // Add background rect for zooming/panning
    svg
      .append('rect')
      .attr('width', defaultDimensions.width)
      .attr('height', defaultDimensions.height)
      .attr('fill', 'oklch(0.985 0.002 247.839)') // Light grey background
      .attr('cursor', 'move');

    // Create a group for all visualization content
    const container = svg.append('g');

    // Function to apply current transform to the container
    function zoomed({ transform }: d3.D3ZoomEvent<Element, unknown>) {
      container.attr('transform', transform as any);
      // Keep point radius and stroke width constant
      container
        .selectAll('circle')
        .attr('r', function (d: any) {
          const isSelected = selectedPlay && getPlayId(selectedPlay) === getPlayId(d);
          return isSelected ? 8 / transform.k : 5 / transform.k;
        })
        .attr('stroke', function (d: any) {
          const isSelected = selectedPlay && getPlayId(selectedPlay) === getPlayId(d);
          return isSelected ? 'black' : 'white';
        })
        .attr('stroke-width', function (d: any) {
          const isSelected = selectedPlay && getPlayId(selectedPlay) === getPlayId(d);
          return isSelected ? 2 / transform.k : 1 / transform.k;
        });
      container.selectAll('path').attr('stroke-width', 0.8 / transform.k); // contour
    }

    // Set up zoom behavior
    const zoom = d3.zoom().scaleExtent([1, 50]).on('zoom', zoomed);
    svg.call(zoom as any).on('dblclick.zoom', null);

    // Apply zoom behavior to svg but filter so it only works on background or when using mousewheel
    function reset() {
      svg
        .transition()
        .duration(750)
        .call(zoom.transform as any, d3.zoomIdentity);
    }

    Object.assign(svg.node() as {}, {
      zoomIn: () => svg.transition().call(zoom.scaleBy as any, 2),
      zoomOut: () => svg.transition().call(zoom.scaleBy as any, 0.5),
      zoomReset: reset,
    });

    function zoomIntoCluster(cluster: string) {
      const clusterPoints = data.filter((d) => String(d.cluster) === cluster);

      // Get extent of cluster points in x and y dimensions
      const xExtend = d3.extent(clusterPoints, (d) => d.x);
      const yExtent = d3.extent(clusterPoints, (d) => d.y);
      if (!xExtend[0] || !yExtent[0]) return;
      const [x0, x1] = xExtend.map(xScale);
      const [y1, y0] = yExtent.map(yScale);

      // Calculate the zoom parameters (scale and translate)
      const k =
        0.9 * Math.min(defaultDimensions.width / (x1 - x0), defaultDimensions.height / (y1 - y0));
      const tx = (defaultDimensions.width - k * (x0 + x1)) / 2;
      const ty = (defaultDimensions.height - k * (y0 + y1)) / 2;

      // Create the transform and animate to it
      const transform = d3.zoomIdentity.translate(tx, ty).scale(k);

      // Apply the transform with a smooth transition
      svg
        .transition()
        .duration(750)
        .call(zoom.transform as any, transform);
    }

    d3.selectAll('.legendItem').on('click', (event) => {
      const cluster = event.target.value;
      if (cluster) zoomIntoCluster(cluster);
    });

    // Function to handle point dragging
    const handlePointDrag = (selection: d3.Selection<any, Point, any, any>) => {
      const drag = d3
        .drag<SVGCircleElement, Point>()
        .on('start', function (event, d) {
          // Prevent zoom during drag
          event.sourceEvent.stopPropagation();
          d3.select(this)
            .raise()
            .attr('r', 7)
            .attr('stroke-width', 2)
            .attr('stroke', 'black')
            .attr('cursor', 'grabbing');
        })
        .on('drag', function (event, d) {
          // Prevent zoom during drag
          event.sourceEvent.stopPropagation();

          // Get mouse position relative to the container
          const [x, y] = d3.pointer(event, container.node());

          // Update data coordinates
          d.x = xScale.invert(x);
          d.y = yScale.invert(y);

          // Update circle position
          d3.select(this).attr('cx', xScale(d.x)).attr('cy', yScale(d.y));
        })
        .on('end', function (event, d) {
          event.sourceEvent.stopPropagation();

          d3.select(this).attr('stroke-width', 1).attr('stroke', 'white').attr('cursor', 'grab');

          // Update cluster assignment
          const clusters = d3.groups(data, (d) => d.cluster);
          let minDist = Infinity;
          let closest = d.cluster;

          for (const [id, points] of clusters) {
            const cx = d3.mean(points, (p) => p.x)!;
            const cy = d3.mean(points, (p) => p.y)!;
            const dist = (cx - d.x) ** 2 + (cy - d.y) ** 2;
            if (dist < minDist) {
              minDist = dist;
              closest = +id;
            }
          }

          // Update the cluster if it changed
          if (d.cluster !== closest) {
            d.cluster = closest;

            // Instead of redrawing everything, update just this point's fill color
            d3.select(this).attr('fill', color(String(d.cluster)) as string);
          }
        });

      // Apply drag behavior to the selection
      selection.call(drag as any);
      return selection;
    };

    // Create contours for each cluster first to be drawn below the points
    const grouped = d3.groups(data, (d) => d.cluster);
    for (const [clusterId, points] of grouped) {
      if (points.length >= 3) {
        try {
          const density = d3
            .contourDensity<Point>()
            .x((d) => xScale(d.x))
            .y((d) => yScale(d.y))
            .size([defaultDimensions.width, defaultDimensions.height])
            .bandwidth(15)(points);

          const outermost = density.slice(0, 1);
          container
            .append('g')
            .attr('class', 'contour')
            .selectAll('path')
            .data(outermost)
            .join('path')
            .attr('d', d3.geoPath())
            .attr('fill-opacity', 0.05)
            .attr('fill', color(String(clusterId)))
            .attr('stroke', color(String(clusterId)))
            .attr('stroke-width', 0.8)
            .attr('opacity', 0.7);
        } catch (error) {
          console.error('Error creating contour for cluster', clusterId, error);
        }
      }
    }

    if (!tooltipRef.current) {
      tooltipRef.current = d3
        .select('body')
        .append('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('visibility', 'hidden')
        .style('background-color', 'white')
        .style('border', 'solid')
        .style('border-width', '1px')
        .style('border-radius', '5px')
        .style('padding', '10px')
        .style('pointer-events', 'none')
        .style('max-width', '300px') // Limit width
        .style('word-wrap', 'break-word');
    }
    const tooltip = tooltipRef.current;

    function centerOnPoint(d: Point) {
      const x = xScale(d.x);
      const y = yScale(d.y);

      // Calculate the transform to center this point
      const transform = d3.zoomIdentity
        .translate(defaultDimensions.width / 2, defaultDimensions.height / 2)
        .scale(1)
        .translate(-x, -y);

      svg
        .transition()
        .duration(750)
        .call(zoom.transform as any, transform);
    }

    // Add points to the visualization
    container
      .selectAll('circle')
      .data(data)
      .join('circle')
      .attr('r', (d) => {
        const isSelected = selectedPlay && getPlayId(selectedPlay) === getPlayId(d);
        return isSelected ? 8 : 5;
      })
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('fill', (d) => color(String(d.cluster)))
      .attr('stroke', (d) => {
        const isSelected = selectedPlay && getPlayId(selectedPlay) === getPlayId(d);
        return isSelected ? 'black' : 'white';
      })
      .attr('stroke-width', (d) => {
        const isSelected = selectedPlay && getPlayId(selectedPlay) === getPlayId(d);
        return isSelected ? 2 : 1;
      })
      .attr('cursor', 'pointer')
      .on('click', (event, play) => {
        if (!selectedPlay || getPlayId(selectedPlay) !== getPlayId(play)) {
          // Update state with event_id and game_id when point is clicked
          console.log('Clicked play:', play);
          updatePlay(play);
          centerOnPoint(play);
        }
      })
      // TODO the handlePointDrag is not thought out and needs to be reworked (reassign in backend etc.)
      // .attr('cursor', 'grab')
      // .call(handlePointDrag)
      .on('mouseover', (event, d) => {
        // Use d to create custom content for this specific point
        tooltip?.html(`
      <div>
        <p>Type: ${d.play_type || 'Unknown'}</p>
        <p>Description: ${d.description || 'N/A'}</p>
        <p>Cluster: ${d.cluster}</p>
      </div>
    `);
        return tooltip?.style('visibility', 'visible');
      })
      .on('mousemove', (event) => {
        return tooltip
          ?.style('top', event.pageY + 10 + 'px')
          .style('left', event.pageX + 10 + 'px');
      })
      .on('mouseout', () => {
        return tooltip?.style('visibility', 'hidden');
      });
  }, [data, selectedPlay]);

  const navigation = useNavigation();
  const isLoading = Boolean(navigation.location);
  const clusters = Array.from(new Set(data.map((d) => String(d.cluster)))).sort();
  return (
    <div className="flex flex-col">
      {teamID && data.length === 0 ? (
        <div className="py-4 text-center">No play data available for this team.</div>
      ) : (
        <div className="relative">
          <Filters teamID={teamID} />
          <Legend clusters={clusters} color={color} />
          <ZoomControls svgRef={svgRef} />
          {isLoading ? (
            <ScatterPlotSkeleton />
          ) : (
            <svg
              ref={svgRef}
              viewBox={`0 0 ${defaultDimensions.width} ${defaultDimensions.height}`}
              className="w-full border border-r-0 border-gray-200"
            />
          )}
        </div>
      )}
      <InfoBar />
    </div>
  );
};

export default ScatterPlot;
