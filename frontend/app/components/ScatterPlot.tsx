import * as d3 from 'd3';
import { Circle, Info, Minus, Move, Plus, RefreshCcw, ZoomIn } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useLoaderData, useNavigation } from 'react-router';
import { Label } from '~/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '~/components/ui/select';
import { EventType } from '~/lib/const';
import { useDashboardStore } from '~/lib/stateStore';
import { getPointId } from '~/lib/utils';
import type { clientLoader } from '~/routes/_index';
import type { Point } from '~/types/data';
import Filters from './Filters';
import { ScatterPlotSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';

const color = d3.scaleOrdinal([...d3.schemeCategory10, d3.schemeSet3]); // 24 colors
const defaultDimensions = { width: 500, height: 400 };
const margin = { top: 60, right: 20, bottom: 20, left: 20 };

const getZoomMethod =
  (svgRef: React.RefObject<SVGSVGElement | null>, method: string) =>
  (...args: any[]) => {
    const svg = svgRef?.current;
    if (svg && typeof (svg as any)[method] === 'function') {
      (svg as any)[method](...args);
    }
  };

function Legend({
  zoomedCluster,
  onSelectCluster,
}: {
  zoomedCluster: string | null;
  onSelectCluster: (clusterId: string) => void;
}) {
  const scatterData = useDashboardStore((state) => state.clusters);
  const clusters =
    scatterData
      ?.filter((c) => c.points.length)
      ?.map(({ cluster_id, cluster_label }) => ({ cluster_id, cluster_label })) ?? [];
  const navigation = useNavigation();
  const isLoading = Boolean(navigation.location);
  if (isLoading) return null;

  return (
    <div className="absolute top-2 right-2 z-10">
      <Select onValueChange={onSelectCluster} value={zoomedCluster ?? ''}>
        <SelectTrigger className="gap-1 border border-gray-400 bg-white">
          <Label htmlFor="timeframe" className="text-xs">
            Legend:
          </Label>
          <SelectValue placeholder="View Cluster" />
        </SelectTrigger>
        <SelectContent>
          {clusters.map((cluster, index) => {
            const c = color(String(cluster.cluster_id)) as string;
            return (
              <SelectItem key={index} value={cluster.cluster_id}>
                <span className="flex items-center gap-1" style={{ color: c }}>
                  <Circle fill={c} stroke={c} width={10} height={10} />
                  Cluster {cluster.cluster_label}
                </span>
              </SelectItem>
            );
          })}
        </SelectContent>
      </Select>
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
              <p>
                Zoom in using controls or mouse wheel. Pan by dragging the background. Click on a
                play to select it. Reset zoom with the refresh button.
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      {/* Control hints */}
      <div className="flex items-center gap-3">
        {/* <TooltipProvider>
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
        </TooltipProvider> */}

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

const ScatterPlot = () => {
  const { teamID } = useLoaderData<typeof clientLoader>();
  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  const updateSelectedPoint = useDashboardStore((state) => state.updateSelectedPoint);
  const updateSelectedCluster = useDashboardStore((state) => state.updateSelectedCluster);
  const resetSelectedPoint = useDashboardStore((state) => state.resetSelectedPoint);
  const scatterData = useDashboardStore((state) => state.clusters);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const [currentTransform, setCurrentTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);
  const [zoomedCluster, setZoomedCluster] = useState<string | null>(null);

  const zoomed = useCallback(
    ({ transform }: d3.D3ZoomEvent<Element, unknown> | { transform: d3.ZoomTransform }) => {
      setCurrentTransform(transform);

      const container = d3.select(svgRef.current).select('g');
      container.attr('transform', transform as any);

      // Keep point radius and stroke width constant
      container
        .selectAll('circle')
        .attr('r', function (d: any) {
          const isTagged = d?.tags?.length;
          const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
          return isSelected ? 8 / transform.k : isTagged ? 6 / transform.k : 5 / transform.k;
        })
        .attr('stroke', function (d: any) {
          const isTagged = d?.tags?.length;
          const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
          return isSelected || isTagged ? 'black' : 'white';
        })
        .attr('stroke-width', function (d: any) {
          const isTagged = d?.tags?.length;
          const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
          return isSelected || isTagged ? 2 / transform.k : 1 / transform.k;
        });
      container.selectAll('path').attr('stroke-width', 0.8 / transform.k); // contour
    },
    [selectedPoint],
  );
  const zoom = d3.zoom().scaleExtent([1, 50]).on('zoom', zoomed);

  const zoomIntoCluster = useCallback(
    (clusterId: string) => {
      const svg = d3.select(svgRef.current);
      const pointsInCluster = scatterData.find((c) => c.cluster_id === clusterId)?.points;

      if (!pointsInCluster?.length) return;

      const xScale = d3
        .scaleLinear()
        .domain(d3.extent(pointsInCluster, (d) => d.x) as [number, number])
        .range([margin.left, defaultDimensions.width - margin.right]);
      const yScale = d3
        .scaleLinear()
        .domain(d3.extent(pointsInCluster, (d) => d.y) as [number, number])
        .range([defaultDimensions.height - margin.bottom, margin.top]);

      const [xMin, xMax] = d3.extent(pointsInCluster, (d) => d.x) as [number, number];
      const [yMin, yMax] = d3.extent(pointsInCluster, (d) => d.y) as [number, number];

      const x0 = xScale(xMin);
      const x1 = xScale(xMax);
      const y0 = yScale(yMax); // y-scale is inverted!
      const y1 = yScale(yMin);

      const clusterWidth = x1 - x0;
      const clusterHeight = y1 - y0;

      const scale =
        0.9 *
        Math.min(
          (defaultDimensions.width - margin.left - margin.right) / clusterWidth,
          (defaultDimensions.height - margin.top - margin.bottom) / clusterHeight,
        );

      const midX = (x0 + x1) / 2;
      const midY = (y0 + y1) / 2;

      // Compute translate
      const translateX = defaultDimensions.width / 2 - scale * midX;
      const translateY = defaultDimensions.height / 2 - scale * midY;
      const transform = d3.zoomIdentity.translate(translateX, translateY).scale(scale);

      svg
        .transition()
        .duration(750)
        .call(zoom.transform as any, transform);
      setZoomedCluster(clusterId);
    },
    [scatterData, zoom],
  );

  const handlePointDrag = (selection: d3.Selection<any, Point, any, any>) => {
    function dragstarted(
      this: SVGCircleElement,
      event: d3.D3DragEvent<SVGCircleElement, Point, any>,
      d: Point,
    ) {
      d3.select(this).raise().attr('stroke', 'black');
    }
    function dragged(
      this: SVGCircleElement,
      event: d3.D3DragEvent<SVGCircleElement, Point, any>,
      d: Point,
    ) {
      d3.select(this).attr('cx', event.x).attr('cy', event.y);
    }
    function dragended(
      this: SVGCircleElement,
      event: d3.D3DragEvent<SVGCircleElement, Point, any>,
      d: Point,
    ) {
      d3.select(this).attr('stroke', null);
    }
    return d3
      .drag<SVGCircleElement, Point>()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  };

  useEffect(() => {
    if (!svgRef.current || !scatterData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const container = svg.append('g').attr('class', 'all-content');
    svg.call(zoom as any).on('dblclick.zoom', null);
    const allPoints = scatterData.flatMap((c) => c.points);
    const xScale = d3
      .scaleLinear()
      .domain(d3.extent(allPoints, (d) => d.x) as [number, number])
      .range([margin.left, defaultDimensions.width - margin.right]);
    const yScale = d3
      .scaleLinear()
      .domain(d3.extent(allPoints, (d) => d.y) as [number, number])
      .range([defaultDimensions.height - margin.bottom, margin.top]);

    container
      .append('rect')
      .attr('width', defaultDimensions.width)
      .attr('height', defaultDimensions.height)
      .style('fill', 'none')
      .style('pointer-events', 'all')
      .on('click', () => {
        // if (resetSelectedPoint) resetSelectedPoint();
      });

    function reset() {
      setZoomedCluster(null);
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

    scatterData.forEach((cluster) => {
      if (cluster.points.length > 2) {
        const density = d3
          .contourDensity<Point>()
          .x((d) => xScale(d.x))
          .y((d) => yScale(d.y))
          .size([defaultDimensions.width, defaultDimensions.height])
          .bandwidth(8)(cluster.points);

        const outermost = density.slice(0, 1);
        container
          .append('g')
          .attr('class', 'contour')
          .selectAll('path')
          .data(outermost)
          .join('path')
          .attr('d', d3.geoPath())
          .attr('fill-opacity', 0.05)
          .attr('fill', color(cluster.cluster_id))
          .attr('stroke', color(cluster.cluster_id))
          .attr('stroke-width', 0.8)
          .attr('opacity', 0.7)
          .attr('cursor', 'move');
      }
    });
    const tooltip = d3.select('.tooltip');

    scatterData.forEach(({ cluster_id, cluster_label, points }) => {
      container
        .selectAll(`circle.cluster-${cluster_id}`)
        .data(points)
        .join('circle')
        .attr('r', (d) => {
          const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
          const isTagged = d?.tags?.length;
          return isSelected ? 8 : isTagged ? 6 : 5;
        })
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('class', (d) => `cluster-${cluster_id}`)
        .attr('fill', (d) => {
          const baseColor = color(cluster_id);
          if (d.recency) {
            const rgb = d3.rgb(baseColor as string);
            const alpha = 0.5 + 0.5 * d.recency; // opacity from 0.5 to 1.0
            return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
          }
          return baseColor;
        })
        .attr('stroke', (d) => {
          const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
          const isTagged = d?.tags?.length;
          return isSelected || isTagged ? 'black' : 'white';
        })
        .attr('stroke-width', (d) => {
          const isTagged = d?.tags?.length;
          const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
          return isSelected || isTagged ? 2 : 1;
        })
        .attr('cursor', 'pointer')
        .on('click', (event, play) => {
          event.stopPropagation();
          if (!selectedPoint || getPointId(selectedPoint) !== getPointId(play)) {
            setZoomedCluster(cluster_id);
            updateSelectedPoint(play);
            updateSelectedCluster({ cluster_id, cluster_label });
          }
        })
        .on('mouseover', (event, d) => {
          container.selectAll('circle').attr('opacity', 0.4);
          container.selectAll(`circle.cluster-${cluster_id}`).attr('opacity', 1);
          tooltip?.html(`
        <div>
          <p>Outcome: ${EventType[d.event_type] ?? 'N/A'}</p>
          <p>Description: ${d.event_desc_home !== 'nan' ? d.event_desc_home : ''} ${d.event_desc_away !== 'nan' ? `, ${d.event_desc_away}` : ''}</p>
          <p>Cluster: ${cluster_label}</p> 
        </div>
      `);
          return tooltip.style('visibility', 'visible');
        })
        .on('mousemove', (event) => {
          return tooltip
            .style('top', event.clientY + 10 + 'px')
            .style('left', event.clientX + 10 + 'px');
        })
        .on('mouseout', () => {
          container.selectAll('circle').attr('opacity', 1);
          return tooltip?.style('visibility', 'hidden');
        });
    });
  }, [scatterData, selectedPoint, svgRef.current, updateSelectedPoint, resetSelectedPoint]);

  useEffect(() => {
    zoomed({ transform: currentTransform });
  }, [scatterData, selectedPoint, currentTransform]);

  const navigation = useNavigation();
  const isLoading = Boolean(navigation.location);

  return (
    <div className="flex flex-col">
      {teamID && scatterData?.length === 0 && !isLoading ? (
        <div className="py-4 text-center">
          No play data available for this team or current filters.
        </div>
      ) : (
        <div className="relative">
          <Filters teamID={teamID} />
          <Legend zoomedCluster={zoomedCluster} onSelectCluster={zoomIntoCluster} />
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
