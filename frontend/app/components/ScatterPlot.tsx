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
import { useDashboardStore } from '~/lib/stateStore';
import { getPointId } from '~/lib/utils';
import type { clientLoader } from '~/routes/_index';
import type { Point } from '~/types/data';
import Filters from './Filters';
import { ScatterPlotSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';

const defaultDimensions = { width: 500, height: 400 };
const margin = { top: 40, right: 10, bottom: 10, left: 10 };

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
  zoomedCluster,
  onSelectCluster,
}: {
  clusters: string[];
  color: d3.ScaleOrdinal<string, string>;
  zoomedCluster: string | null;
  onSelectCluster: (cluster: string) => void;
}) {
  const navigation = useNavigation();
  const isLoading = Boolean(navigation.location);
  if (isLoading) return null;

  return (
    <div className="absolute top-2 right-2 z-10">
      <Select onValueChange={onSelectCluster} value={zoomedCluster ?? undefined}>
        <SelectTrigger className="gap-1 border border-gray-400 bg-white">
          <Label htmlFor="timeframe" className="text-xs">
            Legend:
          </Label>
          <SelectValue placeholder="View Cluster" />
        </SelectTrigger>
        <SelectContent>
          {clusters.map((cluster, index) => {
            const c = color(String(cluster));
            return (
              <SelectItem key={index} value={cluster}>
                <span className="flex items-center gap-1">
                  <Circle fill={c} stroke={c} width={10} height={10} />
                  Cluster {cluster}
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

const ScatterPlot = ({ teamID }: { teamID: string }) => {
  const loaderData = useLoaderData<typeof clientLoader>();
  const clusterData = loaderData?.scatterData?.points ?? [];
  const games = loaderData?.games ?? [];
  const { timeframe } = loaderData;

  const selectedPoint = useDashboardStore((state) => state.selectedPoint);
  const updatePoint = useDashboardStore((state) => state.updatePoint);
  const resetPoint = useDashboardStore((state) => state.resetPoint);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const [currentTransform, setCurrentTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);
  const [zoomedCluster, setZoomedCluster] = useState<string | null>(null);
  const [plotData, setPlotData] = useState<Point[]>(clusterData);

  useEffect(() => {
    if (selectedPoint) {
      setPlotData((currentPlotData) =>
        currentPlotData.map((p) =>
          getPointId(p) === getPointId(selectedPoint)
            ? { ...p, cluster: selectedPoint.cluster }
            : p,
        ),
      );
    }
  }, [selectedPoint]);

  const color = d3.scaleOrdinal(d3.schemeCategory10);

  const zoomed = useCallback(
    ({ transform }: d3.D3ZoomEvent<Element, unknown> | { transform: d3.ZoomTransform }) => {
      setCurrentTransform(transform);

      const container = d3.select(svgRef.current).select('g');
      container.attr('transform', transform as any);

      // Keep point radius and stroke width constant
      container
        .selectAll('circle')
        .attr('r', function (d: any) {
          const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
          return isSelected ? 8 / transform.k : 5 / transform.k;
        })
        .attr('stroke', function (d: any) {
          const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
          return isSelected ? 'black' : 'white';
        })
        .attr('stroke-width', function (d: any) {
          const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
          return isSelected ? 2 / transform.k : 1 / transform.k;
        });
      container.selectAll('path').attr('stroke-width', 0.8 / transform.k); // contour
    },
    [selectedPoint],
  );
  const zoom = d3.zoom().scaleExtent([1, 50]).on('zoom', zoomed);

  const zoomIntoCluster = useCallback(
    (clusterId: string) => {
      const svg = d3.select(svgRef.current);
      const g = svg.select<SVGGElement>('g.all-content');
      if (g.empty() || !plotData || !plotData.length) return;

      const pointsInCluster = plotData.filter((d) => String(d.cluster) === clusterId);
      if (!pointsInCluster.length) return;

      const xScale = d3
        .scaleLinear()
        .domain(d3.extent(plotData, (d) => d.x) as [number, number])
        .range([0, defaultDimensions.width]);
      const yScale = d3
        .scaleLinear()
        .domain(d3.extent(plotData, (d) => d.y) as [number, number])
        .range([defaultDimensions.height, 0]);

      const xExtent = d3.extent(pointsInCluster, (d) => d.x) as [number, number];
      const yExtent = d3.extent(pointsInCluster, (d) => d.y) as [number, number];

      const viewBoxWidth = defaultDimensions.width;
      const viewBoxHeight = defaultDimensions.height;

      const scaleX = viewBoxWidth / (xScale(xExtent[1]) - xScale(xExtent[0]));
      const scaleY = viewBoxHeight / (yScale(yExtent[0]) - yScale(yExtent[1]));
      const k = Math.min(scaleX, scaleY) * 0.9;

      const tx = viewBoxWidth / 2 - (k * (xScale(xExtent[0]) + xScale(xExtent[1]))) / 2;
      const ty = viewBoxHeight / 2 - (k * (yScale(yExtent[0]) + yScale(yExtent[1]))) / 2;
      const transform = d3.zoomIdentity.translate(tx, ty).scale(k);

      svg
        .transition()
        .duration(750)
        .call(zoom.transform as any, transform);
      setZoomedCluster(clusterId);
    },
    [plotData, zoom],
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
    if (!svgRef.current || !plotData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const container = svg.append('g').attr('class', 'all-content');
    svg.call(zoom as any).on('dblclick.zoom', null);

    const xScale = d3
      .scaleLinear()
      .domain(d3.extent(plotData, (d) => d.x) as [number, number])
      .range([0, defaultDimensions.width]);
    const yScale = d3
      .scaleLinear()
      .domain(d3.extent(plotData, (d) => d.y) as [number, number])
      .range([defaultDimensions.height, 0]);

    container
      .append('rect')
      .attr('width', defaultDimensions.width)
      .attr('height', defaultDimensions.height)
      .style('fill', 'none')
      .style('pointer-events', 'all')
      .on('click', () => {
        // if (resetPoint) resetPoint();
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

    const uniqueClusters = Array.from(new Set(plotData.map((d) => d.cluster)));
    for (const clusterId of uniqueClusters) {
      const pointsInThisCluster = plotData.filter((d) => d.cluster === clusterId);
      if (pointsInThisCluster.length > 2) {
        const density = d3
          .contourDensity<Point>()
          .x((d) => xScale(d.x))
          .y((d) => yScale(d.y))
          .size([defaultDimensions.width, defaultDimensions.height])
          .bandwidth(15)(pointsInThisCluster);

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
          .attr('opacity', 0.7)
          .attr('cursor', 'move');
      }
    }
    const tooltip = d3.select('.tooltip');

    container
      .selectAll('circle')
      .data(plotData)
      .join('circle')
      .attr('r', (d) => {
        const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
        return isSelected ? 8 : 5;
      })
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('class', (d) => `cluster-${d.cluster}`)
      .attr('fill', (d) => color(String(d.cluster)))
      .attr('stroke', (d) => {
        const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
        return isSelected ? 'black' : 'white';
      })
      .attr('stroke-width', (d) => {
        const isSelected = selectedPoint && getPointId(selectedPoint) === getPointId(d);
        return isSelected ? 2 : 1;
      })
      .attr('cursor', 'pointer')
      .on('click', (event, play) => {
        event.stopPropagation();
        if (!selectedPoint || getPointId(selectedPoint) !== getPointId(play)) {
          updatePoint(play);
        }
      })
      .on('mouseover', (event, d) => {
        const clusterClass = `.cluster-${d.cluster}`;
        container.selectAll('circle').attr('opacity', 0.4);
        container.selectAll(clusterClass).attr('opacity', 1);
        tooltip?.html(`
      <div>
        <p>Type: ${d.event_type || 'Unknown'}</p>
        <p>Home: ${d.event_desc_home || 'N/A'}</p>
        <p>Away: ${d.event_desc_away || 'N/A'}</p>
        <p>Cluster: ${d.cluster}</p> 
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
  }, [plotData, selectedPoint, svgRef.current, updatePoint, resetPoint]);

  useEffect(() => {
    zoomed({ transform: currentTransform });
  }, [plotData, selectedPoint, currentTransform]);

  const navigation = useNavigation();
  const isLoading = Boolean(navigation.location);
  const clusters = Array.from(new Set(plotData.map((d) => String(d.cluster)))).sort();

  return (
    <div className="flex flex-col">
      {teamID && plotData.length === 0 && !isLoading ? (
        <div className="py-4 text-center">
          No play data available for this team or current filters.
        </div>
      ) : (
        <div className="relative">
          <Filters teamID={teamID} />
          <Legend
            clusters={clusters}
            color={color}
            zoomedCluster={zoomedCluster}
            onSelectCluster={zoomIntoCluster}
          />
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
