import * as d3 from 'd3';
import { Circle, GrabIcon, Info, Minus, Move, Plus, RefreshCcw, Tag, ZoomIn } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useLoaderData, useNavigation, useSearchParams } from 'react-router';
import type { clientLoader } from '~/routes/_index';
import type { Point } from '~/types/data';
import Filters from './Filters';
import { ScatterPlotSkeleton } from './LoaderSkeletons';
import { Button } from './ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { useDashboardStore } from '~/lib/stateStore';
import { getPlayId } from '~/lib/utils';
import { applyTagsToData } from '~/lib/tag-store';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '~/components/ui/select';
import { Label } from '~/components/ui/label';

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
  tags,
  color,
  zoomedTag,
  onSelectTag,
}: {
  tags: string[];
  color: d3.ScaleOrdinal<string, string>;
  zoomedTag: string | null;
  onSelectTag: (tag: string) => void;
}) {
  const navigation = useNavigation();
  const isLoading = Boolean(navigation.location);
  if (isLoading) return null;

  return (
    <div className="absolute top-2 right-2 z-10">
      <Select onValueChange={onSelectTag} value={zoomedTag ?? undefined}>
        <SelectTrigger className="gap-1 border border-gray-400 bg-white">
          <Label htmlFor="timeframe" className="text-xs">
            Legend:
          </Label>
          <SelectValue placeholder="View Tag" />
        </SelectTrigger>
        <SelectContent>
          {tags.map((tag, index) => {
            const c = color(tag);
            return (
              <SelectItem key={index} value={tag}>
                <span className="flex items-center gap-1">
                  <Circle fill={c} stroke={c} width={10} height={10} />
                  {tag}
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
              <p>Color brightness indicates recency - brighter points are from more recent games.</p>
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
  // Apply client-side tags to the scatter plot data
  const data = applyTagsToData(loaderData?.scatterData?.points ?? []);
  const selectedPlay = useDashboardStore((state) => state.selectedPlay);
  const updatePlay = useDashboardStore((state) => state.updatePlay);
  // Listen for tag updates to refresh the visualization
  const tagUpdateCounter = useDashboardStore((state) => state.tagUpdateCounter);
  const [zoomedTag, setZoomedTag] = useState<string | null>(null);
  const color = d3.scaleOrdinal(d3.schemeCategory10);
  const [currentTransform, setCurrentTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);

  const zoomed = useCallback(
    ({ transform }: d3.D3ZoomEvent<Element, unknown> | { transform: d3.ZoomTransform }) => {
      setCurrentTransform(transform);

      const container = d3.select(svgRef.current).select('g');
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
    },
    [selectedPlay],
  );
  const zoom = d3.zoom().scaleExtent([1, 50]).on('zoom', zoomed);

  const zoomIntoTag = useCallback(
    (tag: string) => {
      if (!svgRef.current) return;
      const svg = d3.select(svgRef.current);

      // The class is based on the tag - we need to make sure it's a valid CSS class
      // Replace any non-alphanumeric characters with hyphens
      const tagClass = tag.replace(/[^a-z0-9]/gi, '-').toLowerCase();
      const circles = svg.selectAll(`circle.tag-${tagClass}`).nodes() as SVGCircleElement[];

      const cxValues = circles.map((c) => c.cx.baseVal.value);
      const cyValues = circles.map((c) => c.cy.baseVal.value);

      if (!cxValues.length || !cyValues.length) return;

      const x0 = d3.min(cxValues)!;
      const x1 = d3.max(cxValues)!;
      const y0 = d3.min(cyValues)!;
      const y1 = d3.max(cyValues)!;

      const k =
        0.9 * Math.min(defaultDimensions.width / (x1 - x0), defaultDimensions.height / (y1 - y0));
      const tx = (defaultDimensions.width - k * (x0 + x1)) / 2;
      const ty = (defaultDimensions.height - k * (y0 + y1)) / 2;

      const transform = d3.zoomIdentity.translate(tx, ty).scale(k);

      svg
        .transition()
        .duration(750)
        .call(zoom.transform as any, transform);

      setZoomedTag(tag);
    },
    [zoom],
  );

  // Main rendering effect when data or dimensions change
  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);

    // Set up scales with original domains
    const xExtent = d3.extent(data, (d) => d.x) as [number, number];
    const yExtent = d3.extent(data, (d) => d.y) as [number, number];

    const xScale = d3
      .scaleLinear()
      .domain(xExtent)
      .range([margin.left + margin.right, defaultDimensions.width - margin.left - margin.right]);

    const yScale = d3
      .scaleLinear()
      .domain(yExtent)
      .range([defaultDimensions.height - margin.bottom - margin.top, margin.top + margin.bottom]);

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

    svg.call(zoom as any).on('dblclick.zoom', null);
    // Apply zoom behavior to svg but filter so it only works on background or when using mousewheel
    function reset() {
      setZoomedTag(null);
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

    // Group data by tag for drawing contours
    // If a point doesn't have a tag, use its cluster value as fallback
    const pointTags = data.map(d => d.tag || `${d.cluster}`);
    const tagGroups = d3.groups(data, (d) => d.tag || `${d.cluster}`);

    // Create contours for each tag group first to be drawn below the points
    for (const [tag, points] of tagGroups) {
      if (points.length >= 3) {
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
          .attr('fill', color(tag))
          .attr('stroke', color(tag))
          .attr('stroke-width', 0.8)
          .attr('opacity', 0.7)
          .attr('cursor', 'move');
      }
    }
    const tooltip = d3.select('.tooltip');

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
      .attr('class', (d) => {
        // Create a CSS-safe class name from the tag
        const tagClass = (d.tag || `${d.cluster}`).replace(/[^a-z0-9]/gi, '-').toLowerCase();
        return `tag-${tagClass}`;
      })
      .attr('fill', (d) => {
        const baseColor = color(d.tag || `${d.cluster}`);
        // If recency is provided, use it to adjust color brightness
        if (d.recency !== undefined) {
          // Convert to HSL for easier brightness adjustment
          const hsl = d3.hsl(baseColor);
          // Adjust lightness based on recency (higher recency = less lightness adjustment)
          // This ensures newer plays are more vibrant
          hsl.l = Math.min(0.9, hsl.l + 0.5 * (1 - d.recency));
          return hsl.toString();
        }
        return baseColor;
      })
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
        event.stopPropagation();
        if (!selectedPlay || getPlayId(selectedPlay) !== getPlayId(play)) {
          // Update state with event_id and game_id when point is clicked
          updatePlay(play);
        }
      })
      .on('mouseover', (event, d) => {
        const tag = d.tag || `${d.cluster}`;
        const tagClass = `.tag-${tag.replace(/[^a-z0-9]/gi, '-').toLowerCase()}`;
        container.selectAll('circle').attr('opacity', 0.4);
        container.selectAll(tagClass).attr('opacity', 1);
        tooltip?.html(`
      <div>
        <p>Type: ${d.play_type || 'Unknown'}</p>
        <p>Description: ${d.description || 'N/A'}</p>
        <p><span style="display:inline-flex;align-items:center;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg> Tag:</span> ${tag}</p>
        ${d.recency !== undefined ? `<p>Recency: ${Math.round(d.recency * 100)}%</p>` : ''}
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
  }, [data, selectedPlay, svgRef.current, tagUpdateCounter]);

  useEffect(() => {
    zoomed({ transform: currentTransform });
  }, [selectedPlay, currentTransform]);

  const navigation = useNavigation();
  const isLoading = Boolean(navigation.location);
  
  // Extract unique tags from data points, falling back to clusters if tag is not available
  const tags = Array.from(new Set(data.map((d) => d.tag || `${d.cluster}`))).sort();
  
  return (
    <div className="flex flex-col">
      {teamID && data.length === 0 ? (
        <div className="py-4 text-center">No play data available for this team.</div>
      ) : (
        <div className="relative">
          <Filters teamID={teamID} />
          <Legend
            tags={tags}
            color={color}
            zoomedTag={zoomedTag}
            onSelectTag={zoomIntoTag}
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
