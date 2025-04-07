import * as d3 from 'd3';
import { useEffect, useRef, useState } from 'react';
import { getPanelElement } from 'react-resizable-panels';
import { useDashboardStore } from '~/lib/stateStore';

type Point = {
  x: number;
  y: number;
  cluster: number;
  event_id?: string;
  play_type?: string;
  description?: string;
  period?: number;
  game_id?: string;
};

const ScatterPlot = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<Point[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 500, height: 400 });
  // Keep a reference to the current transform for zooming
  const transformRef = useRef(d3.zoomIdentity);

  const homeTeamId = useDashboardStore((state) => state.homeTeamId);

  const fetchPlayData = async (teamId: string) => {
    if (!teamId) {
      setData([]);
      return;
    }
    setIsLoading(true);
    setError(null);

    try {
      const scatterUrl = `/api/v1/teams/${teamId}/plays/scatter`;
      console.log(`Fetching scatter data from: ${scatterUrl}`);

      const response = await fetch(scatterUrl);
      console.log(`Scatter API response status: ${response.status}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch data: ${response.status}`);
      }

      const playData = await response.json();
      console.log(`Retrieved ${playData.length} play data points`);
      setData(playData);
      setIsLoading(false);
    } catch (err) {
      console.error("Error fetching play data:", err);
      setError('Failed to fetch play data: ' + (err instanceof Error ? err.message : String(err)));
      setIsLoading(false);
      setData([]);
    }
  };

  useEffect(() => {
    if (homeTeamId) {
      fetchPlayData(homeTeamId);
    } else {
      setData([]);
      setError(null);
      setIsLoading(false);
    }
  }, [homeTeamId]);

  useEffect(() => {
    const panel = getPanelElement('left-panel');
    if (!panel) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height });
      }
    });

    resizeObserver.observe(panel);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    // Reset the transform when new data arrives
    if (data.length > 0) {
      transformRef.current = d3.zoomIdentity;
    }
  }, [data]);

  // Main rendering effect when data or dimensions change
  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    // Force reset transform when rendering
    transformRef.current = d3.zoomIdentity;

    const svg = d3.select(svgRef.current);
    const margin = 30;

    // Set up scales with original domains
    const xScale = d3
      .scaleLinear()
      .domain([0, 500])
      .range([margin, dimensions.width - margin]);
    
    const yScale = d3
      .scaleLinear()
      .domain([0, 400])
      .range([dimensions.height - margin, margin]);

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // Clear the SVG
    svg.selectAll('*').remove();

    // Add background rect for zooming/panning
    const background = svg.append('rect')
      .attr('width', dimensions.width)
      .attr('height', dimensions.height)
      .attr('fill', '#f0f9ff') // Light blue background
      .attr('cursor', 'move');

    // Create a group for all visualization content
    const container = svg.append('g')
      .attr('class', 'zoom-container');

    // Function to apply current transform to the container
    const applyTransform = (transform: d3.ZoomTransform) => {
      container.attr('transform', transform.toString());
      transformRef.current = transform;
    };

    // Set up zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        applyTransform(event.transform);
      });

    // Apply zoom behavior to svg but filter so it only works on background or when using mousewheel
    svg.call(zoom as any)
      .on('dblclick.zoom', null); // Disable double-click zoom which can be confusing
      
    // Draw all elements first
    // Create contours for each cluster
    const grouped = d3.groups(data, (d) => d.cluster);
    for (const [clusterId, points] of grouped) {
      if (points.length >= 3) {
        try {
          const density = d3
            .contourDensity<Point>()
            .x((d) => xScale(d.x))
            .y((d) => yScale(d.y))
            .size([dimensions.width, dimensions.height])
            .bandwidth(15)(points);

          const outermost = density.slice(0, 1);
          container
            .append('g')
            .attr('class', 'contour')
            .selectAll('path')
            .data(outermost)
            .join('path')
            .attr('d', d3.geoPath())
            .attr('fill', 'none')
            .attr('stroke', color(String(clusterId)))
            .attr('stroke-width', 2)
            .attr('opacity', 0.8);
        } catch (error) {
          console.error("Error creating contour for cluster", clusterId, error);
        }
      }
    }
    
    // Function to handle point dragging
    const handlePointDrag = (selection: d3.Selection<any, Point, any, any>) => {
      const drag = d3.drag<SVGCircleElement, Point>()
        .on('start', function (event, d) {
          // Prevent zoom during drag
          event.sourceEvent.stopPropagation();
          d3.select(this)
            .raise()
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
          d3.select(this)
            .attr('cx', xScale(d.x))
            .attr('cy', yScale(d.y));
        })
        .on('end', function (event, d) {
          event.sourceEvent.stopPropagation();

          d3.select(this)
            .attr('stroke-width', 1)
            .attr('stroke', 'white')
            .attr('cursor', 'grab');

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

    // Add points to the visualization
    container
      .selectAll('circle')
      .data(data)
      .join('circle')
      .attr('r', 5)
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('fill', (d) => color(String(d.cluster)) as string)
      .attr('stroke', 'white')
      .attr('stroke-width', 1)
      .attr('cursor', 'grab')
      .call(handlePointDrag)
      .on('mouseover', function (event, d) {
        // Show tooltip on hover
        d3.select(this)
          .attr('r', 7)
          .attr('stroke-width', 2);

        const tooltip = container.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${xScale(d.x) + 10},${yScale(d.y) - 10})`);

        tooltip.append('rect')
          .attr('rx', 4)
          .attr('ry', 4)
          .attr('width', 160)
          .attr('height', 50)
          .attr('fill', 'white')
          .attr('stroke', '#ccc')
          .attr('stroke-width', 1)
          .attr('opacity', 0.9);

        tooltip.append('text')
          .attr('x', 5)
          .attr('y', 15)
          .text(`Type: ${d.play_type || 'Unknown'}`)
          .attr('font-size', '10px');

        tooltip.append('text')
          .attr('x', 5)
          .attr('y', 30)
          .text(`Description: ${d.description || 'N/A'}`)
          .attr('font-size', '10px');

        tooltip.append('text')
          .attr('x', 5)
          .attr('y', 45)
          .text(`Cluster: ${d.cluster}`)
          .attr('font-size', '10px');
      })
      .on('mouseout', function () {
        container.selectAll('.tooltip').remove();
        d3.select(this)
          .attr('r', 5)
          .attr('stroke-width', 1);
      });

    // Add zoom controls
    const controls = svg.append('g')
      .attr('class', 'zoom-controls')
      .attr('transform', `translate(${dimensions.width - 70}, 20)`);

    // Zoom in button
    const zoomInBtn = controls.append('g').attr('class', 'zoom-btn');
    zoomInBtn.append('circle')
      .attr('r', 12)
      .attr('fill', 'white')
      .attr('stroke', '#ccc');
    zoomInBtn.append('text')
      .attr('text-anchor', 'middle')
      .attr('alignment-baseline', 'middle')
      .attr('font-size', '16px')
      .text('+');
    zoomInBtn.style('cursor', 'pointer')
      .on('click', (event) => {
        event.stopPropagation();
        svg.transition().duration(300).call(zoom.scaleBy as any, 1.3);
      });

    // Zoom out button
    const zoomOutBtn = controls.append('g')
      .attr('class', 'zoom-btn')
      .attr('transform', 'translate(0, 30)');
    zoomOutBtn.append('circle')
      .attr('r', 12)
      .attr('fill', 'white')
      .attr('stroke', '#ccc');
    zoomOutBtn.append('text')
      .attr('text-anchor', 'middle')
      .attr('alignment-baseline', 'middle')
      .attr('font-size', '16px')
      .text('−');
    zoomOutBtn.style('cursor', 'pointer')
      .on('click', (event) => {
        event.stopPropagation();
        svg.transition().duration(300).call(zoom.scaleBy as any, 0.7);
      });

    // Reset zoom button
    const resetBtn = controls.append('g')
      .attr('class', 'zoom-btn')
      .attr('transform', 'translate(0, 60)');
    resetBtn.append('circle')
      .attr('r', 12)
      .attr('fill', 'white')
      .attr('stroke', '#ccc');
    resetBtn.append('text')
      .attr('text-anchor', 'middle')
      .attr('alignment-baseline', 'middle')
      .attr('font-size', '14px')
      .text('R');
    resetBtn.style('cursor', 'pointer')
      .on('click', (event) => {
        event.stopPropagation();
        svg.transition().duration(300).call(zoom.transform as any, d3.zoomIdentity);
      });
      
    // Center view on data after everything is rendered
    if (data.length > 0) {
      // Use requestAnimationFrame to ensure DOM is fully updated
      requestAnimationFrame(() => {
        // Find data extents
        const xExtent = d3.extent(data, d => d.x) as [number, number];
        const yExtent = d3.extent(data, d => d.y) as [number, number];
        
        // Calculate center point of the data
        const centerX = (xExtent[0] + xExtent[1]) / 2;
        const centerY = (yExtent[0] + yExtent[1]) / 2;
        
        // Calculate translation to center this point
        const tx = dimensions.width / 2 - xScale(centerX);
        const ty = dimensions.height / 2 - yScale(centerY);
        
        // Apply centering transform
        const centerTransform = d3.zoomIdentity.translate(tx, ty);
        svg.call(zoom.transform as any, centerTransform);
      });
    }

    return () => {
      // Clean up any event handlers
      svg.on('.zoom', null);
    };
  }, [data, dimensions]);

  return (
    <div className="flex flex-col">
      {isLoading && <div className="text-center py-4">Loading play data...</div>}

      {error && <div className="text-red-500 py-2">{error}</div>}

      {!homeTeamId && !isLoading && !error && (
        <div className="text-center py-4">Please select a team in the header to view play data.</div>
      )}

      {homeTeamId && data.length === 0 && !isLoading && !error && (
        <div className="text-center py-4">No play data available for this team.</div>
      )}

      <div className="">
        <svg ref={svgRef} viewBox="0 0 600 500" className="w-full border bg-blue-50" />
      </div>

      <div className="mt-2 text-sm text-gray-600">
        <p>Basketball play positions clustered by similarity. Drag points to reassign clusters.</p>
        <p>Colors represent different types of plays. Use mouse wheel to zoom, drag the background to pan.</p>
      </div>
    </div>
  );
};

export default ScatterPlot;
