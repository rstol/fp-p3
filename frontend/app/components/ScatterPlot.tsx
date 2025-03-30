import * as d3 from 'd3';
import { useEffect, useRef, useState } from 'react';
import { getPanelElement } from 'react-resizable-panels';

type Point = {
  x: number;
  y: number;
  cluster: number;
};

const generateData = () => {
  //TODO use data from the backend

  const clusters = 3;
  const pointsPerCluster = 20;
  const data: Point[] = [];

  for (let c = 0; c < clusters; c++) {
    const cx = 100 + 150 * c;
    const cy = 150 + Math.random() * 100;

    for (let i = 0; i < pointsPerCluster; i++) {
      data.push({
        x: cx + Math.random() * 40 - 20,
        y: cy + Math.random() * 40 - 20,
        cluster: c,
      });
    }
  }

  return data;
};

const ScatterPlot = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const dataRef = useRef<Point[]>(generateData());
  const [dimensions, setDimensions] = useState({ width: 500, height: 400 });

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
    const svg = d3.select(svgRef.current);
    const margin = 30;

    const xScale = d3
      .scaleLinear()
      .domain([0, 500])
      .range([margin, dimensions.width - margin]);
    const yScale = d3
      .scaleLinear()
      .domain([0, 400])
      .range([dimensions.height - margin, margin]);

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const draw = () => {
      svg.selectAll('*').remove();
      const data = dataRef.current;

      // Draw contours
      const grouped = d3.groups(data, (d) => d.cluster);
      for (const [clusterId, points] of grouped) {
        const density = d3
          .contourDensity<Point>()
          .x((d) => xScale(d.x))
          .y((d) => yScale(d.y))
          .size([dimensions.width, dimensions.height])
          .bandwidth(5)(points);

        const outermost = density.slice(0, 1);
        svg
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
      }

      // DRAG behavior
      const drag = d3
        .drag<SVGCircleElement, Point>()
        .on('start', function (event, d) {
          // Bring to front
          d3.select(this).raise().attr('class', 'stroke-black stroke-[2px] cursor-grabbing');
        })
        .on('drag', function (event, d) {
          d.x = xScale.invert(event.x);
          d.y = yScale.invert(event.y);
          d3.select(this).attr('cx', xScale(d.x)).attr('cy', yScale(d.y));
        })
        .on('end', function (event, d) {
          d3.select(this).attr('class', 'stroke-white stroke-[1px] cursor-grab');
          assignCluster(d); // possibly updates d.cluster
          draw(); // redraw everything (could be optimized)
        });

      // Draw points
      svg
        .selectAll('circle')
        .data(data)
        .join('circle')
        .attr('r', 5)
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('fill', (d) => color(String(d.cluster)) as string)
        .attr('stroke', 'white')
        .attr('stroke-width', 1)
        .style('cursor', 'grab')
        .call(drag as any);
    };

    const assignCluster = (d: Point) => {
      //TODO do this in the backend
      const clusters = d3.groups(dataRef.current, (d) => d.cluster);
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

      d.cluster = closest;
    };

    draw();
  }, []);

  return (
    <div className="">
      <svg ref={svgRef} viewBox="0 0 600 500" className="w-full border" />
    </div>
  );
};

export default ScatterPlot;
