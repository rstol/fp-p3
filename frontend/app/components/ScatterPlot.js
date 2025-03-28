'use strict';
var __rest =
  (this && this.__rest) ||
  function (s, e) {
    var t = {};
    for (var p in s)
      if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0) t[p] = s[p];
    if (s != null && typeof Object.getOwnPropertySymbols === 'function')
      for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
        if (e.indexOf(p[i]) < 0 && Object.prototype.propertyIsEnumerable.call(s, p[i]))
          t[p[i]] = s[p[i]];
      }
    return t;
  };
Object.defineProperty(exports, '__esModule', { value: true });
var react_1 = require('react');
var d3 = require('d3');
var utils_1 = require('./utils');
var ScatterPlot = function (props) {
  var ref = (0, react_1.createRef)();
  var data = props.data,
    style = __rest(props, ['data']);
  (0, react_1.useEffect)(
    function () {
      var root = ref.current;
      if (data && root) {
        renderScatterPlot(root, data, style);
      }
    },
    [props],
  );
  return (
    <div className="scatterPlot">
      <svg width={props.width} height={props.height} ref={ref} />
    </div>
  );
};
/**
 * Render the scatterplot
 * @param root the root SVG element
 * @param data the data for visualization
 * @param props the parameters of the scatterplot
 */
function renderScatterPlot(root, data, props) {
  var margin = (0, utils_1.getMargin)(props.margin);
  var height = props.height - margin.top - margin.bottom;
  var width = props.width - margin.left - margin.right;
  var visRoot = d3.select(root);
  var base = (0, utils_1.getChildOrAppend)(visRoot, 'g', 'base').attr(
    'transform',
    'translate('.concat(margin.left, ', ').concat(margin.top, ')'),
  );
  var xValues = data.map(function (d) {
    return d.X1;
  });
  var x = d3
    .scaleLinear()
    .domain([d3.min(xValues) || 0, d3.max(xValues) || 1])
    .range([0, width]);
  var yValues = data.map(function (d) {
    return d.X2;
  });
  var y = d3
    .scaleLinear()
    .domain([d3.min(yValues) || 0, d3.max(yValues) || 1])
    .range([height, 0]);
  var colors = d3.scaleOrdinal(['1', '2'], ['blue', 'red']);
  base
    .selectAll('circle.dot')
    .data(data)
    .join(
      function (enter) {
        return enter.append('circle').attr('class', 'dot');
      },
      function (update) {
        return update;
      },
      function (exit) {
        return exit.remove();
      },
    )
    .attr('cx', function (d) {
      return x(d.X1);
    })
    .attr('cy', function (d) {
      return y(d.X2);
    })
    .attr('r', 5)
    .style('fill', function (d) {
      return colors(d.cluster) || '#fff';
    });
  (0, utils_1.getChildOrAppend)(base, 'g', 'y-axis-base').call(d3.axisLeft(y).ticks(4));
  (0, utils_1.getChildOrAppend)(base, 'g', 'x-axis-base')
    .attr('transform', 'translate(0, '.concat(height, ')'))
    .call(d3.axisBottom(x).ticks(5));
}
exports.default = ScatterPlot;
