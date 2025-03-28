'use strict';
var __assign =
  (this && this.__assign) ||
  function () {
    __assign =
      Object.assign ||
      function (t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
          s = arguments[i];
          for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p)) t[p] = s[p];
        }
        return t;
      };
    return __assign.apply(this, arguments);
  };
Object.defineProperty(exports, '__esModule', { value: true });
exports.getMargin = exports.DEFAULT_MARGIN = void 0;
exports.getChildOrAppend = getChildOrAppend;
function getChildOrAppend(root, tag, className) {
  var node = root.selectAll(''.concat(tag, '.').concat(className));
  node.data([tag]).enter().append(tag).attr('class', className);
  return root.select(''.concat(tag, '.').concat(className));
}
exports.DEFAULT_MARGIN = {
  top: 20,
  left: 40,
  bottom: 20,
  right: 20,
};
var getMargin = function (margin) {
  if (!margin) return exports.DEFAULT_MARGIN;
  else return __assign(__assign({}, exports.DEFAULT_MARGIN), margin);
};
exports.getMargin = getMargin;
