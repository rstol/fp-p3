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
exports.ResizablePanelGroup = ResizablePanelGroup;
exports.ResizablePanel = ResizablePanel;
exports.ResizableHandle = ResizableHandle;
var React = require('react');
var lucide_react_1 = require('lucide-react');
var ResizablePrimitive = require('react-resizable-panels');
var utils_1 = require('~/lib/utils');
function ResizablePanelGroup(_a) {
  var className = _a.className,
    props = __rest(_a, ['className']);
  return (
    <ResizablePrimitive.PanelGroup
      data-slot="resizable-panel-group"
      className={(0, utils_1.cn)(
        'flex h-full w-full data-[panel-group-direction=vertical]:flex-col',
        className,
      )}
      {...props}
    />
  );
}
function ResizablePanel(_a) {
  var props = __rest(_a, []);
  return <ResizablePrimitive.Panel data-slot="resizable-panel" {...props} />;
}
function ResizableHandle(_a) {
  var withHandle = _a.withHandle,
    className = _a.className,
    props = __rest(_a, ['withHandle', 'className']);
  return (
    <ResizablePrimitive.PanelResizeHandle
      data-slot="resizable-handle"
      className={(0, utils_1.cn)(
        'bg-border focus-visible:ring-ring relative flex w-px items-center justify-center after:absolute after:inset-y-0 after:left-1/2 after:w-1 after:-translate-x-1/2 focus-visible:ring-1 focus-visible:ring-offset-1 focus-visible:outline-hidden data-[panel-group-direction=vertical]:h-px data-[panel-group-direction=vertical]:w-full data-[panel-group-direction=vertical]:after:left-0 data-[panel-group-direction=vertical]:after:h-1 data-[panel-group-direction=vertical]:after:w-full data-[panel-group-direction=vertical]:after:translate-x-0 data-[panel-group-direction=vertical]:after:-translate-y-1/2 [&[data-panel-group-direction=vertical]>div]:rotate-90',
        className,
      )}
      {...props}
    >
      {withHandle && (
        <div className="bg-border z-10 flex h-4 w-3 items-center justify-center rounded-xs border">
          <lucide_react_1.GripVerticalIcon className="size-2.5" />
        </div>
      )}
    </ResizablePrimitive.PanelResizeHandle>
  );
}
