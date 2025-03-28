'use strict';
Object.defineProperty(exports, '__esModule', { value: true });
exports.links = void 0;
exports.Layout = Layout;
exports.HydrateFallback = HydrateFallback;
exports.default = App;
exports.ErrorBoundary = ErrorBoundary;
var react_router_1 = require('react-router');
require('./app.css');
var links = function () {
  return [
    { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
    {
      rel: 'preconnect',
      href: 'https://fonts.gstatic.com',
      crossOrigin: 'anonymous',
    },
    {
      rel: 'stylesheet',
      href: 'https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap',
    },
  ];
};
exports.links = links;
function Layout(_a) {
  var children = _a.children;
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <react_router_1.Meta />
        <react_router_1.Links />
      </head>
      <body>
        {children}
        <react_router_1.ScrollRestoration />
        <react_router_1.Scripts />
      </body>
    </html>
  );
}
function HydrateFallback() {
  return <div>Loading...</div>;
}
function App() {
  return <react_router_1.Outlet />;
}
function ErrorBoundary(_a) {
  var error = _a.error;
  var message = 'Oops!';
  var details = 'An unexpected error occurred.';
  var stack;
  if ((0, react_router_1.isRouteErrorResponse)(error)) {
    message = error.status === 404 ? '404' : 'Error';
    details =
      error.status === 404 ? 'The requested page could not be found.' : error.statusText || details;
  } else if (import.meta.env.DEV && error && error instanceof Error) {
    details = error.message;
    stack = error.stack;
  }
  return (
    <main className="container mx-auto p-4 pt-16">
      <h1>{message}</h1>
      <p>{details}</p>
      {stack && (
        <pre className="w-full overflow-x-auto p-4">
          <code>{stack}</code>
        </pre>
      )}
    </main>
  );
}
