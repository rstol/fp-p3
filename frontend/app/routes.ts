import { type RouteConfig, index, route } from '@react-router/dev/routes';

export default [
  index('routes/home.tsx'),
  route('mockup', 'routes/mockup.tsx'),
  route('dummy', 'routes/dummy.tsx'),
] satisfies RouteConfig;
