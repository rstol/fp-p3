import { reactRouter } from '@react-router/dev/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  server: {
    port: 3000,
    host: '0.0.0.0',
    strictPort: true,
    hmr: {
      host: '0.0.0.0',
      protocol: 'ws',
      overlay: true,
    },
    watch: {
      useFsEvents: true,
      alwaysStat: true,
    },
  },
  plugins: [tailwindcss(), reactRouter(), tsconfigPaths()],
});
