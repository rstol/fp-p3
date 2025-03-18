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
      port: 3010,
      host: '0.0.0.0',
      protocol: 'ws',
      clientPort: 3010,
    },
    watch: {
      usePolling: true,
      interval: 100,
    },
  },
  plugins: [tailwindcss(), reactRouter(), tsconfigPaths()],
});
