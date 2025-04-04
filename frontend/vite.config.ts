import { reactRouter } from '@react-router/dev/vite';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';
import { defineConfig } from 'vite';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  server: {
    port: 3000,
    host: 'localhost',
    strictPort: true,
    hmr: {
      host: 'localhost',
      clientPort: 3000,
      protocol: 'ws',
      overlay: true,
    },
    watch: {
      useFsEvents: true,
      alwaysStat: true,
      // usePolling: true,
      // interval: 100,
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  resolve: {
    alias: {
      '~': path.resolve(__dirname, 'app'),
    },
  },
  plugins: [tailwindcss(), reactRouter(), tsconfigPaths()],
});
