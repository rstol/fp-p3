import { reactRouter } from '@react-router/dev/vite';
import tailwindcss from '@tailwindcss/vite';
import { execSync } from 'child_process';
import path from 'path';
import { defineConfig } from 'vite';
import tsconfigPaths from 'vite-tsconfig-paths';

const commitHash =
  process.env.NODE_ENV === 'production'
    ? (process.env.VITE_COMMIT_HASH ?? 'unknown')
    : execSync('git rev-parse --short HEAD').toString().trim();

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
  define: {
    __COMMIT_HASH__: JSON.stringify(commitHash),
  },
  plugins: [tailwindcss(), reactRouter(), tsconfigPaths()],
});
