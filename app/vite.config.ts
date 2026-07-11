import {
  fileURLToPath,
  URL,
} from 'node:url';

import { defineConfig } from 'vite';

export default defineConfig({
  base: '/vk_practice/',
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(
          new URL(
            './index.html',
            import.meta.url,
          ),
        ),
        benchmark: fileURLToPath(
          new URL(
            './benchmark.html',
            import.meta.url,
          ),
        ),
      },
    },
  },
});