import {
  fileURLToPath,
  URL,
} from 'node:url';

import { defineConfig } from 'vite';

export default defineConfig({
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
