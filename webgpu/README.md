# WebGPU LiveRenderer demos

Standalone demos of the new WebGPU rendering backbone for SlicerLive (a TypeScript
port of [slicer-wgpu](https://github.com/pieper/slicer-wgpu), destined to replace the
vtk.js path once at parity). The `.js` here is a **committed bundle** — no build step
is needed to serve the site (same convention as `slicerlive-bundle.js`).

## Source of truth
The TypeScript source lives in the SlicerLive repo under `render/`. The same code runs
headless under Deno (render-to-PNG) and in the browser here.

## Rebuild the bundle
```bash
# from the SlicerLive repo root
deno run -A npm:esbuild render/demos/dvr-sphere-browser.ts \
  --bundle --format=esm --outfile=/path/to/live/webgpu/dvr-sphere.js
```

## Demos
- `dvr-sphere.html` — single-volume DVR of a synthetic volume (drag to orbit, scroll to zoom).
  Requires WebGPU with the `float32-filterable` feature (desktop Chrome/Edge/Safari).
