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
# from the SlicerLive repo root — one per demo (entry -> outfile)
deno run -A npm:esbuild render/demos/dvr-sphere-browser.ts \
  --bundle --format=esm --outfile=/path/to/live/webgpu/dvr-sphere.js
deno run -A npm:esbuild render/demos/real-browser.ts \
  --bundle --format=esm --outfile=/path/to/live/webgpu/real.js
```

## Demos
- `nnlive.html` — **interactive segmentation**: a real CT in the 4-up where each click runs nnLive's ONNX model
  (ORT-Web WebGPU, via `nnlive-worker.js`) on a 64³ patch → mask → ColorizeVolume overlay. The model loads from its
  weights host; `?model=<url>` overrides, and a region-grow stub (`SyntheticSegmenter`) is the fallback. The
  SlicerLive-side encode/splat/bake/render is in `render/live-segmenter.ts` (headless-tested via `test/render-nnlive.ts`).
- `real.html` — **real data**: a live SlicerLive scene (MRHead) whose chunked OME-Zarr volume is
  streamed from the Jetstream2 bucket and gunzipped in-browser, then rendered as 3 MPR planes + a 3D
  volume render using the scene's transfer function and true rotated IJK→RAS geometry. `?scene=<url>`
  loads a different scene json. The direct replacement for the legacy vtk.js MRHead.
- `fourup.html` — synthetic Slicer-style 4-up (3 MPR + 3D ColorizeVolume).
- `fiducials.html` — selftest port: markup control points as procedural SDF spheres composited in the DVR.
- `colorize.html` — a segmentation labelmap baked (palette → separable Gaussian → rgba16float) and ray-marched.
- `multi-volume.html` — two volumes composited in one ray-march via the field SceneRenderer.
- `dvr-sphere.html` — single-volume DVR of a synthetic volume (drag to orbit, scroll to zoom).

All require WebGPU with the `float32-filterable` feature (desktop Chrome/Edge/Safari).
