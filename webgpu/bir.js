// render/device.ts
async function initDevice() {
  const gpu = navigator.gpu;
  if (!gpu) throw new Error("WebGPU not available (need Chrome/Edge/Safari or Deno --unstable-webgpu)");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const want = ["float32-filterable", "timestamp-query", "shader-f16"].filter((f) => adapter.features.has(f));
  const lim = adapter.limits;
  const requiredLimits = {};
  const raise = (k) => {
    const v = lim[k];
    if (typeof v === "number") requiredLimits[k] = v;
  };
  raise("maxBufferSize");
  raise("maxStorageBufferBindingSize");
  raise("maxTextureDimension3D");
  const device = await adapter.requestDevice({ requiredFeatures: want, requiredLimits });
  return { adapter, device, features: new Set(want) };
}

// render/mat4.ts
function identity() {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}
function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}
function perspectiveZO(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[11] = -1;
  m[10] = far / (near - far);
  m[14] = far * near / (near - far);
  return m;
}
function perspectiveZOTile(fovy, viewW, viewH, x, y, w, h, near, far) {
  const t = near * Math.tan(fovy / 2), b = -t;
  const r = t * (viewW / viewH), l = -r;
  const l2 = l + (r - l) * x / viewW, r2 = l + (r - l) * (x + w) / viewW;
  const t2 = t - (t - b) * y / viewH, b2 = t - (t - b) * (y + h) / viewH;
  const m = new Float32Array(16);
  m[0] = 2 * near / (r2 - l2);
  m[5] = 2 * near / (t2 - b2);
  m[8] = (r2 + l2) / (r2 - l2);
  m[9] = (t2 + b2) / (t2 - b2);
  m[10] = far / (near - far);
  m[11] = -1;
  m[14] = far * near / (near - far);
  return m;
}
function lookAt(eye, center, up) {
  let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
  let zl = Math.hypot(zx, zy, zz) || 1;
  zx /= zl;
  zy /= zl;
  zz /= zl;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  let xl = Math.hypot(xx, xy, xz) || 1;
  xx /= xl;
  xy /= xl;
  xz /= xl;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  const m = new Float32Array(16);
  m[0] = xx;
  m[4] = xy;
  m[8] = xz;
  m[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  m[1] = yx;
  m[5] = yy;
  m[9] = yz;
  m[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  m[2] = zx;
  m[6] = zy;
  m[10] = zz;
  m[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  m[15] = 1;
  return m;
}
function invert(a) {
  const m = a;
  const b00 = m[0] * m[5] - m[1] * m[4], b01 = m[0] * m[6] - m[2] * m[4];
  const b02 = m[0] * m[7] - m[3] * m[4], b03 = m[1] * m[6] - m[2] * m[5];
  const b04 = m[1] * m[7] - m[3] * m[5], b05 = m[2] * m[7] - m[3] * m[6];
  const b06 = m[8] * m[13] - m[9] * m[12], b07 = m[8] * m[14] - m[10] * m[12];
  const b08 = m[8] * m[15] - m[11] * m[12], b09 = m[9] * m[14] - m[10] * m[13];
  const b10 = m[9] * m[15] - m[11] * m[13], b11 = m[10] * m[15] - m[11] * m[14];
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return identity();
  det = 1 / det;
  const o = new Float32Array(16);
  o[0] = (m[5] * b11 - m[6] * b10 + m[7] * b09) * det;
  o[1] = (m[2] * b10 - m[1] * b11 - m[3] * b09) * det;
  o[2] = (m[13] * b05 - m[14] * b04 + m[15] * b03) * det;
  o[3] = (m[10] * b04 - m[9] * b05 - m[11] * b03) * det;
  o[4] = (m[6] * b08 - m[4] * b11 - m[7] * b07) * det;
  o[5] = (m[0] * b11 - m[2] * b08 + m[3] * b07) * det;
  o[6] = (m[14] * b02 - m[12] * b05 - m[15] * b01) * det;
  o[7] = (m[8] * b05 - m[10] * b02 + m[11] * b01) * det;
  o[8] = (m[4] * b10 - m[5] * b08 + m[7] * b06) * det;
  o[9] = (m[1] * b08 - m[0] * b10 - m[3] * b06) * det;
  o[10] = (m[12] * b04 - m[13] * b02 + m[15] * b00) * det;
  o[11] = (m[9] * b02 - m[8] * b04 - m[11] * b00) * det;
  o[12] = (m[5] * b07 - m[4] * b09 - m[6] * b06) * det;
  o[13] = (m[0] * b09 - m[1] * b07 + m[2] * b06) * det;
  o[14] = (m[13] * b01 - m[12] * b03 - m[14] * b00) * det;
  o[15] = (m[8] * b03 - m[9] * b01 + m[10] * b00) * det;
  return o;
}
function patientToTexture(dims, spacing, center = [0, 0, 0]) {
  const m = new Float32Array(16);
  for (let a = 0; a < 3; a++) {
    const s = 1 / (spacing[a] * dims[a]);
    m[a * 4 + a] = s;
    m[12 + a] = 0.5 - center[a] * s;
  }
  m[15] = 1;
  return m;
}
function volumeAABB(dims, spacing, center = [0, 0, 0]) {
  const ext = [dims[0] * spacing[0] / 2, dims[1] * spacing[1] / 2, dims[2] * spacing[2] / 2];
  return [
    [center[0] - ext[0], center[1] - ext[1], center[2] - ext[2]],
    [center[0] + ext[0], center[1] + ext[1], center[2] + ext[2]]
  ];
}
function transpose4(m) {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) o[c * 4 + r] = m[r * 4 + c];
  return o;
}
function patientToTextureFromIjkToRAS(ijkToRAS, dims) {
  return invert(texToRASFromIjkToRAS(ijkToRAS, dims));
}
function texToRASFromIjkToRAS(ijkToRAS, dims) {
  const M = transpose4(ijkToRAS);
  const A = new Float32Array(16);
  for (let a = 0; a < 3; a++) {
    A[a * 4 + a] = dims[a];
    A[12 + a] = -0.5;
  }
  A[15] = 1;
  return multiply(M, A);
}
function volumeAABBFromIjkToRAS(ijkToRAS, dims) {
  const t2r = texToRASFromIjkToRAS(ijkToRAS, dims);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let c = 0; c < 8; c++) {
    const u = c & 1, v = c >> 1 & 1, w = c >> 2 & 1;
    for (let r = 0; r < 3; r++) {
      const p = t2r[r] * u + t2r[4 + r] * v + t2r[8 + r] * w + t2r[12 + r];
      if (p < lo[r]) lo[r] = p;
      if (p > hi[r]) hi[r] = p;
    }
  }
  return [lo, hi];
}
function applyMat4(m, p) {
  const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
  const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
  const z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15] || 1;
  return [x / w, y / w, z / w];
}
function spacingFromIjkToRAS(ijkToRAS) {
  const col = (c) => Math.hypot(ijkToRAS[c], ijkToRAS[4 + c], ijkToRAS[8 + c]);
  return [col(0), col(1), col(2)];
}

// render/slice-renderer.ts
var DEFAULT_FORMAT = "rgba8unorm-srgb";
var SHADER = (
  /* wgsl */
  `
struct U {
  p2t : mat4x4<f32>,     // RAS -> texture[0,1] (folds in ijkToRAS: rotation + anisotropy)
  origin : vec4<f32>,    // RAS of the plane center (for the current scrub offset)
  uvec : vec4<f32>,      // RAS vector spanning the view width  (isotropic mm)
  vvec : vec4<f32>,      // RAS vector spanning the view height (isotropic mm)
  params : vec4<f32>,    // win, lev, fillOpacity, outlineOpacity
  size : vec4<f32>,      // sizeX, sizeY, labelOverlayMode, _
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var s_lin : sampler;
@group(0) @binding(2) var t_scalar : texture_3d<f32>;
@group(0) @binding(3) var t_overlay : texture_3d<f32>;
@group(0) @binding(4) var s_nn : sampler;   // NEAREST \u2014 labelmap overlay is per-voxel crisp (matches Slicer)
// Label-overlay mode (size.z > 0.5): instead of a pre-coloured rgba volume, take the segment
// number from a u8 label volume and its colour+opacity from the same 256x2 palette the
// ColorizeField uses. A coloured overlay of a 509x365x299 CT would be 222 MB; label + palette
// is 55 MB and, because it shares the palette, hiding an organ group in 3D hides it here too.
@group(0) @binding(5) var t_labels : texture_3d<u32>;
@group(0) @binding(6) var t_palette : texture_2d<f32>;

struct V { @builtin(position) position : vec4<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> V {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : V; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92; let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
/** The overlay colour at a texture coordinate, from whichever source is configured. */
fn ov_tex(t : vec3<f32>) -> vec4<f32> {
  if (u.size.z > 0.5) {
    let d = vec3<f32>(textureDimensions(t_labels));
    let vi = vec3<i32>(clamp(floor(t * d), vec3<f32>(0.0), d - vec3<f32>(1.0)));
    let lab = i32(textureLoad(t_labels, vi, 0).r);
    if (lab == 0) { return vec4<f32>(0.0); }
    return textureLoad(t_palette, vec2<i32>(lab, 1), 0);
  }
  return textureSampleLevel(t_overlay, s_nn, t, 0.0);
}
fn ov_at(ras : vec3<f32>) -> vec4<f32> {   // overlay at a RAS point (0 outside the volume)
  let t = (u.p2t * vec4<f32>(ras, 1.0)).xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec4<f32>(0.0); }
  return ov_tex(t);
}
@fragment
fn fs_main(v : V) -> @location(0) vec4<f32> {
  let uv = v.position.xy / u.size.xy;                 // [0,1], y down
  let ras = u.origin.xyz + u.uvec.xyz * (uv.x - 0.5) + u.vvec.xyz * (0.5 - uv.y);
  let t4 = u.p2t * vec4<f32>(ras, 1.0);
  let tex = t4.xyz;
  if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let val = textureSampleLevel(t_scalar, s_lin, tex, 0.0).r;
  let win = max(u.params.x, 1e-6);
  let g = clamp((val - (u.params.y - win * 0.5)) / win, 0.0, 1.0);
  var col = vec3<f32>(g);
  let ov = ov_tex(tex);
  // Slicer-style 2D segmentation: a semi-transparent per-voxel FILL plus a brighter boundary
  // OUTLINE, with independent opacities (params.z = fill, params.w = outline). The outline is
  // screen-space (constant pixel width under zoom), drawn in the segment's own colour along its
  // inner edge \u2014 at both label\u2194label and label\u2194background boundaries.
  let fillA = clamp(ov.a * u.params.z, 0.0, 1.0);
  var outA = 0.0;
  if (u.params.w > 0.0) {
    let du = u.uvec.xyz / u.size.x * 1.5;   // ~1.5 px right, in RAS
    let dv = u.vvec.xyz / u.size.y * 1.5;   // ~1.5 px up
    let n0 = ov_at(ras + du); let n1 = ov_at(ras - du); let n2 = ov_at(ras + dv); let n3 = ov_at(ras - dv);
    let e = max(max(distance(n0.rgb, ov.rgb) + abs(n0.a - ov.a), distance(n1.rgb, ov.rgb) + abs(n1.a - ov.a)),
                max(distance(n2.rgb, ov.rgb) + abs(n2.a - ov.a), distance(n3.rgb, ov.rgb) + abs(n3.a - ov.a)));
    let edge = clamp((e - 0.03) * 12.0, 0.0, 1.0);   // 0 in the interior, 1 at a colour/label edge
    outA = clamp(ov.a * u.params.w * edge, 0.0, 1.0);
  }
  col = mix(col, ov.rgb, max(fillA, outA));
  return vec4<f32>(srgb2physical(col), 1.0);
}
`
);
var BASES = {
  axial: { uDir: [-1, 0, 0], vDir: [0, 1, 0], uAxis: 0, vAxis: 1, nAxis: 2 },
  coronal: { uDir: [-1, 0, 0], vDir: [0, 0, 1], uAxis: 0, vAxis: 2, nAxis: 1 },
  sagittal: { uDir: [0, -1, 0], vDir: [0, 0, 1], uAxis: 1, vAxis: 2, nAxis: 0 }
};
function ijkAxisForRasAxis(ijkToRAS, rasAxis) {
  let best = 0, bestMag = -1;
  for (let c = 0; c < 3; c++) {
    const mag = Math.abs(ijkToRAS[rasAxis * 4 + c]);
    if (mag > bestMag) {
      bestMag = mag;
      best = c;
    }
  }
  return best;
}
function slicerDefaultOffset01(orient, dims, ijkToRAS, rasLo, rasHi) {
  const b = BASES[orient];
  const n = b.nAxis;
  const a = ijkAxisForRasAxis(ijkToRAS, n);
  const m = Math.floor((dims[a] - 1) / 2);
  const ijk = [(dims[0] - 1) / 2, (dims[1] - 1) / 2, (dims[2] - 1) / 2];
  ijk[a] = m;
  const ras = ijkToRAS[n * 4 + 0] * ijk[0] + ijkToRAS[n * 4 + 1] * ijk[1] + ijkToRAS[n * 4 + 2] * ijk[2] + ijkToRAS[n * 4 + 3];
  const span = rasHi[n] - rasLo[n];
  return span === 0 ? 0.5 : (ras - rasLo[n]) / span;
}
var SliceRenderer = class {
  dev;
  format;
  pipeline;
  sampler;
  nnSampler;
  ubuf;
  u = new Float32Array(36);
  // p2t(16) + origin(4) + uvec(4) + vvec(4) + params(4) + size(4)
  bind;
  overlay;
  labels;
  palette;
  scalarTex;
  // actual in-plane extents (mm) spanned by the LAST rendered viewport, aspect-corrected so
  // pixels stay isotropic on a non-square view (0 until first render → fall back to the square span).
  uSpanMm = 0;
  vSpanMm = 0;
  // volume geometry + current plane
  p2t = new Float32Array(16);
  rasLo = [-1, -1, -1];
  rasHi = [1, 1, 1];
  orient = "axial";
  offset01 = 0.5;
  // Per-orientation pan (mm along the plane's uDir/vDir) + zoom (1 = fitted). Slicer-style
  // slice navigation: pan translates the in-plane view centre, zoom scales the field of view.
  viewState = {
    axial: { panU: 0, panV: 0, zoom: 1 },
    coronal: { panU: 0, panV: 0, zoom: 1 },
    sagittal: { panU: 0, panV: 0, zoom: 1 }
  };
  cX = [0, 0, 0];
  // in-plane centre of the LAST rendered frame (for viewToTex picking)
  constructor(gpu, format = DEFAULT_FORMAT) {
    this.dev = gpu.device;
    this.format = format;
    const m = this.dev.createShaderModule({ code: SHADER });
    this.pipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: m, entryPoint: "vs_main" },
      fragment: { module: m, entryPoint: "fs_main", targets: [{ format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.sampler = this.dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.nnSampler = this.dev.createSampler({ magFilter: "nearest", minFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.ubuf = this.dev.createBuffer({ size: this.u.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.setWindowLevel(255, 127);
    this.setOverlayOpacity(0.55);
  }
  /** 1x1x1 stand-ins so the label-overlay bindings always exist. The pipeline layout is fixed,
   *  so every caller must bind them even when it only wants a plain MPR. */
  emptyLabels;
  emptyPalette;
  noLabels() {
    if (!this.emptyLabels) {
      this.emptyLabels = this.dev.createTexture({ size: [1, 1, 1], dimension: "3d", format: "r8uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.dev.queue.writeTexture({ texture: this.emptyLabels }, new Uint8Array(1), { bytesPerRow: 1, rowsPerImage: 1 }, [1, 1, 1]);
    }
    return this.emptyLabels;
  }
  noPalette() {
    if (!this.emptyPalette) {
      this.emptyPalette = this.dev.createTexture({ size: [256, 2], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.dev.queue.writeTexture({ texture: this.emptyPalette }, new Uint8Array(256 * 2 * 4), { bytesPerRow: 256 * 4 }, [256, 2]);
    }
    return this.emptyPalette;
  }
  emptyOverlay;
  transparentOverlay() {
    if (!this.emptyOverlay) {
      this.emptyOverlay = this.dev.createTexture({ size: [1, 1, 1], dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.dev.queue.writeTexture({ texture: this.emptyOverlay }, new Uint16Array(4), { bytesPerRow: 8, rowsPerImage: 1 }, [1, 1, 1]);
    }
    return this.emptyOverlay;
  }
  /** Volume geometry: patientToTexture (RAS->tex[0,1], encodes ijkToRAS) + the RAS
   *  bounding box (for plane extents/scrub range). Get both from the ImageField. */
  setVolume(p2t, rasLo, rasHi) {
    this.p2t = p2t;
    this.rasLo = rasLo;
    this.rasHi = rasHi;
    this.u.set(p2t, 0);
  }
  /** Set the grayscale scalar (r32float 3d) and, optionally, a colored overlay
   *  (rgba16float 3d) — which MUST share the same geometry (ijkToRAS/dims) so the
   *  same RAS->tex mapping addresses both. Omit overlay for a plain MPR. */
  setTextures(scalar, overlay) {
    this.overlay = overlay ?? this.transparentOverlay();
    this.scalarTex = scalar;
    this.rebind();
  }
  /** Colour the overlay from a u8 label volume + the 256x2 palette (row 1 = colour/opacity),
   *  instead of a pre-coloured rgba volume. Same geometry requirement as setTextures. Pass
   *  nulls to go back to the rgba overlay. */
  setLabelOverlay(labels, palette) {
    this.labels = labels ?? void 0;
    this.palette = palette ?? void 0;
    this.u[34] = labels && palette ? 1 : 0;
    if (this.scalarTex) this.rebind();
  }
  rebind() {
    if (!this.scalarTex) return;
    this.bind = this.dev.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.scalarTex.createView() },
        { binding: 3, resource: (this.overlay ?? this.transparentOverlay()).createView() },
        { binding: 4, resource: this.nnSampler },
        { binding: 5, resource: (this.labels ?? this.noLabels()).createView() },
        { binding: 6, resource: (this.palette ?? this.noPalette()).createView() }
      ]
    });
  }
  // Uniform float layout: p2t[0..15] origin[16..19] uvec[20..23] vvec[24..27] params[28..31] size[32..35]
  /** Select the anatomical plane and scrub position (0..1 along the plane normal, RAS bbox). */
  setPlane(orient, offset01) {
    this.orient = orient;
    this.offset01 = Math.max(0, Math.min(1, offset01));
  }
  setWindowLevel(win, lev) {
    this.u[28] = win;
    this.u[29] = lev;
  }
  /** Overlay FILL opacity (per-voxel coloured regions). 0 hides the fill. */
  setOverlayOpacity(o) {
    this.u[30] = o;
  }
  /** Overlay OUTLINE opacity (boundary line, composited over the fill). 0 hides the outline. */
  setOutlineOpacity(o) {
    this.u[31] = o;
  }
  /** Convenience toggle: outline on (opacity 1) / off (0). Composites over the fill. */
  setOverlayOutline(on) {
    this.u[31] = on ? 1 : 0;
  }
  /** Physical size (mm) of the square view for the current plane (isotropic, letterboxed).
   *  Matches Slicer's FitSliceToBackground: the field of view is exactly the volume's
   *  extent along the limiting in-plane axis — NO extra margin. (Verified against
   *  Slicer: Red FOV=[891.78,256] at viewport 634x182 -> vertical FOV == the 256mm
   *  A-extent, horizontal follows viewport aspect.) */
  viewSpanMm() {
    const b = BASES[this.orient];
    const uExt = this.rasHi[b.uAxis] - this.rasLo[b.uAxis];
    const vExt = this.rasHi[b.vAxis] - this.rasLo[b.vAxis];
    return Math.max(uExt, vExt);
  }
  /** The fitted in-plane extent (mm) used for a given orientation — the value directly
   *  comparable to a Slicer slice node's fitted fieldOfView. */
  spanMmFor(orient) {
    const prev = this.orient;
    this.orient = orient;
    const s = this.viewSpanMm();
    this.orient = prev;
    return s;
  }
  /** Letterbox fit at zoom=1 (Slicer's FitSliceToVolume): the in-plane FOV (uS0×vS0) that
   *  exactly contains the slice's bounding box in a viewport of the given aspect — the whole
   *  slice is visible and the LIMITING axis touches the window edge (so the largest fitting
   *  axis fills the window, no needless margin). Replaces the old max(uExt,vExt) span, which
   *  under-zoomed whenever the larger extent wasn't on the viewport's limiting axis. */
  fitUV(orient, aspectWH) {
    const b = BASES[orient];
    const uExt = this.rasHi[b.uAxis] - this.rasLo[b.uAxis];
    const vExt = this.rasHi[b.vAxis] - this.rasLo[b.vAxis];
    const uS0 = Math.max(uExt, vExt * aspectWH);
    return { uS0, vS0: uS0 / aspectWH };
  }
  /** The complete in-plane view frame for an orientation at a given viewport aspect, folding
   *  in pan (mm along uDir/vDir) + zoom. Single source of truth shared by drawInto, rasToView,
   *  viewToRas — so the rendered image and the markup projection stay pixel-aligned under
   *  pan/zoom. Returns the plane centre `c` (RAS, incl. scrub offset + pan) and the half-... no:
   *  uS/vS are the FULL in-plane extents mapped across the viewport width/height. */
  frameFor(orient, offset01, aspectWH) {
    const b = BASES[orient];
    const vs = this.viewState[orient];
    const { uS0, vS0 } = this.fitUV(orient, aspectWH);
    const uS = uS0 / vs.zoom, vS = vS0 / vs.zoom;
    const c = [(this.rasLo[0] + this.rasHi[0]) / 2, (this.rasLo[1] + this.rasHi[1]) / 2, (this.rasLo[2] + this.rasHi[2]) / 2];
    c[b.nAxis] = this.rasLo[b.nAxis] + Math.max(0, Math.min(1, offset01)) * (this.rasHi[b.nAxis] - this.rasLo[b.nAxis]);
    c[0] += b.uDir[0] * vs.panU + b.vDir[0] * vs.panV;
    c[1] += b.uDir[1] * vs.panU + b.vDir[1] * vs.panV;
    c[2] += b.uDir[2] * vs.panU + b.vDir[2] * vs.panV;
    return { b, c, uS, vS };
  }
  /** Zoom factor for an orientation (1 = fitted). */
  zoom(orient) {
    return this.viewState[orient].zoom;
  }
  /** Pan the in-plane view by a pixel delta (drag): the anatomy under the cursor follows it. */
  panByPixels(orient, dxPx, dyPx, w, h) {
    const z = this.viewState[orient].zoom;
    const { uS0, vS0 } = this.fitUV(orient, w / h);
    const uS = uS0 / z, vS = vS0 / z;
    this.viewState[orient].panU -= dxPx / w * uS;
    this.viewState[orient].panV += dyPx / h * vS;
  }
  /** Zoom by `factor` (>1 zooms in) about a pivot (u,v in [0,1]); the pivot point stays fixed. */
  zoomAbout(orient, factor, pu, pv, w, h) {
    const vs = this.viewState[orient];
    const { uS0, vS0 } = this.fitUV(orient, w / h);
    const z = Math.max(0.2, Math.min(50, vs.zoom * factor));
    vs.panU += (pu - 0.5) * (uS0 / vs.zoom - uS0 / z);
    vs.panV += (0.5 - pv) * (vS0 / vs.zoom - vS0 / z);
    vs.zoom = z;
  }
  /** Reset pan/zoom for an orientation to the fitted view. */
  resetView(orient) {
    this.viewState[orient] = { panU: 0, panV: 0, zoom: 1 };
  }
  /** Snapshot per-orientation pan+zoom (e.g. to persist a view across reloads). */
  getViewState() {
    return structuredClone(this.viewState);
  }
  /** Restore a (possibly partial) snapshot from getViewState(). */
  setViewState(vs) {
    for (const k of Object.keys(vs)) {
      const v = vs[k];
      if (v && Number.isFinite(v.zoom) && v.zoom > 0) this.viewState[k] = { ...v };
    }
  }
  /** Mirror Slicer's in-plane navigation for an orientation: drive pan + zoom from the slice
   *  node's RAS centre and field of view (mm). zoom = extent/FOV on the limiting axis (== 1 when
   *  Slicer is fitted, per FitSliceToBackground's no-margin fit), so SlicerLive tracks Slicer's
   *  zoom proportionally; pan is the centre's offset from the volume centre projected onto the
   *  plane's in-plane axes. The out-of-plane offset is applied separately via setPlane. */
  setMirrorFrame(orient, centerRAS, fovX, fovY) {
    const b = BASES[orient];
    const uExt = this.rasHi[b.uAxis] - this.rasLo[b.uAxis];
    const vExt = this.rasHi[b.vAxis] - this.rasLo[b.vAxis];
    const zoom = Math.max(uExt / Math.max(fovX, 1e-6), vExt / Math.max(fovY, 1e-6));
    const volC = [(this.rasLo[0] + this.rasHi[0]) / 2, (this.rasLo[1] + this.rasHi[1]) / 2, (this.rasLo[2] + this.rasHi[2]) / 2];
    const d = [centerRAS[0] - volC[0], centerRAS[1] - volC[1], centerRAS[2] - volC[2]];
    const panU = d[0] * b.uDir[0] + d[1] * b.uDir[1] + d[2] * b.uDir[2];
    const panV = d[0] * b.vDir[0] + d[1] * b.vDir[1] + d[2] * b.vDir[2];
    this.viewState[orient] = { panU, panV, zoom: Math.max(1e-3, zoom) };
  }
  /** Map a view (u,v) in [0,1] (y down) to normalized texture coords for the current
   *  plane — for click picking. Returns the tex coord; the caller converts to IJK via
   *  ijk = tex*dims - 0.5. Anisotropy/rotation are handled by the same p2t the shader uses. */
  viewToTex(u, v) {
    const b = BASES[this.orient];
    const uS = this.uSpanMm || this.viewSpanMm();
    const vS = this.vSpanMm || this.viewSpanMm();
    const c = this.cX;
    const ras = [
      c[0] + b.uDir[0] * (u - 0.5) * uS + b.vDir[0] * (0.5 - v) * vS,
      c[1] + b.uDir[1] * (u - 0.5) * uS + b.vDir[1] * (0.5 - v) * vS,
      c[2] + b.uDir[2] * (u - 0.5) * uS + b.vDir[2] * (0.5 - v) * vS
    ];
    return applyMat4(this.p2t, ras);
  }
  /** Project a RAS point onto a plane's view: returns u,v in [0,1] (y down, matching the
   *  rendered pixels for a viewport of aspect w/h) and the signed distance (mm) from the
   *  point to the plane along its normal. Inverse of viewToTex; used to place 2D markup
   *  glyphs and hit-test clicks on them. */
  rasToView(orient, offset01, ras, aspectWH) {
    const { b, c, uS, vS } = this.frameFor(orient, offset01, aspectWH);
    const d = [ras[0] - c[0], ras[1] - c[1], ras[2] - c[2]];
    const u = 0.5 + (d[0] * b.uDir[0] + d[1] * b.uDir[1] + d[2] * b.uDir[2]) / uS;
    const v = 0.5 - (d[0] * b.vDir[0] + d[1] * b.vDir[1] + d[2] * b.vDir[2]) / vS;
    return { u, v, distMm: d[b.nAxis] };
  }
  /** Map a view (u,v in [0,1], y down) on a plane back to a RAS point ON that plane —
   *  the exact inverse of rasToView (same pan/zoom/aspect). Used to drag a 2D markup:
   *  the point lands on the current slice (its out-of-plane coord becomes the plane offset). */
  viewToRas(orient, offset01, u, v, aspectWH) {
    const { b, c, uS, vS } = this.frameFor(orient, offset01, aspectWH);
    const du = (u - 0.5) * uS, dv = (0.5 - v) * vS;
    return [
      c[0] + b.uDir[0] * du + b.vDir[0] * dv,
      c[1] + b.uDir[1] * du + b.vDir[1] * dv,
      c[2] + b.uDir[2] * du + b.vDir[2] * dv
    ];
  }
  drawInto(view, w, h) {
    const { b, c, uS, vS } = this.frameFor(this.orient, this.offset01, w / h);
    this.uSpanMm = uS;
    this.vSpanMm = vS;
    this.cX = c;
    this.u.set(this.p2t, 0);
    this.u[16] = c[0];
    this.u[17] = c[1];
    this.u[18] = c[2];
    this.u[19] = 0;
    this.u[20] = b.uDir[0] * uS;
    this.u[21] = b.uDir[1] * uS;
    this.u[22] = b.uDir[2] * uS;
    this.u[23] = 0;
    this.u[24] = b.vDir[0] * vS;
    this.u[25] = b.vDir[1] * vS;
    this.u[26] = b.vDir[2] * vS;
    this.u[27] = 0;
    this.u[32] = w;
    this.u[33] = h;
    this.dev.queue.writeBuffer(this.ubuf, 0, this.u);
    const enc = this.dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.draw(3);
    pass.end();
    this.dev.queue.submit([enc.finish()]);
  }
  renderToView(view, w, h) {
    this.drawInto(view, w, h);
  }
  async renderToRGBA(w, h) {
    const target = this.dev.createTexture({ size: [w, h], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.drawInto(target.createView(), w, h);
    const bpr = Math.ceil(w * 4 / 256) * 256;
    const buf = this.dev.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: h }, [w, h]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) out.set(padded.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
    buf.unmap();
    target.destroy();
    buf.destroy();
    return out;
  }
};

// render/slice-interactor.ts
var NORMAL = {
  axial: { axis: 2, sign: 1 },
  // sliceToRAS col2 = +S
  coronal: { axis: 1, sign: 1 },
  // sliceToRAS col2 = +A
  sagittal: { axis: 0, sign: -1 }
  // sliceToRAS col2 = -R
};
function ijkAxisForRasAxis2(ijkToRAS, rasAxis) {
  let best = 0, bestMag = -1;
  for (let c = 0; c < 3; c++) {
    const mag = Math.abs(ijkToRAS[rasAxis * 4 + c]);
    if (mag > bestMag) {
      bestMag = mag;
      best = c;
    }
  }
  return best;
}
function sliceSpacingFor(orient, ijkToRAS) {
  const n = NORMAL[orient].axis;
  const a = ijkAxisForRasAxis2(ijkToRAS, n);
  return Math.hypot(ijkToRAS[a], ijkToRAS[4 + a], ijkToRAS[8 + a]);
}
function sliceBoundsFor(orient, rasLo, rasHi) {
  const { axis, sign } = NORMAL[orient];
  return sign > 0 ? [rasLo[axis], rasHi[axis]] : [-rasHi[axis], -rasLo[axis]];
}
function offset01ToMm(orient, offset01, rasLo, rasHi) {
  const { axis, sign } = NORMAL[orient];
  return sign * (rasLo[axis] + offset01 * (rasHi[axis] - rasLo[axis]));
}
function mmToOffset01(orient, mm, rasLo, rasHi) {
  const { axis, sign } = NORMAL[orient];
  const ras = sign * mm;
  const span = rasHi[axis] - rasLo[axis];
  return span === 0 ? 0.5 : (ras - rasLo[axis]) / span;
}
var SliceInteractor = class {
  constructor(geom) {
    this.geom = geom;
  }
  setGeometry(g) {
    this.geom = g;
  }
  spacing(orient) {
    return sliceSpacingFor(orient, this.geom.ijkToRAS);
  }
  bounds(orient) {
    return sliceBoundsFor(orient, this.geom.rasLo, this.geom.rasHi);
  }
  /** vtkMRMLSliceIntersectionWidget::MoveSlice — returns the NEW offset01, or the
   *  unchanged one if the step would leave the slice bounds (Slicer rejects, not clamps). */
  moveSlice(orient, offset01, deltaMm) {
    const { rasLo, rasHi } = this.geom;
    const cur = offset01ToMm(orient, offset01, rasLo, rasHi);
    const next = cur + deltaMm;
    const [lo, hi] = this.bounds(orient);
    if (next < lo || next > hi) return offset01;
    return mmToOffset01(orient, next, rasLo, rasHi);
  }
  incrementSlice(orient, offset01) {
    return this.moveSlice(orient, offset01, this.spacing(orient));
  }
  decrementSlice(orient, offset01) {
    return this.moveSlice(orient, offset01, -this.spacing(orient));
  }
  /** Map a wheel event to a step. Returns the new offset01. */
  wheel(orient, offset01, forward) {
    return forward ? this.incrementSlice(orient, offset01) : this.decrementSlice(orient, offset01);
  }
  /** Slicer's slice-view keyboard bindings. Returns the new offset01 (unchanged if the
   *  key isn't a stepping key). `key` is a DOM KeyboardEvent.key value. */
  key(orient, offset01, key) {
    switch (key) {
      case "f":
      case "F":
      case "ArrowRight":
      case "ArrowUp":
        return this.incrementSlice(orient, offset01);
      case "b":
      case "B":
      case "ArrowLeft":
      case "ArrowDown":
        return this.decrementSlice(orient, offset01);
      default:
        return offset01;
    }
  }
  /** True if this key is one Slicer's slice view consumes for stepping. */
  static isStepKey(key) {
    return ["f", "F", "b", "B", "ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"].includes(key);
  }
};

// render/scene-renderer.ts
var DEFAULT_FORMAT2 = "rgba8unorm-srgb";
var SCENE_FLOATS = 16;
var CLIP_FLOATS = 36;
var SceneRenderer = class _SceneRenderer {
  dev;
  format;
  placed = [];
  pipeline;
  sampler;
  camBuf;
  matBuf;
  mat;
  bind;
  // PICK pass: a 1x1 ray-trace that reuses the field compositing to find the RAS point where
  // front-to-back opacity first crosses 50% (Slicer's 3D volume pick). Ghost handles excluded.
  pickPipeline;
  pickBind;
  pickOff = 0;
  // mat[] offset of the pick_cursor uniform (NDC)
  pickTarget;
  // 1x1 rgba32float (wp.xyz, hit)
  pickReadBuf;
  // PRODUCER→RECONSTRUCTOR seam (docs/UNIFIED-RENDERING-PLAN.md M1). The ray-march writes the
  // premultiplied composited sample into `traceTex` (rgba32float, lossless); `resolvePipeline`
  // composites it over the background into the output view. 1:1 for now (byte-identical); the
  // resolve pass is where spatial upsample + temporal accumulation (time-averaged AA) will live.
  resolvePipeline;
  resolveBind;
  resolveBgBuf;
  traceTex;
  traceView;
  traceW = 0;
  traceH = 0;
  // TEMPORAL ACCUMULATION (M2a, docs/UNIFIED-RENDERING-PLAN.md §3). When the view is still, each
  // frame jitters the CAMERA sub-pixel (Halton, via a clip-space translation of invVP — the shader
  // is untouched, so a non-jittered frame is byte-identical) and the Reconstructor folds it into a
  // running mean, converging to a supersampled, time-averaged-AA image. Ping-pong accum + running n.
  baseInvVP = new Float32Array(16);
  // last setCamera invVP (unjittered)
  focalPx = 1;
  // last setCamera focal (view→pixels); used to keep screen-space handles view-sized under low-res trace
  accumPipeline;
  // MRT: trace + prev-accum -> new-accum + presented view
  accumBind = [void 0, void 0];
  accumUniformBuf;
  // (bg.rgb, blend)
  accumTex = [void 0, void 0];
  accumView = [void 0, void 0];
  accumPing = 0;
  accumN = 0;
  lastAccumCam = new Float32Array(16);
  // camera (invVP) of the last accumulated frame
  lastAccumValid = false;
  // false forces a reset (after a rebuild / first frame)
  streamPipeline;
  // trace -> rgba8unorm, for compact sample readback (remote)
  streamBind;
  // its OWN bind group (auto-layout differs from this.pipeline's)
  // RESOLUTION-SCALED reconstruction (M2b): while interacting, trace at a fraction of the view
  // (BudgetController) and Catmull-Rom UPSAMPLE the low-res trace to the view — the client-superres
  // ported from the Python spike. A settled view renders native + accumulates instead.
  superresPipeline;
  superresBind;
  superresBuf;
  // (traceW, traceH, viewW, viewH)
  // The moving/upscale path traces into its OWN low-res target so it never resizes/destroys the
  // full-size traceTex the accumulation bind groups reference (that sharing caused destroyed-texture
  // submits + MRT attachment-size mismatches → 3D flicker/blank during interaction).
  lowTex;
  lowView;
  lowW = 0;
  lowH = 0;
  accumW = 0;
  accumH = 0;
  /** Emit a default AABB-distance skip for fields that don't supply their own bound.
   *
   *  OFF because it MEASURED AS A NET LOSS (render/test/profile-boxskip.ts, 448², M-series):
   *      MultiVolume +8.7%   Volume+Fiducials +7.3%   Segmentation +96.5%   SingleVolume -15.5%
   *  The appealing theory — "Panoramix sits +200mm R of CTACardio, so rays spend much of the
   *  scene box outside one volume" — is true but worthless: ImageField's out-of-box sample was
   *  ALREADY nearly free (it early-returns on the texture-bounds test), so there was no per-step
   *  cost to remove. Meanwhile every field pays a box distance + horizon bookkeeping at every
   *  step it is INSIDE its box, which is most of the march since the scene box is the union of
   *  the field boxes. Fields with their own cheap early-out are hurt worst — SegmentField
   *  (`v<=0.02||v>=0.98`) nearly doubles. The lone SingleVolume win survives warm-up but has no
   *  algorithmic explanation (the box IS the scene box there, so the bound is 0 at every sample)
   *  and is almost certainly a shader-compiler/occupancy artifact — not something to bank on.
   *
   *  Kept behind a flag rather than deleted so the negative result stays reproducible, and
   *  because it may behave differently on other GPUs (NVIDIA/AMD) — re-measure before enabling.
   *  The real win for dense volumes is an occupancy grid over air INSIDE the box, not the box. */
  static boxSkip = false;
  canTime;
  clipOff = 0;
  constructor(gpu, format = DEFAULT_FORMAT2) {
    this.dev = gpu.device;
    this.format = format;
    this.canTime = gpu.features.has("timestamp-query");
    this.sampler = this.dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.camBuf = this.dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.resolveBgBuf = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const rmod = this.dev.createShaderModule({ code: this.resolveWgsl() });
    this.resolvePipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: rmod, entryPoint: "vs_resolve" },
      fragment: { module: rmod, entryPoint: "fs_resolve", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.accumUniformBuf = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const amod = this.dev.createShaderModule({ code: this.accumWgsl() });
    this.accumPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: amod, entryPoint: "vs_resolve" },
      fragment: { module: amod, entryPoint: "fs_accum", targets: [{ format: "rgba32float" }, { format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.superresBuf = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const smod = this.dev.createShaderModule({ code: this.superresWgsl() });
    this.superresPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: smod, entryPoint: "vs_resolve" },
      fragment: { module: smod, entryPoint: "fs_superres", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
  }
  /** RECONSTRUCTOR (upsampling): Catmull-Rom (bicubic, 9 bilinear taps) reconstruction of the
   *  low-res premultiplied trace, composited over the background — the client-superres from the
   *  Python spike (435b28d), on WebGPU. Slight edge sharpening from the negative lobes; premultiplied
   *  so the alpha reconstructs correctly. Used only when the trace is smaller than the view. */
  superresWgsl() {
    return (
      /* wgsl */
      `
@group(0) @binding(0) var t_trace : texture_2d<f32>;
@group(0) @binding(1) var s_lin : sampler;
@group(0) @binding(2) var<uniform> u_sr : vec4<f32>;   // (traceW, traceH, viewW, viewH)
@group(0) @binding(3) var<uniform> u_bg : vec4<f32>;
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
// Catmull-Rom via 9 bilinear taps (Sigg/Hadwiger form).
fn cr(uv : vec2<f32>, texSize : vec2<f32>) -> vec4<f32> {
  let sp = uv * texSize;
  let tp1 = floor(sp - 0.5) + 0.5;
  let f = sp - tp1;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let off12 = w2 / w12;
  let inv = 1.0 / texSize;
  let p0 = (tp1 - 1.0) * inv;
  let p3 = (tp1 + 2.0) * inv;
  let p12 = (tp1 + off12) * inv;
  var r = vec4<f32>(0.0);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p0.x,  p0.y),  0.0) * (w0.x  * w0.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p12.x, p0.y),  0.0) * (w12.x * w0.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p3.x,  p0.y),  0.0) * (w3.x  * w0.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p0.x,  p12.y), 0.0) * (w0.x  * w12.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p12.x, p12.y), 0.0) * (w12.x * w12.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p3.x,  p12.y), 0.0) * (w3.x  * w12.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p0.x,  p3.y),  0.0) * (w0.x  * w3.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p12.x, p3.y),  0.0) * (w12.x * w3.y);
  r += textureSampleLevel(t_trace, s_lin, vec2<f32>(p3.x,  p3.y),  0.0) * (w3.x  * w3.y);
  return r;
}
struct RV { @builtin(position) position : vec4<f32> };
@vertex
fn vs_resolve(@builtin(vertex_index) vi : u32) -> RV {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : RV; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
@fragment
fn fs_superres(v : RV) -> @location(0) vec4<f32> {
  let uv = v.position.xy / u_sr.zw;
  let s = cr(uv, u_sr.xy);
  let a = clamp(s.a, 0.0, 1.0);
  let bg = srgb2physical(u_bg.rgb);
  return vec4<f32>(mix(bg, s.rgb, a), 1.0);
}`
    );
  }
  /** Accumulating RECONSTRUCTOR: fold this frame's traced sample into the running mean (blend =
   *  1/n; blend=1 on reset → mean=this frame) and present it over the background. MRT so one pass
   *  updates the accumulation texture AND the swap-chain view. Frame N jitters the ray sub-pixel,
   *  so the mean over N frames is a supersampled, time-averaged-AA image (still camera). */
  accumWgsl() {
    return (
      /* wgsl */
      `
@group(0) @binding(0) var t_trace : texture_2d<f32>;
@group(0) @binding(1) var t_accum : texture_2d<f32>;
@group(0) @binding(2) var<uniform> u_ra : vec4<f32>;   // (bg.r, bg.g, bg.b, blend)
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
struct RV { @builtin(position) position : vec4<f32> };
@vertex
fn vs_resolve(@builtin(vertex_index) vi : u32) -> RV {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : RV; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
struct FO { @location(0) accum : vec4<f32>, @location(1) present : vec4<f32> };
@fragment
fn fs_accum(v : RV) -> FO {
  let p = vec2<i32>(v.position.xy);
  let cur = textureLoad(t_trace, p, 0);
  let prev = textureLoad(t_accum, p, 0);
  let acc = mix(prev, cur, u_ra.w);        // blend=1 on reset -> acc = cur
  let bg = srgb2physical(u_ra.rgb);
  var o : FO;
  o.accum = acc;
  o.present = vec4<f32>(mix(bg, acc.rgb, acc.a), 1.0);
  return o;
}`
    );
  }
  /** RECONSTRUCTOR (M1: identity resolve). Composites the traced premultiplied sample over the
   *  background — the exact `mix(bg, rgb, a)` the fused fs_main used. `textureLoad` at integer
   *  coords is a 1:1 fetch (no filtering), so the output is byte-identical to the fused path.
   *  M2 replaces this with a spatial-upsample + temporal-accumulate resolve. */
  resolveWgsl() {
    return (
      /* wgsl */
      `
@group(0) @binding(0) var t_trace : texture_2d<f32>;
@group(0) @binding(1) var<uniform> u_bg : vec4<f32>;
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
struct RV { @builtin(position) position : vec4<f32> };
@vertex
fn vs_resolve(@builtin(vertex_index) vi : u32) -> RV {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : RV; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
@fragment
fn fs_resolve(v : RV) -> @location(0) vec4<f32> {
  let s = textureLoad(t_trace, vec2<i32>(v.position.xy), 0);
  let bg = srgb2physical(u_bg.rgb);
  return vec4<f32>(mix(bg, s.rgb, s.a), 1.0);
}`
    );
  }
  /** (Re)allocate the trace target + resolve bind group when the view size changes. */
  ensureTrace(width, height) {
    if (this.traceTex && this.traceW === width && this.traceH === height) return;
    this.traceTex?.destroy();
    this.traceTex = this.dev.createTexture({
      size: [width, height],
      format: "rgba32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
    this.traceView = this.traceTex.createView();
    this.traceW = width;
    this.traceH = height;
    this.resolveBind = this.dev.createBindGroup({
      layout: this.resolvePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.traceView }, { binding: 1, resource: { buffer: this.resolveBgBuf } }]
    });
  }
  /** (Re)allocate the low-res trace target + superres bind group when the moving render size changes.
   *  Separate from traceTex so a moving frame never disturbs the accumulation textures. */
  ensureLow(width, height) {
    if (this.lowTex && this.lowW === width && this.lowH === height) return;
    this.lowTex?.destroy();
    this.lowTex = this.dev.createTexture({ size: [width, height], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.lowView = this.lowTex.createView();
    this.lowW = width;
    this.lowH = height;
    this.superresBind = this.dev.createBindGroup({
      layout: this.superresPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.lowView },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.superresBuf } },
        { binding: 3, resource: { buffer: this.resolveBgBuf } }
      ]
    });
  }
  /** Adaptive (moving-frame) render: trace at `renderW×renderH` and Catmull-Rom upsample to the
   *  `viewW×viewH` output. The caller MUST have set the camera size to renderW×renderH (so the
   *  low-res rays fill the same frustum). Single frame, no accumulation — use while interacting;
   *  switch to renderAccum when the view settles. */
  renderUpscaled(view, renderW, renderH, viewW, viewH) {
    this.ensureLow(renderW, renderH);
    this.flush();
    this.dev.queue.writeBuffer(this.camBuf, 72, new Float32Array([this.focalPx * (viewH / renderH)]));
    this.dev.queue.writeBuffer(this.superresBuf, 0, new Float32Array([renderW, renderH, viewW, viewH]));
    this.dev.queue.writeBuffer(this.resolveBgBuf, 0, this.mat.subarray(12, 16));
    const enc = this.dev.createCommandEncoder();
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.lowView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.draw(3);
    tp.end();
    const sp = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    sp.setPipeline(this.superresPipeline);
    sp.setBindGroup(0, this.superresBind);
    sp.draw(3);
    sp.end();
    this.dev.queue.submit([enc.finish()]);
  }
  /** Encode trace (producer) + resolve (reconstructor) into `enc`, output to `outView`. */
  encodeFrame(enc, outView) {
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.traceView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.draw(3);
    tp.end();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: outView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    rp.setPipeline(this.resolvePipeline);
    rp.setBindGroup(0, this.resolveBind);
    rp.draw(3);
    rp.end();
  }
  /** (Re)allocate the ping-pong accumulation targets + their bind groups on a size change. Tracks its
   *  OWN size and always rebuilds accumBind against the current traceView (which ensureTrace, called
   *  first in renderAccum, has just refreshed) — so the bind never dangles on a destroyed trace. */
  ensureAccum(width, height) {
    if (this.accumTex[0] && this.accumW === width && this.accumH === height) return;
    this.accumW = width;
    this.accumH = height;
    for (let k = 0; k < 2; k++) {
      this.accumTex[k]?.destroy();
      this.accumTex[k] = this.dev.createTexture({ size: [width, height], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.accumView[k] = this.accumTex[k].createView();
    }
    for (let k = 0; k < 2; k++) {
      this.accumBind[k] = this.dev.createBindGroup({
        layout: this.accumPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.traceView },
          { binding: 1, resource: this.accumView[k] },
          { binding: 2, resource: { buffer: this.accumUniformBuf } }
        ]
      });
    }
    this.accumN = 0;
    this.accumPing = 0;
  }
  /** Reset temporal accumulation — call when the view changes (camera move, scene edit, resize). */
  resetAccumulation() {
    this.accumN = 0;
  }
  /** Frames accumulated since the last reset (0 before the first accumulated frame). */
  accumCount() {
    return this.accumN;
  }
  /** Accumulating render: trace this frame (sub-pixel jittered) and fold it into the running mean,
   *  presenting the mean over the background. `reset` (or a view change) restarts the mean at this
   *  frame (n=1, no jitter — byte-identical to renderToView). Call repeatedly while the view is
   *  still to converge to a supersampled, time-averaged-AA image. */
  renderAccum(view, width, height, reset) {
    this.ensureTrace(width, height);
    this.ensureAccum(width, height);
    let camChanged = !this.lastAccumValid;
    const cam = this.baseInvVP;
    for (let i = 0; i < 16 && !camChanged; i++) if (cam[i] !== this.lastAccumCam[i]) camChanged = true;
    if (camChanged) reset = true;
    this.lastAccumCam.set(cam);
    this.lastAccumValid = true;
    if (reset) this.accumN = 0;
    this.accumN += 1;
    const n = this.accumN;
    if (n > 1) {
      const jx = _SceneRenderer.halton(n, 2) - 0.5, jy = _SceneRenderer.halton(n, 3) - 0.5;
      const T = new Float32Array(16);
      T[0] = T[5] = T[10] = T[15] = 1;
      T[12] = 2 * jx / width;
      T[13] = -2 * jy / height;
      this.dev.queue.writeBuffer(this.camBuf, 0, multiply(this.baseInvVP, T));
    } else {
      this.dev.queue.writeBuffer(this.camBuf, 0, this.baseInvVP);
    }
    this.dev.queue.writeBuffer(this.camBuf, 76, new Float32Array([n - 1]));
    this.flush();
    this.dev.queue.writeBuffer(this.accumUniformBuf, 0, new Float32Array([this.mat[12], this.mat[13], this.mat[14], 1 / n]));
    const prev = this.accumPing, next = 1 - this.accumPing;
    const enc = this.dev.createCommandEncoder();
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.traceView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.draw(3);
    tp.end();
    const ap = enc.beginRenderPass({ colorAttachments: [
      { view: this.accumView[next], loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
      { view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }
    ] });
    ap.setPipeline(this.accumPipeline);
    ap.setBindGroup(0, this.accumBind[prev]);
    ap.draw(3);
    ap.end();
    this.dev.queue.submit([enc.finish()]);
    this.accumPing = next;
  }
  /** (Re)build the pipeline for a set of fields. */
  build(fields) {
    const kindCount = {};
    let uoff = SCENE_FLOATS, bbase = 3;
    this.placed = fields.map((field) => {
      const slot = kindCount[field.kind] ?? 0;
      kindCount[field.kind] = slot + 1;
      const p = { field, slot, uoff, bbase };
      uoff += field.uniformFloats();
      bbase += field.bindingCount;
      return p;
    });
    this.clipOff = uoff;
    this.pickOff = uoff + CLIP_FLOATS;
    this.mat = new Float32Array(uoff + CLIP_FLOATS + 12);
    this.matBuf = this.dev.createBuffer({ size: (uoff + CLIP_FLOATS + 12) * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = this.dev.createShaderModule({ code: this.wgsl() });
    this.pipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_trace", targets: [{ format: "rgba32float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.pickPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_pick", targets: [{ format: "rgba32float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.streamPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_trace", targets: [{ format: "rgba8unorm" }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.bind = this.dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    this.streamBind = this.dev.createBindGroup({ layout: this.streamPipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    if (this.pickPipeline) this.pickBind = this.dev.createBindGroup({ layout: this.pickPipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    this.setBackground(0.07, 0.08, 0.12);
    const step = this.placed.length ? Math.min(...this.placed.map((p) => p.field.sampleStep())) : 1;
    this.setSampleStep(step * 0.7);
    this.recomputeBounds();
    for (const p of this.placed) p.field.fillUniforms(this.mat, p.uoff);
    this.accumN = 0;
    this.lastAccumValid = false;
  }
  wgsl() {
    const members = this.placed.map((p) => p.field.structMembers(p.slot)).join("\n");
    const decls = this.placed.map((p) => p.field.declareBindings(p.slot, p.bbase)).join("\n");
    const modifiers = this.placed.filter((p) => p.field.modifier);
    const receivers = this.placed.filter((p) => !p.field.modifier);
    const modFns = modifiers.map((p) => p.field.samplingWGSL(p.slot)).join("\n");
    const slotOf = new Map(this.placed.map((p) => [p.field, p.slot]));
    const tpFns = receivers.map((p) => {
      const tf = p.field.transform;
      const tfSlot = tf && tf.modifier ? slotOf.get(tf) : void 0;
      const body = tfSlot === void 0 ? "  return wp;" : `  return wp + displacement_grid${tfSlot}(wp);`;
      return `fn transform_point_${p.field.kind}${p.slot}(wp : vec3<f32>) -> vec3<f32> {
${body}
}`;
    }).join("\n");
    const fieldFns = receivers.map((p) => p.field.samplingWGSL(p.slot)).join("\n");
    const wf = (v) => (Number.isFinite(v) ? v : 0).toFixed(6);
    const boxSkipWGSL = (p) => {
      const [lo, hi] = p.field.aabb();
      return `
fn skip_${p.field.kind}${p.slot}(wp : vec3<f32>) -> f32 {
  let q = max(vec3<f32>(${wf(lo[0])}, ${wf(lo[1])}, ${wf(lo[2])}) - wp,
              wp - vec3<f32>(${wf(hi[0])}, ${wf(hi[1])}, ${wf(hi[2])}));
  return length(max(q, vec3<f32>(0.0)));   // 0 inside the box, exact distance outside
}`;
    };
    const ghostFields = receivers.filter((p) => p.field.ghost);
    const normalReceivers = receivers.filter((p) => !p.field.ghost);
    const clipGuard = (p, expr) => p.field.clippable === false ? expr : `if (!clipped) { ${expr} }`;
    const sampleInto = (nm, ghost) => ghost ? `let c = sample_field_${nm}(wp, rd); if (c.a > g_op) { g_op = c.a; g_col = c.rgb / max(c.a, 1e-4); }` : `let c = sample_field_${nm}(wp, rd); sum += c;`;
    const skipBranch = (p, clip, ghost = false) => {
      const nm = `${p.field.kind}${p.slot}`;
      const smp = sampleInto(nm, ghost);
      return `    if (t >= resume_${nm}) {
      let d_${nm} = max(skip_${nm}(wp) - step, 0.0);
      if (d_${nm} > 0.0) { resume_${nm} = t + d_${nm}; }
      else { ${clip ? clipGuard(p, smp) : smp} }
    }
    if (t < resume_${nm}) { jump_t = min(jump_t, resume_${nm}); } else { all_defer = false; }`;
    };
    const plainBranch = (p, clip, ghost = false) => {
      const nm = `${p.field.kind}${p.slot}`;
      const smp = sampleInto(nm, ghost);
      return `    { ${clip ? clipGuard(p, smp) : smp} all_defer = false; }`;
    };
    const normalSkippers = normalReceivers.filter((p) => !p.field.transform).filter((p) => _SceneRenderer.boxSkip || p.field.providesSkip && p.field.skipWGSL);
    const ghostSkippers = ghostFields.filter((p) => p.field.providesSkip && p.field.skipWGSL);
    const canSkip = new Set(normalSkippers.map((p) => p.field));
    const ghostCanSkip = new Set(ghostSkippers.map((p) => p.field));
    const skipFns = [
      ...normalSkippers.map((p) => p.field.providesSkip && p.field.skipWGSL ? p.field.skipWGSL(p.slot) : boxSkipWGSL(p)),
      ...ghostSkippers.map((p) => p.field.skipWGSL(p.slot))
    ].join("\n");
    const fns = [modFns, tpFns, fieldFns, skipFns].filter((s) => s.trim()).join("\n");
    const skipInit = [...normalSkippers, ...ghostSkippers].map((p) => `  var resume_${p.field.kind}${p.slot} : f32 = -1.0e30;`).join("\n");
    const dispatch = normalReceivers.map(
      (p) => canSkip.has(p.field) ? skipBranch(p, true) : plainBranch(p, true)
    ).join("\n");
    const ghostDispatch = ghostFields.map(
      (p) => ghostCanSkip.has(p.field) ? skipBranch(p, false, true) : plainBranch(p, false, true)
    ).join("\n");
    const hasGhost = ghostFields.length > 0;
    const pickDispatch = normalReceivers.map(
      (p) => `    ${clipGuard(p, `{ let c = sample_field_${p.field.kind}${p.slot}(wp, rd); sum += c; }`)}`
    ).join("\n");
    return (
      /* wgsl */
      `
struct Camera { inv_view_proj : mat4x4<f32>, size : vec4<f32>, eye : vec4<f32> };
struct Material {
  bmin : vec4<f32>,
  bmax : vec4<f32>,
  scene : vec4<f32>,   // sample_step, _, _, _
  bg : vec4<f32>,
${members}
  clip_planes : array<vec4<f32>, 8>,   // (nx, ny, nz, offset) inward; tail so field offsets are stable
  clip_count : vec4<f32>,              // (count, _, _, _)
  pick_cursor : vec4<f32>,             // (ndc_x, ndc_y, _, _) \u2014 the ray for fs_pick
  probe_origin : vec4<f32>,            // explicit-ray probe: world origin
  probe_dir : vec4<f32>,               // (dx, dy, dz, enabled) \u2014 w>0 uses this ray instead of the cursor
};
@group(0) @binding(0) var<uniform> u_cam : Camera;
@group(0) @binding(1) var<uniform> u_material : Material;
${this.usesSampler() ? "@group(0) @binding(2) var s_lin : sampler;" : ""}
${decls}

struct Varyings { @builtin(position) position : vec4<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> Varyings {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : Varyings; o.position = vec4<f32>(x, y, 0.0, 1.0); return o;
}
fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
fn ndc_to_world(ndc : vec4<f32>) -> vec3<f32> { let w = u_cam.inv_view_proj * ndc; return w.xyz / w.w; }
fn ign(p : vec2<f32>) -> f32 { return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715)))); }
${fns}

// PRODUCER (fs_trace): march the ray and return the composited PREMULTIPLIED sample
// (integrated.rgb, integrated.a) BEFORE the background composite \u2014 a "traced pixel". The
// Reconstructor (fs_resolve / reconstructor.ts) composites it over the background. Splitting
// trace from assemble is the seam the unified local/remote pipeline turns on (see
// docs/UNIFIED-RENDERING-PLAN.md); the background composite is identical to the fused path, so
// output is byte-identical at full density. An empty slab returns transparent (0) \u2192 resolve = bg.
@fragment
fn fs_trace(v : Varyings) -> @location(0) vec4<f32> {
  let size = u_cam.size.xy;
  let ndc_x = (v.position.x / size.x) * 2.0 - 1.0;
  let ndc_y = 1.0 - (v.position.y / size.y) * 2.0;
  let ro = ndc_to_world(vec4<f32>(ndc_x, ndc_y, 0.0, 1.0));
  let rd = normalize(ndc_to_world(vec4<f32>(ndc_x, ndc_y, 1.0, 1.0)) - ro);

  let inv = vec3<f32>(1.0) / rd;
  let tb = (u_material.bmin.xyz - ro) * inv;
  let tt = (u_material.bmax.xyz - ro) * inv;
  let tmn = min(tt, tb); let tmx = max(tt, tb);
  var t_near = max(max(tmn.x, tmn.y), tmn.z);
  var t_far  = min(min(tmx.x, tmx.y), tmx.z);
  if (t_far <= t_near || t_far <= 0.0) { return vec4<f32>(0.0); }

  let step = max(u_material.scene.x, 1e-3);
  t_near = max(t_near + step, 0.0);
  t_far  = t_far - step;
  if (t_far <= t_near) { return vec4<f32>(0.0); }
  let seed = ign(v.position.xy);
  var t = t_near;
  var integrated = vec4<f32>(0.0);
  var safety : i32 = 0;
  var saturated = false;   // LATCH: once opaque, normal fields stay off even after a ghost
                           // handle dims the accumulation (else the volume behind the handle
                           // would re-opaque over it and re-bury the shine-through).
  var g_op = 0.0;          // ghost (handle) surface: max opacity along the ray (0.5 inactive /
  var g_col = vec3<f32>(0.0);  // 1.0 active) and its colour \u2014 tracked, never accumulated.
${skipInit}
  loop {
    if (t >= t_far || safety >= 5000${hasGhost ? "" : " || integrated.a >= 0.99"}) { break; }
    // Per-(pixel, step, ACCUM FRAME) ray-offset jitter. The frame term (u_cam.size.w, the
    // accumulation index) is what makes temporal AA actually converge: with a frame-invariant
    // offset the jitter turns banding into FIXED-PATTERN noise that averaging can never remove
    // (measured: 32 samples was as grainy as 1). Varying it per frame decorrelates the samples
    // so the mean approaches the true integral \u2014 no banding AND no noise. size.w is 0 for every
    // non-accumulating path, so frame 1 stays byte-identical to a plain renderToView.
    // Base offset: decorrelated per (pixel, step) so a single frame shows noise, not banding.
    let jbase = fract(sin(dot(v.position.xy + vec2<f32>(f32(safety) * 0.7548, f32(safety) * 0.5698), vec2<f32>(12.9898, 78.233))) * 43758.5453);
    // Advance it across accumulation frames by the golden-ratio additive recurrence
    // (Cranley-Patterson rotation). MEASURED: this converges at the same 1/sqrt(n) rate as an
    // independent random offset per frame (high-freq energy 1.36 vs 1.31 at n=64) \u2014 the low-
    // discrepancy walk is NOT faster here, because the variance is dominated by the step size
    // against a sharp transfer function, not by the sequence. Kept because it is deterministic
    // and costs nothing; reduce sampleStep if you need less residual speckle.
    // At size.w = 0 this is exactly jbase, so the first accumulated frame stays byte-identical
    // to a plain renderToView \u2014 the property render/test baselines depend on.
    let js = fract(jbase + u_cam.size.w * 0.6180339887) - 0.5;
    let wp = ro + rd * (t + js * step);
    var sum = vec4<f32>(0.0);
    var all_defer = true;        // every field guarantees emptiness here -> we may leap
    var jump_t = 1.0e30;         // nearest field horizon
    var clipped = false;         // ROI clip: sample on the negative side of any active plane
    let ccount = u32(u_material.clip_count.x);
    for (var ci = 0u; ci < ccount; ci = ci + 1u) {
      let cp = u_material.clip_planes[ci];
      if (dot(wp, cp.xyz) + cp.w < 0.0) { clipped = true; break; }
    }
    // Normal fields stop being sampled once the ray is opaque (latched); GHOST fields keep
    // their skip horizons and keep going, so a handle behind an opaque region still shines
    // through and the ray LEAPS between handles on the ghost skip (early-termination kept).
${hasGhost ? "    if (integrated.a >= 0.99) { saturated = true; }\n    if (!saturated) {" : ""}
${dispatch}
      if (sum.a > 0.0) { integrated = integrated + (1.0 - integrated.a) * vec4<f32>(sum.rgb, clamp(sum.a, 0.0, 1.0)); }
${hasGhost ? "    }" : ""}
${ghostDispatch}
    if (all_defer && jump_t > t + step) { t = jump_t; } else { t = t + step; }
    safety = safety + 1;
  }
  // GHOST x-ray, applied ONCE (never compounding): the volume IN FRONT of a handle is shown
  // at residual = 1 - handle_opacity (50% for an inactive handle at opacity 0.5, 0% for an
  // active/hovered handle at opacity 1.0), then the handle (colour g_col at opacity g_op)
  // draws over it.
  if (g_op > 0.001) {
    let ga = clamp(g_op, 0.0, 1.0);
    let residual = 1.0 - ga;
    let fA = integrated.a * residual;
    integrated = vec4<f32>(integrated.rgb * residual + (1.0 - fA) * g_col * ga, fA + (1.0 - fA) * ga);
  }
  return integrated;   // premultiplied (rgb, a); resolve composites over the background
}

// PICK: trace the cursor ray (pick_cursor NDC) through the SAME field compositing and return the
// world (RAS) position where front-to-back opacity first crosses 50% \u2014 Slicer's 3D volume pick.
// Output: (wp.x, wp.y, wp.z, hit). hit=0 means the ray never reached 50% (empty/miss).
@fragment
fn fs_pick() -> @location(0) vec4<f32> {
  // Two ray sources: the screen cursor (pick) or an explicit world ray (probe). The explicit
  // form exists because the cursor ray can only ever probe what is ON SCREEN \u2014 useless for
  // "how much room is BEHIND me?", which endovascular navigation needs for reverse and for
  // lateral clearance.
  var ro = ndc_to_world(vec4<f32>(u_material.pick_cursor.x, u_material.pick_cursor.y, 0.0, 1.0));
  var rd = normalize(ndc_to_world(vec4<f32>(u_material.pick_cursor.x, u_material.pick_cursor.y, 1.0, 1.0)) - ro);
  if (u_material.probe_dir.w > 0.5) {
    ro = u_material.probe_origin.xyz;
    rd = normalize(u_material.probe_dir.xyz);
  }
  let inv = vec3<f32>(1.0) / rd;
  let tb = (u_material.bmin.xyz - ro) * inv;
  let tt = (u_material.bmax.xyz - ro) * inv;
  let tmn = min(tt, tb); let tmx = max(tt, tb);
  var t_near = max(max(tmn.x, tmn.y), tmn.z);
  var t_far  = min(min(tmx.x, tmx.y), tmx.z);
  if (t_far <= t_near || t_far <= 0.0) { return vec4<f32>(0.0); }
  let step = max(u_material.scene.x, 1e-3);
  t_near = max(t_near + step, 0.0);
  t_far  = t_far - step;
  var t = t_near;
  var acc = 0.0;
  var safety : i32 = 0;
  loop {
    if (t >= t_far || safety >= 5000 || acc >= 0.5) { break; }
    let wp = ro + rd * t;
    var clipped = false;
    let ccount = u32(u_material.clip_count.x);
    for (var ci = 0u; ci < ccount; ci = ci + 1u) {
      let cp = u_material.clip_planes[ci];
      if (dot(wp, cp.xyz) + cp.w < 0.0) { clipped = true; break; }
    }
    var sum = vec4<f32>(0.0);
${pickDispatch}
    if (sum.a > 0.0) {
      let a_new = acc + (1.0 - acc) * clamp(sum.a, 0.0, 1.0);
      if (a_new >= 0.5) { return vec4<f32>(wp, 1.0); }   // 50% crossing -> the pick point
      acc = a_new;
    }
    t = t + step;
  }
  return vec4<f32>(0.0);
}`
    );
  }
  setBackground(r, g, b) {
    this.mat[12] = r;
    this.mat[13] = g;
    this.mat[14] = b;
    this.mat[15] = 1;
  }
  setSampleStep(step) {
    this.mat[8] = step;
  }
  /** Van der Corput / Halton radical inverse in `base`. */
  static halton(i, base) {
    let f = 1, r = 0;
    while (i > 0) {
      f /= base;
      r += f * (i % base);
      i = Math.floor(i / base);
    }
    return r;
  }
  /** Set up to 8 clip planes (nx,ny,nz,offset), inward-normal, keep-side `dot(wp,n)+offset>=0`.
   *  Written into the uniform tail — a Tier-A update the next flush() uploads; no rebuild. */
  setClipPlanes(planes) {
    const n = Math.min(planes.length, 8);
    for (let i = 0; i < n; i++) this.mat.set(planes[i], this.clipOff + i * 4);
    this.mat[this.clipOff + 32] = n;
  }
  clearClip() {
    this.mat[this.clipOff + 32] = 0;
  }
  /** Axis-aligned RAS crop box [lo,hi] → 6 inward planes. offset = -dot(faceOrigin, n). */
  setClipBox(lo, hi) {
    this.setClipPlanes([
      [1, 0, 0, -lo[0]],
      [-1, 0, 0, hi[0]],
      // keep lo.x <= x <= hi.x
      [0, 1, 0, -lo[1]],
      [0, -1, 0, hi[1]],
      [0, 0, 1, -lo[2]],
      [0, 0, -1, hi[2]]
    ]);
  }
  /** Scene AABB = union of field AABBs; also picks a default sample step from the smallest field extent. */
  recomputeBounds() {
    if (!this.placed.length) return;
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (const p of this.placed) {
      const [a, b] = p.field.aabb();
      for (let i = 0; i < 3; i++) {
        mn[i] = Math.min(mn[i], a[i]);
        mx[i] = Math.max(mx[i], b[i]);
      }
    }
    this.mat[0] = mn[0];
    this.mat[1] = mn[1];
    this.mat[2] = mn[2];
    this.mat[4] = mx[0];
    this.mat[5] = mx[1];
    this.mat[6] = mx[2];
  }
  /** Tier-A interactive update: re-pack every field's uniform block into the resident
   *  material buffer WITHOUT recompiling the pipeline or rebuilding the bind group. This is
   *  the render-side of the interaction architecture (ARCHITECTURE-2026-07-24 §7): a
   *  lightweight drag — clip planes, ROI box geometry, fiducial position, TPS displacement
   *  grid — mutates node state, the field re-derives its uniforms, and the SAME per-frame
   *  flush() the renderer already does uploads them. Cost is a CPU re-pack; no shader build.
   *
   *  Also refreshes the scene AABB (which is uniform-resident), so a moved field's ray-clip
   *  bounds stay correct. REQUIRES the field SET and each field's uniformFloats() to be
   *  unchanged since build() — geometry/appearance may change, STRUCTURE may not. A structural
   *  change (add/remove a field, a field that resizes its uniform block, or a texture swap
   *  needing refreshBindings) still goes through build()/refreshBindings(). This is exactly
   *  why moving geometry must be uniform-resident, never baked into generated WGSL — see the
   *  box-skip note above and RENDER-PERFORMANCE.md. */
  syncUniforms() {
    for (const p of this.placed) p.field.fillUniforms(this.mat, p.uoff);
    this.recomputeBounds();
  }
  /** Rebuild the bind group from the fields' current resources (e.g. after a field
   *  swapped a texture) without recompiling the pipeline. Field set/structure must be unchanged. */
  refreshBindings() {
    this.bind = this.dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    this.streamBind = this.dev.createBindGroup({ layout: this.streamPipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
    if (this.pickPipeline) this.pickBind = this.dev.createBindGroup({ layout: this.pickPipeline.getBindGroupLayout(0), entries: this.bindGroupEntries() });
  }
  /** Only fields with texture bindings use the shared sampler. `layout: "auto"` derives the
   *  layout from what the shader ACTUALLY references, so in a scene of purely procedural
   *  fields (e.g. fiducials/markups only) binding 2 is absent from the layout — supplying it
   *  anyway fails validation and the whole view silently renders nothing. Emit the sampler
   *  declaration and its bind entry under the SAME condition so the two can't drift. */
  usesSampler() {
    return this.placed.some((p) => p.field.bindingCount > 0);
  }
  bindGroupEntries() {
    const entries = [
      { binding: 0, resource: { buffer: this.camBuf } },
      { binding: 1, resource: { buffer: this.matBuf } }
    ];
    if (this.usesSampler()) entries.push({ binding: 2, resource: this.sampler });
    for (const p of this.placed) entries.push(...p.field.bindEntries(p.slot, p.bbase));
    return entries;
  }
  setCamera(eye, center, up, fovyDeg, width, height) {
    const view = lookAt(eye, center, up);
    const proj = perspectiveZO(fovyDeg * Math.PI / 180, width / height, 1, 1e5);
    const invVP = invert(multiply(proj, view));
    this.baseInvVP = invVP;
    const cam = new Float32Array(24);
    cam.set(invVP, 0);
    this.focalPx = height / 2 / Math.tan(fovyDeg * Math.PI / 360);
    cam[16] = width;
    cam[17] = height;
    cam[18] = height / 2 / Math.tan(fovyDeg * Math.PI / 360);
    cam[19] = 0;
    cam[20] = eye[0];
    cam[21] = eye[1];
    cam[22] = eye[2];
    this.dev.queue.writeBuffer(this.camBuf, 0, cam);
  }
  /** Camera for ONE TILE of the view: the same rays the full frame would cast for `rect`, into a
   *  rect.w×rect.h target. Screen-space glyph sizing stays keyed to the FULL view height, so a
   *  patch of the gizmo is drawn at exactly the size the full frame drew it. Pair with
   *  traceSamples(rect.w, rect.h) — its focal rewrite is then a no-op. */
  setCameraTile(eye, center, up, fovyDeg, viewW, viewH, rect) {
    const view = lookAt(eye, center, up);
    const proj = perspectiveZOTile(fovyDeg * Math.PI / 180, viewW, viewH, rect.x, rect.y, rect.w, rect.h, 1, 1e5);
    const invVP = invert(multiply(proj, view));
    this.baseInvVP = invVP;
    const cam = new Float32Array(24);
    cam.set(invVP, 0);
    this.focalPx = viewH / 2 / Math.tan(fovyDeg * Math.PI / 360);
    cam[16] = rect.w;
    cam[17] = rect.h;
    cam[18] = this.focalPx;
    cam[19] = 0;
    cam[20] = eye[0];
    cam[21] = eye[1];
    cam[22] = eye[2];
    this.dev.queue.writeBuffer(this.camBuf, 0, cam);
  }
  flush() {
    this.dev.queue.writeBuffer(this.matBuf, 0, this.mat);
  }
  /** Ray-trace the cursor (u,v in [0,1], y down) through the composited fields and return the
   *  RAS point where front-to-back opacity first reaches 50% — Slicer's 3D volume pick. Traces
   *  whatever renders (DVR volumes, SegmentField iso shells, RGBA), EXCLUDING ghost handles.
   *  Uses the camera set by the last setCamera(); returns null if the ray never reaches 50%. */
  async pick(u, v) {
    if (!this.pickPipeline || !this.pickBind || !this.placed.length) return null;
    return this.serialise(async () => {
      this.mat[this.pickOff] = u * 2 - 1;
      this.mat[this.pickOff + 1] = 1 - v * 2;
      this.mat[this.pickOff + 11] = 0;
      this.flush();
      return await this.tracePick();
    });
  }
  /** Trace an EXPLICIT world ray and return the distance (mm) to the first point where
   *  front-to-back opacity reaches 50%, or Infinity if it never does. Unlike pick(), the ray
   *  is independent of the camera, so it can look backwards and sideways — which is what makes
   *  collision "rails" possible in a first-person flythrough. */
  async probe(origin, dir) {
    if (!this.pickPipeline || !this.pickBind || !this.placed.length) return Infinity;
    const l = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    return this.serialise(async () => {
      this.mat[this.pickOff + 4] = origin[0];
      this.mat[this.pickOff + 5] = origin[1];
      this.mat[this.pickOff + 6] = origin[2];
      this.mat[this.pickOff + 8] = dir[0] / l;
      this.mat[this.pickOff + 9] = dir[1] / l;
      this.mat[this.pickOff + 10] = dir[2] / l;
      this.mat[this.pickOff + 11] = 1;
      this.flush();
      const hit = await this.tracePick();
      this.mat[this.pickOff + 11] = 0;
      this.flush();
      if (!hit) return Infinity;
      return Math.hypot(hit[0] - origin[0], hit[1] - origin[1], hit[2] - origin[2]);
    });
  }
  /** Serialises pick/probe. They share ONE uniform buffer and ONE readback buffer, so
   *  concurrent calls would overwrite each other's ray and double-map the buffer — a
   *  Promise.all of probes silently returns garbage. Callers may fire as many as they like;
   *  they queue here. */
  pickChain = Promise.resolve();
  serialise(fn) {
    const next = this.pickChain.then(fn, fn);
    this.pickChain = next.catch(() => {
    });
    return next;
  }
  /** The shared 1x1 render + readback behind pick() and probe(). */
  async tracePick() {
    if (!this.pickTarget) {
      this.pickTarget = this.dev.createTexture({ size: [1, 1], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      this.pickReadBuf = this.dev.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    const enc = this.dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.pickTarget.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    pass.setPipeline(this.pickPipeline);
    pass.setBindGroup(0, this.pickBind);
    pass.draw(3);
    pass.end();
    enc.copyTextureToBuffer({ texture: this.pickTarget }, { buffer: this.pickReadBuf, bytesPerRow: 256, rowsPerImage: 1 }, [1, 1]);
    this.dev.queue.submit([enc.finish()]);
    await this.pickReadBuf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(this.pickReadBuf.getMappedRange().slice(0, 16));
    this.pickReadBuf.unmap();
    return r[3] > 0.5 ? [r[0], r[1], r[2]] : null;
  }
  renderToView(view, width, height) {
    this.ensureTrace(width, height);
    this.flush();
    this.dev.queue.writeBuffer(this.resolveBgBuf, 0, this.mat.subarray(12, 16));
    const enc = this.dev.createCommandEncoder();
    this.encodeFrame(enc, view);
    this.dev.queue.submit([enc.finish()]);
  }
  /** Exact GPU time of the ray-march pass (median ms over `iters`), via timestamp-query.
   *  Times ONLY the render pass — no texture copy/readback — so it reflects shader cost.
   *  Returns NaN if the device lacks timestamp-query. Deno gives full-resolution timestamps;
   *  Chrome quantizes them unless cross-origin isolated, so profile headless for sharp numbers. */
  async timePass(width, height, iters = 40) {
    if (!this.canTime) return NaN;
    this.flush();
    const target = this.dev.createTexture({ size: [width, height], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const view = target.createView();
    const qs = this.dev.createQuerySet({ type: "timestamp", count: 2 });
    const resolve = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const samples = [];
    for (let i = 0; i < iters; i++) {
      const enc = this.dev.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bind);
      pass.draw(3);
      pass.end();
      enc.resolveQuerySet(qs, 0, 2, resolve, 0);
      enc.copyBufferToBuffer(resolve, 0, read, 0, 16);
      this.dev.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(read.getMappedRange());
      const ms = Number(t[1] - t[0]) / 1e6;
      read.unmap();
      if (ms > 0 && Number.isFinite(ms)) samples.push(ms);
    }
    target.destroy();
    qs.destroy();
    resolve.destroy();
    read.destroy();
    if (!samples.length) return NaN;
    samples.sort((a, b) => a - b);
    return samples[samples.length >> 1];
  }
  async renderToRGBA(width, height) {
    this.ensureTrace(width, height);
    this.flush();
    this.dev.queue.writeBuffer(this.resolveBgBuf, 0, this.mat.subarray(12, 16));
    const target = this.dev.createTexture({ size: [width, height], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const enc = this.dev.createCommandEncoder();
    this.encodeFrame(enc, target.createView());
    const bpr = Math.ceil(width * 4 / 256) * 256;
    const buf = this.dev.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: height }, [width, height]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) out.set(padded.subarray(y * bpr, y * bpr + width * 4), y * width * 4);
    buf.unmap();
    target.destroy();
    buf.destroy();
    return out;
  }
  /** REMOTE PRODUCER (M3): trace at width×height and read back the PREMULTIPLIED sample (pre-
   *  background) as tightly-packed rgba8 — the bytes streamed to the remote client, which runs the
   *  same reconstruction (upsample + background composite) the local resolve does. The caller sets
   *  the camera to width×height first (like renderUpscaled). Returns width*height*4 bytes. */
  async traceSamples(width, height, viewH = height) {
    this.flush();
    this.dev.queue.writeBuffer(this.camBuf, 64, new Float32Array([width, height]));
    this.dev.queue.writeBuffer(this.camBuf, 72, new Float32Array([this.focalPx * (viewH / height)]));
    const target = this.dev.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const enc = this.dev.createCommandEncoder();
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.streamPipeline);
    tp.setBindGroup(0, this.streamBind);
    tp.draw(3);
    tp.end();
    const bpr = Math.ceil(width * 4 / 256) * 256;
    const buf = this.dev.createBuffer({ size: bpr * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: height }, [width, height]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) out.set(padded.subarray(y * bpr, y * bpr + width * 4), y * width * 4);
    buf.unmap();
    target.destroy();
    buf.destroy();
    return out;
  }
};

// render/fields.ts
function transformedAABB(m, lo, hi) {
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 8; i++) {
    const c = applyMat4(m, [i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]]);
    for (let a = 0; a < 3; a++) {
      mn[a] = Math.min(mn[a], c[a]);
      mx[a] = Math.max(mx[a], c[a]);
    }
  }
  return [mn, mx];
}
var ImageField = class {
  kind = "img";
  bindingCount = 2;
  // volume (3d) + lut (2d)
  volTex;
  lutTex;
  dev;
  p2t;
  clim;
  shade;
  unit;
  stepMm;
  box;
  normScale = 1;
  // r8unorm samples return raw/255; clim is packed /normScale so shader math is unchanged
  constructor(dev, data, dims, spacing, lut, opts) {
    const center = opts.center ?? [0, 0, 0];
    let src = data, fmt = "r32float", bpe = 4;
    this.normScale = 1;
    if (data instanceof Uint8Array) {
      fmt = "r8unorm";
      bpe = 1;
      this.normScale = 255;
    } else if (data instanceof Uint16Array) {
      src = Float32Array.from(data);
      fmt = "r32float";
      bpe = 4;
    }
    this.volTex = dev.createTexture({ size: dims, dimension: "3d", format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    {
      const bytesPerRow = dims[0] * bpe, rowsPerImage = dims[1], sliceBytes = bytesPerRow * rowsPerImage;
      const CHUNK = 256 * 1024 * 1024;
      const slab = Math.max(1, Math.min(dims[2], Math.floor(CHUNK / Math.max(1, sliceBytes))));
      for (let z = 0; z < dims[2]; z += slab) {
        const depth = Math.min(slab, dims[2] - z);
        dev.queue.writeTexture(
          { texture: this.volTex, origin: { x: 0, y: 0, z } },
          src,
          { offset: z * sliceBytes, bytesPerRow, rowsPerImage },
          [dims[0], dims[1], depth]
        );
      }
    }
    this.lutTex = dev.createTexture({ size: [256, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    dev.queue.writeTexture({ texture: this.lutTex }, lut, { bytesPerRow: 256 * 4 }, [256, 1]);
    if (opts.ijkToRAS) {
      this.p2t = patientToTextureFromIjkToRAS(opts.ijkToRAS, dims);
      this.box = volumeAABBFromIjkToRAS(opts.ijkToRAS, dims);
      this.stepMm = Math.min(...spacingFromIjkToRAS(opts.ijkToRAS));
    } else {
      this.p2t = patientToTexture(dims, spacing, center);
      this.box = volumeAABB(dims, spacing, center);
      this.stepMm = Math.min(...spacing);
    }
    this.clim = opts.clim;
    this.shade = opts.shade ?? [0.35, 0.75, 0.35, 20];
    this.unit = opts.opacityUnitDistance ?? this.stepMm;
    this.dev = dev;
  }
  /** Replace the 256-entry rgba8 color/opacity LUT in place (no texture/bind-group churn).
   *  The bind group holds a stable view of lutTex, so the next render uses the new LUT. */
  setLUT(lut) {
    this.dev.queue.writeTexture({ texture: this.lutTex }, lut, { bytesPerRow: 256 * 4 }, [256, 1]);
  }
  /** The scalar range the LUT spans — window/level for the volume rendering. Re-packed into
   *  the material uniform on the next syncUniforms()/render, so no pipeline rebuild. */
  setClim(lo, hi) {
    this.clim = [lo, hi];
  }
  getClim() {
    return [this.clim[0], this.clim[1]];
  }
  /** Phong shading tuple [ka, kd, ks, shininess] — re-packed into the material uniform next
   *  render (VR presets carry their own lighting). [1,0,0,1] = flat emission (no shading). */
  setShade(shade) {
    this.shade = [shade[0], shade[1], shade[2], shade[3]];
  }
  origP2t;
  // sampling matrix + box at identity, for setWorldTransform
  origBox;
  uniformFloats() {
    return 28;
  }
  // mat4(16) + clim(4) + shade(4) + params(4)
  aabb() {
    return this.box;
  }
  sampleStep() {
    return this.stepMm;
  }
  /** The r32float 3D scalar texture (e.g. to share with a SliceRenderer for MPR). */
  volumeTexture() {
    return this.volTex;
  }
  /** Centre of the volume in world (RAS) at identity — a natural pivot for a transform widget. */
  worldCenter() {
    const [lo, hi] = this.origBox ?? this.box;
    return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  }
  /** Place the volume in the world by a rigid transform M (worldFromLocal): the ray samples
   *  at p2t·M⁻¹·wp, so the volume appears moved/rotated. A Tier-A interactive update — caller
   *  does scene.syncUniforms() (which re-packs p2t AND refreshes the ray-entry AABB). */
  setWorldTransform(m) {
    if (!this.origP2t) {
      this.origP2t = this.p2t;
      this.origBox = this.box;
    }
    this.p2t = multiply(this.origP2t, invert(m));
    this.box = transformedAABB(m, this.origBox[0], this.origBox[1]);
  }
  /** RAS(patient) -> texture[0,1] matrix (encodes the real ijkToRAS geometry). */
  patientToTexture() {
    return this.p2t;
  }
  structMembers(s) {
    return [
      `  img${s}_p2t : mat4x4<f32>,`,
      `  img${s}_clim : vec4<f32>,`,
      // lo, hi, _, _
      `  img${s}_shade : vec4<f32>,`,
      // ka, kd, ks, shininess
      `  img${s}_params : vec4<f32>,`
      // opacity_unit_distance, _, _, _
    ].join("\n");
  }
  declareBindings(s, base) {
    return [
      `@group(0) @binding(${base}) var t_vol_img${s} : texture_3d<f32>;`,
      `@group(0) @binding(${base + 1}) var t_lut_img${s} : texture_2d<f32>;`
    ].join("\n");
  }
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn sampc_img${s}(wp : vec3<f32>) -> f32 {
  let t4 = u_material.img${s}_p2t * vec4<f32>(transform_point_img${s}(wp), 1.0);
  return textureSampleLevel(t_vol_img${s}, s_lin, clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r;
}
fn sample_field_img${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let t4 = u_material.img${s}_p2t * vec4<f32>(transform_point_img${s}(wp), 1.0);
  let tex = t4.xyz;
  if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) { return vec4<f32>(0.0); }
  let val = textureSampleLevel(t_vol_img${s}, s_lin, tex, 0.0).r;
  let lo = u_material.img${s}_clim.x; let hi = u_material.img${s}_clim.y;
  let tf = textureSampleLevel(t_lut_img${s}, s_lin, vec2<f32>(clamp((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0), 0.5), 0.0);
  let step = u_material.scene.x;
  let unit = max(u_material.img${s}_params.x, 1e-3);
  let opacity = clamp(1.0 - pow(1.0 - clamp(tf.a, 0.0, 1.0), step / unit), 0.0, 1.0);
  if (opacity <= 0.001) { return vec4<f32>(0.0); }
  let h = step * 2.0;   // wider central difference -> smoother normals (less shading aliasing on coarse volumes)
  let g = vec3<f32>(
    sampc_img${s}(wp + vec3<f32>(h,0,0)) - sampc_img${s}(wp - vec3<f32>(h,0,0)),
    sampc_img${s}(wp + vec3<f32>(0,h,0)) - sampc_img${s}(wp - vec3<f32>(0,h,0)),
    sampc_img${s}(wp + vec3<f32>(0,0,h)) - sampc_img${s}(wp - vec3<f32>(0,0,h))) / (2.0 * h);
  let glen = length(g);
  let ka = u_material.img${s}_shade.x; let kd = u_material.img${s}_shade.y;
  let ks = u_material.img${s}_shade.z; let sh = u_material.img${s}_shade.w;
  var lit_srgb = tf.rgb * ka;
  if (glen > 1e-6) {
    var n = g / glen;
    if (dot(n, -rd) < 0.0) { n = -n; }
    let view_dir = normalize(-rd);
    let ldotn = dot(view_dir, n);
    if (ldotn > 0.0) {
      let refl = normalize(2.0 * ldotn * n - view_dir);
      let rdotv = max(0.0, dot(refl, view_dir));
      lit_srgb = tf.rgb * (ka + kd * ldotn) + vec3<f32>(ks * pow(rdotv, sh));
    }
  }
  let lit = srgb2physical(clamp(lit_srgb, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * opacity, opacity);
}`
    );
  }
  fillUniforms(out, off) {
    out.set(this.p2t, off);
    out[off + 16] = this.clim[0] / this.normScale;
    out[off + 17] = this.clim[1] / this.normScale;
    out[off + 20] = this.shade[0];
    out[off + 21] = this.shade[1];
    out[off + 22] = this.shade[2];
    out[off + 23] = this.shade[3];
    out[off + 24] = this.unit;
  }
  bindEntries(_s, base) {
    return [
      { binding: base, resource: this.volTex.createView() },
      { binding: base + 1, resource: this.lutTex.createView() }
    ];
  }
};
var SegmentField = class {
  kind = "seg";
  bindingCount;
  // 1 (value texture) + 1 when an sdf attr (opacity) texture is bound
  clippable;
  tex;
  attrTex;
  // sdf per-voxel attributes (.r = opacity)
  p2t;
  box;
  color;
  opacity;
  shade;
  bandMm;
  stepMm;
  mode;
  colorFromTex;
  interfaceMode;
  constructor(tex, dims, spacing, opts) {
    this.tex = tex;
    const center = opts.center ?? [0, 0, 0];
    let voxelMm;
    if (opts.ijkToRAS) {
      this.p2t = patientToTextureFromIjkToRAS(opts.ijkToRAS, dims);
      this.box = volumeAABBFromIjkToRAS(opts.ijkToRAS, dims);
      voxelMm = Math.min(...spacingFromIjkToRAS(opts.ijkToRAS));
    } else {
      this.p2t = patientToTexture(dims, spacing, center);
      this.box = volumeAABB(dims, spacing, center);
      voxelMm = Math.min(...spacing);
    }
    this.color = opts.color;
    this.opacity = opts.opacity ?? 1;
    this.shade = opts.shade ?? [0.2, 0.85, 0.3, 32];
    this.bandMm = opts.bandMm ?? voxelMm;
    this.stepMm = opts.sampleStepMm ?? Math.max(0.5 * voxelMm, 0.1);
    this.clippable = opts.clippable ?? true;
    this.mode = opts.mode ?? "iso";
    this.colorFromTex = opts.colorFromTexture ?? false;
    this.interfaceMode = this.mode === "sdf" && (opts.interfaceMode ?? false);
    this.attrTex = this.mode === "sdf" ? opts.attrTexture : void 0;
    this.bindingCount = this.attrTex ? 2 : 1;
  }
  /** Field-level opacity (multiplies every segment's per-label opacity in the shader). Live global
   *  segmentation opacity — the caller does scene.syncUniforms() + redraw to apply. */
  setOpacity(o) {
    this.opacity = Math.max(0, Math.min(1, o));
  }
  uniformFloats() {
    return 28;
  }
  // mat4(16) + color(4) + shade(4) + params(4)
  aabb() {
    return this.box;
  }
  sampleStep() {
    return this.stepMm;
  }
  setTexture(tex, destroyPrev = true) {
    if (destroyPrev && this.tex !== tex) this.tex.destroy();
    this.tex = tex;
  }
  structMembers(s) {
    return [
      `  seg${s}_p2t : mat4x4<f32>,`,
      `  seg${s}_color : vec4<f32>,`,
      // rgb, opacity
      `  seg${s}_shade : vec4<f32>,`,
      // ka, kd, ks, shininess
      `  seg${s}_params : vec4<f32>,`
      // band_mm, _, _, _
    ].join("\n");
  }
  declareBindings(s, base) {
    const value = `@group(0) @binding(${base}) var t_seg${s} : texture_3d<f32>;`;
    return this.attrTex ? `${value}
@group(0) @binding(${base + 1}) var t_attr${s} : texture_3d<f32>;` : value;
  }
  samplingWGSL(s) {
    if (this.mode === "sdf") {
      return (
        /* wgsl */
        `
fn v_seg${s}(wp : vec3<f32>) -> f32 {   // signed distance (mm)
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return 1e3; }   // far outside \u2192 culled
  return textureSampleLevel(t_seg${s}, s_lin, t, 0.0).a;
}
fn vgrad_seg${s}(wp : vec3<f32>) -> f32 {   // signed distance for the NORMAL finite-difference
  // A gradient tap that steps just outside the (padded) SDF texture must read BACKGROUND, not the
  // out-of-volume cull sentinel (1e3) \u2014 a huge sentinel would fabricate an enormous fake gradient that
  // points out through the volume face and unlights the surface (black speckle at the seg/boundary
  // interface). Clamp to the texture edge: with the padded grid that edge IS background, so the normal
  // near a cap stays correct. This is the "artificial background boundary sample" for the normal.
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  return textureSampleLevel(t_seg${s}, s_lin, t, 0.0).a;
}
fn col_seg${s}(wp : vec3<f32>) -> vec3<f32> {   // per-label colour of the nearest region
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec3<f32>(0.0); }
  let pm = textureSampleLevel(t_seg${s}, s_lin, t, 0.0).rgb;   // PREMULTIPLIED (rgb\xB7opacity)${this.attrTex ? `
  let a = textureSampleLevel(t_attr${s}, s_lin, t, 0.0).r;      // per-segment opacity (crisp) \u2014 un-premultiply to the true colour, so a hidden neighbour's colour doesn't bleed in
  return pm / max(a, 1e-3);` : `
  return pm;`}
}${this.attrTex ? `
fn attr_seg${s}(wp : vec3<f32>) -> vec2<f32> {   // per-segment (.x = opacity, .y = shading mode)
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec2<f32>(0.0); }
  return textureSampleLevel(t_attr${s}, s_lin, t, 0.0).rg;
}` : ""}${this.interfaceMode ? `
fn bdist_seg${s}(wp : vec3<f32>) -> f32 {   // SMOOTH (seam-blurred) interface distance from attr.b \u2014 for the normal only
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  return textureSampleLevel(t_attr${s}, s_lin, t, 0.0).b;
}
fn pres_seg${s}(wp : vec3<f32>) -> f32 {   // CRISP in-segment presence (attr.a), linear-sampled for ~1-voxel AA
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return 0.0; }
  return textureSampleLevel(t_attr${s}, s_lin, t, 0.0).a;
}
fn g1_seg${s}(c : f32, wp : vec3<f32>, dir : vec3<f32>, h : f32) -> f32 {
  // One-sided difference of the SMOOTH interface distance along dir, picking the STEEPER side. The
  // unsigned distance has a V-crease at the surface: a central difference straddling it cancels (fake
  // zero gradient \u2192 degenerate normal). Taking the steeper one-sided difference follows the true \xB11
  // slope AWAY from the interface, so the normal stays well-defined right at the shell \u2014 computed on
  // the fly only at shell samples, no stored normal. Uses the blurred distance (attr.b) so the normal
  // is smooth (no JFA facets); the SHARP sdfTex.a still drives shell membership (surface stays at 0).
  let dp = bdist_seg${s}(wp + dir * h) - c;
  let dm = c - bdist_seg${s}(wp - dir * h);
  return select(dm, dp, abs(dp) > abs(dm)) / h;
}` : ""}
// Shell (surface) contribution at wp: crisp Phong shell around sdf=0. Weighted by (1-mode) so it
// morphs smoothly into the volume contribution across a blurred surface\u2194volume boundary.
fn surface_seg${s}(wp : vec3<f32>, rd : vec3<f32>, sdf : f32, band : f32, step : f32, seg_op : f32, op0 : f32) -> vec4<f32> {
  let d_mm = abs(sdf);
  if (d_mm > band + step) { return vec4<f32>(0.0); }
  let T = clamp(op0 * seg_op, 0.0, 1.0);      // TARGET surface opacity (per-segment \xD7 field)
  if (T <= 0.0) { return vec4<f32>(0.0); }
  let h = step;
${this.interfaceMode ? `  let hg = 1.5 * step;
  let dc = bdist_seg${s}(wp);
  let g = vec3<f32>(
    g1_seg${s}(dc, wp, vec3<f32>(1,0,0), hg),
    g1_seg${s}(dc, wp, vec3<f32>(0,1,0), hg),
    g1_seg${s}(dc, wp, vec3<f32>(0,0,1), hg));` : `  let g = vec3<f32>(
    vgrad_seg${s}(wp + vec3<f32>(h,0,0)) - vgrad_seg${s}(wp - vec3<f32>(h,0,0)),
    vgrad_seg${s}(wp + vec3<f32>(0,h,0)) - vgrad_seg${s}(wp - vec3<f32>(0,h,0)),
    vgrad_seg${s}(wp + vec3<f32>(0,0,h)) - vgrad_seg${s}(wp - vec3<f32>(0,0,h))) / (2.0 * h);`}
  let glen = length(g);
  if (glen < 1e-5) { return vec4<f32>(0.0); }
  var n = g / glen;
  if (dot(n, -rd) < 0.0) { n = -n; }
  // SURFACE opacity (Slicer polydata parity): the shell is a THIN surface of opacity T, not a solid
  // band. A raymarch crosses it in several samples; giving each \u03B1=T lets the front-to-back OVER
  // saturate toward opaque (50% looked like ~100%). Instead accumulate OPTICAL DEPTH with a shell
  // profile \u03C1 = a/band that integrates to 1 across the crossing, scaled by -ln(1-T): \u03A3d\u03C4 = -ln(1-T),
  // so net opacity = 1-e^(-\u03A3d\u03C4) = T EXACTLY \u2014 independent of band thickness and sample rate, and T\u21921
  // stays crisply opaque. |dot(rd,n)| converts ray-step to shell-normal distance (\u21920 at grazing =
  // built-in silhouette AA).
  let a = max(1.0 - d_mm / band, 0.0);
  if (a <= 0.0) { return vec4<f32>(0.0); }
  // Convert ray-step to d_mm-distance. Outer: RAW gradient projection |dot(rd,g)| = |d(d_mm)/ds|
  // (includes |grad sdf|, which the distance blur pulls below 1). Interface: the re-signed gradient has
  // an arbitrary magnitude (a sign jump at the interface), so use the UNIT normal cosine |dot(rd,n)| \u2014
  // still \u21920 at grazing (silhouette AA), and keeps opacity thickness-consistent.
  let rate = max(abs(dot(rd, ${this.interfaceMode ? "n" : "g"})), 1e-3);
  let tau = -log(1.0 - min(T, 0.9999)) * (a / band) * (step * rate);
  var op = 1.0 - exp(-tau);
${this.interfaceMode ? `  op = op * pres_seg${s}(wp);   // gate to GENUINELY in-segment voxels \u2014 kills the bled colour/opacity halo beyond the real edge` : ""}
  if (op <= 0.0004) { return vec4<f32>(0.0); }
  let ka = u_material.seg${s}_shade.x; let kd = u_material.seg${s}_shade.y;
  let ks = u_material.seg${s}_shade.z; let sh = u_material.seg${s}_shade.w;
  let ldn = max(dot(-rd, n), 0.0);
  let refl = normalize(2.0 * ldn * n + rd);
  let rdv = max(dot(refl, -rd), 0.0);
  let col = col_seg${s}(wp);
  var lit = col * ka + col * (kd * ldn) + vec3<f32>(ks * pow(rdv, max(sh, 1.0)));
  lit = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * op, op);
}
fn sample_field_seg${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let op0 = u_material.seg${s}_color.a;
  if (op0 <= 0.0) { return vec4<f32>(0.0); }
  let sdf = v_seg${s}(wp);
  let band = max(u_material.seg${s}_params.x, 1e-3);
  let step = max(u_material.scene.x, 1e-3);
${this.attrTex ? `  let at = attr_seg${s}(wp);        // (opacity, shading mode)
  let seg_op = at.x;
  if (seg_op <= 0.0) { return vec4<f32>(0.0); }
  let mode = clamp(at.y, 0.0, 1.0);
  // Surface and volume are BLENDED by the (seam-blurred, fractional) mode, so an opaque-surface
  // segment and a translucent-volume segment meet with a smooth transition instead of a jagged,
  // voxel-quantized classification edge.
  var acc = vec4<f32>(0.0);
  if (mode > 0.001 && sdf < 0.0) {
    // VOLUME: translucent DVR fill of the interior (~24 mm opacity-unit-distance).
    let vop = clamp(op0 * seg_op * step / 24.0, 0.0, 1.0);
    if (vop > 0.0) {
      let vcol = srgb2physical(clamp(col_seg${s}(wp), vec3<f32>(0.0), vec3<f32>(1.0)));
      acc += mode * vec4<f32>(vcol * vop, vop);
    }
  }
  if (mode < 0.999) {
    acc += (1.0 - mode) * surface_seg${s}(wp, rd, sdf, band, step, seg_op, op0);
  }
  return acc;` : `  return surface_seg${s}(wp, rd, sdf, band, step, 1.0, op0);`}
}`
      );
    }
    const alphaWGSL = this.mode === "surface" ? (
      /* wgsl */
      `
  let step = max(u_material.scene.x, 1e-3);
  let op = clamp(op0 * glen * step, 0.0, 1.0);
  if (op <= 0.0) { return vec4<f32>(0.0); }`
    ) : (
      /* wgsl */
      `
  // Local first-order signed distance to the v=0.5 isosurface (mm), then a
  // 1-voxel opacity band around it: crisp opaque shell, sub-voxel anti-aliased.
  let d_mm = abs((v - 0.5) / glen);
  let band = max(u_material.seg${s}_params.x, 1e-3);
  let a = 1.0 - clamp(d_mm / band, 0.0, 1.0);
  if (a <= 0.0) { return vec4<f32>(0.0); }
  let op = clamp(a * op0, 0.0, 1.0);`
    );
    const colWGSL = this.colorFromTex ? (
      /* wgsl */
      `
fn col_seg${s}(wp : vec3<f32>) -> vec3<f32> {
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec3<f32>(0.0); }
  return textureSampleLevel(t_seg${s}, s_lin, t, 0.0).rgb;
}`
    ) : "";
    const colExpr = this.colorFromTex ? `col_seg${s}(wp)` : `u_material.seg${s}_color.rgb`;
    return (
      /* wgsl */
      `
fn v_seg${s}(wp : vec3<f32>) -> f32 {
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return 0.0; }
  return textureSampleLevel(t_seg${s}, s_lin, t, 0.0).a;   // Gaussian-smoothed presence in .a
}${colWGSL}
fn sample_field_seg${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let op0 = u_material.seg${s}_color.a;
  if (op0 <= 0.0) { return vec4<f32>(0.0); }
  let v = v_seg${s}(wp);
  // Skip deep interior / exterior: |grad| ~ 0 there so no shell to emit.
  if (v <= 0.02 || v >= 0.98) { return vec4<f32>(0.0); }
  let h = max(u_material.scene.x, 1e-3);
  let g = vec3<f32>(
    v_seg${s}(wp + vec3<f32>(h,0,0)) - v_seg${s}(wp - vec3<f32>(h,0,0)),
    v_seg${s}(wp + vec3<f32>(0,h,0)) - v_seg${s}(wp - vec3<f32>(0,h,0)),
    v_seg${s}(wp + vec3<f32>(0,0,h)) - v_seg${s}(wp - vec3<f32>(0,0,h))) / (2.0 * h);
  let glen = length(g);
  if (glen < 1e-5) { return vec4<f32>(0.0); }${alphaWGSL}
  // Phong from the same gradient, normal flipped to face the camera.
  var n = g / glen;
  if (dot(n, -rd) < 0.0) { n = -n; }
  let ka = u_material.seg${s}_shade.x; let kd = u_material.seg${s}_shade.y;
  let ks = u_material.seg${s}_shade.z; let sh = u_material.seg${s}_shade.w;
  let ldn = max(dot(-rd, n), 0.0);
  let refl = normalize(2.0 * ldn * n + rd);
  let rdv = max(dot(refl, -rd), 0.0);
  let col = ${colExpr};
  var lit = col * ka + col * (kd * ldn) + vec3<f32>(ks * pow(rdv, max(sh, 1.0)));
  lit = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * op, op);
}`
    );
  }
  fillUniforms(out, off) {
    out.set(this.p2t, off);
    out[off + 16] = this.color[0];
    out[off + 17] = this.color[1];
    out[off + 18] = this.color[2];
    out[off + 19] = this.opacity;
    out[off + 20] = this.shade[0];
    out[off + 21] = this.shade[1];
    out[off + 22] = this.shade[2];
    out[off + 23] = this.shade[3];
    out[off + 24] = this.bandMm;
  }
  bindEntries(_s, base) {
    const e = [{ binding: base, resource: this.tex.createView() }];
    if (this.attrTex) e.push({ binding: base + 1, resource: this.attrTex.createView() });
    return e;
  }
};

// render/bake.ts
var INIT_WGSL = (
  /* wgsl */
  `
struct U { dims : vec4<u32> };
@group(0) @binding(0) var t_label : texture_3d<u32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u_pal : array<vec4<f32>, 256>;
@group(0) @binding(3) var<uniform> u : U;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let label = textureLoad(t_label, vec3<i32>(gid), 0).r;
  let pal = u_pal[label & 255u];
  let present = select(0.0, 1.0, label != 0u);
  textureStore(t_out, vec3<i32>(gid), vec4<f32>(pal.rgb, present * pal.a));
}`
);
var BLUR_WGSL = (
  /* wgsl */
  `
struct U { dims : vec4<u32>, axis_r : vec4<u32>, w : array<vec4<f32>, 4> };  // axis, radius; half-kernel weights
@group(0) @binding(0) var t_in : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u : U;
fn wt(i : u32) -> f32 { return u.w[i >> 2u][i & 3u]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var av = vec3<i32>(0);
  if (u.axis_r.x == 0u) { av = vec3<i32>(1,0,0); } else if (u.axis_r.x == 1u) { av = vec3<i32>(0,1,0); } else { av = vec3<i32>(0,0,1); }
  let center = textureLoad(t_in, c, 0);
  var asum = center.a * wt(0u);
  let R = i32(u.axis_r.y);
  for (var i = 1; i <= R; i = i + 1) {
    let o = av * i;
    let p1 = clamp(c + o, vec3<i32>(0), dmax);
    let p2 = clamp(c - o, vec3<i32>(0), dmax);
    asum = asum + wt(u32(i)) * (textureLoad(t_in, p1, 0).a + textureLoad(t_in, p2, 0).a);
  }
  textureStore(t_out, c, vec4<f32>(center.rgb, asum));
}`
);
function gaussHalfKernel(sigma) {
  const radius = Math.max(1, Math.min(15, Math.ceil(3 * sigma)));
  const raw = new Float32Array(radius + 1);
  let total = 0;
  for (let i = 0; i <= radius; i++) {
    raw[i] = Math.exp(-(i * i) / (2 * sigma * sigma));
    total += (i === 0 ? 1 : 2) * raw[i];
  }
  const w = new Float32Array(16);
  for (let i = 0; i <= radius; i++) w[i] = raw[i] / total;
  return { radius, w };
}
function bakeColorizeRGBA(dev, labelmap, dims, palette, sigmaVoxels = 1.5) {
  const [dx, dy, dz] = dims;
  const storageUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
  const labelTex = dev.createTexture({ size: dims, dimension: "3d", format: "r8uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  dev.queue.writeTexture({ texture: labelTex }, labelmap, { bytesPerRow: dx, rowsPerImage: dy }, dims);
  const texA = dev.createTexture({ size: dims, dimension: "3d", format: "rgba16float", usage: storageUsage });
  const texB = dev.createTexture({ size: dims, dimension: "3d", format: "rgba16float", usage: storageUsage });
  const palBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const palData = new Float32Array(256 * 4);
  palData.set(palette.subarray(0, Math.min(palette.length, 256 * 4)));
  dev.queue.writeBuffer(palBuf, 0, palData);
  const dimsBuf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(dimsBuf, 0, new Uint32Array([dx, dy, dz, 0]));
  const gx = Math.ceil(dx / 4), gy = Math.ceil(dy / 4), gz = Math.ceil(dz / 4);
  const initPipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: INIT_WGSL }), entryPoint: "main" } });
  const initBind = dev.createBindGroup({ layout: initPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: labelTex.createView() },
    { binding: 1, resource: texA.createView() },
    { binding: 2, resource: { buffer: palBuf } },
    { binding: 3, resource: { buffer: dimsBuf } }
  ] });
  const enc = dev.createCommandEncoder();
  {
    const p = enc.beginComputePass();
    p.setPipeline(initPipe);
    p.setBindGroup(0, initBind);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
  }
  if (sigmaVoxels <= 0) {
    dev.queue.submit([enc.finish()]);
    labelTex.destroy();
    texB.destroy();
    return texA;
  }
  const { radius, w } = gaussHalfKernel(sigmaVoxels);
  const blurPipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: BLUR_WGSL }), entryPoint: "main" } });
  const passes = [[texA, texB, 0], [texB, texA, 1], [texA, texB, 2]];
  for (const [src, dst, axis] of passes) {
    const ub = dev.createBuffer({ size: 16 + 16 + 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(ub, 0, new Uint32Array([dx, dy, dz, 0, axis, radius, 0, 0]));
    dev.queue.writeBuffer(ub, 32, w);
    const b = dev.createBindGroup({ layout: blurPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: src.createView() },
      { binding: 1, resource: dst.createView() },
      { binding: 2, resource: { buffer: ub } }
    ] });
    const p = enc.beginComputePass();
    p.setPipeline(blurPipe);
    p.setBindGroup(0, b);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
  }
  dev.queue.submit([enc.finish()]);
  labelTex.destroy();
  texA.destroy();
  return texB;
}
var ColorizeBaker = class {
  /** `label` is either a CPU labelmap (baker allocates + uploads its own r8uint texture, the classic
   *  path) OR an EXTERNAL r8uint 3D texture the baker only READS (the shared-buffer path used by
   *  `algorithms/EditableSegmentation` — a compute effect writes the label texture on-GPU and the baker
   *  re-colorizes from it, no CPU round-trip). An external texture must be `r8uint` with at least
   *  TEXTURE_BINDING usage; the baker never writes or destroys it. */
  constructor(dev, label, dims) {
    this.dev = dev;
    this.dims = dims;
    const [dx, dy, dz] = dims;
    if (label instanceof GPUTexture) {
      this.labelTex = label;
      this.ownsLabel = false;
    } else {
      this.labelTex = dev.createTexture({ size: dims, dimension: "3d", format: "r8uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      dev.queue.writeTexture({ texture: this.labelTex }, label, { bytesPerRow: dx, rowsPerImage: dy }, dims);
      this.ownsLabel = true;
    }
    this.palBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dimsBuf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(this.dimsBuf, 0, new Uint32Array([dx, dy, dz, 0]));
    this.initPipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: INIT_WGSL }), entryPoint: "main" } });
    this.blurPipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: BLUR_WGSL }), entryPoint: "main" } });
    this.g = [Math.ceil(dx / 4), Math.ceil(dy / 4), Math.ceil(dz / 4)];
  }
  labelTex;
  ownsLabel;
  // false when the label texture is owned externally (shared buffer)
  scratch;
  // blur ping-pong (lazy; only when sigma > 0)
  palBuf;
  dimsBuf;
  initPipe;
  blurPipe;
  g;
  /** Re-upload an EDITED labelmap (same dims). Follow with bakeInto() to re-colorize into the caller's
   *  existing output textures — an in-place replace (no re-allocation, so a segmentation edit updates
   *  smoothly with no flash). */
  updateLabelmap(labelmap) {
    if (!this.ownsLabel) throw new Error("ColorizeBaker.updateLabelmap: label texture is external (write it via the owner, e.g. a compute effect), then call bakeInto()");
    const [dx, dy] = this.dims;
    this.dev.queue.writeTexture({ texture: this.labelTex }, labelmap, { bytesPerRow: dx, rowsPerImage: dy }, this.dims);
  }
  /** Allocate an output texture sized/typed for this baker's labelmap (caller owns it). */
  output() {
    return this.dev.createTexture({ size: this.dims, dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
  }
  /** (Re)colorize into `out` with `palette` (256*4 f32: rgb + presence*opacity) and Gaussian
   *  `sigmaVoxels` (0 = crisp, for the 2D slice overlay). In place — reuses everything resident. */
  bakeInto(out, palette, sigmaVoxels = 1.5) {
    const dev = this.dev, [gx, gy, gz] = this.g, [dx, dy, dz] = this.dims;
    const palData = new Float32Array(256 * 4);
    palData.set(palette.subarray(0, Math.min(palette.length, 256 * 4)));
    dev.queue.writeBuffer(this.palBuf, 0, palData);
    const enc = dev.createCommandEncoder();
    const smooth = sigmaVoxels > 0;
    if (smooth && !this.scratch) this.scratch = this.output();
    const initDst = smooth ? this.scratch : out;
    const initBind = dev.createBindGroup({ layout: this.initPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.labelTex.createView() },
      { binding: 1, resource: initDst.createView() },
      { binding: 2, resource: { buffer: this.palBuf } },
      { binding: 3, resource: { buffer: this.dimsBuf } }
    ] });
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.initPipe);
      p.setBindGroup(0, initBind);
      p.dispatchWorkgroups(gx, gy, gz);
      p.end();
    }
    if (smooth) {
      const s = this.scratch;
      const { radius, w } = gaussHalfKernel(sigmaVoxels);
      const passes = [[s, out, 0], [out, s, 1], [s, out, 2]];
      for (const [src, dst, axis] of passes) {
        const ub = dev.createBuffer({ size: 16 + 16 + 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        dev.queue.writeBuffer(ub, 0, new Uint32Array([dx, dy, dz, 0, axis, radius, 0, 0]));
        dev.queue.writeBuffer(ub, 32, w);
        const b = dev.createBindGroup({ layout: this.blurPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: src.createView() },
          { binding: 1, resource: dst.createView() },
          { binding: 2, resource: { buffer: ub } }
        ] });
        const p = enc.beginComputePass();
        p.setPipeline(this.blurPipe);
        p.setBindGroup(0, b);
        p.dispatchWorkgroups(gx, gy, gz);
        p.end();
      }
    }
    dev.queue.submit([enc.finish()]);
  }
  destroy() {
    if (this.ownsLabel) this.labelTex.destroy();
    this.scratch?.destroy();
    this.palBuf.destroy();
    this.dimsBuf.destroy();
  }
};

// render/fiducial-field.ts
var MAX = 64;
var FiducialField = class {
  kind = "fid";
  bindingCount = 0;
  // procedural — all state lives in the uniform block
  spheres = new Float32Array(MAX * 4);
  // (cx,cy,cz,radius)
  colors = new Float32Array(MAX * 4);
  // (r,g,b,a)
  n = 0;
  maxR = 0;
  // largest radius in this field (for the skip bound)
  active = -1;
  // hovered/active sphere index (ghost mode: it goes full opacity)
  clippable;
  ghost;
  providesSkip;
  // off in screen-space mode (radius varies with the camera)
  screen;
  sh;
  ka;
  kd;
  ks;
  light;
  constructor(spheres = [], opts = {}) {
    this.setSpheres(spheres);
    this.sh = opts.shininess ?? 80;
    this.ka = opts.kAmbient ?? 0.2;
    this.kd = opts.kDiffuse ?? 0.85;
    this.ks = opts.kSpecular ?? 0.5;
    this.light = opts.lightColor ?? [1, 1, 1];
    this.clippable = opts.clippable ?? true;
    this.ghost = opts.ghost ?? false;
    this.screen = opts.screenSpace ?? false;
    this.providesSkip = true;
  }
  setSpheres(list) {
    this.n = Math.min(list.length, MAX);
    this.spheres.fill(0);
    this.colors.fill(0);
    this.maxR = 0;
    for (let i = 0; i < this.n; i++) {
      const s = list[i];
      this.spheres.set([s.center[0], s.center[1], s.center[2], s.radius], i * 4);
      this.colors.set(s.color, i * 4);
      this.maxR = Math.max(this.maxR, s.radius);
    }
  }
  get count() {
    return this.n;
  }
  /** Hovered/active sphere (ghost mode only): it renders at full opacity while the others stay
   *  half-visible (partially hidden inside the volume). Pass null/-1 to clear. */
  setActive(i) {
    this.active = i ?? -1;
  }
  get activeIndex() {
    return this.active;
  }
  uniformFloats() {
    return 12 + MAX * 4 * 2;
  }
  // params(4)+params2(4)+light(4) + spheres + colors
  sampleStep() {
    return 1;
  }
  aabb() {
    if (this.n === 0) return [[-1, -1, -1], [1, 1, 1]];
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < this.n; i++) {
      const r = this.screen ? 0 : this.spheres[i * 4 + 3];
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], this.spheres[i * 4 + a] - r);
        hi[a] = Math.max(hi[a], this.spheres[i * 4 + a] + r);
      }
    }
    if (this.screen) {
      const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
      const m = Math.max(40, diag * 0.15);
      for (let a = 0; a < 3; a++) {
        lo[a] -= m;
        hi[a] += m;
      }
    }
    return [lo, hi];
  }
  structMembers(s) {
    return [
      `  fid${s}_params : vec4<f32>,`,
      // n_spheres, visible, shininess, k_ambient
      `  fid${s}_params2 : vec4<f32>,`,
      // k_diffuse, k_specular, max_radius, _
      `  fid${s}_light : vec4<f32>,`,
      // light_color.rgb, _
      `  fid${s}_spheres : array<vec4<f32>, ${MAX}>,`,
      `  fid${s}_colors : array<vec4<f32>, ${MAX}>,`
    ].join("\n");
  }
  declareBindings(_s, _base) {
    return "";
  }
  bindEntries(_s, _base) {
    return [];
  }
  // --- empty-space skipping -------------------------------------------------
  // The spheres are an exact SDF, so we can hand the ray-marcher a real distance to
  // leap. Conservative form: nearest-CENTRE distance minus the field's LARGEST radius.
  // Since min_j(d_j) <= d_k and max_r >= r_k for every k, this never exceeds the true
  // min_k(d_k - r_k) — so it can't skip over a sphere — and it costs only squared
  // distances in the loop plus ONE sqrt at the end (cheaper than the sampling loop).
  // (providesSkip is false in screen-space mode — the world radius varies with the camera.)
  skipWGSL(s) {
    if (this.screen) {
      return (
        /* wgsl */
        `
fn skip_fid${s}(wp : vec3<f32>) -> f32 {
  let n = i32(u_material.fid${s}_params.x);
  if (n <= 0) { return 1.0e6; }
  var best = 1.0e12;
  for (var k = 0; k < n; k = k + 1) {
    let sp = u_material.fid${s}_spheres[k];
    if (sp.w <= 0.0) { continue; }
    let r = sp.w * length(u_cam.eye.xyz - sp.xyz) / max(u_cam.size.z, 1.0);
    best = min(best, length(wp - sp.xyz) - r);
  }
  return max(best, 0.0);
}`
      );
    }
    return (
      /* wgsl */
      `
fn skip_fid${s}(wp : vec3<f32>) -> f32 {
  let n = i32(u_material.fid${s}_params.x);
  if (n <= 0) { return 1.0e6; }        // nothing here: unbounded empty space
  var min_d2 = 1.0e12;
  for (var k = 0; k < n; k = k + 1) {
    let sp = u_material.fid${s}_spheres[k];
    if (sp.w <= 0.0) { continue; }
    let dv = wp - sp.xyz;
    min_d2 = min(min_d2, dot(dv, dv));
  }
  return max(sqrt(min_d2) - u_material.fid${s}_params2.z, 0.0);
}`
    );
  }
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn sample_field_fid${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  // an attached TransformField warps where the spheres appear (slicer_wgpu parity)
  let wp_r = transform_point_fid${s}(wp);
  let n = i32(u_material.fid${s}_params.x);
  var best_depth = -1.0;
  var best_center = vec3<f32>(0.0);
  var best_color = vec4<f32>(0.0);
  var best_k = -1;
  var found = false;
  for (var k = 0; k < n; k = k + 1) {
    let sp = u_material.fid${s}_spheres[k];
    if (sp.w <= 0.0) { continue; }
    // screen-space: sp.w is a PIXEL radius -> world radius = px * distance(eye) / focal_px,
    // so the sphere stays a constant size on screen. Otherwise sp.w is a world radius.
    ${this.screen ? `let r = sp.w * length(u_cam.eye.xyz - sp.xyz) / max(u_cam.size.z, 1.0);` : `let r = sp.w;`}
    let depth = r - length(wp_r - sp.xyz);   // > 0 -> inside this sphere
    if (depth > best_depth) { best_depth = depth; best_center = sp.xyz; best_color = u_material.fid${s}_colors[k]; best_k = k; found = true; }
  }
  if (!found || best_depth <= 0.0) { return vec4<f32>(0.0); }

  let to_wp = wp_r - best_center;
  var n_hat = to_wp / max(length(to_wp), 1e-6);
  if (dot(n_hat, -rd) < 0.0) { n_hat = -n_hat; }
  let view_dir = normalize(-rd);            // headlight (== normalize(ray_origin - wp) for t>0)
  let ldotn = max(dot(view_dir, n_hat), 0.0);
  let refl = normalize(2.0 * ldotn * n_hat - view_dir);
  let rdotv = max(dot(refl, view_dir), 0.0);

  let sh = u_material.fid${s}_params.z;
  let ka = u_material.fid${s}_params.w; let kd = u_material.fid${s}_params2.x; let ks = u_material.fid${s}_params2.y;
  let base = best_color.rgb;
  let highlight = mix(base, u_material.fid${s}_light.rgb, 0.85);
  let lit = base * ka + base * (kd * ldotn) + highlight * (ks * pow(rdotv, sh));
  let col = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  // Ghost mode: a non-active glyph emits HALF opacity so the ghost compositor leaves 50% of the
  // volume in front of it (partially hidden inside the render); the hovered one emits full (0%
  // residual -> fully visible). Same trick the transform gizmo uses for its active handle.
  ${this.ghost ? `let ghostScale = select(0.5, 1.0, best_k == i32(u_material.fid${s}_params2.w));` : `let ghostScale = 1.0;`}
  let opacity = clamp(best_color.a, 0.0, 1.0) * ghostScale;
  return vec4<f32>(col * opacity, opacity);
}`
    );
  }
  fillUniforms(out, off) {
    out[off + 0] = this.n;
    out[off + 1] = 1;
    out[off + 2] = this.sh;
    out[off + 3] = this.ka;
    out[off + 4] = this.kd;
    out[off + 5] = this.ks;
    out[off + 6] = this.maxR;
    out[off + 7] = this.active;
    out[off + 8] = this.light[0];
    out[off + 9] = this.light[1];
    out[off + 10] = this.light[2];
    out.set(this.spheres, off + 12);
    out.set(this.colors, off + 12 + MAX * 4);
  }
};

// render/roi-box-field.ts
var RoiBoxField = class {
  kind = "roi";
  bindingCount = 0;
  // procedural — all state in the uniform block
  clippable = false;
  // the frame sits on the clip planes; never clip it
  providesSkip = true;
  // sparse SDF -> cheap via empty-space skipping
  center;
  half;
  color;
  opacity;
  bar;
  constructor(center, half, opts = {}) {
    this.center = [...center];
    this.half = [...half];
    this.color = opts.color ?? [1, 0.85, 0.25];
    this.opacity = opts.opacity ?? 1;
    this.bar = opts.barHalfMm ?? 1.5;
  }
  /** Update the box (a drag) — caller does scene.syncUniforms() + redraw. */
  setBox(center, half) {
    this.center = [...center];
    this.half = [...half];
  }
  get boxCenter() {
    return [...this.center];
  }
  get boxHalf() {
    return [...this.half];
  }
  uniformFloats() {
    return 16;
  }
  // center(4) + half(4) + color(4) + params(4)
  sampleStep() {
    return Math.max(0.5 * this.bar, 0.25);
  }
  aabb() {
    const m = this.bar + 0.5;
    return [
      [this.center[0] - this.half[0] - m, this.center[1] - this.half[1] - m, this.center[2] - this.half[2] - m],
      [this.center[0] + this.half[0] + m, this.center[1] + this.half[1] + m, this.center[2] + this.half[2] + m]
    ];
  }
  structMembers(s) {
    return [
      `  roi${s}_center : vec4<f32>,`,
      // cx,cy,cz,_
      `  roi${s}_half : vec4<f32>,`,
      // hx,hy,hz,_
      `  roi${s}_color : vec4<f32>,`,
      // rgb, opacity
      `  roi${s}_params : vec4<f32>,`
      // bar_half, _, _, _
    ].join("\n");
  }
  declareBindings() {
    return "";
  }
  bindEntries() {
    return [];
  }
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn sd_box_frame${s}(p0 : vec3<f32>, b : vec3<f32>, e : f32) -> f32 {
  let p = abs(p0) - b;
  let q = abs(p + vec3<f32>(e)) - vec3<f32>(e);
  return min(min(
    length(max(vec3<f32>(p.x, q.y, q.z), vec3<f32>(0.0))) + min(max(p.x, max(q.y, q.z)), 0.0),
    length(max(vec3<f32>(q.x, p.y, q.z), vec3<f32>(0.0))) + min(max(q.x, max(p.y, q.z)), 0.0)),
    length(max(vec3<f32>(q.x, q.y, p.z), vec3<f32>(0.0))) + min(max(q.x, max(q.y, p.z)), 0.0));
}
fn sd_roi${s}(wp : vec3<f32>) -> f32 {
  return sd_box_frame${s}(wp - u_material.roi${s}_center.xyz, u_material.roi${s}_half.xyz, u_material.roi${s}_params.x);
}
fn skip_roi${s}(wp : vec3<f32>) -> f32 {
  // exact exterior distance to the bars, minus a bar-width margin (stays conservative)
  return max(sd_roi${s}(wp) - u_material.roi${s}_params.x, 0.0);
}
fn sample_field_roi${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let op0 = u_material.roi${s}_color.a;
  if (op0 <= 0.0) { return vec4<f32>(0.0); }
  let sd = sd_roi${s}(wp);
  // crisp opaque bar: ~1 inside, AA-ramp to 0 across ~half a sample step at the surface
  let op = clamp(0.5 - sd / max(u_material.scene.x, 1e-3), 0.0, 1.0) * op0;
  if (op <= 0.0) { return vec4<f32>(0.0); }
  let col = srgb2physical(u_material.roi${s}_color.rgb);   // flat/unlit, the Slicer widget look
  return vec4<f32>(col * op, op);
}`
    );
  }
  skipWGSL(s) {
    return "";
  }
  // skip_roi<s> is emitted by samplingWGSL above
  fillUniforms(out, off) {
    out[off + 0] = this.center[0];
    out[off + 1] = this.center[1];
    out[off + 2] = this.center[2];
    out[off + 4] = this.half[0];
    out[off + 5] = this.half[1];
    out[off + 6] = this.half[2];
    out[off + 8] = this.color[0];
    out[off + 9] = this.color[1];
    out[off + 10] = this.color[2];
    out[off + 11] = this.opacity;
    out[off + 12] = this.bar;
  }
};

// render/demos/roi-widget.ts
function createRoiWidget(lo, hi, opts = {}) {
  const MIN_HALF = opts.minHalfMm ?? 5;
  const cov = opts.coverage ?? 0.35;
  const center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const half = [(hi[0] - lo[0]) * cov, (hi[1] - lo[1]) * cov, (hi[2] - lo[2]) * cov];
  const hR = Math.max(3, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 0.012);
  const bar = Math.max(1.2, hR * 0.35);
  const box = new RoiBoxField(center, half, { color: [1, 0.85, 0.25], barHalfMm: bar });
  const handles = new FiducialField([], { shininess: 60, kSpecular: 0.4, clippable: false, screenSpace: true, ghost: true });
  let hover = null;
  const metas = [];
  for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) metas.push({ kind: "face", axis, sign });
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) metas.push({ kind: "corner", s: [sx, sy, sz] });
  metas.push({ kind: "center" });
  const worldOf = (m) => {
    if (m.kind === "center") return [...center];
    if (m.kind === "face") {
      const w = [...center];
      w[m.axis] += m.sign * half[m.axis];
      return w;
    }
    return [center[0] + m.s[0] * half[0], center[1] + m.s[1] * half[1], center[2] + m.s[2] * half[2]];
  };
  const refreshHandles = () => {
    const pins = metas.map((m, i) => {
      const on = i === hover;
      const base = m.kind === "center" ? [0.4, 1, 0.5] : [0.35, 0.8, 1];
      return { center: worldOf(m), radius: on ? 13 : 8, color: on ? [1, 0.9, 0.3, 1] : [base[0], base[1], base[2], 0.5] };
    });
    handles.setSpheres(pins);
  };
  refreshHandles();
  const moveFace = (axis, sign, box0, deltaAxis) => {
    const opp = box0.center[axis] - sign * box0.half[axis];
    let face = box0.center[axis] + sign * box0.half[axis] + deltaAxis;
    face = sign > 0 ? Math.max(face, opp + 2 * MIN_HALF) : Math.min(face, opp - 2 * MIN_HALF);
    return [(face + opp) / 2, Math.abs(face - opp) / 2];
  };
  return {
    box,
    handles,
    center,
    half,
    lo: () => [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
    hi: () => [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
    handleList: () => metas.map((m, i) => ({
      id: i,
      world: worldOf(m),
      data: m,
      cursor: m.kind === "center" ? "move" : "grab"
    })),
    applyDrag(meta, box0, delta) {
      if (meta.kind === "center") {
        for (let a = 0; a < 3; a++) center[a] = box0.center[a] + delta[a];
      } else if (meta.kind === "face") {
        const [c, h] = moveFace(meta.axis, meta.sign, box0, delta[meta.axis]);
        center[meta.axis] = c;
        half[meta.axis] = h;
      } else {
        for (let a = 0; a < 3; a++) {
          const [c, h] = moveFace(a, meta.s[a], box0, delta[a]);
          center[a] = c;
          half[a] = h;
        }
      }
      box.setBox(center, half);
      refreshHandles();
    },
    setHover(i) {
      hover = i;
      refreshHandles();
    },
    snapshot: () => ({ center: [...center], half: [...half] }),
    setBox(c, h) {
      for (let a = 0; a < 3; a++) {
        center[a] = c[a];
        half[a] = h[a];
      }
      box.setBox(center, half);
      refreshHandles();
    }
  };
}

// algorithms/geom.ts
function labelmapHasInternalBoundary(lab, dims) {
  const [nx, ny, nz] = dims;
  const at = (x, y, z) => lab[(z * ny + y) * nx + x];
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const v = at(x, y, z);
    if (v === 0) continue;
    if (x + 1 < nx) {
      const n = at(x + 1, y, z);
      if (n !== 0 && n !== v) return true;
    }
    if (y + 1 < ny) {
      const n = at(x, y + 1, z);
      if (n !== 0 && n !== v) return true;
    }
    if (z + 1 < nz) {
      const n = at(x, y, z + 1);
      if (n !== 0 && n !== v) return true;
    }
  }
  return false;
}
function resampleIsotropic(lab, dims, ijkToRAS, maxDim, maxVoxels = Infinity) {
  const sp = spacingFromIjkToRAS2(ijkToRAS);
  const ext = [dims[0] * sp[0], dims[1] * sp[1], dims[2] * sp[2]];
  let vox = Math.max(...ext) / maxDim;
  const count = (v) => Math.max(1, Math.round(ext[0] / v)) * Math.max(1, Math.round(ext[1] / v)) * Math.max(1, Math.round(ext[2] / v));
  let coarsened = false;
  if (count(vox) > maxVoxels) {
    vox = Math.cbrt(ext[0] * ext[1] * ext[2] / maxVoxels);
    coarsened = true;
  }
  const cd = [Math.max(1, Math.round(ext[0] / vox)), Math.max(1, Math.round(ext[1] / vox)), Math.max(1, Math.round(ext[2] / vox))];
  const [nx, ny, nz] = dims, [cx, cy, cz] = cd;
  const out = new Uint32Array(cx * cy * cz);
  if (cx === nx && cy === ny && cz === nz) {
    for (let i = 0; i < out.length; i++) out[i] = lab[i];
    return { lab: out, dims, ijkToRAS, vox, coarsened };
  }
  for (let z = 0; z < cz; z++) {
    const sz = Math.min(nz - 1, Math.floor((z + 0.5) * nz / cz));
    for (let y = 0; y < cy; y++) {
      const sy = Math.min(ny - 1, Math.floor((y + 0.5) * ny / cy));
      for (let x = 0; x < cx; x++) {
        const sx = Math.min(nx - 1, Math.floor((x + 0.5) * nx / cx));
        out[(z * cy + y) * cx + x] = lab[(sz * ny + sy) * nx + sx];
      }
    }
  }
  const r = [nx / cx, ny / cy, nz / cz], m = ijkToRAS.slice();
  for (let row = 0; row < 3; row++) {
    m[row * 4] *= r[0];
    m[row * 4 + 1] *= r[1];
    m[row * 4 + 2] *= r[2];
  }
  return { lab: out, dims: cd, ijkToRAS: m, vox, coarsened };
}
function spacingFromIjkToRAS2(ijkToRAS) {
  const col = (c) => Math.hypot(ijkToRAS[c], ijkToRAS[4 + c], ijkToRAS[8 + c]);
  return [col(0), col(1), col(2)];
}

// algorithms/editable-segmentation.ts
var EditableSegmentation = class {
  dims;
  ijkToRAS;
  device;
  // effects (algorithms/effects/*) build their own pipelines against this
  labelTex;
  // master (r32uint, STORAGE) — the shared buffer effects write
  dirtyCbs = [];
  constructor(device, dims, opts) {
    this.device = device;
    this.dims = dims;
    this.ijkToRAS = Array.from(opts.ijkToRAS);
    this.labelTex = device.createTexture({
      size: dims,
      dimension: "3d",
      format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
    });
  }
  /** The master labelmap (r32uint storage). The logic layer reads it (to derive a presence texture);
   *  editing effects write it on-GPU (A-1+). */
  masterTexture() {
    return this.labelTex;
  }
  /** Register a callback fired after any edit — the logic layer rebakes + redraws. Returns an
   *  unsubscribe (so a logic can be swapped/disposed without leaking a stale rebake). */
  onDirty(cb) {
    this.dirtyCbs.push(cb);
    return () => {
      const i = this.dirtyCbs.indexOf(cb);
      if (i >= 0) this.dirtyCbs.splice(i, 1);
    };
  }
  /** Signal that the master was edited (effects call this after writing the label texture on-GPU). */
  markDirty() {
    for (const cb of this.dirtyCbs) cb();
  }
  /** Voxel spacing (mm) from the geometry — for mm↔voxel effect params. */
  spacingMm() {
    return spacingFromIjkToRAS2(this.ijkToRAS);
  }
  /** Load a full labelmap (ids 0..255) from CPU into the master, then notify. */
  loadLabelmap(data) {
    const [dx, dy, dz] = this.dims;
    const u32 = data instanceof Uint32Array ? data : Uint32Array.from(data);
    this.device.queue.writeTexture({ texture: this.labelTex }, u32, { bytesPerRow: dx * 4, rowsPerImage: dy }, [dx, dy, dz]);
    this.markDirty();
  }
  /** Region-limited labelmap write (`lo`/`size` in label-grid ijk; data x-fastest, tightly packed).
   *  Deliberately NO dirty notification: the caller pairs it with a region-limited rebake
   *  (SegmentationLogic.rebakeShellRegion) — an onDirty full rebake would defeat the point. */
  writeLabelRegion(data, lo, size) {
    this.device.queue.writeTexture(
      { texture: this.labelTex, origin: lo },
      data,
      { bytesPerRow: size[0] * 4, rowsPerImage: size[1] },
      size
    );
  }
  /** Read the master labelmap back to CPU (ids per voxel, x-fastest). Handles WebGPU's 256-byte
   *  bytesPerRow alignment. For tests + zarr serialization (A-7); not on the interactive path. */
  async readLabelmap() {
    const [dx, dy, dz] = this.dims;
    const bpr = Math.ceil(dx * 4 / 256) * 256;
    const rowU32 = bpr / 4;
    const buf = this.device.createBuffer({ size: bpr * dy * dz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.labelTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint32Array(buf.getMappedRange());
    const out = new Uint32Array(dx * dy * dz);
    for (let z = 0; z < dz; z++) for (let y = 0; y < dy; y++) {
      const src = (z * dy + y) * rowU32, dst = (z * dy + y) * dx;
      for (let x = 0; x < dx; x++) out[dst + x] = padded[src + x];
    }
    buf.unmap();
    buf.destroy();
    return out;
  }
  destroy() {
    this.labelTex.destroy();
  }
};

// render/sdf-bake.ts
var INIT_WGSL2 = (
  /* wgsl */
  `
struct U { ijkToRAS : mat4x4<f32>, dims : vec4<u32>, params : vec4<f32>, origin : vec4<i32> };
@group(0) @binding(0) var t_label : texture_3d<u32>;
@group(0) @binding(1) var t_seed_out : texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u : U;
// PADDED SDF GRID: the seed/SDF textures are LARGER than the labelmap by 'pad' voxels on every side
// (dims.w). Coord c is a padded-grid coord; the label lives at c-pad, and everything outside the label
// range is background (0). This gives a real spatial margin of background BEYOND the segmentation, so
// a segment touching the labelmap edge closes as a genuine capped surface with room for the SDF to go
// positive \u2014 and gradient/finite-difference samples near that cap stay in-bounds (they read real
// background) instead of hitting the out-of-volume cull sentinel, which used to poison the normal.
fn labelAt(c : vec3<i32>) -> u32 {
  let pad = i32(u.dims.w);
  let ld = vec3<i32>(u.dims.xyz) - vec3<i32>(2 * pad);   // label dims = padded dims - 2\xB7pad
  let lc = c - vec3<i32>(pad);
  if (any(lc < vec3<i32>(0)) || any(lc >= ld)) { return 0u; }
  return textureLoad(t_label, lc, 0).r;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;   // region-limited dispatch offsets into the grid
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let my = labelAt(c);
  let meIn = my != 0u;
  let allMode = u.params.y > 0.5;                   // 0 = outer boundary only; 1 = ANY label change (multi-material interfaces)
  var boundary = false;
  var region = my;                                  // inside voxel \u2192 own label
  let offs = array<vec3<i32>, 6>(vec3<i32>(1,0,0), vec3<i32>(-1,0,0), vec3<i32>(0,1,0), vec3<i32>(0,-1,0), vec3<i32>(0,0,1), vec3<i32>(0,0,-1));
  for (var i = 0; i < 6; i = i + 1) {
    let nl = labelAt(c + offs[i]);
    // outer mode: boundary at inside\u2194outside (segment\u2194background). all mode: boundary at ANY label change
    // (segment\u2194background AND segment\u2194segment) so embedded/nested structures get an interface shell too.
    let isChange = select((nl != 0u) != meIn, nl != my, allMode);
    // outer: a background boundary voxel adopts the neighbour's label (so the outer shell renders on both
    // sides). all: every voxel keeps its OWN label (background stays 0 = transparent), so the region
    // COLOUR changes across EVERY interface \u2014 including the outer one \u2014 which the shader's on-the-fly
    // re-signing needs to recover a clean inside/outside normal there too.
    if (isChange) { boundary = true; if (my == 0u && !allMode) { region = nl; } }
  }
  var seed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (boundary) { seed = vec4<f32>((u.ijkToRAS * vec4<f32>(vec3<f32>(c), 1.0)).xyz, f32(region)); }
  textureStore(t_seed_out, c, seed);
}`
);
var JFA_WGSL = (
  /* wgsl */
  `
struct U { ijkToRAS : mat4x4<f32>, dims : vec4<u32>, params : vec4<f32>, origin : vec4<i32> };
@group(0) @binding(0) var t_seed_in : texture_3d<f32>;
@group(0) @binding(1) var t_seed_out : texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u : U;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;   // region-limited dispatch offsets into the grid
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let p = (u.ijkToRAS * vec4<f32>(vec3<f32>(c), 1.0)).xyz;
  let step = i32(u.params.x);
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var best = textureLoad(t_seed_in, c, 0);
  var bestD = select(1e30, distance(p, best.xyz), best.w > 0.5);
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        if (dx == 0 && dy == 0 && dz == 0) { continue; }
        let nc = clamp(c + vec3<i32>(dx, dy, dz) * step, vec3<i32>(0), dmax);
        let s = textureLoad(t_seed_in, nc, 0);
        if (s.w > 0.5) {
          let d = distance(p, s.xyz);
          if (d < bestD) { bestD = d; best = s; }
        }
      }
    }
  }
  textureStore(t_seed_out, c, best);
}`
);
var FINAL_WGSL = (
  /* wgsl */
  `
struct U { ijkToRAS : mat4x4<f32>, dims : vec4<u32>, params : vec4<f32>, origin : vec4<i32> };
@group(0) @binding(0) var t_seed_in : texture_3d<f32>;
@group(0) @binding(1) var t_label : texture_3d<u32>;
@group(0) @binding(2) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var<uniform> u : U;
@group(0) @binding(4) var<uniform> u_pal : array<vec4<f32>, 256>;
@group(0) @binding(5) var t_attr : texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var<uniform> u_mode : array<vec4<f32>, 256>;   // .x = shading mode (0 surface, 1 volume)
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;   // region-limited dispatch offsets into the grid
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let p = (u.ijkToRAS * vec4<f32>(vec3<f32>(c), 1.0)).xyz;
  let s = textureLoad(t_seed_in, c, 0);
  let valid = s.w > 0.5;
  let dist = select(1e3, distance(p, s.xyz), valid);
  let pad = i32(u.dims.w);                                       // label is offset by pad; pad region = background (outside)
  let lc = c - vec3<i32>(pad);
  let ld = vec3<i32>(u.dims.xyz) - vec3<i32>(2 * pad);
  let inRange = all(lc >= vec3<i32>(0)) && all(lc < ld);
  let ins = inRange && textureLoad(t_label, lc, 0).r != 0u;
  // outer mode: SIGNED (neg inside the union) \u2014 smooth zero-crossing = clean normals. all mode:
  // UNSIGNED distance to the nearest interface (every label change is a wall; no global inside/outside).
  let sdf = select(select(dist, -dist, ins), dist, u.params.y > 0.5);
  let lbl = u32(s.w + 0.5) & 255u;
  let pal = select(vec4<f32>(0.0), u_pal[lbl], valid);
  let mode = select(0.0, u_mode[lbl].x, valid);
  // PREMULTIPLIED colour (rgb\xB7opacity): the colour-seam blur then can't bleed a HIDDEN (opacity 0)
  // segment's colour into a visible neighbour \u2014 an invisible organ was still tinting the organ it
  // abutted (looked like half-opacity). The shader divides by the per-segment opacity to recover the
  // true colour, so a 0-opacity region contributes nothing to the blend.
  textureStore(t_out, c, vec4<f32>(pal.rgb * pal.a, sdf));
  // .r = opacity, .g = shading mode, .b = distance (seam-blurred \u2192 SMOOTH distance for the interface-mode
  // normal; sdfTex.a stays sharp for shell membership), .a = CRISP presence (1 inside a real segment, 0
  // background) \u2014 the FULLBLUR carries it unblurred so the shader can tell a genuine in-segment voxel
  // from one that merely caught BLED colour/opacity outside any segment, and gate the shell to the real edge.
  textureStore(t_attr, c, vec4<f32>(pal.a, mode, sdf, select(0.0, 1.0, ins)));
}`
);
var BLUR_WGSL2 = (
  /* wgsl */
  `
struct BU { dims : vec4<u32>, axis_r : vec4<u32>, w : array<vec4<f32>, 4>, origin : vec4<i32> };
@group(0) @binding(0) var t_in : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u : BU;
fn wt(i : u32) -> f32 { return u.w[i >> 2u][i & 3u]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var av = vec3<i32>(0);
  if (u.axis_r.x == 0u) { av = vec3<i32>(1,0,0); } else if (u.axis_r.x == 1u) { av = vec3<i32>(0,1,0); } else { av = vec3<i32>(0,0,1); }
  let center = textureLoad(t_in, c, 0);
  var sum = center.a * wt(0u);
  let R = i32(u.axis_r.y);
  for (var i = 1; i <= R; i = i + 1) {
    sum = sum + wt(u32(i)) * (textureLoad(t_in, clamp(c + av * i, vec3<i32>(0), dmax), 0).a
                            + textureLoad(t_in, clamp(c - av * i, vec3<i32>(0), dmax), 0).a);
  }
  textureStore(t_out, c, vec4<f32>(center.rgb, sum));
}`
);
var COLBLUR_WGSL = (
  /* wgsl */
  `
struct BU { dims : vec4<u32>, axis_r : vec4<u32>, w : array<vec4<f32>, 4>, origin : vec4<i32> };
@group(0) @binding(0) var t_in : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u : BU;
fn wt(i : u32) -> f32 { return u.w[i >> 2u][i & 3u]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var av = vec3<i32>(0);
  if (u.axis_r.x == 0u) { av = vec3<i32>(1,0,0); } else if (u.axis_r.x == 1u) { av = vec3<i32>(0,1,0); } else { av = vec3<i32>(0,0,1); }
  let center = textureLoad(t_in, c, 0);
  var sum = center.rgb * wt(0u);
  let R = i32(u.axis_r.y);
  for (var i = 1; i <= R; i = i + 1) {
    sum = sum + wt(u32(i)) * (textureLoad(t_in, clamp(c + av * i, vec3<i32>(0), dmax), 0).rgb
                            + textureLoad(t_in, clamp(c - av * i, vec3<i32>(0), dmax), 0).rgb);
  }
  textureStore(t_out, c, vec4<f32>(sum, center.a));
}`
);
var FULLBLUR_WGSL = (
  /* wgsl */
  `
struct BU { dims : vec4<u32>, axis_r : vec4<u32>, w : array<vec4<f32>, 4>, origin : vec4<i32> };
@group(0) @binding(0) var t_in : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u : BU;
fn wt(i : u32) -> f32 { return u.w[i >> 2u][i & 3u]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid) + u.origin.xyz;
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var av = vec3<i32>(0);
  if (u.axis_r.x == 0u) { av = vec3<i32>(1,0,0); } else if (u.axis_r.x == 1u) { av = vec3<i32>(0,1,0); } else { av = vec3<i32>(0,0,1); }
  let center = textureLoad(t_in, c, 0);
  var gb = center.gb * wt(0u);
  let R = i32(u.axis_r.y);
  for (var i = 1; i <= R; i = i + 1) {
    gb = gb + wt(u32(i)) * (textureLoad(t_in, clamp(c + av * i, vec3<i32>(0), dmax), 0).gb
                          + textureLoad(t_in, clamp(c - av * i, vec3<i32>(0), dmax), 0).gb);
  }
  textureStore(t_out, c, vec4<f32>(center.r, gb.x, gb.y, center.a));
}`
);
function gaussHalfKernel2(sigma) {
  const radius = Math.max(1, Math.min(15, Math.ceil(3 * sigma)));
  const raw = new Float32Array(radius + 1);
  let total = 0;
  for (let i = 0; i <= radius; i++) {
    raw[i] = Math.exp(-(i * i) / (2 * sigma * sigma));
    total += (i === 0 ? 1 : 2) * raw[i];
  }
  const w = new Float32Array(16);
  for (let i = 0; i <= radius; i++) w[i] = raw[i] / total;
  return { radius, w };
}
var JfaSdfBaker = class {
  // 0 = outer boundary (signed); 1 = any label change (unsigned, multi-material)
  constructor(dev, labelTex, dims, ijkToRAS, smoothSigmaVoxels = 1, pad = 2, boundaryMode = "outer") {
    this.labelTex = labelTex;
    this.dims = dims;
    this.ijkToRAS = ijkToRAS;
    this.dev = dev;
    this.smoothSigma = smoothSigmaVoxels;
    this.bmode = boundaryMode === "all" ? 1 : 0;
    this.pad = pad;
    this.labelDims = [dims[0], dims[1], dims[2]];
    this.dims = [dims[0] + 2 * pad, dims[1] + 2 * pad, dims[2] + 2 * pad];
    const m = ijkToRAS.slice();
    for (let r = 0; r < 3; r++) m[r * 4 + 3] -= pad * (m[r * 4] + m[r * 4 + 1] + m[r * 4 + 2]);
    this.ijkToRAS = m;
    const [dx, dy, dz] = this.dims;
    const mk = (fmt, extra = 0) => dev.createTexture({ size: this.dims, dimension: "3d", format: fmt, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | extra });
    const seedUsage = GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    this.seed = [mk("rgba32float", seedUsage), mk("rgba32float", seedUsage)];
    this.sdfTex = mk("rgba16float", GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC);
    this.attrTex = mk("rgba16float", GPUTextureUsage.COPY_DST);
    this.attrScratch = mk("rgba16float", GPUTextureUsage.COPY_SRC);
    this.sdfScratch = mk("rgba16float", GPUTextureUsage.COPY_SRC);
    this.uni = dev.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.palBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.modeBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const mod = (code) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } });
    this.initPipe = mod(INIT_WGSL2);
    this.jfaPipe = mod(JFA_WGSL);
    this.finalPipe = mod(FINAL_WGSL);
    this.blurPipe = mod(BLUR_WGSL2);
    this.colBlurPipe = mod(COLBLUR_WGSL);
    this.fullBlurPipe = mod(FULLBLUR_WGSL);
    this.g = [Math.ceil(dx / 4), Math.ceil(dy / 4), Math.ceil(dz / 4)];
    const maxDim = Math.max(dx, dy, dz);
    const steps = [];
    for (let s = 1 << Math.floor(Math.log2(maxDim - 1)); s >= 1; s >>= 1) steps.push(s);
    this.steps = steps;
  }
  dev;
  seed;
  // rgba32float ping-pong (RAS seed xyz + regionLabel)
  sdfTex;
  // rgba16float: .rgb = per-label colour, .a = signed dist (mm) — sampled by SegmentField
  attrTex;
  // rgba16float: .r = per-segment opacity, .g = shading mode — sampled by SegmentField
  attrScratch;
  // rgba16float attr-blur ping-pong
  lastSeed = 0;
  // seed buffer the last sweep finalized from
  sdfScratch;
  // rgba16float blur ping-pong
  uni;
  palBuf;
  // 256 × vec4 label→colour palette (.a = opacity)
  modeBuf;
  // 256 × vec4 label→shading mode (.x = 0 surface / 1 volume)
  initPipe;
  jfaPipe;
  finalPipe;
  blurPipe;
  // blurs .a (distance), carries .rgb
  colBlurPipe;
  // blurs .rgb (colour), carries .a
  fullBlurPipe;
  // blurs all channels — the attr texture (opacity + mode)
  g;
  steps;
  smoothSigma;
  pad;
  // background margin (voxels) padded around the labelmap
  labelDims;
  // original (label) dims, before padding
  // `pad` voxels of background are added on every side so segments touching the labelmap edge get a
  // real cap + in-bounds gradient neighbourhood (docs/ALGORITHMS.md border artifact). The SDF textures,
  // dispatch, seeds, blur and readback all run on the PADDED grid; `dims`/`ijkToRAS` become the padded
  // grid's, and `sdfDims()`/`sdfIjkToRAS()` expose them to the SegmentField.
  bmode;
  /** The resident colorized-SDF texture (rgba16float: .rgb = per-label colour, .a = signed mm).
   *  Identity stable across bakes → the SceneRenderer bind group stays valid; a live edit updates in
   *  place. */
  sdfTexture() {
    return this.sdfTex;
  }
  /** The resident per-segment attribute texture (rgba16float; .r = opacity). Identity stable. */
  attrTexture() {
    return this.attrTex;
  }
  /** The PADDED grid the SDF/attr textures live on (labelDims + 2·pad), and the ijkToRAS that maps it
   *  to RAS — hand these to the SegmentField so its patient→texture transform covers the padded extent. */
  sdfDims() {
    return this.dims;
  }
  sdfIjkToRAS() {
    return this.ijkToRAS;
  }
  /** Background margin (voxels) padded on each side; readDistance() is on the padded grid, so a label
   *  voxel (x,y,z) is at padded (x+pad, y+pad, z+pad). */
  padVoxels() {
    return this.pad;
  }
  /** Read back the per-voxel signed distance (sdfTex .a, mm) to CPU. For accuracy comparison/tests. */
  async readDistance() {
    const [dx, dy, dz] = this.dims;
    const bpr = Math.ceil(dx * 8 / 256) * 256;
    const rowU16 = bpr / 2;
    const buf = this.dev.createBuffer({ size: bpr * dy * dz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.sdfTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const u16 = new Uint16Array(buf.getMappedRange());
    const h2f = (h) => {
      const s = h & 32768 ? -1 : 1, e = (h & 31744) >> 10, f = h & 1023;
      if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
      if (e === 31) return f ? NaN : s * Infinity;
      return s * Math.pow(2, e - 15) * (1 + f / 1024);
    };
    const out = new Float32Array(dx * dy * dz);
    for (let z = 0; z < dz; z++) for (let y = 0; y < dy; y++) for (let x = 0; x < dx; x++) {
      out[(z * dy + y) * dx + x] = h2f(u16[(z * dy + y) * rowU16 + x * 4 + 3]);
    }
    buf.unmap();
    buf.destroy();
    return out;
  }
  /** Set the label→colour palette (256 × rgba f32: rgb = colour, a = opacity). Call before bake(). */
  setPalette(palette) {
    const pal = new Float32Array(256 * 4);
    pal.set(palette.subarray(0, Math.min(palette.length, 256 * 4)));
    this.dev.queue.writeBuffer(this.palBuf, 0, pal);
  }
  /** Set the per-label shading mode palette (256 × vec4; .x = 0 surface shell / 1 volume DVR fill). */
  setModePalette(modes) {
    const m = new Float32Array(256 * 4);
    m.set(modes.subarray(0, Math.min(modes.length, 256 * 4)));
    this.dev.queue.writeBuffer(this.modeBuf, 0, m);
  }
  writeUni(step, origin = [0, 0, 0]) {
    const ab = new ArrayBuffer(112);
    const f = new Float32Array(ab), u = new Uint32Array(ab);
    f.set(transpose4(this.ijkToRAS), 0);
    u[16] = this.dims[0];
    u[17] = this.dims[1];
    u[18] = this.dims[2];
    u[19] = this.pad;
    f[20] = step;
    f[21] = this.bmode;
    f[22] = 0;
    f[23] = 0;
    const i32v = new Int32Array(ab);
    i32v[24] = origin[0];
    i32v[25] = origin[1];
    i32v[26] = origin[2];
    i32v[27] = 0;
    this.dev.queue.writeBuffer(this.uni, 0, ab);
  }
  /** FAST bake for LIVE editing: plain JFA (approximate) + a light distance-only blur (crisp colour
   *  seams). Cheap, so it keeps up with an in-progress stroke; the seams stay voxel-jagged until the
   *  edit settles and refine() runs. */
  bake() {
    this.sweep([], this.smoothSigma, 0);
  }
  /** REFINE for a STATIC labelmap (run once the edit settles): JFA+2 extra passes → a near-exact
   *  Voronoi/SDF (fixes the small JFA mistakes near close/overlapping segments) and a colour-seam blur
   *  so neighbouring-label boundaries are smooth, not a voxel staircase. Distance blur stays at the
   *  same σ (dropping it re-introduces Voronoi facets — crispness comes from the render band, not from
   *  under-smoothing). Higher quality lives in the resident texture, so camera renders stay cheap. */
  refine() {
    this.sweep([2, 1], this.smoothSigma, 1);
  }
  /** REGION-LIMITED refine: re-flood ONLY `regionIjk` (padded-grid coords) after a labelmap edit
   *  confined to it — a per-vertebra visibility flip re-bakes a few % of the grid instead of the
   *  whole volume, which is what makes level stepping feel instant. The seed ping-pong pair is
   *  first made consistent with a full-texture copy, so region passes can ping-pong while JFA
   *  taps read valid exterior seeds (surfaces just outside the region flood in correctly).
   *  Exterior distances that referenced a surface REMOVED inside the region go stale, but only
   *  ≫band away from any visible shell — invisible, and the next full sweep cleans them. */
  refineRegion(regionIjk) {
    const dev = this.dev, [dx, dy, dz] = this.dims;
    const lo = [Math.max(0, regionIjk.lo[0]), Math.max(0, regionIjk.lo[1]), Math.max(0, regionIjk.lo[2])];
    const hi = [Math.min(dx, regionIjk.hi[0]), Math.min(dy, regionIjk.hi[1]), Math.min(dz, regionIjk.hi[2])];
    const [rx, ry, rz] = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    if (rx <= 0 || ry <= 0 || rz <= 0) return;
    const g = [Math.ceil(rx / 4), Math.ceil(ry / 4), Math.ceil(rz / 4)];
    const region = { lo, hi };
    let src = this.lastSeed;
    let enc = dev.createCommandEncoder();
    enc.copyTextureToTexture({ texture: this.seed[src] }, { texture: this.seed[src ^ 1] }, this.dims);
    dev.queue.submit([enc.finish()]);
    this.writeUni(0, lo);
    enc = dev.createCommandEncoder();
    {
      const b = dev.createBindGroup({ layout: this.initPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.labelTex.createView() },
        { binding: 1, resource: this.seed[src].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.initPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(g[0], g[1], g[2]);
      p2.end();
    }
    dev.queue.submit([enc.finish()]);
    const maxDim = Math.max(rx, ry, rz);
    const steps = [];
    for (let s = 1 << Math.floor(Math.log2(Math.max(2, maxDim - 1))); s >= 1; s >>= 1) steps.push(s);
    steps.push(2, 1);
    for (const step of steps) {
      this.writeUni(step, lo);
      const dst = src ^ 1;
      enc = dev.createCommandEncoder();
      const b = dev.createBindGroup({ layout: this.jfaPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.seed[src].createView() },
        { binding: 1, resource: this.seed[dst].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.jfaPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(g[0], g[1], g[2]);
      p2.end();
      dev.queue.submit([enc.finish()]);
      src = dst;
    }
    this.lastSeed = src;
    this.writeUni(0, lo);
    enc = dev.createCommandEncoder();
    const bf = dev.createBindGroup({ layout: this.finalPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seed[src].createView() },
      { binding: 1, resource: this.labelTex.createView() },
      { binding: 2, resource: this.sdfTex.createView() },
      { binding: 3, resource: { buffer: this.uni } },
      { binding: 4, resource: { buffer: this.palBuf } },
      { binding: 5, resource: this.attrTex.createView() },
      { binding: 6, resource: { buffer: this.modeBuf } }
    ] });
    const p = enc.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(g[0], g[1], g[2]);
    p.end();
    dev.queue.submit([enc.finish()]);
    this.blurStage(this.blurPipe, this.smoothSigma, this.sdfTex, this.sdfScratch, region);
    this.blurStage(this.colBlurPipe, 1, this.sdfTex, this.sdfScratch, region);
    this.blurStage(this.fullBlurPipe, 1, this.attrTex, this.attrScratch, region);
  }
  /** ATTR-ONLY rebake for palette/opacity changes (per-segment visibility): the distance field
   *  doesn't move, so re-run ONLY the finalize (from the last sweep's seed) with its sdf writes
   *  routed to the scratch texture (discarded — the blurred resident sdfTex stays pristine) and
   *  re-blur the attribute seams. ~4 passes instead of the ~20-pass init+JFA+blur sweep, which is
   *  what makes per-vertebra focus switching real-time. */
  rebakeAttr(blurSeams = false, regionIjk) {
    const dev = this.dev, [dx, dy, dz] = this.dims;
    const region = regionIjk && {
      lo: [Math.max(0, regionIjk.lo[0]), Math.max(0, regionIjk.lo[1]), Math.max(0, regionIjk.lo[2])],
      hi: [Math.min(dx, regionIjk.hi[0]), Math.min(dy, regionIjk.hi[1]), Math.min(dz, regionIjk.hi[2])]
    };
    const [gx, gy, gz] = region ? [Math.ceil((region.hi[0] - region.lo[0]) / 4), Math.ceil((region.hi[1] - region.lo[1]) / 4), Math.ceil((region.hi[2] - region.lo[2]) / 4)] : this.g;
    if (region && (gx <= 0 || gy <= 0 || gz <= 0)) return;
    this.writeUni(0, region ? region.lo : [0, 0, 0]);
    const enc = dev.createCommandEncoder();
    const bf = dev.createBindGroup({ layout: this.finalPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seed[this.lastSeed].createView() },
      { binding: 1, resource: this.labelTex.createView() },
      { binding: 2, resource: this.sdfScratch.createView() },
      // sdf writes discarded
      { binding: 3, resource: { buffer: this.uni } },
      { binding: 4, resource: { buffer: this.palBuf } },
      { binding: 5, resource: this.attrTex.createView() },
      { binding: 6, resource: { buffer: this.modeBuf } }
    ] });
    const p = enc.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc.finish()]);
    if (blurSeams) this.blurStage(this.fullBlurPipe, 1, this.attrTex, this.attrScratch, region);
  }
  /** Blur the attribute seams of the CURRENT attr texture in place (no re-finalize) — the
   *  cheapest possible settle after a run of rebakeAttr(false) visibility steps. */
  blurAttrOnly() {
    this.blurStage(this.fullBlurPipe, 1, this.attrTex, this.attrScratch);
  }
  /** One full sweep: init → JFA (schedule + extra) → finalize → blur .a → optional blur .rgb. */
  sweep(extraSteps, distSigma, colorSigma) {
    const dev = this.dev, [gx, gy, gz] = this.g;
    this.writeUni(0);
    let enc = dev.createCommandEncoder();
    {
      const b = dev.createBindGroup({ layout: this.initPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.labelTex.createView() },
        { binding: 1, resource: this.seed[0].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.initPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(gx, gy, gz);
      p2.end();
    }
    dev.queue.submit([enc.finish()]);
    let src = 0;
    for (const step of [...this.steps, ...extraSteps]) {
      this.writeUni(step);
      const dst = src ^ 1;
      enc = dev.createCommandEncoder();
      const b = dev.createBindGroup({ layout: this.jfaPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.seed[src].createView() },
        { binding: 1, resource: this.seed[dst].createView() },
        { binding: 2, resource: { buffer: this.uni } }
      ] });
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.jfaPipe);
      p2.setBindGroup(0, b);
      p2.dispatchWorkgroups(gx, gy, gz);
      p2.end();
      dev.queue.submit([enc.finish()]);
      src = dst;
    }
    this.lastSeed = src;
    enc = dev.createCommandEncoder();
    const bf = dev.createBindGroup({ layout: this.finalPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seed[src].createView() },
      { binding: 1, resource: this.labelTex.createView() },
      { binding: 2, resource: this.sdfTex.createView() },
      { binding: 3, resource: { buffer: this.uni } },
      { binding: 4, resource: { buffer: this.palBuf } },
      { binding: 5, resource: this.attrTex.createView() },
      { binding: 6, resource: { buffer: this.modeBuf } }
    ] });
    const p = enc.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc.finish()]);
    if (distSigma > 0) this.blurStage(this.blurPipe, distSigma, this.sdfTex, this.sdfScratch);
    if (colorSigma > 0) this.blurStage(this.colBlurPipe, colorSigma, this.sdfTex, this.sdfScratch);
    if (colorSigma > 0) this.blurStage(this.fullBlurPipe, colorSigma, this.attrTex, this.attrScratch);
  }
  /** 3 separable Gaussian passes with the given pipeline (which channels it blurs), tex↔scratch,
   *  ending in scratch → copied back to `tex` so its identity stays stable for the renderer. */
  blurStage(pipe, sigma, tex, scratch, region) {
    const dev = this.dev, [dx, dy, dz] = this.dims;
    const { radius, w } = gaussHalfKernel2(sigma);
    const passes = [[tex, scratch, 0], [scratch, tex, 1], [tex, scratch, 2]];
    const enc = dev.createCommandEncoder();
    let passIdx = 0;
    for (const [srcT, dstT, axis] of passes) {
      const expand = region ? (2 - passIdx) * radius : 0;
      const lo = region ? [Math.max(0, region.lo[0] - expand), Math.max(0, region.lo[1] - expand), Math.max(0, region.lo[2] - expand)] : [0, 0, 0];
      const hi = region ? [Math.min(dx, region.hi[0] + expand), Math.min(dy, region.hi[1] + expand), Math.min(dz, region.hi[2] + expand)] : [dx, dy, dz];
      const ab = new ArrayBuffer(112);
      const u32 = new Uint32Array(ab), f32 = new Float32Array(ab), i32 = new Int32Array(ab);
      u32[0] = dx;
      u32[1] = dy;
      u32[2] = dz;
      u32[4] = axis;
      u32[5] = radius;
      f32.set(w, 8);
      i32[24] = lo[0];
      i32[25] = lo[1];
      i32[26] = lo[2];
      const ub = dev.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(ub, 0, ab);
      const b = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: srcT.createView() },
        { binding: 1, resource: dstT.createView() },
        { binding: 2, resource: { buffer: ub } }
      ] });
      const bp = enc.beginComputePass();
      bp.setPipeline(pipe);
      bp.setBindGroup(0, b);
      bp.dispatchWorkgroups(Math.ceil((hi[0] - lo[0]) / 4), Math.ceil((hi[1] - lo[1]) / 4), Math.ceil((hi[2] - lo[2]) / 4));
      bp.end();
      passIdx++;
    }
    if (region) {
      const sz = [region.hi[0] - region.lo[0], region.hi[1] - region.lo[1], region.hi[2] - region.lo[2]];
      enc.copyTextureToTexture({ texture: scratch, origin: region.lo }, { texture: tex, origin: region.lo }, sz);
    } else {
      enc.copyTextureToTexture({ texture: scratch }, { texture: tex }, this.dims);
    }
    dev.queue.submit([enc.finish()]);
  }
  destroy() {
    this.seed[0].destroy();
    this.seed[1].destroy();
    this.sdfTex.destroy();
    this.attrTex.destroy();
    this.attrScratch.destroy();
    this.sdfScratch.destroy();
    this.uni.destroy();
    this.palBuf.destroy();
    this.modeBuf.destroy();
  }
};

// logic/segmentation-logic.ts
function invertAffine(m) {
  const r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
  const i = [
    (r[4] * r[8] - r[5] * r[7]) / det,
    (r[2] * r[7] - r[1] * r[8]) / det,
    (r[1] * r[5] - r[2] * r[4]) / det,
    (r[5] * r[6] - r[3] * r[8]) / det,
    (r[0] * r[8] - r[2] * r[6]) / det,
    (r[2] * r[3] - r[0] * r[5]) / det,
    (r[3] * r[7] - r[4] * r[6]) / det,
    (r[1] * r[6] - r[0] * r[7]) / det,
    (r[0] * r[4] - r[1] * r[3]) / det
  ];
  const t = [m[3], m[7], m[11]];
  return [
    i[0],
    i[1],
    i[2],
    -(i[0] * t[0] + i[1] * t[1] + i[2] * t[2]),
    i[3],
    i[4],
    i[5],
    -(i[3] * t[0] + i[4] * t[1] + i[5] * t[2]),
    i[6],
    i[7],
    i[8],
    -(i[6] * t[0] + i[7] * t[1] + i[8] * t[2]),
    0,
    0,
    0,
    1
  ];
}
var SegmentationLogic = class {
  constructor(device, seg, opts = {}) {
    this.seg = seg;
    this.renderMode = opts.renderMode ?? "sdf";
    this.sigma = opts.sigmaVoxels ?? 1;
    this.bandMm = opts.bandMm;
    this.opacity = opts.opacity ?? 1;
    this.refineDelayMs = opts.refineDelayMs ?? 180;
    this.boundaryMode = opts.boundaryMode ?? "outer";
    this.clippable = opts.clippable ?? false;
    this.setLabelColor(1, opts.color ?? [0.3, 0.85, 0.55]);
    if (this.renderMode === "sdf") {
      this.sdf = new JfaSdfBaker(device, seg.masterTexture(), seg.dims, seg.ijkToRAS, 1, 2, this.boundaryMode);
    } else {
      this.baker = new ColorizeBaker(device, seg.masterTexture(), seg.dims);
      this.presenceTex = this.baker.output();
    }
    this.rebake();
    this.scheduleRefine();
    this.unsubDirty = seg.onDirty(() => {
      this.rebake();
      for (const cb of this.redrawCbs) cb();
      this.scheduleRefine();
    });
  }
  renderMode;
  clippable;
  attrSettleTimer;
  sdf;
  // sdf path
  baker;
  // surface path
  presenceTex;
  sigma;
  bandMm;
  opacity;
  palette = new Float32Array(256 * 4);
  // label id → (r,g,b, opacity); shared by both paths
  modePalette = new Float32Array(256 * 4);
  // label id → (.x = shading mode: 0 surface / 1 volume) — sdf only
  segField;
  redrawCbs = [];
  unsubDirty;
  refineTimer;
  refineDelayMs;
  // quiescence before the settle-refine (sdf mode; capability-tuned)
  boundaryMode;
  /** Assign a display colour to a label id (0..255). Keeps the current opacity (defaults to 1 =
   *  opaque). Takes effect on the next rebake. */
  setLabelColor(id, rgb) {
    if (id < 1 || id > 255) return;
    const o = id * 4;
    this.palette[o] = rgb[0];
    this.palette[o + 1] = rgb[1];
    this.palette[o + 2] = rgb[2];
    if (this.palette[o + 3] === 0) this.palette[o + 3] = 1;
  }
  /** Per-segment opacity (0 = hidden, 1 = opaque) — palette alpha. Enables translucent surface-model
   *  rendering (see through outer segments to inner ones). Rebake/refine to apply. */
  setLabelOpacity(id, opacity) {
    if (id < 1 || id > 255) return;
    this.palette[id * 4 + 3] = Math.max(0, Math.min(1, opacity));
  }
  /** Per-segment shading (sdf mode): "surface" = crisp SDF shell (surface model), "volume" = DVR fill
   *  of the interior (translucent cloud). Rebake/refine to apply. */
  setLabelShading(id, shading) {
    if (id < 1 || id > 255) return;
    this.modePalette[id * 4] = shading === "volume" ? 1 : 0;
  }
  /** Re-derive the render texture from the current master + palette (FAST, in place). */
  rebake() {
    if (this.sdf) {
      this.sdf.setPalette(this.palette);
      this.sdf.setModePalette(this.modePalette);
      this.sdf.bake();
    } else this.baker.bakeInto(this.presenceTex, this.palette, this.sigma);
  }
  /** Schedule the settle-refine after quiescence (debounced; sdf mode only). */
  scheduleRefine() {
    if (!this.sdf) return;
    if (this.refineTimer !== void 0) clearTimeout(this.refineTimer);
    this.refineTimer = setTimeout(() => {
      this.refineTimer = void 0;
      this.refineNow();
    }, this.refineDelayMs);
  }
  /** Run the settle-refine now (JFA+2 + tighter distance blur + colour-seam blur), then redraw.
   *  Public so a test — or an app that knows the edit is done — can force the high-quality bake. */
  refineNow() {
    if (this.refineTimer !== void 0) {
      clearTimeout(this.refineTimer);
      this.refineTimer = void 0;
    }
    if (this.sdf) {
      this.sdf.setPalette(this.palette);
      this.sdf.setModePalette(this.modePalette);
      this.sdf.refine();
      for (const cb of this.redrawCbs) cb();
    }
  }
  /** FAST per-segment opacity refresh: attr-only rebake (no JFA re-sweep) — for visibility
   *  toggles where the labelmap and colours are unchanged.
   *
   *  With `regionRAS` (the bbox of the labels whose opacity changed): the finalize AND the
   *  seam blur run region-limited in one shot — full settled quality lands immediately, no
   *  two-phase. Without it: full-volume fast pass + a debounced full-volume seam blur. */
  /** RAS bbox → padded-SDF-grid ijk bbox with an M-voxel margin (shell band + blur radii). */
  regionToIjk(regionRAS, M = 8) {
    const inv = invertAffine(this.sdf.sdfIjkToRAS());
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const x of [regionRAS.lo[0], regionRAS.hi[0]]) for (const y of [regionRAS.lo[1], regionRAS.hi[1]]) for (const z of [regionRAS.lo[2], regionRAS.hi[2]]) {
      const i = inv[0] * x + inv[1] * y + inv[2] * z + inv[3];
      const j = inv[4] * x + inv[5] * y + inv[6] * z + inv[7];
      const k = inv[8] * x + inv[9] * y + inv[10] * z + inv[11];
      lo[0] = Math.min(lo[0], i);
      lo[1] = Math.min(lo[1], j);
      lo[2] = Math.min(lo[2], k);
      hi[0] = Math.max(hi[0], i);
      hi[1] = Math.max(hi[1], j);
      hi[2] = Math.max(hi[2], k);
    }
    return {
      lo: [Math.floor(lo[0]) - M, Math.floor(lo[1]) - M, Math.floor(lo[2]) - M],
      hi: [Math.ceil(hi[0]) + M, Math.ceil(hi[1]) + M, Math.ceil(hi[2]) + M]
    };
  }
  /** REGION-LIMITED settle-refine after a labelmap edit confined to `regionRAS` (e.g. a
   *  per-vertebra visibility flip written via EditableSegmentation.writeLabelRegion): the full
   *  refine quality — JFA re-flood, finalize, seam blurs — over just the region, immediately.
   *  Small regions bake in ~ms, so stepping through per-label visibility stays real-time. */
  rebakeShellRegion(regionRAS) {
    if (!this.sdf) {
      this.rebake();
      return;
    }
    if (this.refineTimer !== void 0) {
      clearTimeout(this.refineTimer);
      this.refineTimer = void 0;
    }
    this.sdf.setPalette(this.palette);
    this.sdf.setModePalette(this.modePalette);
    this.sdf.refineRegion(this.regionToIjk(regionRAS));
    for (const cb of this.redrawCbs) cb();
  }
  refreshOpacity(regionRAS) {
    if (!this.sdf) {
      this.rebake();
      return;
    }
    this.sdf.setPalette(this.palette);
    if (regionRAS) {
      this.sdf.rebakeAttr(true, this.regionToIjk(regionRAS));
      for (const cb of this.redrawCbs) cb();
      return;
    }
    this.sdf.rebakeAttr(false);
    for (const cb of this.redrawCbs) cb();
    if (this.attrSettleTimer !== void 0) clearTimeout(this.attrSettleTimer);
    this.attrSettleTimer = setTimeout(() => {
      this.attrSettleTimer = void 0;
      if (this.sdf) {
        this.sdf.blurAttrOnly();
        for (const cb of this.redrawCbs) cb();
      }
    }, 600);
  }
  /** A SegmentField bound to the shared render texture — hand this to the SceneRenderer once; edits
   *  update it in place. Colour comes from the texture (per-label); the uniform supplies opacity. */
  field() {
    if (!this.segField) {
      const tex = this.sdf ? this.sdf.sdfTexture() : this.presenceTex;
      const voxelMm = Math.min(...this.seg.spacingMm());
      const interfaceMode = this.renderMode === "sdf" && this.boundaryMode === "all";
      const band = this.bandMm ?? (this.renderMode === "sdf" ? (interfaceMode ? 1.5 : 0.65) * voxelMm : void 0);
      const fdims = this.sdf ? this.sdf.sdfDims() : this.seg.dims;
      const fijk = this.sdf ? this.sdf.sdfIjkToRAS() : this.seg.ijkToRAS;
      this.segField = new SegmentField(tex, fdims, [1, 1, 1], {
        color: [1, 1, 1],
        opacity: this.opacity,
        ijkToRAS: fijk,
        mode: this.renderMode === "sdf" ? "sdf" : "surface",
        colorFromTexture: true,
        bandMm: band,
        clippable: this.clippable,
        attrTexture: this.sdf ? this.sdf.attrTexture() : void 0,
        // per-segment opacity (sdf)
        interfaceMode
      });
    }
    return this.segField;
  }
  /** Live GLOBAL segmentation opacity (0..1) — the field-level multiplier over every segment's own
   *  opacity. Caller does scene.syncUniforms() + redraw. */
  setGlobalOpacity(o) {
    this.opacity = o;
    this.segField?.setOpacity(o);
  }
  /** Notified after every edit (post-rebake) so the app can redraw. */
  onRedraw(cb) {
    this.redrawCbs.push(cb);
  }
  destroy() {
    if (this.refineTimer !== void 0) clearTimeout(this.refineTimer);
    this.unsubDirty();
    this.sdf?.destroy();
    this.baker?.destroy();
    this.presenceTex?.destroy();
  }
};

// render/demos/segroulette-scene.ts
var SDF_MAX_DIM = 256;
function modalityLUT(modality, maxAlpha = 0.42) {
  const lut = new Uint8Array(256 * 4);
  const m = (modality ?? "CT").toUpperCase();
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let r, g, b, a;
    if (m === "PET" || m === "PT") {
      r = Math.min(1, t * 3);
      g = Math.min(1, Math.max(0, t * 3 - 1));
      b = Math.min(1, Math.max(0, t * 3 - 2));
      a = Math.max(0, (t - 0.25) / 0.75) * 0.9;
    } else {
      r = g = b = t;
      let aa = Math.max(0, (t - 0.42) / 0.58);
      aa *= aa;
      a = Math.min(maxAlpha, aa);
    }
    lut[i * 4] = Math.round(r * 255);
    lut[i * 4 + 1] = Math.round(g * 255);
    lut[i * 4 + 2] = Math.round(b * 255);
    lut[i * 4 + 3] = Math.round(a * 255);
  }
  return lut;
}
function buildSegrouletteScene(gpu, format, ct, seg, opts = {}) {
  const dev = gpu.device;
  const dims = ct.dims;
  const data = ct.vol instanceof Float32Array ? ct.vol : Float32Array.from(ct.vol);
  const clim = [ct.lev - ct.win / 2, ct.lev + ct.win / 2];
  const grayLut = modalityLUT(ct.modality);
  const grayClim = [clim[0], clim[1]];
  const grayShade = [0.25, 0.7, 0.45, 20];
  let baseVolLut = grayLut;
  let baseClim = [clim[0], clim[1]];
  let baseShade = [grayShade[0], grayShade[1], grayShade[2], grayShade[3]];
  let volShift = 0;
  const scaledVolLut = (o) => {
    const l = baseVolLut.slice();
    for (let i = 0; i < 256; i++) l[i * 4 + 3] = Math.round(l[i * 4 + 3] * o);
    return l;
  };
  const volumeField = new ImageField(dev, data, dims, [1, 1, 1], baseVolLut, {
    clim,
    ijkToRAS: ct.ijkToRAS,
    shade: baseShade
  });
  const segments = [];
  const palette = new Float32Array(256 * 4);
  if (seg) {
    const total = dims[0] * dims[1] * dims[2];
    for (const [num, r, g, b] of seg.colors) {
      if (num === 0 || r === 0 && g === 0 && b === 0) continue;
      let n = 0;
      for (let i = 0; i < seg.lab.length; i++) if (seg.lab[i] === num) n++;
      if (!n || n > total * 0.6) continue;
      if (num < 256) {
        palette[num * 4] = r;
        palette[num * 4 + 1] = g;
        palette[num * 4 + 2] = b;
        palette[num * 4 + 3] = 1;
      }
      segments.push({ num, name: seg.names[num] ?? `Segment ${num}`, color: [r, g, b], voxels: n });
    }
  }
  const segOpacity = /* @__PURE__ */ new Map();
  const opacityOf = (num) => segOpacity.get(num) ?? 1;
  const visPalette = () => {
    const p = palette.slice();
    for (const s of segments) if (s.num < 256) p[s.num * 4 + 3] = opacityOf(s.num);
    return p;
  };
  let colorTex = seg ? bakeColorizeRGBA(dev, seg.lab, dims, visPalette(), 1.5) : void 0;
  let mode = "volume";
  let segLogic;
  let editable;
  if (seg && segments.length > 0) {
    const cap = resampleIsotropic(seg.lab, dims, ct.ijkToRAS, opts.sdfMaxDim ?? SDF_MAX_DIM, opts.sdfMaxVoxels);
    if (cap.coarsened) console.warn(`SEGRoulette: SDF grid coarsened to fit the device memory budget (${cap.dims.join("\xD7")}, ${cap.vox.toFixed(2)}mm) \u2014 a remote render server would allow full resolution.`);
    editable = new EditableSegmentation(dev, cap.dims, { ijkToRAS: cap.ijkToRAS });
    const boundaryMode = labelmapHasInternalBoundary(cap.lab, cap.dims) ? "all" : "outer";
    segLogic = new SegmentationLogic(dev, editable, { renderMode: "sdf", opacity: 1, boundaryMode, refineDelayMs: opts.refineDelayMs });
    for (const s of segments) {
      segLogic.setLabelColor(s.num, s.color);
      segLogic.setLabelOpacity(s.num, opacityOf(s.num));
    }
    editable.loadLabelmap(cap.lab);
    segLogic.refineNow();
    mode = "sdf";
  }
  const hasSeg = !!segLogic;
  const scene = new SceneRenderer(gpu, format);
  const [rasLo, rasHi] = volumeField.aabb();
  const roi = createRoiWidget(rasLo, rasHi, { coverage: 0.35 });
  let volumeOpacity = 1, segLayerOpacity = 1;
  let showVolume = true, showSeg = hasSeg;
  let roiEnabled = false, roiVisible = false;
  const currentSegFields = () => segLogic ? [segLogic.field()] : [];
  const rebuild = () => {
    const f = [];
    if (showVolume) f.push(volumeField);
    if (showSeg) f.push(...currentSegFields());
    if (roiVisible) {
      f.push(roi.box, roi.handles);
    }
    scene.build(f);
    scene.setBackground(0.05, 0.06, 0.09);
    if (roiEnabled) scene.setClipBox(roi.lo(), roi.hi());
    else scene.clearClip();
  };
  rebuild();
  const rebakeColorized = () => {
    if (!seg) return;
    const nt = bakeColorizeRGBA(dev, seg.lab, dims, visPalette(), 1.5);
    const old = colorTex;
    colorTex = nt;
    slice.setTextures(volumeField.volumeTexture(), colorTex);
    old?.destroy();
  };
  const slice = new SliceRenderer(gpu, format);
  slice.setVolume(volumeField.patientToTexture(), rasLo, rasHi);
  slice.setTextures(volumeField.volumeTexture(), colorTex);
  slice.setWindowLevel(ct.win, ct.lev);
  slice.setOverlayOpacity(seg ? 0.5 : 0);
  const center = [(rasLo[0] + rasHi[0]) / 2, (rasLo[1] + rasHi[1]) / 2, (rasLo[2] + rasHi[2]) / 2];
  const radius = Math.hypot(rasHi[0] - rasLo[0], rasHi[1] - rasLo[1], rasHi[2] - rasLo[2]) / 2;
  return {
    scene,
    slice,
    center,
    radius,
    rasLo,
    rasHi,
    ijkToRAS: ct.ijkToRAS,
    dims,
    win: ct.win,
    lev: ct.lev,
    segments,
    mode,
    hasSeg,
    roi,
    setLayers(sv, ss) {
      this.setVolumeOpacity(sv ? 1 : 0);
      this.setSegOpacity(ss ? 1 : 0);
    },
    setVolumeOpacity(o) {
      volumeOpacity = Math.max(0, Math.min(1, o));
      const was = showVolume;
      showVolume = volumeOpacity > 1e-3;
      volumeField.setLUT(scaledVolLut(volumeOpacity));
      if (was !== showVolume) rebuild();
    },
    volumeOpacity: () => volumeOpacity,
    setVolumePreset(bake) {
      if (bake) {
        baseVolLut = bake.lut;
        baseClim = [bake.clim[0], bake.clim[1]];
        baseShade = bake.shade;
      } else {
        baseVolLut = grayLut;
        baseClim = [grayClim[0], grayClim[1]];
        baseShade = grayShade;
      }
      volumeField.setLUT(scaledVolLut(volumeOpacity));
      volumeField.setClim(baseClim[0] + volShift, baseClim[1] + volShift);
      volumeField.setShade(baseShade);
      scene.syncUniforms();
    },
    setVolumeShift(hu) {
      volShift = hu;
      volumeField.setClim(baseClim[0] + volShift, baseClim[1] + volShift);
      scene.syncUniforms();
    },
    volumeShift: () => volShift,
    setSegOpacity(o) {
      segLayerOpacity = Math.max(0, Math.min(1, o));
      const was = showSeg;
      showSeg = hasSeg && segLayerOpacity > 1e-3;
      segLogic?.setGlobalOpacity(segLayerOpacity);
      if (was !== showSeg) rebuild();
      else scene.syncUniforms();
    },
    segOpacity: () => segLayerOpacity,
    setSegmentOpacity(num, opacity) {
      const o = Math.max(0, Math.min(1, opacity));
      if (o >= 1) segOpacity.delete(num);
      else segOpacity.set(num, o);
      rebakeColorized();
      segLogic?.setLabelOpacity(num, o);
      segLogic?.refineNow();
    },
    segmentOpacity: (num) => opacityOf(num),
    setSegmentVisible(num, visible) {
      this.setSegmentOpacity(num, visible ? 1 : 0);
    },
    isSegmentVisible: (num) => opacityOf(num) > 0,
    setRoiEnabled(on) {
      roiEnabled = on;
      rebuild();
    },
    setRoiVisible(on) {
      roiVisible = on;
      rebuild();
    },
    roiEnabled: () => roiEnabled,
    roiVisible: () => roiVisible,
    reclip() {
      if (roiEnabled) scene.setClipBox(roi.lo(), roi.hi());
      else scene.clearClip();
      scene.syncUniforms();
    },
    destroy() {
      segLogic?.destroy();
      editable?.destroy();
    }
  };
}

// render/demos/crosshair.ts
function createCrosshair(visible = true) {
  const listeners = /* @__PURE__ */ new Set();
  const notify = () => {
    for (const cb of listeners) cb();
  };
  const st = {
    ras: null,
    visible,
    set(ras) {
      this.ras = ras;
      notify();
    },
    toggle(on) {
      this.visible = on ?? !this.visible;
      notify();
    },
    onChange(cb) {
      listeners.add(cb);
      return () => listeners.delete(cb);
    }
  };
  return st;
}
function drawCross(ctx, x, y, opts = {}) {
  const size = opts.size ?? 11, gap = opts.gap ?? 3;
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = opts.color ?? "rgba(120,220,255,0.95)";
  ctx.shadowColor = "rgba(0,0,0,0.8)";
  ctx.shadowBlur = 2;
  ctx.beginPath();
  ctx.moveTo(x - size, y);
  ctx.lineTo(x - gap, y);
  ctx.moveTo(x + gap, y);
  ctx.lineTo(x + size, y);
  ctx.moveTo(x, y - size);
  ctx.lineTo(x, y - gap);
  ctx.moveTo(x, y + gap);
  ctx.lineTo(x, y + size);
  ctx.stroke();
  ctx.restore();
}
function rasToScreen3D(cam, ras, w, h) {
  const vp = multiply(perspectiveZO(cam.viewAngle * Math.PI / 180, w / h, 1, 1e5), lookAt(cam.position, cam.focalPoint, cam.viewUp));
  const cw = vp[3] * ras[0] + vp[7] * ras[1] + vp[11] * ras[2] + vp[15];
  if (cw <= 0) return null;
  return {
    x: (vp[0] * ras[0] + vp[4] * ras[1] + vp[8] * ras[2] + vp[12]) / cw * 0.5 + 0.5,
    y: 1 - ((vp[1] * ras[0] + vp[5] * ras[1] + vp[9] * ras[2] + vp[13]) / cw * 0.5 + 0.5)
  };
}
var uvOf = (canvas, e) => {
  const r = canvas.getBoundingClientRect();
  return { u: (e.clientX - r.left) / r.width, v: (e.clientY - r.top) / r.height, aspect: r.width / r.height };
};
var isShiftHover = (e) => e.shiftKey && e.buttons === 0;
function mountCrosshair(cfg) {
  const state = createCrosshair(cfg.visible ?? true);
  const slices = ["axial", "coronal", "sagittal"];
  const all = [...slices, "threeD"];
  const ctx = {};
  for (const cell of all) {
    const o = document.createElement("canvas");
    o.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;border-radius:6px;background:transparent;";
    cfg.cells[cell].parentElement.appendChild(o);
    ctx[cell] = { c: o, g: o.getContext("2d") };
  }
  const redraw = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const cell of all) {
      const { c, g } = ctx[cell];
      const w = cfg.cells[cell].clientWidth, h = cfg.cells[cell].clientHeight;
      if (!w || !h) continue;
      if (c.width !== Math.floor(w * dpr)) {
        c.width = Math.floor(w * dpr);
        c.height = Math.floor(h * dpr);
      }
      g.setTransform(c.width / w, 0, 0, c.height / h, 0, 0);
      g.clearRect(0, 0, w, h);
      if (!state.visible || !state.ras) continue;
      if (cell === "threeD") {
        const s = rasToScreen3D(cfg.getCamera(), state.ras, w, h);
        if (s) drawCross(g, s.x * w, s.y * h);
      } else {
        const pr = cfg.getSlice().rasToView(cell, cfg.getOffset(cell), state.ras, w / h);
        if (pr.u >= 0 && pr.u <= 1 && pr.v >= 0 && pr.v <= 1) drawCross(g, pr.u * w, pr.v * h);
      }
    }
  };
  state.onChange(redraw);
  let inFlight = false, queued = null;
  const pick3d = async (u, v) => {
    inFlight = true;
    const ras = await cfg.getScene().pick(u, v);
    inFlight = false;
    if (ras) {
      state.set(ras);
      cfg.onJump(ras);
    }
    if (queued) {
      const q = queued;
      queued = null;
      pick3d(q.u, q.v);
    }
  };
  cfg.cells.threeD.addEventListener("pointermove", (e) => {
    if (!isShiftHover(e)) return;
    const { u, v } = uvOf(cfg.cells.threeD, e);
    if (inFlight) queued = { u, v };
    else pick3d(u, v);
  });
  for (const cell of slices) {
    cfg.cells[cell].addEventListener("pointermove", (e) => {
      if (!isShiftHover(e)) return;
      const { u, v, aspect } = uvOf(cfg.cells[cell], e);
      const ras = cfg.getSlice().viewToRas(cell, cfg.getOffset(cell), u, v, aspect);
      state.set(ras);
      cfg.onJump(ras);
    });
  }
  return { state, redraw };
}

// render/vtk-camera.ts
var sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
var add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
var scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
var cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0]
];
var dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
var norm = (a) => Math.hypot(a[0], a[1], a[2]);
var normalize = (a) => {
  const n = norm(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};
function rotateAboutAxis(v, axis, deg) {
  const k = normalize(axis);
  const t = deg * Math.PI / 180;
  const c = Math.cos(t), s = Math.sin(t);
  const kv = cross(k, v);
  const kd = dot(k, v);
  return [
    v[0] * c + kv[0] * s + k[0] * kd * (1 - c),
    v[1] * c + kv[1] * s + k[1] * kd * (1 - c),
    v[2] * c + kv[2] * s + k[2] * kd * (1 - c)
  ];
}
var VtkCamera = class _VtkCamera {
  position;
  focalPoint;
  viewUp;
  viewAngle;
  // degrees (vtkCamera default 30)
  parallelProjection = false;
  parallelScale = 1;
  constructor(position = [0, 0, 1], focalPoint = [0, 0, 0], viewUp = [0, 1, 0], viewAngle = 30) {
    this.position = [...position];
    this.focalPoint = [...focalPoint];
    this.viewUp = [...viewUp];
    this.viewAngle = viewAngle;
  }
  /** Slicer's default 3D camera (vtkMRMLCameraNode): (0,500,0) -> origin, +S up, 30 deg. */
  static slicerDefault() {
    return new _VtkCamera([0, 500, 0], [0, 0, 0], [0, 0, 1], 30);
  }
  clone() {
    const c = new _VtkCamera(this.position, this.focalPoint, this.viewUp, this.viewAngle);
    c.parallelProjection = this.parallelProjection;
    c.parallelScale = this.parallelScale;
    return c;
  }
  get distance() {
    return norm(sub(this.focalPoint, this.position));
  }
  /** normalize(focalPoint - position) — vtkCamera::DirectionOfProjection. */
  get directionOfProjection() {
    return normalize(sub(this.focalPoint, this.position));
  }
  /** Rows of the view transform, per vtkTransform::SetupCamera. */
  basis(viewUp = this.viewUp) {
    const back = normalize(sub(this.position, this.focalPoint));
    const right = normalize(cross(viewUp, back));
    const up = cross(back, right);
    return { right, up, back };
  }
  /** vtkCamera::Azimuth — rotate position about viewUp through the focal point. */
  azimuth(deg) {
    const rel = sub(this.position, this.focalPoint);
    this.position = add(this.focalPoint, rotateAboutAxis(rel, this.viewUp, deg));
  }
  /** vtkCamera::Elevation — rotate position about -right through the focal point.
   *  Returns the rotated view-up VTK uses internally (see class comment); callers that
   *  mirror Slicer follow with orthogonalizeViewUp(rotatedUp). */
  elevation(deg) {
    const axis = scale(this.basis().right, -1);
    const rotatedUp = rotateAboutAxis(this.viewUp, axis, deg);
    const rel = sub(this.position, this.focalPoint);
    this.position = add(this.focalPoint, rotateAboutAxis(rel, axis, deg));
    return rotatedUp;
  }
  /** vtkCamera::OrthogonalizeViewUp — viewUp = row1 of the view transform. */
  orthogonalizeViewUp(usingUp = this.viewUp) {
    this.viewUp = this.basis(usingUp).up;
  }
  /** vtkCamera::Dolly — factor > 1 moves the camera toward the focal point. */
  dolly(factor) {
    if (factor <= 0) return;
    if (this.parallelProjection) {
      this.parallelScale = this.parallelScale / factor;
      return;
    }
    const d = this.distance / factor;
    const dop = this.directionOfProjection;
    this.position = sub(this.focalPoint, scale(dop, d));
  }
  /** Translate both position and focal point (used by pan). */
  translate(v) {
    this.position = add(this.position, v);
    this.focalPoint = add(this.focalPoint, v);
  }
  /** Half-height of the view plane at the focal point (perspective). */
  focalPlaneHalfHeight() {
    return this.parallelProjection ? this.parallelScale : this.distance * Math.tan(this.viewAngle * Math.PI / 360);
  }
  /** Pan by a display-space delta, moving the world under the cursor 1:1 at focal depth.
   *  Equivalent to vtkMRMLCameraWidget::ProcessTranslate's focal-depth unprojection, but
   *  expressed directly in the camera basis (exact for a centred perspective view).
   *  dxDisplay/dyDisplay are in VTK display convention (y UP). */
  panByDisplayDelta(dxDisplay, dyDisplay, viewportWidth, viewportHeight) {
    const halfH = this.focalPlaneHalfHeight();
    const mmPerPixel = 2 * halfH / viewportHeight;
    const { right, up } = this.basis();
    const motion = add(scale(right, -dxDisplay * mmPerPixel), scale(up, -dyDisplay * mmPerPixel));
    this.translate(motion);
  }
  /** Project a world (RAS) point to display pixels (y DOWN, origin top-left) for a w×h viewport.
   *  Vertical-FOV perspective matching SceneRenderer.setCamera (perspectiveZO(fovy, w/h)). `depth`
   *  is the distance along the view direction (>0 in front of the camera). Used to hit-test
   *  screen-space markup glyphs. */
  worldToDisplay(p, w, h) {
    const { right, up } = this.basis();
    const dop = this.directionOfProjection;
    const rel = sub(p, this.position);
    const depth = dot(rel, dop);
    const halfH = Math.max(1e-6, depth) * Math.tan(this.viewAngle * Math.PI / 360);
    const aspect = w / h;
    const ndcx = dot(rel, right) / (halfH * aspect);
    const ndcy = dot(rel, up) / halfH;
    return { x: (ndcx * 0.5 + 0.5) * w, y: (0.5 - ndcy * 0.5) * h, depth };
  }
  /** Inverse of worldToDisplay at a FIXED view-depth: the world point under display pixel (x,y)
   *  lying in the plane perpendicular to the view at `depth`. Dragging a 3D handle in this plane
   *  keeps its distance from the camera, so it tracks the cursor without depth ambiguity. */
  displayToWorldAtDepth(x, y, depth, w, h) {
    const { right, up } = this.basis();
    const dop = this.directionOfProjection;
    const halfH = Math.max(1e-6, depth) * Math.tan(this.viewAngle * Math.PI / 360);
    const aspect = w / h;
    const ndcx = x / w * 2 - 1;
    const ndcy = 1 - y / h * 2;
    const offset = add(scale(right, ndcx * halfH * aspect), scale(up, ndcy * halfH));
    return add(add(this.position, scale(dop, depth)), offset);
  }
  /** vtkCamera-comparable snapshot for the harness. */
  state() {
    return {
      position: [...this.position],
      focalPoint: [...this.focalPoint],
      viewUp: [...this.viewUp],
      viewAngle: this.viewAngle,
      distance: this.distance
    };
  }
};

// render/vtk-interactor.ts
var MOTION_FACTOR = 10;
var MOUSE_WHEEL_MOTION_FACTOR = 1;
function actionForButton(button, m = {}) {
  const shift = !!m.shift, ctrl = !!m.ctrl, alt = !!m.alt;
  if (button === 0) {
    if (shift && ctrl) return "scale";
    if (ctrl) return "spin";
    if (shift) return "translate";
    return "rotate";
  }
  if (button === 1) return "translate";
  if (button === 2) return "scale";
  return "none";
}
var CameraInteractor = class _CameraInteractor {
  camera;
  action = "none";
  prev = null;
  // previous position, VTK display coords
  onChange;
  constructor(camera, onChange) {
    this.camera = camera;
    this.onChange = onChange;
  }
  /** Convert browser (cssX, cssY within the view) to VTK display coords (y up). */
  static toDisplay(cssX, cssY, height) {
    return [cssX, height - cssY];
  }
  start(button, cssX, cssY, height, m = {}) {
    this.action = actionForButton(button, m);
    this.prev = _CameraInteractor.toDisplay(cssX, cssY, height);
  }
  end() {
    this.action = "none";
    this.prev = null;
  }
  /** Mouse move while dragging. width/height are the view size in CSS pixels. */
  move(cssX, cssY, width, height) {
    if (this.action === "none" || !this.prev) return;
    const [x, y] = _CameraInteractor.toDisplay(cssX, cssY, height);
    const dx = x - this.prev[0];
    const dy = y - this.prev[1];
    if (dx === 0 && dy === 0) return;
    switch (this.action) {
      case "rotate":
        this.rotate(dx, dy, width, height);
        break;
      case "translate":
        this.camera.panByDisplayDelta(dx, dy, width, height);
        break;
      case "scale":
        this.scale(dy, height);
        break;
      case "spin":
        this.spin(x, y, this.prev[0], this.prev[1], width, height);
        break;
    }
    this.prev = [x, y];
    this.onChange?.();
  }
  /** vtkMRMLCameraWidget::ProcessRotate */
  rotate(dx, dy, width, height) {
    const deltaAzimuth = -20 / width;
    const deltaElevation = -20 / height;
    const rxf = dx * deltaAzimuth * MOTION_FACTOR;
    const ryf = dy * deltaElevation * MOTION_FACTOR;
    this.camera.azimuth(rxf);
    const rotatedUp = this.camera.elevation(ryf);
    this.camera.orthogonalizeViewUp(rotatedUp);
  }
  /** vtkMRMLCameraWidget::ProcessScale — note the sign flip vs plain VTK. */
  scale(dy, height) {
    const centerY = height / 2;
    const dyf = MOTION_FACTOR * dy / centerY;
    this.camera.dolly(Math.pow(1.1, -dyf));
  }
  /** vtkMRMLCameraWidget::ProcessSpin — roll about the view plane normal. */
  spin(x, y, px, py, width, height) {
    const cx = width / 2, cy = height / 2;
    const newAngle = Math.atan2(y - cy, x - cx) * 180 / Math.PI;
    const oldAngle = Math.atan2(py - cy, px - cx) * 180 / Math.PI;
    this.roll(newAngle - oldAngle);
  }
  /** vtkCamera::Roll — rotate viewUp about the direction of projection. */
  roll(deg) {
    const cam = this.camera;
    const axis = cam.directionOfProjection;
    const t = deg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
    const v = cam.viewUp;
    const k = axis;
    const kv = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]];
    const kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    cam.viewUp = [
      v[0] * c + kv[0] * s + k[0] * kd * (1 - c),
      v[1] * c + kv[1] * s + k[1] * kd * (1 - c),
      v[2] * c + kv[2] * s + k[2] * kd * (1 - c)
    ];
    cam.orthogonalizeViewUp();
    this.onChange?.();
  }
  /** Mouse wheel. `forward` = wheel away from the user = zoom in. */
  wheel(forward) {
    const e = 0.2 * MOTION_FACTOR * MOUSE_WHEEL_MOTION_FACTOR;
    this.camera.dolly(Math.pow(1.1, forward ? e : -e));
    this.onChange?.();
  }
};

// render/demos/camera-control.ts
function attachCameraControls(canvas, camera, opts = {}) {
  const interactor = new CameraInteractor(camera, opts.onChange);
  const local = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  canvas.style.touchAction = "none";
  const docEl = (canvas.ownerDocument ?? document).documentElement;
  if (docEl) docEl.style.overscrollBehavior = "none";
  canvas.addEventListener("touchmove", (e) => e.preventDefault(), { passive: false });
  const pointers = /* @__PURE__ */ new Map();
  let pinch = null;
  let triple = null;
  const centroid = () => {
    let mx = 0, my = 0;
    for (const p of pointers.values()) {
      mx += p.x;
      my += p.y;
    }
    const n = pointers.size || 1;
    return { mx: mx / n, my: my / n };
  };
  const pinchState = () => {
    const [a, b] = [...pointers.values()];
    return { dist: Math.hypot(b.x - a.x, b.y - a.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
  };
  const on = () => opts.enabled?.() ?? true;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    if (!on()) return;
    const { x, y } = local(e);
    pointers.set(e.pointerId, { x, y });
    canvas.setPointerCapture(e.pointerId);
    if (pointers.size === 1) {
      interactor.start(e.button, x, y, canvas.clientHeight, { shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey, alt: e.altKey });
      opts.onLog?.("cameraStart", { action: interactor.action, x, y, button: e.button, shift: e.shiftKey, ctrl: e.ctrlKey, alt: e.altKey });
    } else if (pointers.size === 2) {
      interactor.end();
      pinch = pinchState();
    } else if (pointers.size === 3) {
      pinch = null;
      const c = centroid();
      triple = { mx: c.mx, my: c.my };
      opts.onVolumeDragStart?.();
    }
  });
  const endPointer = (e) => {
    if (!pointers.delete(e.pointerId)) return;
    if (!on()) {
      interactor.end();
      pinch = null;
      return;
    }
    canvas.releasePointerCapture?.(e.pointerId);
    if (pointers.size < 2) pinch = null;
    if (pointers.size < 3 && triple) {
      triple = null;
      opts.onVolumeDragEnd?.();
    }
    if (pointers.size === 1) {
      const p = [...pointers.values()][0];
      interactor.start(0, p.x, p.y, canvas.clientHeight, { shift: false, ctrl: false, alt: false });
    } else if (pointers.size === 0) {
      interactor.end();
    }
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("pointermove", (e) => {
    if (!on()) return;
    if (!pointers.has(e.pointerId)) return;
    const { x, y } = local(e);
    pointers.set(e.pointerId, { x, y });
    if (pointers.size >= 3) {
      const c = centroid();
      if (triple) opts.onVolumeDrag?.(c.mx - triple.mx, c.my - triple.my);
      return;
    }
    if (pointers.size === 2) {
      const p = pinchState();
      if (pinch) {
        if (p.dist > 0 && pinch.dist > 0) camera.dolly(p.dist / pinch.dist);
        camera.panByDisplayDelta(p.mx - pinch.mx, pinch.my - p.my, canvas.clientWidth, canvas.clientHeight);
        opts.onChange?.();
      }
      pinch = p;
    } else if (interactor.action !== "none") {
      interactor.move(x, y, canvas.clientWidth, canvas.clientHeight);
    }
  });
  canvas.addEventListener("wheel", (e) => {
    if (!on()) return;
    e.preventDefault();
    interactor.wheel(e.deltaY < 0);
    opts.onLog?.("cameraWheel", { deltaY: e.deltaY, distance: camera.distance });
  }, { passive: false });
  return interactor;
}
function framedCamera(center, radius, distMul = 2.6) {
  return new VtkCamera(
    [center[0], center[1] + radius * distMul, center[2]],
    [...center],
    [0, 0, 1],
    30
  );
}

// render/demos/slice-control.ts
function attachSliceControls(canvas, cfg) {
  const SCROLL_PX = cfg.scrollPx ?? 7;
  const h = cfg.hooks ?? {};
  const uv = (e) => {
    const r = canvas.getBoundingClientRect();
    return { u: (e.clientX - r.left) / r.width, v: (e.clientY - r.top) / r.height, w: r.width, h: r.height };
  };
  let lastDown = 0, lastX = 0, lastY = 0;
  let view = null;
  let scroll = null;
  let grabbed = null;
  let wlDrag = null;
  const onContext = (e) => e.preventDefault();
  const onWheel = (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      const { u, v, w, h: hh } = uv(e);
      cfg.getSlice().zoomAbout(cfg.orient, Math.exp(-e.deltaY * 15e-4), u, v, w, hh);
      cfg.redraw();
      h.onZoom?.();
      return;
    }
    cfg.step(e.deltaY < 0);
    cfg.redraw();
    h.onScroll?.(e.deltaY < 0);
  };
  const onDown = (e) => {
    if (e.button === 0) {
      const now = e.timeStamp, dbl = now - lastDown < 350 && Math.hypot(e.clientX - lastX, e.clientY - lastY) < 6;
      lastDown = dbl ? 0 : now;
      lastX = e.clientX;
      lastY = e.clientY;
      if (dbl && (e.ctrlKey || e.metaKey) && cfg.wl?.enabled() && cfg.wl.reset) {
        e.preventDefault();
        cfg.wl.reset();
        cfg.redraw();
        return;
      }
      if (dbl && h.onDoubleClick?.()) {
        e.preventDefault();
        return;
      }
    }
    const wantPan = e.button === 1 || e.button === 0 && e.shiftKey;
    const wantZoom = e.button === 2;
    if (wantPan || wantZoom) {
      e.preventDefault();
      const { u: u2, v: v2 } = uv(e);
      view = { mode: wantZoom ? "zoom" : "pan", x: e.clientX, y: e.clientY, pu: u2, pv: v2 };
      canvas.style.cursor = wantZoom ? "ns-resize" : "grabbing";
      canvas.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button !== 0) return;
    e.preventDefault();
    const { u, v, w, h: hh } = uv(e);
    const mode = cfg.leftMode?.() ?? (cfg.wl?.enabled() ? "wl" : "scroll");
    if (h.onLeftGrab?.(u, v, w, hh)) {
      grabbed = { moved: 0 };
    } else if (mode === "wl" && cfg.wl) {
      const [win, lev] = cfg.wl.get();
      wlDrag = { x: e.clientX, y: e.clientY, win, lev };
      canvas.style.cursor = "crosshair";
    } else if (mode === "zoom" || mode === "pan") {
      view = { mode, x: e.clientX, y: e.clientY, pu: u, pv: v };
      canvas.style.cursor = mode === "zoom" ? "ns-resize" : "grabbing";
    } else scroll = { x: e.clientX, y: e.clientY, acc: 0 };
    canvas.setPointerCapture(e.pointerId);
  };
  const onMove = (e) => {
    if (view) {
      const dx = e.clientX - view.x, dy = e.clientY - view.y;
      const r = canvas.getBoundingClientRect();
      if (view.mode === "pan") cfg.getSlice().panByPixels(cfg.orient, dx, dy, r.width, r.height);
      else cfg.getSlice().zoomAbout(cfg.orient, Math.exp(dy * 6e-3), 0.5, 0.5, r.width, r.height);
      view.x = e.clientX;
      view.y = e.clientY;
      cfg.redraw();
      return;
    }
    if (grabbed) {
      grabbed.moved += Math.abs(e.movementX) + Math.abs(e.movementY);
      const { u, v, w, h: hh } = uv(e);
      h.onLeftDrag?.(u, v, w, hh);
      return;
    }
    if (wlDrag && cfg.wl) {
      const [lo, hi] = cfg.wl.range();
      const r = canvas.getBoundingClientRect();
      const gain = (hi - lo) / Math.max(1, Math.min(r.width, r.height));
      let win = wlDrag.win + gain * (e.clientX - wlDrag.x);
      if (win < 0) win = 0;
      let lev = wlDrag.lev + gain * (wlDrag.y - e.clientY);
      if (lev < lo - win / 2) lev = lo - win / 2;
      if (lev > hi + win / 2) lev = hi + win / 2;
      cfg.wl.set(win, lev);
      wlDrag = { x: e.clientX, y: e.clientY, win, lev };
      cfg.redraw();
      return;
    }
    if (scroll) {
      scroll.acc += e.clientX - scroll.x - (e.clientY - scroll.y);
      scroll.x = e.clientX;
      scroll.y = e.clientY;
      while (Math.abs(scroll.acc) >= SCROLL_PX) {
        const f = scroll.acc > 0;
        cfg.step(f);
        scroll.acc -= f ? SCROLL_PX : -SCROLL_PX;
      }
      cfg.redraw();
      return;
    }
    if (e.buttons === 0 && h.onHover) {
      const { u, v, w, h: hh } = uv(e);
      h.onHover(u, v, w, hh);
    }
  };
  const onUp = (e) => {
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
    }
    if (view) {
      view = null;
      canvas.style.cursor = "default";
      return;
    }
    if (grabbed) {
      const m = grabbed.moved;
      grabbed = null;
      h.onLeftDrop?.(m);
      return;
    }
    if (wlDrag) {
      wlDrag = null;
      canvas.style.cursor = "default";
      return;
    }
    scroll = null;
  };
  canvas.addEventListener("contextmenu", onContext);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  return {
    resetView() {
      cfg.getSlice().resetView(cfg.orient);
      cfg.redraw();
    },
    detach() {
      canvas.removeEventListener("contextmenu", onContext);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
    }
  };
}

// render/demos/view-grid.ts
function attachViewGrid(grid, cells, onResize) {
  let maxed = null;
  const cellDiv = (cell) => grid.querySelector(`.cell[data-cell="${cell}"]`);
  return {
    toggleMax(cell) {
      maxed = maxed === cell ? null : cell;
      for (const n of cells) cellDiv(n).classList.toggle("max", n === maxed);
      grid.classList.toggle("has-max", maxed !== null);
      requestAnimationFrame(onResize);
    },
    isMax(cell) {
      return maxed === cell;
    },
    maxCell: () => maxed
  };
}
function attachDoubleClick(canvas, onDbl) {
  let last = 0, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => {
    const dbl = e.timeStamp - last < 350 && Math.hypot(e.clientX - lx, e.clientY - ly) < 6;
    last = dbl ? 0 : e.timeStamp;
    lx = e.clientX;
    ly = e.clientY;
    if (dbl) {
      e.preventDefault();
      onDbl();
    }
  });
}

// render/demos/widget-control.ts
var sub2 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
var dot2 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function camMatrices(cam, w, h) {
  const view = lookAt(cam.position, cam.focalPoint, cam.viewUp);
  const proj = perspectiveZO(cam.viewAngle * Math.PI / 180, w / h, 1, 1e5);
  const vp = multiply(proj, view);
  return { vp, invVp: invert(vp) };
}
function worldToClip(vp, p) {
  return [
    vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12],
    vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13],
    vp[2] * p[0] + vp[6] * p[1] + vp[10] * p[2] + vp[14],
    vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15]
  ];
}
function attachWidgetControls(canvas, camera, opts) {
  const cursorCss = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top, rw: r.width, rh: r.height };
  };
  const project = (vp, world, rw, rh) => {
    const c = worldToClip(vp, world);
    if (c[3] <= 0) return null;
    const ndcx = c[0] / c[3], ndcy = c[1] / c[3];
    return { x: (ndcx * 0.5 + 0.5) * rw, y: (1 - (ndcy * 0.5 + 0.5)) * rh };
  };
  const unprojectToPlane = (invVp, px, py, rw, rh, planePt) => {
    const ndcx = px / rw * 2 - 1, ndcy = 1 - py / rh * 2;
    const near = applyMat4(invVp, [ndcx, ndcy, 0]);
    const far = applyMat4(invVp, [ndcx, ndcy, 1]);
    const ro = near, rd = sub2(far, near);
    const n = sub2(camera.position, camera.focalPoint);
    const denom = dot2(rd, n);
    if (Math.abs(denom) < 1e-9) return [...planePt];
    const t = dot2(sub2(planePt, ro), n) / denom;
    return [ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t];
  };
  const pick = (e) => {
    const { x, y, rw, rh } = cursorCss(e);
    const { w, h } = opts.getSize();
    const { vp } = camMatrices(camera, w, h);
    const touch = e.pointerType === "touch";
    let best = null, bestD = Infinity;
    for (const hnd of opts.getHandles()) {
      const s = project(vp, hnd.world, rw, rh);
      if (!s) continue;
      const r = (hnd.pickPx ?? 16) * (touch ? 2.75 : 1);
      const d = Math.hypot(s.x - x, s.y - y);
      if (d < r && d < bestD) {
        bestD = d;
        best = hnd;
      }
    }
    return best;
  };
  let grabbed = null, hovered = null, grabbedId = -1;
  const release = (pointerId) => {
    if (!grabbed) return;
    const g = grabbed;
    grabbed = null;
    grabbedId = -1;
    try {
      canvas.releasePointerCapture(pointerId);
    } catch {
    }
    window.removeEventListener("pointermove", onMove, true);
    window.removeEventListener("pointerup", onUp, true);
    window.removeEventListener("pointercancel", onCancel, true);
    canvas.style.cursor = "";
    opts.onDragEnd?.(g);
  };
  const onDown = (e) => {
    if (e.button !== 0) return;
    if (grabbed) {
      release(e.pointerId);
      return;
    }
    if (e.isPrimary === false) return;
    const h = pick(e);
    if (!h) return;
    e.stopPropagation();
    e.preventDefault();
    grabbed = h;
    grabbedId = e.pointerId;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = h.cursor ? h.cursor : "grabbing";
    opts.onDragStart?.(h);
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
    window.addEventListener("pointercancel", onCancel, true);
  };
  const onMove = (e) => {
    if (!grabbed || e.pointerId !== grabbedId) return;
    e.stopPropagation();
    const { x, y, rw, rh } = cursorCss(e);
    const { w, h } = opts.getSize();
    const { invVp } = camMatrices(camera, w, h);
    const world = unprojectToPlane(invVp, x, y, rw, rh, grabbed.world);
    opts.onDrag(grabbed, world);
    opts.onChange?.();
  };
  const onUp = (e) => {
    if (!grabbed || e.pointerId !== grabbedId) return;
    e.stopPropagation();
    release(e.pointerId);
  };
  const onCancel = (e) => {
    if (grabbed && e.pointerId === grabbedId) release(e.pointerId);
  };
  const onHoverMove = (e) => {
    if (grabbed) return;
    const h = pick(e);
    if (h !== hovered) {
      hovered = h;
      canvas.style.cursor = h ? h.cursor ?? "grab" : "";
      opts.onHover?.(h);
      opts.onChange?.();
    }
  };
  canvas.addEventListener("pointerdown", onDown, true);
  canvas.addEventListener("pointermove", onHoverMove);
  return {
    detach() {
      canvas.removeEventListener("pointerdown", onDown, true);
      canvas.removeEventListener("pointermove", onHoverMove);
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      window.removeEventListener("pointercancel", onCancel, true);
    }
  };
}

// render/demos/mosaic.ts
function createMosaic(host) {
  const wrap = document.createElement("div");
  wrap.style.cssText = "position:absolute;inset:0;z-index:20;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;background:radial-gradient(60% 60% at 50% 45%,rgba(20,24,38,.35),rgba(6,8,14,.75));opacity:1;transition:opacity 350ms ease-out;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "max-width:min(70vmin,620px);max-height:70vh;border-radius:8px;box-shadow:0 10px 40px rgba(0,0,0,.5);image-rendering:auto;";
  const cap = document.createElement("div");
  cap.style.cssText = "font:600 12px ui-monospace,Menlo,monospace;color:#9fe9ff;text-shadow:0 0 4px #000;";
  wrap.appendChild(canvas);
  wrap.appendChild(cap);
  const hostPos = getComputedStyle(host).position;
  if (hostPos === "static") host.style.position = "relative";
  host.appendChild(wrap);
  const ctx = canvas.getContext("2d");
  const tmp = document.createElement("canvas");
  const tctx = tmp.getContext("2d");
  let cols = 1, rows = 1, cw = 0, ch = 0, count = 0, filled = 0;
  const layout = (n, aspect = 1) => {
    count = Math.max(1, n);
    cols = Math.ceil(Math.sqrt(count));
    rows = Math.ceil(count / cols);
    const cellW = 96;
    cw = cellW;
    ch = Math.round(cellW / aspect);
    canvas.width = cols * cw;
    canvas.height = rows * ch;
    ctx.fillStyle = "#0b0e14";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };
  return {
    setCount(n) {
      layout(n);
    },
    thumb(n, w, h, rgba) {
      if (!count || cols * rows < count) layout(Math.max(count, n), w / h);
      if (ch !== Math.round(cw / (w / h))) {
        ch = Math.round(cw / (w / h));
        canvas.height = rows * ch;
        ctx.fillStyle = "#0b0e14";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      tmp.width = w;
      tmp.height = h;
      tctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
      const idx = Math.max(0, (n | 0) - 1) % (cols * rows);
      const cx = idx % cols * cw, cy = Math.floor(idx / cols) * ch;
      ctx.drawImage(tmp, cx, cy, cw, ch);
      filled++;
      cap.textContent = `streaming ${filled}${count ? " / " + count : ""} slices\u2026`;
    },
    status(msg) {
      cap.textContent = msg;
    },
    done() {
      wrap.style.opacity = "0";
      setTimeout(() => wrap.remove(), 400);
    },
    reset() {
      filled = 0;
      count = 0;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      wrap.style.opacity = "1";
      if (!wrap.isConnected) host.appendChild(wrap);
    }
  };
}

// render/demos/sl-logo.ts
var SL_LOGO = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAA8CAIAAABTt4VhAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAARGVYSWZNTQAqAAAACAABh2kABAAAAAEAAAAaAAAAAAADoAEAAwAAAAEAAQAAoAIABAAAAAEAAAA5oAMABAAAAAEAAAA8AAAAAH9xBdAAAAHLaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWERpbWVuc2lvbj41MDA8L2V4aWY6UGl4ZWxYRGltZW5zaW9uPgogICAgICAgICA8ZXhpZjpQaXhlbFlEaW1lbnNpb24+NTIwPC9leGlmOlBpeGVsWURpbWVuc2lvbj4KICAgICAgPC9yZGY6RGVzY3JpcHRpb24+CiAgIDwvcmRmOlJERj4KPC94OnhtcG1ldGE+ConTBbQAABmbSURBVGgFjZpZkB3XWcd7775919k1o2VGsjZLthw5sR3HiZ3EGMcJJqRIXKmiqAKTByh4yEN4pQJFUVBUUUWRQIViCVQZQ0IWJyGLYyeyY8lYkrXYlmxJtnbNSLPeO3fpvZvfd/pKOOSFnjt97+0+fc7//L/1fOfqjbFZjaPItCLVNU3Xdc3gj3fd0DTP1bZtsD/5yNwnfvmef/23Z1860VlY1dJcM7lrFAYnXePEWeMD/ci/HEWha3mRa0XOvzrzIS+0jAtylbci40XDnNZyq5B3+aQ+FwXf5UKhGZZhWHzmX/qnZ66XAGV0EHNNF0zcWm/3gl77Pfu2Hzv9qmvrZloYprQBpMnEmJl6VB4qDzUEGPJc5yNnBTTPcs3IDb5lZVPOQDZkPvBCe8Ys6FDgCgAQ8cctdegWX9QMNMWQ3FT3ChCYhuHYuu9oFUczbbtSqTiW5jl6bOiWyWRBWWItcUuHJVohhpdg5SVEpvDFma8pf0ZKC0CkhW7qaUY/Q7jyCA8K3BK6dKJkLV0LrxyCT+YgQKESoUKbY2sNX9805e3fv3V6Zvrgy6fTwrRNgGoiFUvOMiU4pj0Pqlmq7mVIAXoTIoCyTLCm4ABcljNKyoBZgexuwlWS5aKSP9wpphU+JXWwCtsykCioUG+YcGwAqOrokyPW3u2jO7ZtPvDcz469di0rTMfKhVRLGticBavBiw+i63QFCvQyK7IsFzqBqFAmqZamWmxqZponCUgYi3ENmYGhw7Rp6Ggw3YhGiB4Ig5kisTxZdF8OUZJiWgIUtmxT8z1jy7R//727PN+bv97pBaDQKq6JyoLSsQ3bpmGexPFgEIVRVuhukhZx1KfNxFijWa8oA9IS5J4VcWrESR4lehzroQ7jJqwqmoZwmd5NuAgEFRGzM0WPwSiHJQrB7ORrYSFNaBV11BxL91297pueYw36Sa8fMgfXFi4rHmLPozBY6iS5XhuZ2rVtz50zs7tWFuePHzmQDHrdOFq9tLBhJKj6HijLl/AK9gLNoU+hLBTicAoiVdFoU3g1DQQCHmYzhDtEClbhVb1MET0odT5Ylu7aYDJqnul7rikM5r4nU6FBp7PeT5wNs+/7yKOP7L7j3ompGdetHD/83Ksv/3Dj3I5N2/ZkSfzakefnz70wmsSmLXqH5on+pdpqR69WR5rNeoG0tVzBzZRkFVxRAww2x2UoZcALiVqWhA5tCxC4J+hSQDXXQsrwao6P1sbGm1EQjTTsRi1N1vpLq9nc3k/c/9FP37bzTsdx0ixlBudOHzl04JmK39i8dXeep8y56lc/+Uj1yc94k2OmIOUwtYtXsy9/NT56yvBdA8srncWQXTUXYVcwgga4eDOQivaXWnATq5iUbg2lbyBoDz9q6J313oWzp2a3TG/dMnH2/Fm/Pvdrv/q597z/Y7phpmmSxBEeYNDvvnH8hTgJ53bthSrRe8ZCmUTxbzoHdRHDgA68XsUBKxDQRkg1IiEO/kTaelrgK7RM/IVosKalQ5sd+iycAGqk44ag0zEN18FhQbTWWY+uXl3ZvHlza2LLvQ+9d27/b/mtqTCK0SnBhEaZ1qV3Xl9cuDQ6MeN5Xs51y0jiJEtjaFOjCIKhGBGdobmOgdWimhIsS5UFtD60+JhOgYswMoQsuqsCgIx2k1elqaipY4kBebZZcXXQOo7tN0ZTb+vcPR/d6t7WG6QAJSLghiQiaXo86IE1yZKZiRlINU1z+frFs68fXJ4/v2cy0XVXoSn1TXiDaSSmdAAlEF4hEEPiM7qqJUDSY2YmpmakWY4Ry6RktrewltI3DczfdUwClecaWOv6enDqfDFz9z3TznbbKUZsK0n0fpAPgiwU3TLnb1xaWbpWq7dc16O7teX5tcV3PvLoY1E4qAZfL4qB2I2SrRouJ3zg8iouvIrPQimEUZErnyEPt4CvykRHyRW4esuzacRYDon7OFSxJ/ChADhI39GhMNEmP3zP79rNfaud0HW1mm/7vm1ZZCD0h/YXN+bfjsLe2Mw2IKVxtL5y5Y79+x9+/LOHnvuOvohzgkZDghMHUcATh0LnjSrxBJUTKXO2yBJkSim806ucxR4JXzxLhCs1SOUugMWr2ZBqa+KqHMNz9SxLB2nr8c/+4W17HgjCSCZO+EmwpxS4vm+iBmvd3tryVRwyvBI/1teut1r1Pe+5x/ereZaeOBEvXFz3wIoQcax60etrb17Q1oP2wmqfuBDTYZrzgZ5rPtruaY6oiWp9k10BXuqy4pX7+CxIdS3TJVNxUZK0EzoPfvzJ/fc9HEdRSTxqJGZuGJ5XQI/j6v2ri0Fv1a/WLNfN8QvR+ty22Y2z2wiuzP6iN3O8vrPwHMlfoAatNIJG5dIHvcHeCQ/CEDSEM4eXLnXPDiIbZw5nkkVIWig3RBUM8gc5iiGvotigRJMwUkvPuqG2bc/DDz76RBbH0pz4LpqihClaispIgOl3FopsUG+Mo4W9oOs41syW2Ypfy1S8d2cn6w9s11pVBgcrfMXX2yNZ8IFs9Vd2jyghy2VMi4zg+uWImKOyR5UIKE1QMaSARzmwfvUmgUB4BautR3Hm1Gcf/dTv2KaJJgBxeICaQ0IGc8NEsjxYdu3cqzfwROGgU6/7k9ObyvboH27HSBMjlsjKRZI3DFsX3sRuJAcT8oj2wiVai99FUXISXFEaU/EDzMyIlXUKr5JJig64eCvUpcjDzH3woU/PbNyaJJHcEnzqrKByIgA6tuO6ccUJm3XH8v2E/DMNmiMbGiPjknGopJ1YYNEvmi6iFIQSGjF5rnBwFlqV9SvLxk74qlIG8a44LeWARYgKhdIBZoEQbVtSliQuWlO33/fAY6jfLaDSNYcaIsvzGOJtLDcZHXEnJ0YCw0n7oWXmrdaI79eAypggkexWnhF0opsEMz5zCZZ5Q4cFrjQRazElmDFJ1g5K19Rgco9Qp8RS6gCXeRbp83ysVR64/+P15ojEz5v4pEFJreqBcXECg07btu2JDdNrfbfbbaPojdaIaVpERnlOVji6acmbkrp8BRj4FVbVtUxAeVjSZfFl3MQ0WMYRI5iaoAY3CiZTHtqW+FchlVy42tpy5/4PlqSqptJM/hVKnpJE2zFJSRcXFhbnr3jNyVartrhwuVpx6s0WvnBImcTzXB9E0IJJih5wI4iZZZDlnVDWh6Wp4h9CyOSruGwWaASFgvQQrKaZu5iWUI/DLW2LSQC2yJPc2rHzfSPjk2kSl6SWQAWq4hVxMkuiKylpd319ZbU9WZ30G0TiXDfdarXO9IczxFhPX/BefQ3wspCB21zDGvtB9JSWffOtNqsGXBLZCfq9FqS91Dh1LeGSePG8kNVOWmRJNDPVwI5Up6XPktgrXk0za7vuuLfksRzy3Wexk5JlIY/0fhCEKZGtbpnVipnpVsX3S6wwQYcP+v5DFaeJvDB55TDnw/Qn9vrohPPAhqr4VyaWSch85uyN7+n3J9uf0L0max8UDF1OoqBz8K9M85Kon5LqMHcRdSh0vzG1acuOoZ9St8EmpArJqJaIgyhgu5bGiiWLmF6sXECrUclNslkyFdoOHYFvGWOm07JsoRQiioKgUo3MlmfN1ByqA9JpVtiG1nQM26gWjUnDHUO8KmTpWhSYdsVhTNFeOYb+lXmTK05MztWbo9K1Okq0fBS46ookYg7GwUFCnRFmcDFREFarrlUBlXjIUglKr4W7vPmwqBy30CLOQjQdStMyoaYVOs01EpZyYkyZxBXzEBnSLU1RJtqxuGFQc3JqzrJthWp4EogKpuKXWUg7wYGqa6lKGomvdsWvVKo+CaGotaiKWBiwRBDDLyIXeYmHEMg3LUA6lkvyJy+aq2/DrwRUMSVuqFvyJnqu22OSg0qj4fG/n4YXUPwwTHECzN7QEvxUrVaR2JiReOJJBGt5CAABd2tkASfHrSvDccoH1GU1NN/LS9KQ8IFPHnYpa0NhVqAaOJ3RoQzVLG+OerOtij2ie6ziqR74BqtcX1KTrNdZa7gteJWkQ/5wF3AqVRmxMrnAY+I+cbi2wdIDkUv+hYWxbqe2wEx1y9UdT5IDmuZ4vIwuZHbclWOYuzAEKyjL86tiBD9/DKel3mAK7mRMwxyfmjKN0PEtsvc4iuM4AR/kpcyFSJknl/udF9LYN230RkJmXqxEydlB75pmkWSSnIiNyAKwOL7Y6xanU/1bpldP44wkETWPozBpXzWmUbYh2GF9QKYoM8Dkfh4rEEuUcl+0EHsKw0TLY6cyMrXZ0e1We2WZp8AjKk9BRtfOvXFw8dLr1p73XW9M4JqUKLgpvmYaq8yKC8AU+QCKa8X4Lv0xLeuszL/6ysENm2+vtybgLnGTE2a/34soTMEBT9/yWaKzjDaEpqgVkDIH8VZIDNqk2GHiWQvKLL2FS426Pr5lDgJcqjSS0kjKeePiievvHLrvQ4/ddd/jlo0X+/8eT/3T3/Qi8/HPfmF60w4MqdNZ+os/+tyNS0dTTRZtgFG8ok64NVYPqaTV7z5KW+EMXRWP9IJUTWOmcaK3O2EU9OqTA782Um00yGXo4ezJFxbePrT37kd33vUI0/p5Gb27Y/l8a6iV5cWn/+XLL7/44mOf/nxrbJqKjunYF8+9QegOo6TAMynNHPIKGkJs0O8qQx12Kpyq/jAP0sVqxSYwBv00UdXJdjdei1ZGZ5Zbo5Ou6wRZeubYs5fPHNl3z8d27HsYoP8X2rvAlbdYEF25eO6F57934NnvVurTTzz5x5u37SFvwEMlUfSz5/8TlSNxG6gUGomXWLmLx4w7ndVbc72FV9hXwxC4Cd/EbHISPAERK+xH/V7YGNHwyldePxDl9fse/Mzu/R8dPvsLb7KG6/dWlheuXDh37MiLb7/5xura+uTMzo/9+hd27n2/5bhpmmH6eN/vf+Pvz711bPfeexcuHhksD8qewKroE+1NVxav/qLU5Iq4IQ5RVgk5fCUl1Vg1gJ5ZZzguM1tyLXdt6cLRF//DIzEo9F4/CPphf9Drddc7a8urK0urq8u9bjfLdNcf2bxt/4Mf/4ONc7vrzQkaU2vCXBzHW19f/vbTXzr002c+9RufX1ldWb58WHyvHGWuTRop4SFbWrgYRQF1PLFYoVo1UWeUAd+eoKpizuIzSFVt03I9CjTR2NSWOz/4uUsXrl5552wQDlzPxpGtrQUR9RMWO07Nr8/M7b5rb3OiOTJVq4+7fs0yHeaZMt2c3M9iqb+6ev3oS99//vtPkcz+9u//6Z3v+/A//90XWYxJ3FJcyRpGOJPVRb66fLmzsjQ+NSOZGYe6BUrJUVItCDIUXRZtgtv0XIoItu/lKIBR3bYJf7u9CMKM+hDuhFwPhVHec9jT0HOJ78JdMeFcSsx53qb48darJw8/f/bNo65Xf+jRJ/bd+0u2V11dW1ycP2+TwKG2Uk26WX+FRqiO+8uXLp6enNlS5vYKrYyEL4yoUElEKXMOKrVEV+qGPVOLdWdbWjTJvOjBccRiw1A6EHCCWcQjCwDeSfuLPA7D9fby5fNvXDh3kvPq8oLr1bbffu9v/t6fTc/uYr3e6barGMDFc8H6jaZyeqUeqhhLB7laCxWDM6+/8t77HhGANw+GZGIyttIKhsQtoDPNVi1OSAoW29f+e3F+vjD8xti0ZXuF7iQpNdqAtJoENwwHfTx6t9NpL62t3GivLGDBlML9amty4233fOiTc9vvGp3cjDJQGFnrLKIUCM51/TOvHzGLLhESfRSKWLnwzovp08IysitvH1+8fmVyw2ZSXlFZhCXuTJIcnpGgwEVZ8+Tj03d4mzY6btFbu9bvzK+tXO8utyyTOnFg2NVOp3vt2o3lVVwFqYBrOnXPH5mcnt2x9wOt8U0jY9ONkSnLcoT5HJRtJJ5mlBbFjhyn0mu3z5896lmUupwsT2jG6EP/SrxOMqlZ9/sLJw7/5LFPPYlh3mRWJsOBDIENXFJB1vJ16m3eBPO2/F07Rx+MQtZJGRlsGW4lj9H1KMolHstiFb8xrKmI92DXQ9bDIV0TQQZBT+wAz6oGoueXD/xX0L4yUiWDkS07IbKMXTRgiRPH8Gy6ZnTy8A+Xb8xLxULZ1q0zzQBKRbJeNeuUBWydZVkah6QrttRxkiSFVOmExdAgZM+jIPay6QDZOG9CTSqvME0jkAkAgaCHYRhFAyVYIY99tCgIjr/8o4odwoOsO8rSnYgWHVAuE3dEVkEFrr924eBPvknapSYpHap/xavUxIlh7MwZJETRgGIcvBSsbLnITEoIfOBiGGe8sPVG1W7WbCr6koiJTqlD+pUKX7/fRlmHjGhaqzV++KUfri+dpWIJnoikSykAzcHKWfwLtZxQseLb6YlXvvvOmZOGI/p062AMSV/YaWQhiRQpEMtOFUU3yZfQH1Y4Mj56b4s7QsvDMO8PWD9mTJyCISUItQ12q0ut11sP40BZj5Baq43cmL/yyoFv1NyYmgFABSteTh0q4ZUcSVZ5Ucw9IamIrv/gm18ZdNfF5ktNUApAzYtKVlamw2p3hnSaxXGMULAJxVwUp1yxHanQYZOULHv9dL1LcCiASw9lGMJIB2G/P+iINxbBivRd2/3Bt/6hCOc9147TIkyKNFELXigo2AesNGCDujtLRlkhienIxsby4rVeP96z736ZklqUU0RyqAQVuYxH7FA1lSQpgohqIrVO2aBDXlEoO29MybJMqKI+gvvHHhgFr0xn3MXpxkm01l6KE2xfgJqmPT624dnvPvXmkWdGG7INisaTkYdJ3h0Amlaa6QrWguE9h5xWvBQ4bJvNwezi+TOGVdu+e7/Ka6U2g7PDpFlmIWmaAQIRR5iK4BDRE6mQGhMAEFegFj5lvU9YZheO9QpyyrU4zVbXrkdxKEFMgFoT45uOHPzxT7/3ldFabJpOEOZBDFCknVNgpQrBIbzyRq4MtaiMLFCGBkSMjt86fdL1R+e23yHkiPdTBVqEyxLANJKsYGsTKExa7dBiK6JIYmFlqUq1pBn4FLOF45DkZ+vd1W6/SxtaWqY9ObHp5JEXv/P0X7a8vuu4gzgbwGgEnVS9tUEsEZ7JgrWupIMHkDgr61v546AiDd3hqZNHTNvfumMfRsxq3WP3EDnGFIVAruOk8ASIFs8AkTCK3mPWokvskNEglRK7BCOokHNGjTlNCWokLoXr+ePjmw7/7EfffurPW26v4ntBlPdDfEheGg8SCGMtUT84oLTRQDo4UwSICig75g0Sh3AdMzx14tCg179t936/wu6shFz2iOGVFxIX7TQpMuP+hWlWOGgtD7MWR4OhmMb0BotR0o8Ga6aZsCcFmkp1vOK1fvTMvz737b8eq8aViofoYTTAG7JMQpdSIlQepSiSToQ3SW1QUoa3JGoIQNEEtIvOuY8qOw4VoGvnjyxdfWvL3M6xiQ0sNMFHLsuM8BuJ1AYk9SmnihcTy0abdKKlbEHCaxhHvd7aoL8G0bJB7lT8+vTywo2n//FPXnv561OjFiuzQYToxVIxgIiAi4hwv5kWZ7htYdF0PIp7yJPtJrDKv1CreEXp1FeUwZoac2v6dbe47Pq+4UzabgW9plofBPzOQsIg6QNPUs5gomxpQy2Mor4sw5Kk224v9QckImatRuVzynUqLz//79/46hc7i6fZ8kUwfYCK6DPUFKBRIls00g/7c9K/xAHTFl5BhxMQrCVQSBGgwjH2rrEqvG1z8yMPbB+pJm+/9tzlc8ebo2N+fTLOXbIIqcLIIbMkHFDEQMmIt0EYkDpFYcd1igqa6I3Um1PY55VzR8+f/Nq5Y1/TsoFhV9BONvewejRVRD8EmknPKHqBDFmMCLhhIAUT9Eh1RPZZuYFtmypjlqIIKQ6lYAzjjdOXmq1GEZ+7dvJvTx8eC/VtzZm7JzbuqtZHTEruLAIkjJLni4sypHLokZJ7ZKNG3Lv89qGDPz519Plg/Z27947duXv62BsLK71gEGb9EHsHJT6ESSIWAYqgZLtPORmhQWEVOsXJqAqwxjCSg0MwgsV95ZEsHSRqtdfDziC3K/xew/XMYKX9+pm3f3r24pcKZ3JseufG2Z2bNs7Obtk4MTWJ3SyvdLu4pc5K0LlSs1Y3TmYLl86cOXk6ilEff2E5uWtfa9NMcHUxxAGHsUhfWZIAJXxgDzg6ASq5tRKZYBWogrrMqWnEFoTQCjPiYwyP3/1QHPDddifIdWdhJUqj0NuzwXKrptWfnHD6Qbdz7dDyxQOnjGJqrHrnHXOMcfy1CzdW+mQwjZp51+7JPVPbalvclZWpMxf764MMZ7yy1t26ZfzM+dVrSxFJotgov+CQX5ugoyVQngaoYJMDXNiWsl+UUwK/cgMKPcahdkqoDGyc9G/fMbGy2tMN+613VtbWIxxts1ltd5P5ZRwlqy/X8/x6vdqoOfxEhh9GIFam6VaqFZ+9V61R9xr1yko7WGCnMciIi9UK6YE9PjFyfam/uBqhCfAqQCVvk7CnGBUmyTJLOpW+ivsTtkHK5gKCV7RK9i8Pi96I11xY6jdHmgQ9FoBhInkJ/bIOG4TMCh0lcuqUDVHTMNWYxlovgS1+4sGPdAjoCzfW6brZ9NZ6HeTeaLZOnLpG4Z4opVyp+GxxxIpU8EqchE2BJaRygFW+wKksp8TqpZEUd9SeDe+o/Hova/eSxbVwEGu3bZs69tolSi8EQ5vapMCVWEWyQneu50IDsJbbUW+QkyslhD629Ayj0+1R8JmbHcWD9np9KphBbBw7cXWxk+EH3g1UlgiCRSmrAqpg6/8DnlUhNsYFwKsAAAAASUVORK5CYII=";

// render/demos/sl-chrome.ts
var DEFAULT_HELP = [
  { title: "3D view", rows: [
    ["Left-drag", "Rotate"],
    ["Right-drag", "Zoom"],
    ["Middle / Shift+Left-drag", "Pan"],
    ["Wheel / two-finger", "Zoom (dolly)"],
    ["Double-click", "Maximize / restore"],
    ["Shift + move", "Pick \u2192 jump slices to the point"]
  ] },
  { title: "Endovascular flight (fly-inside / endo demo)", rows: [
    ["Up / Down", "Move in / out along the view axis"],
    ["Left / Right", "Yaw"],
    ["Shift + Left/Right", "Pitch"],
    ["Ctrl + Left/Right", "Roll"],
    ["Space", "Toggle forward cruise"],
    ["Shift + Space", "Toggle reverse cruise"],
    ["Escape", "Stop"],
    ["Left-drag", "Look around"],
    ["Shift + click", "Autopilot target"],
    ["Speed slider", "Travel speed in mm/s (live, applies mid-flight)"]
  ] },
  { title: "Slice views", rows: [
    ["Wheel / Left-drag", "Scroll through slices"],
    ["Right-drag / \u2318-wheel", "Zoom this slice"],
    ["Middle / Shift+Left-drag", "Pan"],
    ["Double-click", "Maximize / restore"],
    ["R", "Reset pan/zoom"],
    ["Shift + move", "Jump the other views to the point under the cursor"]
  ] }
];
function glass(el, extra = "") {
  el.style.cssText += ";background:linear-gradient(135deg,rgba(58,64,88,.55),rgba(20,24,38,.66));backdrop-filter:blur(20px) saturate(1.6);-webkit-backdrop-filter:blur(20px) saturate(1.6);border:1px solid rgba(255,255,255,.2);box-shadow:0 18px 50px rgba(0,0,0,.55);" + extra;
}
function installChrome(opts) {
  const controls = opts.controls ?? [];
  const host = opts.container ?? document.body;
  const help = (opts.help === false ? [] : opts.help) ?? DEFAULT_HELP;
  let helpBtn = null;
  if (opts.help !== false) {
    helpBtn = document.createElement("button");
    helpBtn.textContent = "?";
    helpBtn.title = "Controls & key bindings";
    helpBtn.style.cssText = "position:fixed;top:12px;left:12px;z-index:74;width:32px;height:32px;padding:0;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;border-radius:50%;color:#cfe6ff;font:700 15px -apple-system,system-ui,sans-serif;";
    glass(helpBtn);
    helpBtn.onclick = openHelp;
    host.appendChild(helpBtn);
  }
  let helpEl = null;
  function openHelp() {
    if (helpEl) return;
    helpEl = document.createElement("div");
    helpEl.style.cssText = "position:fixed;inset:0;z-index:96;display:flex;align-items:center;justify-content:center;background:rgba(6,8,14,.55);font:13px/1.5 -apple-system,system-ui,sans-serif;color:#e8eeff;";
    helpEl.addEventListener("mousedown", (e) => {
      if (e.target === helpEl) closeHelp();
    });
    const panel = document.createElement("div");
    panel.style.cssText = "max-width:min(640px,92vw);max-height:86vh;overflow-y:auto;padding:22px 26px;border-radius:16px;color:#eaf0ff;";
    glass(panel);
    panel.innerHTML = `<div style="font:800 20px -apple-system,system-ui,sans-serif;margin-bottom:4px">SlicerLive \u2014 controls</div>`;
    for (const sec of help) {
      const rows2 = sec.rows.map(([k, d]) => `<div style="font:600 12px ui-monospace,Menlo,monospace;color:#fff5d6;white-space:nowrap">${k}</div><div style="color:rgba(232,238,255,.85)">${d}</div>`).join("");
      panel.innerHTML += `<div style="margin-top:14px;padding:12px 14px;border-radius:10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)"><div style="font:700 11px -apple-system,system-ui,sans-serif;letter-spacing:1.1px;text-transform:uppercase;color:#9fe9ff;margin-bottom:9px">${sec.title}</div><div style="display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;align-items:baseline">${rows2}</div></div>`;
    }
    panel.innerHTML += `<div style="margin-top:16px;font-size:12px;color:rgba(232,238,255,.55)">Press <b style="color:#fff5d6">esc</b> or click outside to dismiss.</div>`;
    helpEl.appendChild(panel);
    host.appendChild(helpEl);
    document.addEventListener("keydown", escClose, true);
  }
  function escClose(e) {
    if (e.key === "Escape") closeHelp();
  }
  function closeHelp() {
    if (helpEl) {
      helpEl.remove();
      helpEl = null;
      document.removeEventListener("keydown", escClose, true);
    }
  }
  const logo = document.createElement("div");
  logo.id = "sl-badge";
  logo.title = "SlicerLive \u2014 visualization";
  logo.style.cssText = "position:fixed;z-index:74;cursor:pointer;user-select:none;display:flex;flex-direction:column;align-items:center;gap:4px;padding:7px 12px 6px;border-radius:14px;background:#121826;border:1px solid rgba(255,255,255,.12);box-shadow:0 10px 30px rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.06);transition:transform 120ms ease-out;";
  const mark = document.createElement("img");
  mark.src = SL_LOGO;
  mark.alt = "SlicerLive";
  mark.style.cssText = "height:40px;width:auto;display:block;filter:drop-shadow(0 0 5px rgba(255,200,80,.5));";
  const word = document.createElement("div");
  word.innerHTML = 'Slicer<b style="color:#ffd34d">Live</b>';
  word.style.cssText = "font:800 12px/1 -apple-system,system-ui,sans-serif;letter-spacing:.5px;color:#eef7ff;text-shadow:0 0 14px rgba(255,210,90,.4);";
  logo.appendChild(mark);
  logo.appendChild(word);
  host.appendChild(logo);
  const place = () => {
    const a = opts.anchor;
    const r = a && a.getClientRects().length ? a.getBoundingClientRect() : null;
    if (r && r.width > 2 && r.height > 2) {
      logo.style.top = Math.round(r.top + 8) + "px";
      logo.style.right = Math.round(globalThis.innerWidth - r.right + 8) + "px";
    } else {
      logo.style.top = "10px";
      logo.style.right = "12px";
    }
  };
  place();
  requestAnimationFrame(place);
  globalThis.addEventListener("resize", place);
  const anchorRO = opts.anchor && "ResizeObserver" in globalThis ? new ResizeObserver(place) : null;
  anchorRO?.observe(opts.anchor);
  const pop = document.createElement("div");
  pop.id = "sl-popup";
  pop.style.cssText = "position:fixed;z-index:73;min-width:210px;max-width:300px;max-height:84vh;overflow-y:auto;padding:10px 12px;border-radius:12px;color:#eaf0ff;font:13px -apple-system,system-ui,sans-serif;opacity:0;pointer-events:none;transform:translateY(-6px);transition:opacity 120ms ease-out,transform 120ms ease-out;";
  glass(pop);
  host.appendChild(pop);
  const paintSw = (sw, on) => {
    sw.style.background = on ? "linear-gradient(180deg,#9fe9ff,#54c6f0)" : "rgba(255,255,255,.18)";
    sw.innerHTML = `<span style="position:absolute;top:2px;left:${on ? 17 : 2}px;width:15px;height:15px;border-radius:50%;background:#fff;transition:left 120ms;box-shadow:0 1px 3px rgba(0,0,0,.4)"></span>`;
  };
  const afterPaint = (fn) => requestAnimationFrame(() => requestAnimationFrame(fn));
  const paintTri = (box, level, color) => {
    const pct = Math.round(level * 100);
    const c = `rgb(${Math.round(color[0] * 255)},${Math.round(color[1] * 255)},${Math.round(color[2] * 255)})`;
    box.style.opacity = level < 0.02 ? "0.75" : "1";
    box.innerHTML = `<span style="position:absolute;left:0;top:0;bottom:0;width:${pct}%;background:${c};opacity:.9"></span><span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font:700 10px -apple-system,system-ui,sans-serif;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.75)">${pct}%</span>`;
  };
  const triNext = (v) => v > 0.66 ? 0.5 : v > 0.04 ? 0 : 1;
  const attachOpacity = (box, get, set, color, onChange) => {
    box.style.cursor = "ew-resize";
    box.title = "Click: 100% \u2192 50% \u2192 off \xB7 Drag sideways for a live opacity slider";
    const paint = () => paintTri(box, get(), color);
    paint();
    let startX = 0, startV = 0, dragged = false, id = -1;
    box.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      startX = e.clientX;
      startV = get();
      dragged = false;
      id = e.pointerId;
      try {
        box.setPointerCapture(id);
      } catch {
      }
    });
    box.addEventListener("pointermove", (e) => {
      if (id < 0) return;
      const dx = e.clientX - startX;
      if (Math.abs(dx) > 3) dragged = true;
      if (dragged) {
        set(Math.max(0, Math.min(1, startV + dx / 130)));
        paint();
        onChange();
      }
    });
    const end = () => {
      if (id < 0) return;
      if (!dragged) {
        set(triNext(get()));
        paint();
        onChange();
      }
      try {
        box.releasePointerCapture(id);
      } catch {
      }
      id = -1;
    };
    box.addEventListener("pointerup", end);
    box.addEventListener("pointercancel", end);
    return paint;
  };
  const OPBOX_CSS = "width:44px;height:18px;border-radius:6px;position:relative;overflow:hidden;flex:0 0 auto;background:rgba(255,255,255,.14);box-shadow:inset 0 0 0 1px rgba(255,255,255,.18);touch-action:none;";
  const heading = (text, first) => {
    const h = document.createElement("div");
    h.textContent = text;
    h.style.cssText = "font:700 10px -apple-system,system-ui,sans-serif;letter-spacing:1.1px;text-transform:uppercase;color:#9fe9ff;margin:" + (first ? "0 0 8px" : "12px 0 6px") + ";" + (first ? "" : "border-top:1px solid rgba(255,255,255,.12);padding-top:10px;");
    pop.appendChild(h);
  };
  const selects = opts.selects ?? [];
  const selEls = [];
  let sectionSeen = null;
  let firstHead = true;
  for (const c of selects) {
    const sec = c.section ?? "Visualization";
    if (sec !== sectionSeen) {
      heading(sec, firstHead);
      sectionSeen = sec;
      firstHead = false;
    }
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:5px 0;";
    const lab = document.createElement("span");
    lab.textContent = c.label;
    const sel = document.createElement("select");
    sel.style.cssText = "flex:1 1 auto;max-width:60%;border-radius:7px;padding:4px 6px;cursor:pointer;font:500 12px -apple-system,system-ui,sans-serif;color:#e8eeff;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.20);";
    for (const o of c.options) {
      const op = document.createElement("option");
      op.value = o.value;
      op.textContent = o.label;
      op.style.cssText = "background:#1b2030;color:#e8eeff;";
      sel.appendChild(op);
    }
    sel.value = c.get();
    sel.onclick = (e) => e.stopPropagation();
    sel.onchange = () => {
      c.set(sel.value);
      opts.onChange?.();
      refresh();
    };
    row.appendChild(lab);
    row.appendChild(sel);
    pop.appendChild(row);
    selEls.push({ c, el: sel });
  }
  const rows = [];
  if (controls.length) {
    for (const c of controls) {
      const sec = c.section ?? "Visualization";
      if (sec !== sectionSeen) {
        heading(sec, firstHead);
        sectionSeen = sec;
        firstHead = false;
      }
      const row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:14px;padding:5px 0;";
      if (c.slider) {
        row.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:6px 0;";
        const top = document.createElement("div");
        top.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
        const lab2 = document.createElement("span");
        lab2.textContent = c.label;
        const val = document.createElement("span");
        val.style.cssText = "font:600 11px ui-monospace,Menlo,monospace;color:#9fe9ff;font-variant-numeric:tabular-nums;";
        top.appendChild(lab2);
        top.appendChild(val);
        const inp = document.createElement("input");
        inp.type = "range";
        inp.min = String(c.slider.min);
        inp.max = String(c.slider.max);
        inp.step = String(c.slider.step ?? 1);
        inp.value = String(c.slider.get());
        inp.style.cssText = "width:100%;accent-color:#54c6f0;cursor:pointer;";
        const fmt = c.slider.format ?? ((v) => String(Math.round(v)));
        const paint = () => {
          val.textContent = fmt(c.slider.get());
        };
        inp.oninput = () => {
          c.slider.set(parseFloat(inp.value));
          paint();
          opts.onChange?.();
        };
        inp.onpointerdown = (e) => e.stopPropagation();
        paint();
        row.appendChild(top);
        row.appendChild(inp);
        pop.appendChild(row);
        rows.push({ c, row, repaint: () => {
          inp.value = String(c.slider.get());
          paint();
        } });
        continue;
      }
      const lab = document.createElement("span");
      lab.textContent = c.label;
      row.appendChild(lab);
      if (c.button) {
        const pill = document.createElement("span");
        pill.style.cssText = "max-width:60%;border-radius:7px;padding:4px 10px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font:600 12px -apple-system,system-ui,sans-serif;color:#eaf0ff;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.20);";
        pill.textContent = c.button.text();
        pill.onclick = (e) => {
          e.stopPropagation();
          c.button.run();
        };
        pill.onpointerdown = (e) => e.stopPropagation();
        row.appendChild(pill);
        pop.appendChild(row);
        rows.push({ c, row, repaint: () => {
          pill.textContent = c.button.text();
        } });
        continue;
      }
      if (c.getOpacity && c.setOpacity) {
        const box = document.createElement("span");
        box.style.cssText = OPBOX_CSS;
        row.appendChild(box);
        const paint = attachOpacity(box, c.getOpacity, (o) => c.setOpacity(o), c.color ?? [0.62, 0.9, 1], () => opts.onChange?.());
        rows.push({ c, row, repaint: paint });
      } else {
        row.style.cursor = "pointer";
        const sw = document.createElement("span");
        sw.style.cssText = "width:34px;height:19px;border-radius:999px;position:relative;transition:background 120ms;flex:0 0 auto;";
        row.appendChild(sw);
        row.onclick = () => {
          if (c.disabled?.()) return;
          const next = !c.get();
          paintSw(sw, next);
          afterPaint(() => {
            c.set(next);
            opts.onChange?.();
            refresh();
          });
        };
        rows.push({ c, row, sw });
      }
      pop.appendChild(row);
    }
  } else if (opts.about === false && !opts.segments && !selects.length) {
    pop.textContent = "SlicerLive \u2014 WebGPU renderer";
  }
  const segHost = document.createElement("div");
  pop.appendChild(segHost);
  const segRows = [];
  function buildSegments() {
    const S = opts.segments;
    segRows.length = 0;
    segHost.innerHTML = "";
    if (!S) return;
    const list = S.list();
    if (!list.length) return;
    const wrap = document.createElement("div");
    wrap.style.cssText = "margin-top:6px;border-top:1px solid rgba(255,255,255,.12);padding-top:6px;" + (list.length > 6 ? "max-height:210px;overflow-y:auto;" : "");
    for (const s of list) {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:4px 2px;";
      const left = document.createElement("span");
      left.style.cssText = "display:flex;align-items:center;gap:8px;min-width:0;";
      const swatch = document.createElement("span");
      swatch.style.cssText = `flex:0 0 auto;width:11px;height:11px;border-radius:3px;box-shadow:0 0 0 1px rgba(255,255,255,.25);background:rgb(${Math.round(s.color[0] * 255)},${Math.round(s.color[1] * 255)},${Math.round(s.color[2] * 255)})`;
      const lab = document.createElement("span");
      lab.textContent = s.name;
      lab.style.cssText = "font:500 12.5px -apple-system,system-ui,sans-serif;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
      left.appendChild(swatch);
      left.appendChild(lab);
      const box = document.createElement("span");
      box.style.cssText = OPBOX_CSS;
      row.appendChild(left);
      row.appendChild(box);
      const paint = attachOpacity(box, () => S.get(s.num), (o) => {
        if (!(S.enabled && !S.enabled())) S.set(s.num, o);
      }, s.color, () => opts.onChange?.());
      wrap.appendChild(row);
      segRows.push({ num: s.num, box, color: s.color, paint });
    }
    segHost.appendChild(wrap);
    paintSegments();
  }
  function paintSegments() {
    const S = opts.segments;
    if (!S) return;
    const dis = S.enabled ? !S.enabled() : false;
    segHost.style.opacity = dis ? "0.4" : "1";
    for (const r of segRows) r.paint();
  }
  if (opts.about !== false) {
    const about = document.createElement("div");
    const aLabel = opts.about?.label ?? "About SlicerLive";
    const aURL = opts.about?.url ?? "https://github.com/pieper/SlicerLive";
    about.textContent = aLabel;
    about.style.cssText = "cursor:pointer;border-radius:9px;padding:9px 8px 3px;margin-top:4px;" + (controls.length || opts.segments ? "border-top:1px solid rgba(255,255,255,.12);" : "") + "font:600 13px -apple-system,system-ui,sans-serif;color:#9fe9ff;";
    about.onmouseenter = () => {
      about.style.background = "rgba(255,255,255,.07)";
    };
    about.onmouseleave = () => {
      about.style.background = "transparent";
    };
    about.onclick = (e) => {
      e.stopPropagation();
      globalThis.open(aURL, "_blank", "noopener");
    };
    pop.appendChild(about);
  }
  function refresh() {
    for (const { c, el } of selEls) {
      const v = c.get();
      if (el.value !== v) el.value = v;
    }
    for (const { c, row, sw, repaint } of rows) {
      const dis = c.disabled?.() ?? false;
      row.style.opacity = dis ? "0.4" : "1";
      if (repaint) {
        repaint();
        continue;
      }
      const on = c.get();
      row.style.cursor = dis ? "default" : "pointer";
      sw.style.background = on ? "linear-gradient(180deg,#9fe9ff,#54c6f0)" : "rgba(255,255,255,.18)";
      sw.innerHTML = `<span style="position:absolute;top:2px;left:${on ? 17 : 2}px;width:15px;height:15px;border-radius:50%;background:#fff;transition:left 120ms;box-shadow:0 1px 3px rgba(0,0,0,.4)"></span>`;
    }
    paintSegments();
  }
  refresh();
  const show = () => {
    buildSegments();
    refresh();
    const b = logo.getBoundingClientRect();
    pop.style.top = Math.round(b.bottom + 6) + "px";
    pop.style.right = Math.round(globalThis.innerWidth - b.right) + "px";
    pop.style.opacity = "1";
    pop.style.pointerEvents = "auto";
    pop.style.transform = "translateY(0)";
  };
  const hide = () => {
    pop.style.opacity = "0";
    pop.style.pointerEvents = "none";
    pop.style.transform = "translateY(-6px)";
  };
  let pinned = false;
  logo.onmouseenter = () => {
    logo.style.transform = "scale(1.08)";
    show();
  };
  logo.onclick = () => {
    pinned = !pinned;
    pinned ? show() : hide();
  };
  logo.onmouseleave = () => {
    logo.style.transform = "scale(1)";
    if (!pinned) setTimeout(() => {
      if (!pop.matches(":hover") && !pinned) hide();
    }, 120);
  };
  pop.onmouseleave = () => {
    if (!pinned) hide();
  };
  const destroy = () => {
    globalThis.removeEventListener("resize", place);
    anchorRO?.disconnect();
    document.removeEventListener("keydown", escClose, true);
    helpBtn?.remove();
    helpEl?.remove();
    logo.remove();
    pop.remove();
  };
  return { refresh, destroy };
}

// render/vendor/idc_tools/s3.js
var idcS3 = (bucket) => "https://" + (bucket || "idc-open-data") + ".s3.us-east-1.amazonaws.com/";
async function fetchRetry(url, opts, tries = 6) {
  let err;
  for (let i = 0; i < tries; i++) {
    const ac = new AbortController();
    const to = setTimeout(() => ac.abort(), 2e4);
    try {
      const r = await fetch(url, { ...opts || {}, signal: ac.signal });
      if (!r.ok && r.status !== 206)
        throw new Error("HTTP " + r.status);
      return r;
    } catch (e) {
      err = e;
      if (i < tries - 1)
        await new Promise((res) => setTimeout(res, Math.min(4e3, 250 * 2 ** i) * (0.6 + 0.8 * Math.random())));
    } finally {
      clearTimeout(to);
    }
  }
  throw err;
}
async function s3ListKeys(prefix, bucket) {
  if (!prefix.endsWith("/"))
    prefix += "/";
  const base = idcS3(bucket);
  const keys = [];
  let token = null, more = true;
  while (more) {
    let url = `${base}?list-type=2&prefix=${encodeURIComponent(prefix)}`;
    if (token)
      url += `&continuation-token=${encodeURIComponent(token)}`;
    const xml = new DOMParser().parseFromString(await fetchRetry(url).then((r) => r.text()), "application/xml");
    for (const e of Array.from(xml.getElementsByTagName("Key"))) {
      const k = e.textContent;
      if (k && /\.dcm$/i.test(k))
        keys.push(k);
    }
    more = xml.getElementsByTagName("IsTruncated")[0]?.textContent === "true";
    token = more ? xml.getElementsByTagName("NextContinuationToken")[0]?.textContent ?? null : null;
  }
  return keys;
}
function ohifViewerURL(studyInstanceUID) {
  return studyInstanceUID ? `https://viewer.imaging.datacommons.cancer.gov/viewer/${studyInstanceUID}` : null;
}

// render/vendor/idc_tools/loader.js
var _worker = null;
function resolveWorkerURL(opts) {
  if (opts?.workerUrl)
    return opts.workerUrl;
  return new URL("./idc-worker.js", import.meta.url);
}
function runWorker(ctKeys, segKeys, ctBucket, segBucket, modality, handlers, opts) {
  if (_worker) {
    try {
      _worker.terminate();
    } catch {
    }
    _worker = null;
  }
  return new Promise((resolve, reject) => {
    const w = new Worker(resolveWorkerURL(opts));
    _worker = w;
    let ct, seg;
    let chain = Promise.resolve();
    w.onmessage = (e) => {
      const m = e.data;
      switch (m.t) {
        case "ctinfo":
          handlers.onSliceCount?.(m.count);
          break;
        case "thumb":
          handlers.onThumb?.(m.n, m.w, m.h, m.rgba);
          break;
        case "seg":
          handlers.onSegName?.(m.name);
          break;
        case "progress":
          handlers.onProgress?.({ frac: m.frac, msg: m.msg });
          break;
        case "ct": {
          const Ctor = m.dtype === "float32" ? Float32Array : Int16Array;
          ct = { vol: new Ctor(m.vol), dims: m.dims, ijkToRAS: m.ijkToRAS, win: m.win, lev: m.lev, dtype: m.dtype, modality };
          chain = chain.then(() => handlers.onCT?.(ct)).catch((err) => console.error("[idc_tools] onCT", err));
          break;
        }
        case "labelmap": {
          seg = { lab: new Uint8Array(m.lab), colors: m.colors, names: m.names };
          chain = chain.then(() => handlers.onLabelmap?.(seg)).catch((err) => console.error("[idc_tools] onLabelmap", err));
          break;
        }
        case "error":
          w.terminate();
          if (_worker === w)
            _worker = null;
          reject(new Error(m.error));
          break;
        case "alldone":
          w.terminate();
          if (_worker === w)
            _worker = null;
          chain.then(() => resolve({ ct, seg }));
          break;
      }
    };
    w.onerror = (e) => {
      w.terminate();
      if (_worker === w)
        _worker = null;
      reject(new Error("idc_tools worker: " + (e.message || e)));
    };
    w.postMessage({ ctKeys, segKeys, ctBucket, segBucket, modality });
  });
}
async function loadSeries(entry, handlers = {}, opts) {
  const modality = { CT: "CT", MR: "MR", PT: "PET" }[entry.m] || entry.m;
  const ctKeys = await s3ListKeys(entry.c, entry.cb);
  if (!ctKeys.length)
    throw new Error("idc_tools: no DICOM under CT prefix " + entry.c);
  const segKeys = entry.s ? await s3ListKeys(entry.s, entry.sb) : [];
  const r = await runWorker(ctKeys, segKeys, entry.cb, entry.sb, modality, handlers, opts);
  r.entry = entry;
  return r;
}

// render/demos/bir.ts
var PLANE_COLOR = {
  axial: "#f05a5a",
  sagittal: "#f0d24a",
  coronal: "#5ad07a"
};
function displayClass() {
  const sw = globalThis.screen?.width ?? globalThis.innerWidth;
  const sh = globalThis.screen?.height ?? globalThis.innerHeight;
  const devicePx = sw * (globalThis.devicePixelRatio || 1);
  if (Math.min(sw, sh) < 700) return "phone";
  if (devicePx >= 3400) return "large";
  return "desktop";
}
var I = (body) => `<svg viewBox="0 0 24 24" width="25" height="25" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
var ICONS = {
  // Action tools
  patient: I(`<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 5-6 8-6s6.5 2 8 6"/>`),
  prevStudy: I(`<path d="M3 7h6l2-2h10v14H3z"/><path d="M14 11l-4 3 4 3" fill="currentColor"/>`),
  nextStudy: I(`<path d="M3 7h6l2-2h10v14H3z"/><path d="M11 11l4 3-4 3" fill="currentColor"/>`),
  prevFS: I(`<path d="M12 6l-7 6 7 6M19 6l-7 6 7 6"/>`),
  // «
  nextFS: I(`<path d="M5 6l7 6-7 6M12 6l7 6-7 6"/>`),
  // »
  prevFrame: I(`<path d="M15 5l-8 7 8 7"/>`),
  // ‹
  nextFrame: I(`<path d="M9 5l8 7-8 7"/>`),
  // ›
  layoutIn: I(
    `<rect x="3" y="4" width="14" height="14"/><path d="M10 4v14M3 11h14"/><path d="M19 15l2 3h-4z" fill="currentColor"/>`
  ),
  layoutMulti: I(
    `<rect x="2" y="3" width="8" height="7"/><rect x="12" y="3" width="8" height="7"/><rect x="2" y="12" width="8" height="7"/><path d="M18 16l2 3h-4z" fill="currentColor"/>`
  ),
  invert: I(`<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 010 16z" fill="currentColor"/>`),
  // IEC 5411
  localizer: I(
    `<rect x="3" y="4" width="18" height="16"/><circle cx="12" cy="10" r="2.5"/><path d="M8 14c1-1.5 7-1.5 8 0"/><path d="M3 12h3M8 12h3M13 12h3M18 12h3" stroke-dasharray="2 2"/>`
  ),
  link: I(
    `<rect x="2" y="9" width="6" height="6" rx="3"/><rect x="9" y="9" width="6" height="6" rx="3"/><rect x="16" y="9" width="6" height="6" rx="3"/>`
  ),
  wlMode: I(
    `<circle cx="7" cy="10" r="4.5"/><path d="M7 5.5a4.5 4.5 0 010 9z" fill="currentColor"/><circle cx="17" cy="10" r="1.2" fill="currentColor"/><ellipse cx="17" cy="10" rx="4" ry="1.6"/><path d="M3 19h18"/>`
  ),
  // IEC 5435 + Bohr atom + line
  annotation: I(
    `<path d="M3 4h6M3 7h4M15 4h6M17 7h4M3 17h4M3 20h6M17 17h4M15 20h6"/>`
  ),
  print: I(
    `<rect x="6" y="3" width="12" height="5"/><rect x="3" y="8" width="18" height="8" rx="1"/><rect x="6" y="14" width="12" height="7"/>`
  ),
  reset: I(`<path d="M4 9V4h5"/><path d="M4 4l6 6"/><rect x="8" y="10" width="12" height="10"/>`),
  // IEC 5495
  report: I(`<path d="M4 4h16M4 7h16M4 10h10M4 15h16M4 18h12"/>`),
  rotate: I(`<path d="M19 12a7 7 0 11-2-4.9"/><path d="M17 3v4h4" fill="currentColor"/>`),
  // IEC 5772 one-way
  flip: I(`<path d="M12 3v18" stroke-dasharray="3 2"/><path d="M9 7L4 12l5 5zM15 7l5 5-5 5z" fill="currentColor"/>`),
  // IEC 5408
  cine: I(
    `<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 8h18M6 5v3M10 5v3M14 5v3M18 5v3"/><path d="M10 11l5 3.5-5 3.5z" fill="currentColor"/>`
  ),
  // IEC 1123
  presets: I(
    `<circle cx="10" cy="10" r="5.5"/><path d="M10 4.5a5.5 5.5 0 010 11z" fill="currentColor"/><path d="M17 15l2.5 4h-5z" fill="currentColor"/>`
  ),
  // IEC 5435 + dropdown
  help: I(`<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 114 2c-.9.7-1.5 1.2-1.5 2.3"/><circle cx="12" cy="17" r=".6" fill="currentColor"/>`),
  // IEC 5289
  advanced: I(`<path d="M2 9l10-4 10 4-10 4z" fill="currentColor"/><path d="M6 11v5c2 2.5 10 2.5 12 0v-5"/>`),
  // mortarboard
  // Modal tools
  zoom: I(`<circle cx="10" cy="10" r="6"/><path d="M15 15l6 6"/>`),
  pan: I(
    `<path d="M8 11V5.5a1.4 1.4 0 012.8 0V10m0-5.5V4a1.4 1.4 0 012.8 0v6m0-4.5a1.4 1.4 0 012.8 0V12m0-3a1.4 1.4 0 012.8 0v5c0 4-2.5 7-6.5 7S8.5 19 7 16l-2.4-4.2a1.3 1.3 0 012.2-1.3L8 12.5z"/>`
  ),
  window: I(`<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 010 16z" fill="currentColor"/><path d="M12 4v16" stroke-width="1"/>`),
  // IEC 5435
  scroll: I(
    `<rect x="7" y="3" width="12" height="9"/><rect x="5" y="7" width="12" height="9" fill="#05060a"/><rect x="3" y="11" width="12" height="9" fill="#05060a"/>`
  ),
  // stack of frames
  selectvp: I(`<path d="M5 3l7 16 2.2-6.2L20 10.5z" fill="currentColor"/>`),
  // up-left arrow
  distance: I(`<path d="M4 20L20 4"/><path d="M6 14l2 2M9 11l2 2M12 8l2 2M15 5l2 2"/>`),
  // IEC 5658 ruler
  angle: I(`<path d="M4 20h16L4 6z"/><path d="M10 20a8 8 0 00-2.5-5.5"/><path d="M9 12.5l-1.5 2M10 20l-2.4-.6" stroke-width="1.2"/>`),
  crosshair: I(
    `<circle cx="12" cy="12" r="8"/><path d="M12 4v5M12 15v5M4 12h5M15 12h5" stroke-dasharray="3 2"/>`
  ),
  // Cine controls
  play: I(`<path d="M7 5l12 7-12 7z" fill="currentColor"/>`),
  stop: I(`<rect x="6" y="6" width="12" height="12" fill="currentColor"/>`),
  pause: I(`<rect x="6" y="5" width="4" height="14" fill="currentColor"/><rect x="14" y="5" width="4" height="14" fill="currentColor"/>`),
  toStart: I(`<path d="M6 5v14M19 5l-10 7 10 7z" fill="currentColor"/>`),
  stepFwd: I(`<path d="M5 5l10 7-10 7z" fill="currentColor"/><rect x="17" y="5" width="2.5" height="14" fill="currentColor"/>`),
  // IEC 5471
  toEnd: I(`<path d="M18 5v14M5 5l10 7-10 7z" fill="currentColor"/>`),
  thumbs: I(`<rect x="3" y="3" width="7" height="7"/><rect x="3" y="13" width="7" height="7"/><path d="M13 5h8M13 9h6M13 15h8M13 19h6"/>`),
  // IDC actions
  share: I(`<circle cx="18" cy="5" r="2.6"/><circle cx="6" cy="12" r="2.6"/><circle cx="18" cy="19" r="2.6"/><path d="M8.3 10.8l7.4-4.3M8.3 13.2l7.4 4.3"/>`),
  download: I(`<path d="M12 3v11m0 0l-4-4m4 4l4-4"/><path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"/>`)
};
var STYLE = `
#bir-toolbar { display:flex; align-items:center; gap:2px; padding:3px 8px; background:#0b0e16;
  border-bottom:1px solid #1b2740; flex-wrap:wrap; }
#bir-toolbar .sep { width:1px; height:28px; background:#1b2740; margin:0 6px; }
#bir-toolbar button { display:inline-flex; align-items:center; justify-content:center;
  width:38px; height:36px; color:#9fb3d0; background:none; border:1px solid transparent;
  border-radius:5px; cursor:pointer; padding:0; }
#bir-toolbar button:hover { background:#1b2740; color:#d6e2f2; }
#bir-toolbar button.active { background:#2b6cb0; color:#fff; border-color:#33507e; }
#bir-toolbar button:disabled { opacity:.3; cursor:default; background:none; }
#bir-cine-box { display:flex; align-items:center; gap:2px; margin-left:4px; padding:2px 6px;
  background:#11141d; border:1px solid #33507e; border-radius:5px; }
#bir-cine-box input { width:44px; font:600 11px -apple-system,system-ui,sans-serif; color:#d6e2f2;
  background:#0b0e16; border:1px solid #33507e; border-radius:3px; padding:1px 4px; }
#bir-strip { width:158px; overflow-y:auto; background:#0b0e16; border-right:1px solid #1b2740;
  flex:0 0 auto; }
#bir-strip .head { font:700 10px -apple-system,system-ui,sans-serif; letter-spacing:.6px;
  text-transform:uppercase; color:#5a6b85; padding:6px 8px 4px; }
#bir-strip .item { margin:0 6px 8px; border:1px solid #1b2740; border-radius:5px; cursor:pointer;
  overflow:hidden; background:#05060a; }
#bir-strip .item:hover { border-color:#33507e; }
#bir-strip .item.current { border-color:#2b6cb0; box-shadow:0 0 0 1px #2b6cb0; }
#bir-strip img { width:100%; aspect-ratio:1; object-fit:contain; background:#000; display:block; }
#bir-strip .cap { font:10px -apple-system,system-ui,sans-serif; color:#9fb3d0; padding:3px 6px;
  line-height:1.35; }
.bir-cell-selected { outline:1px solid #2b6cb0; outline-offset:-1px; }
.srv-grid.ann-none .srv-orient, .srv-grid.ann-none .srv-slice-readout,
.srv-grid.ann-min .srv-slice-readout { display:none; }
`;
function mountBir(cfg) {
  if (!document.getElementById("bir-style")) {
    const st = document.createElement("style");
    st.id = "bir-style";
    st.textContent = STYLE;
    document.head.appendChild(st);
  }
  let tool = "scroll";
  let selectedPlane = cfg.planes[0];
  let localizerOn = false;
  let inverted = false;
  let clamped = /^(NM|PT)$/i.test(cfg.modality);
  let annState = 0;
  let crossRas = null;
  const measurements = [];
  let pending = null;
  let layout = "fourUp";
  const overlays = /* @__PURE__ */ new Map();
  for (const p of cfg.planes) {
    const cell = cfg.cellOf(p);
    if (!cell) continue;
    const c = document.createElement("canvas");
    c.className = "bir-overlay";
    c.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;";
    cell.appendChild(c);
    overlays.set(p, c);
  }
  const resizeOverlays = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const [, c] of overlays) {
      c.width = Math.max(1, Math.floor(c.clientWidth * dpr));
      c.height = Math.max(1, Math.floor(c.clientHeight * dpr));
    }
  };
  const rasOfClick = (p, u, v, w, h) => cfg.slice().viewToRas(p, cfg.off01(p), u, v, w / h);
  const toPx = (p, ras, w, h) => {
    const r = cfg.slice().rasToView(p, cfg.off01(p), ras, w / h);
    return { x: r.u * w, y: r.v * h };
  };
  const drawOverlay = (p) => {
    const c = overlays.get(p);
    if (!c) return;
    const ctx = c.getContext("2d");
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = c.width / dpr, h = c.height / dpr;
    ctx.clearRect(0, 0, w, h);
    if (!w || !h) return;
    if (localizerOn) {
      const nAxis = { sagittal: 0, coronal: 1, axial: 2 };
      for (const q of cfg.planes) {
        if (q === p) continue;
        const free = [0, 1, 2].find((a2) => a2 !== nAxis[p] && a2 !== nAxis[q]);
        const mk = (t) => {
          const r = [0, 0, 0];
          r[nAxis[p]] = cfg.offsetMm(p);
          r[nAxis[q]] = cfg.offsetMm(q);
          r[free] = t;
          return r;
        };
        const a = toPx(p, mk(cfg.rasLo[free]), w, h);
        const b = toPx(p, mk(cfg.rasHi[free]), w, h);
        ctx.strokeStyle = PLANE_COLOR[q];
        ctx.globalAlpha = 0.75;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
    }
    if (crossRas) {
      const pt = toPx(p, crossRas, w, h);
      if (pt.x >= 0 && pt.x <= w && pt.y >= 0 && pt.y <= h) {
        drawCross(ctx, pt.x, pt.y, { color: "#6cc4ff", size: 14, gap: 5 });
      }
    }
    const near = (m) => m.plane === p && Math.abs(m.mm - cfg.offsetMm(p)) <= cfg.spacing(p) / 2;
    ctx.font = "700 15px -apple-system,system-ui,sans-serif";
    ctx.lineJoin = "round";
    ctx.textBaseline = "alphabetic";
    const label = (text, x, y) => {
      ctx.lineWidth = 4;
      ctx.strokeStyle = "rgba(6,10,16,.92)";
      ctx.strokeText(text, x, y);
      ctx.fillStyle = "#8dffc4";
      ctx.fillText(text, x, y);
    };
    for (const m of [...measurements.filter(near), ...pending && pending.plane === p ? [pending] : []]) {
      const px = m.pts.map((r) => toPx(p, r, w, h));
      ctx.strokeStyle = "#6ce0a8";
      ctx.fillStyle = "#6ce0a8";
      ctx.lineWidth = 2;
      for (let i = 0; i + 1 < px.length; i += 2) {
        ctx.beginPath();
        ctx.moveTo(px[i].x, px[i].y);
        ctx.lineTo(px[i + 1].x, px[i + 1].y);
        ctx.stroke();
      }
      for (const q of px) {
        ctx.beginPath();
        ctx.arc(q.x, q.y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (m.kind === "distance" && m.pts.length === 2) {
        const d = Math.hypot(
          m.pts[0][0] - m.pts[1][0],
          m.pts[0][1] - m.pts[1][1],
          m.pts[0][2] - m.pts[1][2]
        );
        label(`${d.toFixed(1)} mm`, (px[0].x + px[1].x) / 2 + 8, (px[0].y + px[1].y) / 2 - 8);
      } else if (m.kind === "angle" && m.pts.length === 4) {
        const v1 = [m.pts[1][0] - m.pts[0][0], m.pts[1][1] - m.pts[0][1], m.pts[1][2] - m.pts[0][2]];
        const v2 = [m.pts[3][0] - m.pts[2][0], m.pts[3][1] - m.pts[2][1], m.pts[3][2] - m.pts[2][2]];
        const dot3 = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        const deg = Math.acos(
          Math.min(1, Math.abs(dot3) / (Math.hypot(...v1) * Math.hypot(...v2) || 1))
        ) * 180 / Math.PI;
        label(`${deg.toFixed(1)}\xB0`, px[3].x + 8, px[3].y - 8);
      }
    }
  };
  const redrawOverlays = () => {
    for (const p of cfg.planes) drawOverlay(p);
  };
  const markSelected = () => {
    for (const p of cfg.planes) {
      cfg.cellOf(p)?.classList.toggle("bir-cell-selected", p === selectedPlane);
    }
  };
  for (const p of cfg.planes) {
    cfg.cellOf(p)?.addEventListener("pointerdown", () => {
      if (selectedPlane !== p) {
        selectedPlane = p;
        markSelected();
      }
    });
  }
  markSelected();
  const tb = document.createElement("div");
  tb.id = "bir-toolbar";
  cfg.bar.after(tb);
  const buttons = /* @__PURE__ */ new Map();
  const btn = (id, icon, tip, onClick, opts = {}) => {
    const b = document.createElement("button");
    b.id = "bir-" + id;
    b.innerHTML = ICONS[icon] ?? "?";
    b.title = opts.disabled && opts.disabledTip ? opts.disabledTip : tip;
    b.disabled = !!opts.disabled;
    if (onClick) b.addEventListener("click", onClick);
    tb.appendChild(b);
    buttons.set(id, b);
    return b;
  };
  const sep = () => {
    const s = document.createElement("div");
    s.className = "sep";
    tb.appendChild(s);
  };
  const stepSelected = (forward) => {
    cfg.step(selectedPlane, forward);
    cfg.redraw(selectedPlane);
  };
  btn("patient", "patient", "Select Patient \u2014 back to the patient/study/series browser", cfg.close);
  btn("prev-study", "prevStudy", "Previous Study", () => cfg.nav?.prevStudy?.(), {
    disabled: !cfg.nav?.prevStudy,
    disabledTip: "Previous Study \u2014 none earlier"
  });
  btn("next-study", "nextStudy", "Next Study", () => cfg.nav?.nextStudy?.(), {
    disabled: !cfg.nav?.nextStudy,
    disabledTip: "Next Study \u2014 none later"
  });
  btn("prev-fs", "prevFS", "Previous FrameSet (series) \u2014 Page Up", () => cfg.nav?.prevSeries?.(), {
    disabled: !cfg.nav?.prevSeries,
    disabledTip: "Previous FrameSet \u2014 none"
  });
  btn("next-fs", "nextFS", "Next FrameSet (series) \u2014 Page Down", () => cfg.nav?.nextSeries?.(), {
    disabled: !cfg.nav?.nextSeries,
    disabledTip: "Next FrameSet \u2014 none"
  });
  btn("prev-frame", "prevFrame", "Previous Frame \u2014 Left/Up arrow", () => stepSelected(false));
  btn("next-frame", "nextFrame", "Next Frame \u2014 Right/Down arrow", () => stepSelected(true));
  sep();
  const layoutSel = document.createElement("select");
  layoutSel.id = "bir-layout";
  layoutSel.title = "Layout Multiple Viewports \u2014 hanging protocol";
  layoutSel.style.cssText = "font:600 11px -apple-system,system-ui,sans-serif;color:#d6e2f2;background:#11141d;border:1px solid #33507e;border-radius:4px;padding:2px 4px;cursor:pointer;";
  for (const [v, label] of [
    ["fourUp", "Four-Up (MPR + 3D)"],
    ["twoUp", "2 side-by-side"],
    ["single", "Single viewport"],
    ["browser", "Browser (patient/study/series)"]
  ]) layoutSel.add(new Option(label, v));
  layoutSel.addEventListener("change", () => {
    if (layoutSel.value === "browser") cfg.close();
    else setLayout(layoutSel.value);
  });
  tb.appendChild(layoutSel);
  btn("layout-in", "layoutIn", "Layout Within Viewport", void 0, {
    disabled: true,
    disabledTip: "Layout Within Viewport \u2014 tiled stack layout not yet available (MPR planes shown instead)"
  });
  sep();
  const CURSORS = {
    zoom: "ns-resize",
    pan: "grab",
    wl: "crosshair",
    distance: "crosshair",
    angle: "crosshair",
    crosshair: "crosshair",
    select: "pointer",
    scroll: "default"
  };
  const setTool = (t) => {
    tool = t;
    pending = null;
    for (const [id, tl] of [
      ["tool-scroll", "scroll"],
      ["tool-window", "wl"],
      ["tool-zoom", "zoom"],
      ["tool-pan", "pan"],
      ["tool-select", "select"],
      ["tool-distance", "distance"],
      ["tool-angle", "angle"],
      ["tool-crosshair", "crosshair"]
    ]) buttons.get(id)?.classList.toggle("active", tool === tl);
    for (const p of cfg.planes) cfg.canvases[p].style.cursor = CURSORS[tool] ?? "default";
    cfg.onToolChange?.(tool);
    redrawOverlays();
  };
  btn("tool-scroll", "scroll", "Scroll (stack) \u2014 S", () => setTool("scroll"));
  btn("tool-window", "window", "Window (adjust W/L by dragging) \u2014 W", () => setTool("wl"));
  btn("tool-zoom", "zoom", "Zoom (drag up = out, down = in) \u2014 Z", () => setTool("zoom"));
  btn("tool-pan", "pan", "Pan (translate) \u2014 T", () => setTool("pan"));
  btn("tool-select", "selectvp", "Select Viewport \u2014 V", () => setTool("select"));
  btn("tool-distance", "distance", "Distance measurement (click start, click end) \u2014 D", () => setTool("distance"));
  btn("tool-angle", "angle", "Angle measurement (two segments, four clicks \u2014 Cobb) \u2014 A", () => setTool("angle"));
  btn("tool-crosshair", "crosshair", "Cross-hair (click to localize in all viewports) \u2014 J", () => setTool("crosshair"));
  sep();
  let presetPop = null;
  let presetOrigWL = null;
  let presetEsc = null;
  const closePresetPop = (revertWL) => {
    if (revertWL) cfg.wl.set(revertWL[0], revertWL[1]);
    presetPop?.remove();
    presetPop = null;
    document.removeEventListener("pointerdown", onPresetOutside, true);
    if (presetEsc) document.removeEventListener("keydown", presetEsc, true);
    presetEsc = null;
  };
  const onPresetOutside = (e) => {
    if (presetPop && !presetPop.contains(e.target) && e.target?.id !== "bir-presets") {
      closePresetPop(presetOrigWL ?? void 0);
    }
  };
  const openPresetPop = () => {
    if (presetPop) {
      closePresetPop(presetOrigWL ?? void 0);
      return;
    }
    presetOrigWL = cfg.wl.get();
    const orig = presetOrigWL;
    const anchor = buttons.get("presets").getBoundingClientRect();
    presetPop = document.createElement("div");
    presetPop.id = "bir-preset-pop";
    presetPop.style.cssText = `position:fixed;top:${Math.round(anchor.bottom + 4)}px;left:${Math.round(anchor.left)}px;z-index:1200;min-width:180px;background:#11141d;border:1px solid #33507e;border-radius:8px;padding:5px;box-shadow:0 10px 30px rgba(0,0,0,.6);font:12px -apple-system,system-ui,sans-serif;color:#d6e2f2;`;
    const head = document.createElement("div");
    head.textContent = "Window presets \u2014 hover to preview";
    head.style.cssText = "font-size:10px;color:#5a6b85;padding:3px 8px 5px;text-transform:uppercase;letter-spacing:.5px;";
    presetPop.appendChild(head);
    for (const opt of [...cfg.presetsEl.options].filter((o) => o.value !== "current")) {
      const item = document.createElement("div");
      item.className = "bir-preset-item";
      item.dataset.value = opt.value;
      item.textContent = opt.textContent;
      item.style.cssText = "padding:6px 10px;border-radius:5px;cursor:pointer;white-space:nowrap;";
      item.addEventListener("pointerenter", () => {
        for (const el of presetPop.querySelectorAll(".bir-preset-item")) {
          el.style.background = "transparent";
        }
        item.style.background = "#2b6cb0";
        cfg.presetsEl.value = opt.value;
        cfg.presetsEl.dispatchEvent(new Event("change"));
      });
      item.addEventListener("click", () => closePresetPop());
      presetPop.appendChild(item);
    }
    presetEsc = (e) => {
      if (e.key === "Escape") closePresetPop(orig);
    };
    document.body.appendChild(presetPop);
    setTimeout(() => {
      document.addEventListener("pointerdown", onPresetOutside, true);
      document.addEventListener("keydown", presetEsc, true);
    }, 0);
  };
  btn("presets", "presets", "Window Presets \u2014 hover to preview on the volume, click to keep (CT: Soft Tissue / Lung / Bone / Brain + Auto)", openPresetPop);
  btn("wl-clamp", "wlMode", "Window mode \u2014 Center/Width vs Clamped (lower limit fixed at 0; default for NM/PET)", () => {
    clamped = !clamped;
    buttons.get("wl-clamp").classList.toggle("active", clamped);
    if (clamped) {
      const [w] = cfg.wl.get();
      cfg.wl.set(w, w / 2);
    }
  });
  if (clamped) {
    const [w] = cfg.wl.get();
    cfg.wl.set(w, w / 2);
  }
  buttons.get("wl-clamp").classList.toggle("active", clamped);
  btn("invert", "invert", "Invert grayscale", () => {
    inverted = !inverted;
    buttons.get("invert").classList.toggle("active", inverted);
    for (const p of cfg.planes) cfg.canvases[p].style.filter = inverted ? "invert(1)" : "";
  });
  sep();
  btn("localizer", "localizer", "Localizer lines \u2014 O", () => {
    localizerOn = !localizerOn;
    buttons.get("localizer").classList.toggle("active", localizerOn);
    redrawOverlays();
  });
  btn("linksync", "link", "Link/unlink synchronized scrolling \u2014 L/U", void 0, {
    disabled: true,
    disabledTip: "Link/unlink synchronization \u2014 MPR planes of one series are always linked; enabled with compare viewports"
  });
  btn("annotation", "annotation", "Annotation \u2014 cycle full / minimal / off \u2014 I", () => {
    annState = (annState + 1) % 3;
    cfg.grid.classList.toggle("ann-min", annState === 1);
    cfg.grid.classList.toggle("ann-none", annState === 2);
    buttons.get("annotation").classList.toggle("active", annState !== 0);
  });
  sep();
  let cineBox = null;
  let cineTimer = null;
  const cineStop = () => {
    if (cineTimer !== null) clearInterval(cineTimer);
    cineTimer = null;
  };
  const cineClose = () => {
    cineStop();
    cineBox?.remove();
    cineBox = null;
    buttons.get("cine").classList.remove("active");
  };
  const cinePlay = (fps) => {
    cineStop();
    cineTimer = setInterval(() => {
      const before = cfg.off01(selectedPlane);
      cfg.step(selectedPlane, true);
      if (cfg.off01(selectedPlane) === before) cfg.setOff01(selectedPlane, 0);
      cfg.redraw(selectedPlane);
    }, 1e3 / Math.max(1, fps));
  };
  const cineToggle = () => {
    if (cineBox) {
      cineClose();
      return;
    }
    buttons.get("cine").classList.add("active");
    cineBox = document.createElement("div");
    cineBox.id = "bir-cine-box";
    const fps = document.createElement("input");
    fps.type = "number";
    fps.value = "10";
    fps.title = "Frame rate (frames/second)";
    const cbtn = (icon, tip, fn) => {
      const b = document.createElement("button");
      b.innerHTML = ICONS[icon];
      b.title = tip;
      b.addEventListener("click", fn);
      cineBox.appendChild(b);
      return b;
    };
    cbtn("play", "Cine Play (loop)", () => cinePlay(Number(fps.value) || 10));
    cbtn("pause", "Cine Pause", cineStop);
    cbtn("stop", "Cine Stop (back to first frame)", () => {
      cineStop();
      cfg.setOff01(selectedPlane, 0);
      cfg.redraw(selectedPlane);
    });
    cbtn("toStart", "Go to first frame", () => {
      cfg.setOff01(selectedPlane, 0);
      cfg.redraw(selectedPlane);
    });
    cbtn("stepFwd", "Step one frame", () => stepSelected(true));
    cbtn("toEnd", "Go to last frame", () => {
      cfg.setOff01(selectedPlane, 1);
      cfg.redraw(selectedPlane);
    });
    cineBox.appendChild(fps);
    const close = document.createElement("button");
    close.textContent = "\xD7";
    close.title = "Close cine";
    close.addEventListener("click", cineClose);
    cineBox.appendChild(close);
    tb.appendChild(cineBox);
    fps.addEventListener("change", () => {
      if (cineTimer !== null) cinePlay(Number(fps.value) || 10);
    });
    cinePlay(10);
  };
  btn("cine", "cine", "Cine (play through frames) \u2014 C", cineToggle);
  sep();
  const rotatable = /^(DX|CR)$/i.test(cfg.modality);
  btn("rotate", "rotate", "Rotate 90\xB0 clockwise", void 0, {
    disabled: !rotatable,
    disabledTip: "Rotate 90\xB0 \u2014 only active for DX/CR projection images (safety: BIR 4.16.4.2.2.5.2)"
  });
  btn("flip", "flip", "Flip horizontally", void 0, {
    disabled: !rotatable,
    disabledTip: "Flip \u2014 only active for DX/CR projection images (safety: BIR 4.16.4.2.2.5.2)"
  });
  sep();
  const reset = () => {
    cfg.resetViews();
    cfg.wl.set(...cfg.wl.auto);
    measurements.length = 0;
    pending = null;
    if (inverted) buttons.get("invert").click();
    cfg.redrawAll();
    redrawOverlays();
  };
  btn("reset", "reset", "Display Reset (zoom/pan/window/measurements; keeps scroll position) \u2014 Esc", reset);
  btn("print", "print", "Print selected viewport \u2014 P", () => printViewport());
  btn("report", "report", "Show Report \u2014 R", void 0, {
    disabled: true,
    disabledTip: "Show Report \u2014 no reports available for this patient"
  });
  btn("help", "help", "Help \u2014 tool guide and keyboard shortcuts \u2014 H / F1", () => showHelp());
  btn("advanced", "advanced", "Advanced Mode \u2014 toggle BIR basic (single stack) vs MPR Four-Up + 3D \u2014 F10", () => {
    setLayout(layout === "single" ? "fourUp" : "single");
  });
  if (cfg.extraTools?.length) {
    sep();
    for (const t of cfg.extraTools) btn(t.id, t.icon, t.title, t.run);
  }
  const setLayout = (l) => {
    layout = l;
    const show = {};
    const names = [...cfg.planes, "threeD"];
    if (l === "single") for (const n of names) show[n] = n === selectedPlane;
    else if (l === "twoUp") for (const n of names) show[n] = n === "axial" || n === "sagittal";
    else for (const n of names) show[n] = true;
    for (const n of names) {
      const cell = cfg.cellOf(n);
      if (cell) cell.style.display = show[n] ? "" : "none";
    }
    cfg.grid.style.gridTemplateColumns = l === "single" ? "1fr" : "1fr 1fr";
    cfg.grid.style.gridTemplateRows = l === "fourUp" ? "1fr 1fr" : "1fr";
    layoutSel.value = l;
    buttons.get("advanced").classList.toggle("active", l === "fourUp");
    globalThis.dispatchEvent(new Event("resize"));
  };
  let strip = null;
  if (cfg.strip?.length) {
    strip = document.createElement("div");
    strip.id = "bir-strip";
    strip.innerHTML = `<div class="head">Series</div>`;
    for (const it of cfg.strip) {
      const item = document.createElement("div");
      item.className = "item" + (it.current ? " current" : "");
      item.title = it.lines.join(" \xB7 ") + " \u2014 double-click to load";
      const img = document.createElement("img");
      img.alt = "";
      it.thumb().then((b) => {
        if (b) img.src = URL.createObjectURL(b);
      }).catch(() => {
      });
      const cap = document.createElement("div");
      cap.className = "cap";
      cap.textContent = it.lines.join(" \xB7 ");
      item.append(img, cap);
      item.addEventListener("dblclick", () => it.open());
      strip.appendChild(item);
    }
    const row = document.createElement("div");
    row.id = "bir-row";
    row.style.cssText = "flex:1;display:flex;min-height:0;";
    cfg.grid.parentElement.insertBefore(row, cfg.grid);
    row.append(strip, cfg.grid);
  }
  const stripBtn = btn("thumbs", "thumbs", "Show/hide the series thumbnail strip", () => {
    if (!strip) return;
    strip.style.display = strip.style.display === "none" ? "" : "none";
    stripBtn.classList.toggle("active", strip.style.display !== "none");
    globalThis.dispatchEvent(new Event("resize"));
  });
  if (!strip) {
    stripBtn.disabled = true;
    stripBtn.title = "Series thumbnail strip \u2014 no sibling series available";
  } else stripBtn.classList.add("active");
  const dc = displayClass();
  if (dc === "phone") {
    if (strip) strip.style.display = "none";
    stripBtn.classList.remove("active");
    setLayout("single");
  } else {
    setLayout(cfg.initialLayout ?? "fourUp");
  }
  const printViewport = () => {
    const p = selectedPlane;
    cfg.redraw(p);
    requestAnimationFrame(() => {
      setTimeout(() => {
        const src = cfg.canvases[p];
        const ov = overlays.get(p);
        const out = document.createElement("canvas");
        out.width = src.width;
        out.height = src.height;
        const ctx = out.getContext("2d");
        try {
          ctx.drawImage(src, 0, 0);
          if (inverted) {
            ctx.globalCompositeOperation = "difference";
            ctx.fillStyle = "#fff";
            ctx.fillRect(0, 0, out.width, out.height);
            ctx.globalCompositeOperation = "source-over";
          }
          if (ov) ctx.drawImage(ov, 0, 0, out.width, out.height);
        } catch {
        }
        const [win, lev] = cfg.wl.get();
        const w = globalThis.open("", "_blank", "width=900,height=700");
        if (!w) return;
        w.document.write(
          `<title>SlicerRad print \u2014 ${p}</title><p style="font:12px sans-serif">For Investigational Use Only \u2014 Not for Diagnostic Use \xB7 ${p} \xB7 W ${Math.round(win)} / L ${Math.round(lev)}</p><img style="max-width:100%" src="${out.toDataURL("image/png")}">`
        );
        w.document.close();
        w.focus();
        setTimeout(() => w.print(), 200);
      }, 60);
    });
  };
  const showHelp = () => {
    if (document.getElementById("bir-help-overlay")) return;
    const h = document.createElement("div");
    h.id = "bir-help-overlay";
    h.style.cssText = "position:fixed;inset:0;z-index:2000;background:rgba(5,6,10,.85);display:flex;align-items:center;justify-content:center;";
    h.innerHTML = `
      <div style="background:#11141d;border:1px solid #33507e;border-radius:10px;padding:20px 26px;
        max-width:560px;max-height:80vh;overflow:auto;font:13px -apple-system,system-ui,sans-serif;color:#d6e2f2;">
        <h3 style="margin:0 0 10px;color:#fff;">SlicerRad reader \u2014 IHE Basic Image Review tools</h3>
        <table style="border-collapse:collapse;width:100%;">
        ${[
      ["S", "Scroll through slices (default; wheel always scrolls)"],
      ["W", "Window \u2014 drag to adjust window/level"],
      ["Z / T", "Zoom / Pan (translate)"],
      ["V", "Select viewport (click also selects)"],
      ["D / A", "Distance (2 clicks) / Angle (4 clicks, Cobb) measurements"],
      ["J", "Cross-hair \u2014 click localizes in all viewports"],
      ["O", "Localizer lines on/off"],
      ["I", "Annotations \u2014 full / minimal / off"],
      ["C", "Cine \u2014 plays immediately; controls appear in the toolbar"],
      ["\u2190 \u2191 / \u2192 \u2193", "Previous / next frame"],
      ["PgUp / PgDn", "Previous / next series (FrameSet)"],
      ["Esc", "Display Reset \u2014 zoom/pan/window/measurements (keeps scroll position)"],
      ["P", "Print selected viewport"],
      ["F10", "Advanced Mode \u2014 Four-Up MPR + 3D vs single-stack"],
      ["H / F1", "This help"]
    ].map(
      ([k, d]) => `<tr><td style="padding:3px 14px 3px 0;color:#9fd0b3;white-space:nowrap;font-weight:600;">${k}</td><td style="padding:3px 0;color:#9fb3d0;">${d}</td></tr>`
    ).join("")}
        </table>
        <p style="color:#9fb3d0;margin:14px 0 4px;">This reader implements the IHE Radiology <b>Basic Image Review (BIR)</b> profile \u2014
        its tool set, icons and shortcuts follow the supplement (Rev 1.3 \xA74.16.4.2.2.5.13).</p>
        <p style="margin:0 0 4px;"><a id="bir-docs-link" href="https://profiles.ihe.net/RAD/" target="_blank" rel="noopener"
          style="color:#6cc4ff;">IHE Radiology profiles (Basic Image Review) \u2192</a></p>
        <p style="margin:0;"><a href="https://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_Suppl_BIR_Rev1.3_TI_2016_09-09.pdf" target="_blank" rel="noopener"
          style="color:#6cc4ff;">BIR Technical Framework Supplement (PDF) \u2192</a></p>
        <button id="bir-help-close" style="margin-top:14px;font:600 12px -apple-system,system-ui,sans-serif;
          color:#d6e2f2;background:#1b2740;border:1px solid #33507e;border-radius:4px;padding:5px 14px;cursor:pointer;">Close</button>
      </div>`;
    document.body.appendChild(h);
    h.querySelector("#bir-help-close").addEventListener("click", () => h.remove());
    h.addEventListener("click", (e) => {
      if (e.target === h) h.remove();
    });
  };
  const onKey = (e) => {
    if (!document.body.contains(cfg.overlay)) return;
    const t = e.target;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(t?.tagName ?? "")) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const map = {
      s: () => setTool("scroll"),
      w: () => setTool("wl"),
      z: () => setTool("zoom"),
      t: () => setTool("pan"),
      v: () => setTool("select"),
      d: () => setTool("distance"),
      a: () => setTool("angle"),
      j: () => setTool("crosshair"),
      o: () => buttons.get("localizer").click(),
      i: () => buttons.get("annotation").click(),
      c: () => cineToggle(),
      p: () => printViewport(),
      h: () => showHelp(),
      F1: () => showHelp(),
      F10: () => buttons.get("advanced").click(),
      Escape: () => reset(),
      ArrowLeft: () => stepSelected(false),
      ArrowUp: () => stepSelected(false),
      ArrowRight: () => stepSelected(true),
      ArrowDown: () => stepSelected(true),
      PageUp: () => cfg.nav?.prevSeries?.(),
      PageDown: () => cfg.nav?.nextSeries?.()
    };
    const fn = map[k];
    if (fn) {
      e.preventDefault();
      fn();
    }
  };
  document.addEventListener("keydown", onKey);
  const hooks = (p) => ({
    onLeftGrab: (u, v, w, h) => {
      if (tool === "select") return true;
      if (tool === "crosshair") {
        crossRas = rasOfClick(p, u, v, w, h);
        cfg.jumpAll(crossRas);
        redrawOverlays();
        return true;
      }
      if (tool === "distance" || tool === "angle") {
        const ras = rasOfClick(p, u, v, w, h);
        if (!pending || pending.plane !== p) {
          pending = { kind: tool, plane: p, mm: cfg.offsetMm(p), pts: [ras] };
        } else {
          pending.pts.push(ras);
          const needed = tool === "distance" ? 2 : 4;
          if (pending.pts.length >= needed) {
            measurements.push(pending);
            pending = null;
          }
        }
        drawOverlay(p);
        return true;
      }
      return false;
    }
  });
  resizeOverlays();
  return {
    tool: () => tool,
    setTool,
    measureCount: () => measurements.length,
    leftMode: () => tool === "wl" || tool === "zoom" || tool === "pan" ? tool : "scroll",
    hooks,
    drawOverlay,
    resize: () => {
      resizeOverlays();
      redrawOverlays();
    },
    selected: () => selectedPlane,
    setLayout,
    reset,
    detachKeys: () => {
      document.removeEventListener("keydown", onKey);
      cineStop();
    }
  };
}

// render/demos/idc-share.ts
function studyShareURL(src) {
  const u = new URL(globalThis.location.origin + globalThis.location.pathname);
  if (src.st) u.searchParams.set("StudyInstanceUIDs", src.st);
  u.searchParams.set("series", src.c);
  if (src.cb && src.cb !== "idc-open-data") u.searchParams.set("bucket", src.cb);
  if (src.s) u.searchParams.set("seg", src.s);
  if (src.sb && src.sb !== "idc-open-data") u.searchParams.set("segBucket", src.sb);
  if (src.m && src.m !== "CT") u.searchParams.set("modality", src.m);
  const pid = (src.sd.split("\xB7")[0] || "").trim();
  if (pid && pid !== "IDC") u.searchParams.set("patient", pid);
  if (src.col && src.col !== "IDC") u.searchParams.set("collection", src.col);
  return u.toString();
}
var OVERLAY_CSS = "position:fixed;inset:0;z-index:2000;background:rgba(5,6,10,.85);display:flex;align-items:center;justify-content:center;";
var CARD_CSS = "background:#11141d;border:1px solid #33507e;border-radius:10px;padding:20px 24px;width:600px;max-width:92vw;font:13px -apple-system,system-ui,sans-serif;color:#d6e2f2;";
function shareStudy(src) {
  if (document.getElementById("idc-share")) return;
  const here = studyShareURL(src);
  const portal = src.st && ohifViewerURL(src.st) || "";
  const overlay = document.createElement("div");
  overlay.id = "idc-share";
  overlay.style.cssText = OVERLAY_CSS;
  const esc = (s) => s.replace(/"/g, "&quot;");
  const row = (label, url, hint) => `
    <div style="margin:0 0 14px;">
      <div style="color:#9fd0b3;font-weight:600;margin-bottom:4px;">${label}</div>
      <div style="display:flex;gap:6px;">
        <input readonly value="${esc(url)}" style="flex:1;font:12px ui-monospace,monospace;color:#d6e2f2;
          background:#0b0e16;border:1px solid #33507e;border-radius:4px;padding:6px 8px;">
        <button data-copy="${esc(url)}" style="font:600 12px -apple-system,system-ui,sans-serif;color:#d6e2f2;
          background:#1b2740;border:1px solid #33507e;border-radius:4px;padding:6px 12px;cursor:pointer;">Copy</button>
      </div>
      <div style="color:#5a6b85;font-size:11px;margin-top:3px;">${hint}</div>
    </div>`;
  overlay.innerHTML = `
    <div style="${CARD_CSS}">
      <h3 style="margin:0 0 4px;color:#fff;">Share study</h3>
      <p style="margin:0 0 16px;color:#9fb3d0;">${src.sd}${src.st ? " \xB7 " + src.st : ""}</p>
      ${row("Open in SlicerLive (this viewer)", here, "Reopens this study in the SlicerLive BIR reader (OHIF-style + direct-S3 params).")}
      ${portal ? row("Open in IDC OHIF portal", portal, "The canonical IDC viewer link for the same study.") : ""}
      <div style="text-align:right;margin-top:6px;">
        <button id="idc-share-close" style="font:600 12px -apple-system,system-ui,sans-serif;color:#9fb3d0;
          background:none;border:none;cursor:pointer;padding:6px 4px;">Close</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  const close = () => overlay.remove();
  overlay.addEventListener("click", (e) => e.target === overlay && close());
  overlay.querySelector("#idc-share-close").addEventListener("click", close);
  for (const b of overlay.querySelectorAll("button[data-copy]")) {
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(b.dataset.copy);
        b.textContent = "Copied \u2713";
        setTimeout(() => b.textContent = "Copy", 1400);
      } catch {
        b.textContent = "Copy failed";
      }
    });
  }
}
var DL_LANES = Math.max(2, Math.min(6, (navigator.hardwareConcurrency || 4) - 1));
var safeName = (s) => s.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 80) || "study";
async function downloadStudy(src, seriesList, onProgress, opts) {
  let root;
  if (opts?.root) root = opts.root;
  else {
    const picker = globalThis.showDirectoryPicker;
    if (!picker) throw new Error("File System Access API unavailable \u2014 cannot pick a download folder");
    root = await picker({ mode: "readwrite", id: "slicerlive-idc-download" });
  }
  onProgress(0, 0, 0, "listing series\u2026");
  const studyDir = await root.getDirectoryHandle(
    safeName(`${src.col}_${(src.st || src.c).slice(-12)}`),
    { create: true }
  );
  const tasks = [];
  let total = 0, sn = 0;
  for (const s of seriesList) {
    onProgress(0, total, 0, `listing ${s.modality} ${s.seriesDescription || s.prefix.slice(0, 8) + "\u2026"}\u2026`);
    let keys = await s3ListKeys(s.prefix, s.bucket);
    if (!keys.length) continue;
    if (opts?.maxFilesPerSeries) keys = keys.slice(0, opts.maxFilesPerSeries);
    total += keys.length;
    const dir = await studyDir.getDirectoryHandle(
      safeName(`${s.seriesNumber || ++sn}_${s.modality}_${s.prefix.slice(0, 8)}`),
      { create: true }
    );
    tasks.push({ dir, base: idcS3(s.bucket), keys });
  }
  if (!total) throw new Error("no DICOM objects found to download");
  onProgress(0, total, 0, `downloading ${total} files (${DL_LANES} at a time)\u2026`);
  let done = 0, bytes = 0;
  const errors = [];
  for (const task of tasks) {
    let next = 0;
    const lane = async () => {
      while (next < task.keys.length) {
        const key = task.keys[next++];
        try {
          const resp = await fetchRetry(task.base + key, {});
          const buf = await resp.arrayBuffer();
          const fname = safeName(key.split("/").pop() || `obj-${done}.dcm`);
          const fh = await task.dir.getFileHandle(fname, { create: true });
          const w = await fh.createWritable();
          await w.write(buf);
          await w.close();
          bytes += buf.byteLength;
        } catch (e) {
          errors.push(`${key}: ${e.message}`);
        }
        done++;
        if (done % 5 === 0 || done === total) onProgress(done, total, bytes, `${done}/${total} files`);
      }
    };
    await Promise.all(Array.from({ length: DL_LANES }, lane));
  }
  const mb = (bytes / 1e6).toFixed(0);
  onProgress(
    done,
    total,
    bytes,
    `saved ${done - errors.length}/${total} files (${mb} MB) to \u201C${root.name}/${studyDir.name}\u201D` + (errors.length ? ` \u2014 ${errors.length} failed` : "")
  );
  if (errors.length) console.warn("IDC download errors:", errors.slice(0, 10));
}
async function downloadStudyWithDialog(src, listAll) {
  if (document.getElementById("idc-download")) return;
  const overlay = document.createElement("div");
  overlay.id = "idc-download";
  overlay.style.cssText = OVERLAY_CSS;
  overlay.innerHTML = `
    <div style="${CARD_CSS.replace("width:600px", "width:560px")}">
      <h3 style="margin:0 0 4px;color:#fff;">Downloading study</h3>
      <p style="margin:0 0 6px;color:#9fb3d0;">${src.sd}${src.st ? " \xB7 " + src.st : ""}</p>
      <p style="margin:0 0 14px;color:#5a6b85;font-size:11px;">${src.lic || "NCI Imaging Data Commons \u2014 open data"}</p>
      <div style="height:8px;background:#0b0e16;border:1px solid #1b2740;border-radius:5px;overflow:hidden;">
        <div id="idc-dl-bar" style="height:100%;width:0%;background:#2b6cb0;transition:width .2s;"></div>
      </div>
      <p id="idc-dl-note" style="margin:10px 0 0;color:#9fb3d0;font-variant-numeric:tabular-nums;">starting\u2026</p>
      <div style="text-align:right;margin-top:12px;">
        <button id="idc-dl-close" style="font:600 12px -apple-system,system-ui,sans-serif;color:#9fb3d0;
          background:none;border:none;cursor:pointer;padding:6px 4px;">Close</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  const bar = overlay.querySelector("#idc-dl-bar");
  const note = overlay.querySelector("#idc-dl-note");
  const closeBtn = overlay.querySelector("#idc-dl-close");
  closeBtn.addEventListener("click", () => overlay.remove());
  try {
    const loaded = [{ prefix: src.c, bucket: src.cb, modality: src.m }];
    if (src.s) loaded.push({ prefix: src.s, bucket: src.sb || "idc-open-data", modality: "SEG" });
    const seriesList = listAll ? await listAll().catch(() => loaded) : loaded;
    await downloadStudy(src, seriesList.length ? seriesList : loaded, (done, total, bytes, msg) => {
      bar.style.width = (total ? 100 * done / total : 0).toFixed(1) + "%";
      note.textContent = total ? `${done.toLocaleString("en-US")} / ${total.toLocaleString("en-US")} files \xB7 ${(bytes / 1e6).toFixed(0)} MB \u2014 ${msg}` : msg;
    });
    bar.style.width = "100%";
    closeBtn.textContent = "Done";
  } catch (err) {
    note.textContent = "download failed: " + err.message;
    note.style.color = "#ff6b74";
    closeBtn.textContent = "Close";
  }
}
globalThis.__slicerLiveIdc = {
  studyShareURL,
  shareStudy,
  downloadStudy,
  downloadStudyWithDialog
};

// render/scene-volume.ts
function interpTF(tf, s, comps) {
  if (!tf.length) return new Array(comps).fill(0);
  if (s <= tf[0][0]) return tf[0].slice(1, 1 + comps);
  const last = tf[tf.length - 1];
  if (s >= last[0]) return last.slice(1, 1 + comps);
  for (let i = 1; i < tf.length; i++) {
    if (s <= tf[i][0]) {
      const a = tf[i - 1], b = tf[i];
      const u = (s - a[0]) / Math.max(b[0] - a[0], 1e-9);
      return Array.from({ length: comps }, (_, c) => a[1 + c] + u * (b[1 + c] - a[1 + c]));
    }
  }
  return last.slice(1, 1 + comps);
}
function lutFromTransferFunctions(colorTF, opacityTF, clim) {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const s = clim[0] + i / 255 * (clim[1] - clim[0]);
    const [r, g, b] = interpTF(colorTF, s, 3);
    const [a] = interpTF(opacityTF, s, 1);
    lut[i * 4 + 0] = Math.round(Math.max(0, Math.min(1, r)) * 255);
    lut[i * 4 + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
    lut[i * 4 + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
    lut[i * 4 + 3] = Math.round(Math.max(0, Math.min(1, a)) * 255);
  }
  return lut;
}

// render/ct-vr-presets.ts
var CT_VR_PRESETS = [
  {
    name: "CT-AAA",
    label: "CT AAA (angio)",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [143.556, 0.6157, 0.3569, 0.1843], [166.222, 0.8824, 0.6039, 0.2902], [214.389, 1, 1, 1], [419.736, 1, 0.937, 0.9545], [3071, 0.8275, 0.6588, 1]],
    opacityTF: [[-3024, 0], [143.556, 0], [166.222, 0.6863], [214.389, 0.6961], [419.736, 0.8333], [3071, 0.8039]]
  },
  {
    name: "CT-Bone",
    label: "CT Bone",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [-16.446, 0.7294, 0.2549, 0.302], [641.385, 0.9059, 0.8157, 0.5529], [3071, 1, 1, 1]],
    opacityTF: [[-3024, 0], [-16.446, 0], [641.385, 0.7157], [3071, 0.7059]]
  },
  {
    name: "CT-Bones",
    label: "CT Bones",
    shade: true,
    light: [0.2, 1, 0, 1],
    colorTF: [[-1e3, 0.3, 0.3, 1], [-488, 0.3, 1, 0.3], [463.28, 1, 0, 0], [659.15, 1, 0.9125, 0.0375], [953, 1, 0.3, 0.3]],
    opacityTF: [[-1e3, 0], [152.19, 0], [278.93, 0.1905], [952, 0.2]]
  },
  {
    name: "CT-Cardiac",
    label: "CT Cardiac",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [-77.688, 0.549, 0.251, 0.149], [94.952, 0.8824, 0.6039, 0.2902], [179.052, 1, 0.937, 0.9545], [260.439, 0.6157, 0, 0], [3071, 0.8275, 0.6588, 1]],
    opacityTF: [[-3024, 0], [-77.688, 0], [94.952, 0.2857], [179.052, 0.5536], [260.439, 0.8482], [3071, 0.875]]
  },
  {
    name: "CT-Chest-Contrast-Enhanced",
    label: "CT Chest (contrast)",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [67.011, 0.549, 0.251, 0.149], [251.105, 0.8824, 0.6039, 0.2902], [439.291, 1, 0.937, 0.9545], [3071, 0.8275, 0.6588, 1]],
    opacityTF: [[-3024, 0], [67.011, 0], [251.105, 0.4464], [439.291, 0.625], [3071, 0.6161]]
  },
  {
    name: "CT-Coronary-Arteries",
    label: "CT Coronary",
    shade: false,
    light: [0.2, 1, 0, 1],
    colorTF: [[-2048, 0, 0, 0], [136.47, 0, 0, 0], [159.215, 0.1598, 0.1598, 0.1598], [318.43, 0.7647, 0.7647, 0.7647], [478.693, 1, 1, 1], [3661, 1, 1, 1]],
    opacityTF: [[-2048, 0], [136.47, 0], [159.215, 0.2589], [318.43, 0.5714], [478.693, 0.7768], [3661, 1]]
  },
  {
    name: "CT-Fat",
    label: "CT Fat",
    shade: false,
    light: [0.2, 1, 0, 1],
    colorTF: [[-1e3, 0.3, 0.3, 1], [-497.5, 0.3, 1, 0.3], [-99, 0, 0, 1], [-76.946, 0, 1, 0], [-65.481, 0.8354, 0.8889, 0.0165], [83.89, 1, 0, 0], [463.28, 1, 0, 0], [659.15, 1, 0.9125, 0.0375], [2952, 1, 0.3003, 0.2999]],
    opacityTF: [[-1e3, 0], [-100, 0], [-99, 0.15], [-60, 0.15], [-59, 0], [101.2, 0], [952, 0]]
  },
  {
    name: "CT-Lung",
    label: "CT Lung",
    shade: true,
    light: [0.2, 1, 0, 1],
    colorTF: [[-1e3, 0.3, 0.3, 1], [-600, 0, 0, 1], [-530, 0.1347, 0.7817, 0.0725], [-460, 0.9292, 1, 0.1095], [-400, 0.8889, 0.2549, 0.024], [2952, 1, 0.3, 0.3]],
    opacityTF: [[-1e3, 0], [-600, 0], [-599, 0.15], [-400, 0.15], [-399, 0], [2952, 0]]
  },
  {
    name: "CT-MIP",
    label: "CT MIP",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [-637.62, 1, 1, 1], [700, 1, 1, 1], [3071, 1, 1, 1]],
    opacityTF: [[-3024, 0], [-637.62, 0], [700, 1], [3071, 1]]
  },
  {
    name: "CT-Muscle",
    label: "CT Muscle",
    shade: true,
    light: [0.1, 0.9, 0.2, 10],
    colorTF: [[-3024, 0, 0, 0], [-155.407, 0.549, 0.251, 0.149], [217.641, 0.8824, 0.6039, 0.2902], [419.736, 1, 0.937, 0.9545], [3071, 0.8275, 0.6588, 1]],
    opacityTF: [[-3024, 0], [-155.407, 0], [217.641, 0.6765], [419.736, 0.8333], [3071, 0.8039]]
  },
  {
    name: "CT-Pulmonary-Arteries",
    label: "CT Pulmonary",
    shade: true,
    light: [0.2, 1, 0, 1],
    colorTF: [[-2048, 0, 0, 0], [-568.625, 0, 0, 0], [-364.081, 0.3961, 0.302, 0.1804], [-244.813, 0.6118, 0.3529, 0.0706], [18.277, 0.8431, 0.0157, 0.1569], [447.798, 0.7529, 0.7529, 0.7529], [3592.73, 1, 1, 1]],
    opacityTF: [[-2048, 0], [-568.625, 0], [-364.081, 0.0714], [-244.813, 0.4018], [18.277, 0.6071], [447.798, 0.8304], [3592.73, 0.8393]]
  },
  {
    name: "CT-Soft-Tissue",
    label: "CT Soft Tissue",
    shade: false,
    light: [0.2, 1, 0, 1],
    colorTF: [[-2048, 0, 0, 0], [-167.01, 0, 0, 0], [-160, 0.0556, 0.0556, 0.0556], [240, 1, 1, 1], [3661, 1, 1, 1]],
    opacityTF: [[-2048, 0], [-167.01, 0], [-160, 1], [240, 1], [3661, 1]]
  }
];
function presetLUT(p) {
  const lo = p.colorTF[0][0];
  const hi = p.colorTF[p.colorTF.length - 1][0];
  const clim = [lo, hi];
  const lut = lutFromTransferFunctions(p.colorTF, p.opacityTF, clim);
  const shade = p.shade ? [p.light[0], p.light[1], p.light[2], p.light[3]] : [1, 0, 0, 1];
  return { lut, clim, shade };
}

// render/demos/vr-preset-menu.ts
function openVrPresetMenu(opts) {
  if (document.getElementById("vr-preset-menu")) return;
  const overlay = document.createElement("div");
  overlay.id = "vr-preset-menu";
  overlay.style.cssText = "position:fixed;inset:0;z-index:2100;background:rgba(5,6,10,.82);display:flex;align-items:center;justify-content:center;";
  const card = document.createElement("div");
  card.style.cssText = "background:#11141d;border:1px solid #33507e;border-radius:12px;padding:18px 20px;max-width:min(720px,94vw);max-height:88vh;overflow-y:auto;font:13px -apple-system,system-ui,sans-serif;color:#d6e2f2;";
  card.innerHTML = `<div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:12px;"><h3 style="margin:0;color:#fff;font-size:16px;">Volume rendering preset</h3><span style="color:#5a6b85;font-size:11px;">Slicer CT presets \xB7 thumbnails rendered from this study</span></div>`;
  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;";
  card.appendChild(grid);
  const close = () => overlay.remove();
  for (const it of opts.items) {
    const cell = document.createElement("button");
    const sel = it.name === opts.current;
    cell.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:6px;padding:8px 6px 7px;cursor:pointer;background:" + (sel ? "rgba(84,198,240,.18)" : "rgba(255,255,255,.04)") + ";border:1px solid " + (sel ? "#54c6f0" : "rgba(255,255,255,.12)") + ";border-radius:10px;color:#eaf0ff;";
    it.canvas.style.cssText = "width:110px;height:110px;border-radius:7px;background:#05060a;display:block;";
    const lab = document.createElement("div");
    lab.textContent = it.label;
    lab.style.cssText = "font:600 11.5px -apple-system,system-ui,sans-serif;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:110px;" + (sel ? "color:#9fe9ff;" : "");
    cell.appendChild(it.canvas);
    cell.appendChild(lab);
    cell.onclick = () => {
      opts.onPick(it.name);
      close();
    };
    grid.appendChild(cell);
  }
  overlay.appendChild(card);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });
  const esc = (e) => {
    if (e.key === "Escape") {
      close();
      document.removeEventListener("keydown", esc, true);
    }
  };
  document.addEventListener("keydown", esc, true);
  document.body.appendChild(overlay);
}

// render/demos/bir-browser.ts
var P = new URLSearchParams(location.search);
var KITS_DEFAULT = {
  c: "e3e86cde-da96-44b0-9e3b-b0b7bdd5a675",
  cb: "idc-open-data",
  s: "04a800eb-2f06-4e29-a10d-934a6f5c7d47",
  sb: "idc-open-data",
  m: "CT",
  col: "c4kc_kits",
  st: "1.3.6.1.4.1.14519.5.2.1.6919.4624.368281589441706814147998236429",
  sd: "KiTS-00051 \xB7 noncontrast abdomen + kidney/tumour SEG",
  lic: "CC BY 3.0 \xB7 IDC c4kc_kits \xB7 doi:10.7937/tcia.2019.ix49e8nx"
};
var IDC_INDEX_BASE = globalThis.__IDC_INDEX_BASE || P.get("indexBase") || "https://js2.jetstream-cloud.org:8001/swift/v1/idc-index/";
var GROUPS_URL = new URL("idc-rad-groups.json", IDC_INDEX_BASE).href;
var PARQUET_URL = new URL("idc-rad-slim.parquet", IDC_INDEX_BASE).href;
var HYPARQUET_ESM = "https://cdn.jsdelivr.net/npm/hyparquet@1.28.2/+esm";
var splitList = (v) => (v ?? "").split(",").map((x) => x.trim()).filter(Boolean);
var _dirCache = null;
async function loadGroupDir(onStatus) {
  if (_dirCache) return _dirCache;
  onStatus("fetching the IDC index directory\u2026");
  let resp;
  try {
    const cache = globalThis.caches ? await globalThis.caches.open("idc-rad") : null;
    if (cache) resp = await cache.match(GROUPS_URL);
    if (!resp) {
      resp = await fetch(GROUPS_URL);
      if (resp.ok && cache) await cache.put(GROUPS_URL, resp.clone());
    }
  } catch {
    resp = await fetch(GROUPS_URL);
  }
  if (!resp || !resp.ok) throw new Error(`index directory not reachable at ${GROUPS_URL} (HTTP ${resp?.status ?? "?"})`);
  _dirCache = await resp.json();
  return _dirCache;
}
var SLIM_COLS = [
  "StudyInstanceUID",
  "SeriesInstanceUID",
  "crdc_series_uuid",
  "aws_bucket",
  "Modality",
  "instanceCount",
  "SeriesDescription",
  "PatientID",
  "collection_id",
  "license_short_name",
  "source_DOI"
];
async function readStudyRows(studyUID, onStatus) {
  const dir = await loadGroupDir(onStatus);
  const RGS = dir.rowGroupSize;
  const spans = [];
  for (let i = 0; i < dir.groups.length; i++) {
    const g = dir.groups[i];
    if (studyUID >= g.min && studyUID <= g.max) spans.push([i * RGS, Math.min((i + 1) * RGS, dir.total)]);
  }
  if (!spans.length) {
    throw new Error(
      `StudyInstanceUID not found in the slim IDC index (${dir.version}): ${studyUID}. It may be a non-radiology study, or from a newer index. The direct form always works: ?series=<crdc_series_uuid>&bucket=idc-open-data.`
    );
  }
  onStatus(`range-reading the IDC index (${spans.length} group${spans.length > 1 ? "s" : ""})\u2026`);
  try {
    const hp = await import(HYPARQUET_ESM);
    const file = await hp.asyncBufferFromUrl({ url: PARQUET_URL });
    const metadata = await hp.parquetMetadataAsync(file);
    const parts = await Promise.all(
      spans.map(([rowStart, rowEnd]) => hp.parquetReadObjects({ file, metadata, columns: SLIM_COLS, rowStart, rowEnd }))
    );
    const rows = parts.flat().filter((r) => r.StudyInstanceUID === studyUID);
    if (!rows.length) throw new Error(`StudyInstanceUID not found in the IDC index: ${studyUID}`);
    return rows;
  } catch (e) {
    if (e.message.includes("not found")) throw e;
    throw new Error(
      `couldn't range-read the slim IDC index at ${PARQUET_URL} \u2014 check the CORS bucket (?indexBase=). Meanwhile the direct form works: ?series=<crdc_series_uuid>&bucket=idc-open-data. (${e.message})`
    );
  }
}
async function resolveSource(onStatus) {
  const series = P.get("series");
  if (series) {
    return {
      c: series,
      cb: P.get("bucket") || "idc-open-data",
      s: P.get("seg") || void 0,
      sb: P.get("segBucket") || "idc-open-data",
      m: (P.get("modality") || "CT").toUpperCase(),
      col: P.get("collection") || "IDC",
      st: "",
      sd: `${P.get("patient") || "IDC"} \xB7 ${series.slice(0, 8)}\u2026`,
      lic: "NCI Imaging Data Commons"
    };
  }
  const studyUIDs = [...splitList(P.get("StudyInstanceUIDs")), ...splitList(P.get("studyUID"))];
  if (studyUIDs.length) {
    const want = [
      ...splitList(P.get("SeriesInstanceUIDs")),
      ...splitList(P.get("initialSeriesInstanceUID")),
      ...splitList(P.get("seriesUID"))
    ];
    return await resolveFromIndex(studyUIDs[0], want, onStatus);
  }
  return KITS_DEFAULT;
}
async function resolveFromIndex(studyUID, wantSeries, onStatus) {
  const memoKey = `idc-rad:${studyUID}:${wantSeries.join(",")}`;
  try {
    const hit = localStorage.getItem(memoKey);
    if (hit) return JSON.parse(hit);
  } catch {
  }
  const inStudy = await readStudyRows(studyUID, onStatus);
  const want = new Set(wantSeries);
  const IMG = /* @__PURE__ */ new Set(["CT", "MR", "PT", "PET", "NM"]);
  const imgs = inStudy.filter((r) => IMG.has(String(r.Modality).toUpperCase()));
  const segs = inStudy.filter((r) => String(r.Modality).toUpperCase() === "SEG");
  const chosen = imgs.find((r) => want.has(r.SeriesInstanceUID)) ?? imgs.slice().sort((a, b) => Number(b.instanceCount) - Number(a.instanceCount))[0];
  if (!chosen) throw new Error("no CT/MR/PET/NM image series found in this study");
  const seg = segs.find((r) => want.has(r.SeriesInstanceUID)) ?? segs[0];
  const mod = String(chosen.Modality).toUpperCase();
  const src = {
    c: String(chosen.crdc_series_uuid),
    cb: String(chosen.aws_bucket),
    s: seg ? String(seg.crdc_series_uuid) : void 0,
    sb: seg ? String(seg.aws_bucket) : void 0,
    m: mod === "PET" ? "PT" : mod,
    col: String(chosen.collection_id),
    st: studyUID,
    sd: `${chosen.PatientID} \xB7 ${chosen.SeriesDescription || mod}`,
    lic: `${chosen.license_short_name || "IDC"} \xB7 ${chosen.collection_id}${chosen.source_DOI ? " \xB7 doi:" + chosen.source_DOI : ""}`
  };
  try {
    localStorage.setItem(memoKey, JSON.stringify(src));
  } catch {
  }
  return src;
}
var status = (msg, err = false) => {
  const el = document.getElementById("status");
  if (el) {
    el.textContent = msg;
    el.style.color = err ? "#ff6b74" : "#9fb3d0";
  }
};
var cvEl = (id) => document.getElementById(id);
var PLANES = ["axial", "coronal", "sagittal"];
var SEG_FILL = 0.5;
var CT_WL_PRESETS = [
  { name: "Soft Tissue", win: 400, lev: 40 },
  { name: "Lung", win: 1500, lev: -600 },
  { name: "Bone", win: 1800, lev: 400 },
  { name: "Brain", win: 80, lev: 40 },
  { name: "Abdomen", win: 350, lev: 50 }
];
async function main() {
  if (!navigator.gpu) {
    status("WebGPU not available \u2014 try Chrome/Edge 113+ or Safari 18+.", true);
    return;
  }
  status("loading KiTS case from the NCI Imaging Data Commons\u2026");
  const gpu = await initDevice();
  const preferred = navigator.gpu.getPreferredCanvasFormat();
  const srgb = preferred + "-srgb";
  const names = ["axial", "threeD", "coronal", "sagittal"];
  const cv = {}, cx = {};
  for (const n of names) {
    cv[n] = cvEl("c-" + n);
    cx[n] = cv[n].getContext("webgpu");
    cx[n].configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
  }
  const mosaic = createMosaic(document.getElementById("viewer"));
  mosaic.status("contacting the NCI Imaging Data Commons\u2026");
  const setLoad = (m) => {
    status(m);
    mosaic.status(m);
  };
  const source = await resolveSource(setLoad);
  const res = await loadSeries(source, {
    onProgress: (p) => {
      setLoad(`${p.msg}${p.frac ? ` \u2014 ${Math.round(p.frac * 100)}%` : ""}`);
    },
    onSliceCount: (n) => mosaic.setCount(n),
    onThumb: (n, w, h, rgba) => mosaic.thumb(n, w, h, rgba)
  });
  const sc = buildSegrouletteScene(gpu, srgb, res.ct, res.seg);
  sc.setVolumeOpacity(0.5);
  let sliceOutline = false;
  let roiEnabled = false, roiVisible = false, roiFirstEnable = true;
  let annOn = true;
  const annEls = {};
  const buildAnnotations = () => {
    for (const p of PLANES) {
      const cell = document.querySelector(`[data-cell="${p}"]`);
      if (!cell) continue;
      const mk = (cls, html) => {
        const e = document.createElement("div");
        e.className = "ann " + cls;
        e.innerHTML = html;
        e.style.cssText = "position:absolute;z-index:2;font:600 10px ui-monospace,Menlo,monospace;color:#cfe0f5;text-shadow:0 0 3px #000,0 0 3px #000;pointer-events:none;line-height:1.35;max-width:46%;" + (cls === "tl" ? "top:22px;left:7px;" : cls === "tr" ? "top:22px;right:7px;text-align:right;" : "bottom:22px;right:7px;text-align:right;");
        cell.appendChild(e);
        return e;
      };
      const pid = (source.sd.split("\xB7")[0] || "").trim() || source.col;
      mk("tl", `${pid}<br>${source.col} \xB7 ${res.ct.modality}`);
      mk("tr", `${res.ct.modality} \xB7 ${sc.dims[2]} slices${sc.hasSeg ? " \xB7 SEG" : ""}<br>${sc.dims[0]}\xD7${sc.dims[1]}`);
      annEls[p] = { br: mk("br", "") };
    }
    updateAnnWL();
  };
  const updateAnnWL = () => {
    for (const p of PLANES) {
      if (annEls[p]) annEls[p].br.innerHTML = `W ${Math.round(wl.win)} / L ${Math.round(wl.lev)}`;
    }
  };
  const applyAnn = () => {
    for (const e of document.querySelectorAll(".ann")) e.style.display = annOn ? "" : "none";
  };
  const off = {
    axial: slicerDefaultOffset01("axial", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
    coronal: slicerDefaultOffset01("coronal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
    sagittal: slicerDefaultOffset01("sagittal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi)
  };
  const sliceIx = new SliceInteractor({ ijkToRAS: sc.ijkToRAS, rasLo: sc.rasLo, rasHi: sc.rasHi });
  const camera = framedCamera(sc.center, sc.radius);
  const wl = { win: sc.win, lev: sc.lev };
  let bir = null;
  let xhair = null;
  const drawSlice = (p) => {
    bir?.drawOverlay(p);
    if (!cv[p].width) return;
    sc.slice.setPlane(p, off[p]);
    sc.slice.renderToView(cx[p].getCurrentTexture().createView({ format: srgb }), cv[p].width, cv[p].height);
    updateReadout(p);
  };
  const draw3d = () => {
    if (!cv.threeD.width) return;
    sc.scene.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, cv.threeD.width, cv.threeD.height);
    sc.scene.renderToView(cx.threeD.getCurrentTexture().createView({ format: srgb }), cv.threeD.width, cv.threeD.height);
  };
  const drawAll = () => {
    for (const p of PLANES) drawSlice(p);
    draw3d();
    xhair?.redraw();
  };
  const updateReadout = (p) => {
    const el = document.getElementById("sr-" + p);
    if (!el) return;
    const [lo, hi] = sliceIx.bounds(p);
    const sp = sliceIx.spacing(p);
    const mm = offset01ToMm(p, off[p], sc.rasLo, sc.rasHi);
    el.textContent = `${Math.round((mm - lo) / sp + 0.5)}/${Math.round((hi - lo) / sp)} \xB7 ${mm.toFixed(1)} mm`;
  };
  const resize = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const n of names) {
      cv[n].width = Math.max(1, Math.round(cv[n].clientWidth * dpr));
      cv[n].height = Math.max(1, Math.round(cv[n].clientHeight * dpr));
    }
    bir?.resize();
    drawAll();
  };
  globalThis.addEventListener("resize", resize);
  for (const n of names) new ResizeObserver(() => resize()).observe(cv[n]);
  const viewGrid = attachViewGrid(document.getElementById("grid"), [...names], resize);
  attachDoubleClick(cv.threeD, () => viewGrid.toggleMax("threeD"));
  const setWL = (win, lev) => {
    wl.win = win;
    wl.lev = lev;
    sc.slice.setWindowLevel(win, lev);
    for (const p of PLANES) drawSlice(p);
    xhair?.redraw();
    syncWl();
    updateAnnWL();
  };
  const presets = document.getElementById("wl-readout");
  const syncWl = () => {
    presets.innerHTML = "";
    presets.add(new Option(`W ${Math.round(wl.win)} / L ${Math.round(wl.lev)}`, "current", true, true));
    presets.add(new Option(`Auto (${Math.round(sc.win)}/${Math.round(sc.lev)})`, "auto"));
    for (const q of CT_WL_PRESETS) presets.add(new Option(`${q.name} (${q.win}/${q.lev})`, q.name));
  };
  presets.addEventListener("change", () => {
    if (presets.value === "auto") setWL(sc.win, sc.lev);
    else {
      const q = CT_WL_PRESETS.find((x) => x.name === presets.value);
      if (q) setWL(q.win, q.lev);
    }
  });
  syncWl();
  const nAxisOf = { axial: 2, coronal: 1, sagittal: 0 };
  const jumpAll = (ras) => {
    for (const p of PLANES) {
      const a = nAxisOf[p];
      off[p] = Math.max(0, Math.min(1, (ras[a] - sc.rasLo[a]) / (sc.rasHi[a] - sc.rasLo[a])));
    }
    drawAll();
  };
  const sliceControls = [];
  const listAllStudySeries = source.st ? async () => {
    const rows = await readStudyRows(source.st, () => {
    });
    return rows.map((r) => ({
      prefix: String(r.crdc_series_uuid),
      bucket: String(r.aws_bucket),
      modality: String(r.Modality).toUpperCase(),
      seriesUID: String(r.SeriesInstanceUID),
      seriesDescription: r.SeriesDescription ? String(r.SeriesDescription) : void 0
    }));
  } : void 0;
  bir = mountBir({
    overlay: document.getElementById("viewer"),
    bar: document.getElementById("bir-bar"),
    grid: document.getElementById("grid"),
    planes: PLANES,
    canvases: cv,
    cellOf: (name) => document.querySelector(`[data-cell="${name}"]`),
    slice: () => sc.slice,
    off01: (p) => off[p],
    setOff01: (p, v) => off[p] = Math.max(0, Math.min(1, v)),
    offsetMm: (p) => offset01ToMm(p, off[p], sc.rasLo, sc.rasHi),
    spacing: (p) => sliceIx.spacing(p),
    step: (p, fwd) => off[p] = sliceIx.wheel(p, off[p], fwd),
    redraw: (p) => {
      drawSlice(p);
      xhair?.redraw();
    },
    redrawAll: drawAll,
    rasLo: sc.rasLo,
    rasHi: sc.rasHi,
    wl: { get: () => [wl.win, wl.lev], set: setWL, auto: [sc.win, sc.lev] },
    presetsEl: presets,
    resetViews: () => sliceControls.forEach((c) => c.resetView()),
    close: () => status("This is the SlicerLive Basic Image Review demo \u2014 reload to restart."),
    jumpAll,
    modality: res.ct.modality,
    extraTools: [
      {
        id: "idc-share",
        icon: "share",
        title: "Share \u2014 copy a link that reopens this study here (or in the IDC OHIF portal)",
        run: () => shareStudy(source)
      },
      {
        id: "idc-download",
        icon: "download",
        title: "Download this study's DICOM to a local folder (streamed, parallel)",
        run: () => downloadStudyWithDialog(source, listAllStudySeries)
      }
    ]
  });
  globalThis.__birDbg = {
    ready: () => !!bir,
    dims: () => sc.dims,
    tool: () => bir?.tool(),
    cellOrder: () => [...document.querySelectorAll("#grid .cell")].map((c) => c.dataset.cell),
    // test hooks
    setRoi: (en, vis) => {
      roiEnabled = en;
      roiVisible = vis;
      if (en) roiFirstEnable = false;
      sc.setRoiEnabled(en);
      sc.setRoiVisible(vis);
      draw3d();
    },
    roiState: () => ({ visible: sc.roiVisible(), handles: sc.roi.handleList().length }),
    setSegOpacity: (o) => {
      sc.setSegOpacity(o);
      sc.slice.setOverlayOpacity(o * SEG_FILL);
      drawAll();
    }
  };
  for (const p of PLANES) {
    const hooks = bir.hooks(p);
    sliceControls.push(attachSliceControls(cv[p], {
      orient: p,
      getSlice: () => sc.slice,
      step: (fwd) => off[p] = sliceIx.wheel(p, off[p], fwd),
      redraw: () => {
        drawSlice(p);
        xhair?.redraw();
      },
      wl: {
        enabled: () => bir.tool() === "wl",
        get: () => [wl.win, wl.lev],
        set: setWL,
        range: () => res.ct.range,
        reset: () => setWL(sc.win, sc.lev)
      },
      leftMode: () => bir.leftMode(),
      hooks: {
        onLeftGrab: hooks.onLeftGrab,
        onDoubleClick: () => {
          viewGrid.toggleMax(p);
          return true;
        }
      }
    }));
  }
  attachCameraControls(cv.threeD, camera, { onChange: () => {
    draw3d();
    xhair?.redraw();
  } });
  let roiBox0 = null;
  attachWidgetControls(cv.threeD, camera, {
    getHandles: () => sc.roiVisible() ? sc.roi.handleList().map((h) => ({ id: h.id, world: h.world, data: h.data, cursor: h.cursor })) : [],
    getSize: () => ({ w: cv.threeD.width, h: cv.threeD.height }),
    onDragStart: () => {
      roiBox0 = sc.roi.snapshot();
    },
    onDrag: (h, world) => {
      if (!roiBox0) return;
      const d = [world[0] - h.world[0], world[1] - h.world[1], world[2] - h.world[2]];
      sc.roi.applyDrag(h.data, roiBox0, d);
      sc.reclip();
      draw3d();
      xhair?.redraw();
    },
    onHover: (h) => {
      sc.roi.setHover(h ? h.id : null);
      sc.scene.syncUniforms();
      draw3d();
    },
    onChange: () => draw3d()
  });
  xhair = mountCrosshair({
    cells: { axial: cv.axial, coronal: cv.coronal, sagittal: cv.sagittal, threeD: cv.threeD },
    getScene: () => sc.scene,
    getSlice: () => sc.slice,
    getCamera: () => camera,
    getOffset: (o) => off[o],
    onJump: jumpAll
  });
  let vrPreset = null;
  let vrShift = 0;
  const vd = res.ct.vol;
  let vmin = Infinity, vmax = -Infinity;
  const vstride = Math.max(1, Math.floor(vd.length / 2e6));
  for (let i = 0; i < vd.length; i += vstride) {
    const v = vd[i];
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  const shiftRange = Math.max(200, (vmax - vmin) / 2);
  const bakeOf = (name) => name ? presetLUT(CT_VR_PRESETS.find((p) => p.name === name)) : null;
  const presetLabel = () => vrPreset ? CT_VR_PRESETS.find((p) => p.name === vrPreset)?.label ?? vrPreset : "Default (W/L)";
  const applyVrPreset = (name) => {
    vrPreset = name;
    sc.setVolumePreset(bakeOf(name));
    if (sc.volumeOpacity() <= 0) sc.setVolumeOpacity(1);
    draw3d();
    xhair?.redraw();
  };
  const applyVrShift = (hu) => {
    vrShift = hu;
    sc.setVolumeShift(hu);
    draw3d();
    xhair?.redraw();
  };
  const renderPresetThumbnails = () => {
    const THUMB = 116;
    const savedOp = sc.volumeOpacity(), savedShift = sc.volumeShift(), savedSeg = sc.segOpacity();
    sc.setVolumeShift(0);
    if (savedSeg > 0) sc.setSegOpacity(0);
    if (savedOp < 1) sc.setVolumeOpacity(1);
    const entries = [
      { name: null, label: "Default (W/L)" },
      ...CT_VR_PRESETS.map((p) => ({ name: p.name, label: p.label }))
    ];
    const items = [];
    for (const e of entries) {
      const c = document.createElement("canvas");
      c.width = THUMB;
      c.height = THUMB;
      const cxt = c.getContext("webgpu");
      cxt.configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
      sc.setVolumePreset(bakeOf(e.name));
      sc.scene.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, THUMB, THUMB);
      sc.scene.renderToView(cxt.getCurrentTexture().createView({ format: srgb }), THUMB, THUMB);
      items.push({ name: e.name, label: e.label, canvas: c });
    }
    sc.setVolumePreset(bakeOf(vrPreset));
    sc.setVolumeShift(savedShift);
    if (savedSeg > 0) sc.setSegOpacity(savedSeg);
    if (savedOp < 1) sc.setVolumeOpacity(savedOp);
    draw3d();
    return items;
  };
  Object.assign(globalThis.__birDbg, {
    presets: () => CT_VR_PRESETS.map((p) => p.name),
    setPreset: (name) => applyVrPreset(name),
    setShift: (hu) => applyVrShift(hu),
    vrState: () => ({ preset: vrPreset, shift: vrShift, shiftRange, dataRange: [vmin, vmax] }),
    thumbCount: () => renderPresetThumbnails().length
  });
  const controls = [
    {
      label: "Volume render",
      getOpacity: () => sc.volumeOpacity(),
      setOpacity: (o) => {
        sc.setVolumeOpacity(o);
        draw3d();
        xhair?.redraw();
      },
      color: [0.75, 0.78, 0.85]
    },
    // Slicer's VR "Shift" — a plain slider directly below the opacity control.
    {
      label: "Shift",
      slider: {
        min: -shiftRange,
        max: shiftRange,
        step: Math.max(1, Math.round(shiftRange / 200)),
        get: () => vrShift,
        set: (v) => applyVrShift(v),
        format: (v) => (v > 0 ? "+" : "") + Math.round(v)
      },
      disabled: () => sc.volumeOpacity() <= 0
    },
    // Volume-rendering preset — opens the on-the-fly thumbnail menu (Slicer/OHIF CT presets).
    {
      label: "Preset",
      button: { text: () => presetLabel(), run: () => openVrPresetMenu({ items: renderPresetThumbnails(), current: vrPreset, onPick: applyVrPreset }) }
    },
    {
      label: "Segmentation",
      getOpacity: () => sc.segOpacity(),
      setOpacity: (o) => {
        sc.setSegOpacity(o);
        sc.slice.setOverlayOpacity(o * SEG_FILL);
        drawAll();
      },
      disabled: () => !sc.hasSeg,
      color: [0.62, 0.9, 1]
    },
    // Segmentation overlay: fill (default) vs outline on the MPR planes.
    {
      label: "Segmentation outline",
      get: () => sliceOutline,
      set: (on) => {
        sliceOutline = on;
        sc.slice.setOverlayOutline(on);
        for (const p of PLANES) drawSlice(p);
        xhair?.redraw();
      },
      disabled: () => !sc.hasSeg
    },
    // 3D volume-render crop: enabling it reveals the ROI box the first time (Slicer behaviour).
    {
      label: "Crop volume (3D)",
      get: () => roiEnabled,
      set: (on) => {
        roiEnabled = on;
        if (on && roiFirstEnable) {
          roiVisible = true;
          roiFirstEnable = false;
        }
        sc.setRoiEnabled(roiEnabled);
        sc.setRoiVisible(roiVisible);
        draw3d();
        xhair?.redraw();
      },
      disabled: () => sc.volumeOpacity() <= 0
    },
    {
      label: "Show ROI box",
      get: () => roiVisible,
      set: (on) => {
        roiVisible = on;
        sc.setRoiVisible(on);
        draw3d();
        xhair?.redraw();
      }
    },
    // DICOM corner annotations (IHE BIR) — the "official radiology look".
    {
      label: "Corner annotations",
      get: () => annOn,
      set: (on) => {
        annOn = on;
        applyAnn();
      }
    }
  ];
  installChrome({
    controls,
    anchor: cv.threeD.parentElement ?? void 0,
    segments: {
      list: () => sc.segments.map((s) => ({ num: s.num, name: s.name, color: s.color })),
      get: (num) => sc.segmentOpacity(num),
      set: (num, o) => {
        sc.setSegmentOpacity(num, o);
        drawAll();
      },
      enabled: () => sc.hasSeg
    }
  });
  buildAnnotations();
  applyAnn();
  const info = document.getElementById("info");
  if (info) info.textContent = `${source.sd} \xB7 ${res.ct.modality} ${sc.dims.join("\xD7")} \xB7 ${source.lic}`;
  resize();
  mosaic.done();
  status("KiTS abdomen CT \u2014 scroll to page slices, pick a tool from the toolbar, drag 3D to orbit");
}
main().catch((e) => status("error: " + (e?.message ?? e), true));
