// render/device.ts
async function initDevice() {
  const gpu2 = navigator.gpu;
  if (!gpu2) throw new Error("WebGPU not available (need Chrome/Edge/Safari or Deno --unstable-webgpu)");
  const adapter = await gpu2.requestAdapter({ powerPreference: "high-performance" });
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

// render/scene-renderer.ts
var DEFAULT_FORMAT = "rgba8unorm-srgb";
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
  constructor(gpu2, format = DEFAULT_FORMAT) {
    this.dev = gpu2.device;
    this.format = format;
    this.canTime = gpu2.features.has("timestamp-query");
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
  async traceSamples(width, height) {
    this.flush();
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

// render/slice-renderer.ts
var DEFAULT_FORMAT2 = "rgba8unorm-srgb";
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
function slicerDefaultOffset01(orient, dims, ijkToRAS, rasLo2, rasHi2) {
  const b = BASES[orient];
  const n = b.nAxis;
  const a = ijkAxisForRasAxis(ijkToRAS, n);
  const m = Math.floor((dims[a] - 1) / 2);
  const ijk = [(dims[0] - 1) / 2, (dims[1] - 1) / 2, (dims[2] - 1) / 2];
  ijk[a] = m;
  const ras = ijkToRAS[n * 4 + 0] * ijk[0] + ijkToRAS[n * 4 + 1] * ijk[1] + ijkToRAS[n * 4 + 2] * ijk[2] + ijkToRAS[n * 4 + 3];
  const span = rasHi2[n] - rasLo2[n];
  return span === 0 ? 0.5 : (ras - rasLo2[n]) / span;
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
  constructor(gpu2, format = DEFAULT_FORMAT2) {
    this.dev = gpu2.device;
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
  setVolume(p2t, rasLo2, rasHi2) {
    this.p2t = p2t;
    this.rasLo = rasLo2;
    this.rasHi = rasHi2;
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
  /** Fitted (zoom=1) in-plane extent for an orientation. */
  baseSpan(orient) {
    const b = BASES[orient];
    return Math.max(this.rasHi[b.uAxis] - this.rasLo[b.uAxis], this.rasHi[b.vAxis] - this.rasLo[b.vAxis]);
  }
  /** The complete in-plane view frame for an orientation at a given viewport aspect, folding
   *  in pan (mm along uDir/vDir) + zoom. Single source of truth shared by drawInto, rasToView,
   *  viewToRas — so the rendered image and the markup projection stay pixel-aligned under
   *  pan/zoom. Returns the plane centre `c` (RAS, incl. scrub offset + pan) and the half-... no:
   *  uS/vS are the FULL in-plane extents mapped across the viewport width/height. */
  frameFor(orient, offset01, aspectWH) {
    const b = BASES[orient];
    const vs = this.viewState[orient];
    const span = this.baseSpan(orient) / vs.zoom;
    const uS = span * Math.max(1, aspectWH), vS = span * Math.max(1, 1 / aspectWH);
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
    const span = this.baseSpan(orient) / this.viewState[orient].zoom;
    const uS = span * Math.max(1, w / h), vS = span * Math.max(1, h / w);
    this.viewState[orient].panU -= dxPx / w * uS;
    this.viewState[orient].panV += dyPx / h * vS;
  }
  /** Zoom by `factor` (>1 zooms in) about a pivot (u,v in [0,1]); the pivot point stays fixed. */
  zoomAbout(orient, factor, pu, pv, w, h) {
    const vs = this.viewState[orient];
    const base = this.baseSpan(orient);
    const spanOld = base / vs.zoom;
    const z = Math.max(0.2, Math.min(50, vs.zoom * factor));
    const spanNew = base / z;
    const au = Math.max(1, w / h), av = Math.max(1, h / w);
    vs.panU += (pu - 0.5) * (spanOld - spanNew) * au;
    vs.panV += (0.5 - pv) * (spanOld - spanNew) * av;
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
  constructor(dev, data, dims, spacing, lut, opts) {
    const center = opts.center ?? [0, 0, 0];
    this.volTex = dev.createTexture({ size: dims, dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    dev.queue.writeTexture({ texture: this.volTex }, data, { bytesPerRow: dims[0] * 4, rowsPerImage: dims[1] }, dims);
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
  fillUniforms(out, off2) {
    out.set(this.p2t, off2);
    out[off2 + 16] = this.clim[0];
    out[off2 + 17] = this.clim[1];
    out[off2 + 20] = this.shade[0];
    out[off2 + 21] = this.shade[1];
    out[off2 + 22] = this.shade[2];
    out[off2 + 23] = this.shade[3];
    out[off2 + 24] = this.unit;
  }
  bindEntries(_s, base) {
    return [
      { binding: base, resource: this.volTex.createView() },
      { binding: base + 1, resource: this.lutTex.createView() }
    ];
  }
};

// render/cine-field.ts
function transformedAABB2(m, lo, hi) {
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
var _f32 = new Float32Array(1);
var _u32 = new Uint32Array(_f32.buffer);
function f32tof16(v) {
  _f32[0] = v;
  const u = _u32[0];
  const sign = u >>> 16 & 32768;
  let exp = (u >>> 23 & 255) - 127 + 15;
  const man = u & 8388607;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 31744;
  return sign | exp << 10 | man >>> 13;
}
function toF16Array(src) {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = f32tof16(src[i]);
  return out;
}
var CineField = class {
  kind = "cine";
  bindingCount = 3;
  // volA (3d) + volB (3d) + lut (2d)
  texes = [];
  filled = [];
  dims = [0, 0, 0];
  lutTex;
  dev;
  p2t;
  clim;
  shade;
  unit;
  stepMm;
  box;
  a = 0;
  // index of frame A
  b = 0;
  // index of frame B
  blend = 0;
  // 0 => pure A
  /** Allocates `frameCount` empty frame textures; fill them with setFrameData(i, data) as
   *  they arrive. Progressive loading matters: the first phase can be shown (and the scene
   *  built) after ~1 MB instead of waiting for the whole sequence. Frames not yet supplied
   *  read as zero, so always show a frame you have actually filled. */
  constructor(dev, frames, dims, lut, opts) {
    const count = typeof frames === "number" ? frames : frames.length;
    if (!count) throw new Error("CineField needs at least one frame");
    const size = dims;
    this.dims = dims;
    for (let i = 0; i < count; i++) {
      this.texes.push(dev.createTexture({
        size,
        dimension: "3d",
        format: "r16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      }));
    }
    this.filled = new Array(count).fill(false);
    this.dev = dev;
    if (typeof frames !== "number") frames.forEach((f, i) => this.setFrameData(i, f));
    this.lutTex = dev.createTexture({ size: [256, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    dev.queue.writeTexture({ texture: this.lutTex }, lut, { bytesPerRow: 256 * 4 }, [256, 1]);
    this.p2t = patientToTextureFromIjkToRAS(opts.ijkToRAS, dims);
    this.box = volumeAABBFromIjkToRAS(opts.ijkToRAS, dims);
    this.stepMm = Math.min(...spacingFromIjkToRAS(opts.ijkToRAS));
    this.clim = opts.clim;
    this.shade = opts.shade ?? [0.25, 0.75, 0.5, 24];
    this.unit = opts.opacityUnitDistance ?? this.stepMm;
  }
  /** Upload one phase's voxels (C-order z,y,x) into its preallocated texture. */
  setFrameData(i, data) {
    const [nx, ny, nz] = this.dims;
    this.dev.queue.writeTexture(
      { texture: this.texes[i] },
      toF16Array(data),
      { bytesPerRow: nx * 2, rowsPerImage: ny },
      [nx, ny, nz]
    );
    this.filled[i] = true;
  }
  /** True once setFrameData has been called for this phase. */
  hasFrame(i) {
    return !!this.filled[i];
  }
  get framesLoaded() {
    return this.filled.reduce((n, f) => n + (f ? 1 : 0), 0);
  }
  get frameCount() {
    return this.texes.length;
  }
  get frame() {
    return this.a;
  }
  /** Select the displayed frame. Fractional values interpolate toward the next frame.
   *  Caller then does scene.refreshBindings() (bind group only — no pipeline rebuild)
   *  and scene.syncUniforms() for the blend weight. */
  setFrame(t, loop = true) {
    const n = this.texes.length;
    const wrap = (i) => loop ? (i % n + n) % n : Math.max(0, Math.min(n - 1, i));
    this.a = wrap(Math.floor(t));
    this.b = wrap(Math.floor(t) + 1);
    this.blend = t - Math.floor(t);
  }
  setLUT(lut) {
    this.dev.queue.writeTexture({ texture: this.lutTex }, lut, { bytesPerRow: 256 * 4 }, [256, 1]);
  }
  origP2t;
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
  /** The currently displayed frame's texture (e.g. to share with a SliceRenderer for MPR). */
  volumeTexture() {
    return this.texes[this.a];
  }
  worldCenter() {
    const [lo, hi] = this.origBox ?? this.box;
    return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  }
  setWorldTransform(m) {
    if (!this.origP2t) {
      this.origP2t = this.p2t;
      this.origBox = this.box;
    }
    this.p2t = multiply(this.origP2t, invert(m));
    this.box = transformedAABB2(m, this.origBox[0], this.origBox[1]);
  }
  patientToTexture() {
    return this.p2t;
  }
  structMembers(s) {
    return [
      `  cine${s}_p2t : mat4x4<f32>,`,
      `  cine${s}_clim : vec4<f32>,`,
      // lo, hi, _, _
      `  cine${s}_shade : vec4<f32>,`,
      // ka, kd, ks, shininess
      `  cine${s}_params : vec4<f32>,`
      // opacity_unit_distance, blend, _, _
    ].join("\n");
  }
  declareBindings(s, base) {
    return [
      `@group(0) @binding(${base}) var t_volA_cine${s} : texture_3d<f32>;`,
      `@group(0) @binding(${base + 1}) var t_volB_cine${s} : texture_3d<f32>;`,
      `@group(0) @binding(${base + 2}) var t_lut_cine${s} : texture_2d<f32>;`
    ].join("\n");
  }
  // Mirrors ImageField.samplingWGSL exactly, except the scalar is a lerp of two frames.
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn sampc_cine${s}(wp : vec3<f32>) -> f32 {
  let t4 = u_material.cine${s}_p2t * vec4<f32>(transform_point_cine${s}(wp), 1.0);
  let tex = clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  let va = textureSampleLevel(t_volA_cine${s}, s_lin, tex, 0.0).r;
  let vb = textureSampleLevel(t_volB_cine${s}, s_lin, tex, 0.0).r;
  return mix(va, vb, u_material.cine${s}_params.y);
}
fn sample_field_cine${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let t4 = u_material.cine${s}_p2t * vec4<f32>(transform_point_cine${s}(wp), 1.0);
  let tex = t4.xyz;
  if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) { return vec4<f32>(0.0); }
  let va = textureSampleLevel(t_volA_cine${s}, s_lin, tex, 0.0).r;
  let vb = textureSampleLevel(t_volB_cine${s}, s_lin, tex, 0.0).r;
  let val = mix(va, vb, u_material.cine${s}_params.y);
  let lo = u_material.cine${s}_clim.x; let hi = u_material.cine${s}_clim.y;
  let tf = textureSampleLevel(t_lut_cine${s}, s_lin, vec2<f32>(clamp((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0), 0.5), 0.0);
  let step = u_material.scene.x;
  let unit = max(u_material.cine${s}_params.x, 1e-3);
  let opacity = clamp(1.0 - pow(1.0 - clamp(tf.a, 0.0, 1.0), step / unit), 0.0, 1.0);
  if (opacity <= 0.001) { return vec4<f32>(0.0); }
  let h = step * 2.0;
  let g = vec3<f32>(
    sampc_cine${s}(wp + vec3<f32>(h,0,0)) - sampc_cine${s}(wp - vec3<f32>(h,0,0)),
    sampc_cine${s}(wp + vec3<f32>(0,h,0)) - sampc_cine${s}(wp - vec3<f32>(0,h,0)),
    sampc_cine${s}(wp + vec3<f32>(0,0,h)) - sampc_cine${s}(wp - vec3<f32>(0,0,h))) / (2.0 * h);
  let glen = length(g);
  let ka = u_material.cine${s}_shade.x; let kd = u_material.cine${s}_shade.y;
  let ks = u_material.cine${s}_shade.z; let sh = u_material.cine${s}_shade.w;
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
  fillUniforms(out, off2) {
    out.set(this.p2t, off2);
    out[off2 + 16] = this.clim[0];
    out[off2 + 17] = this.clim[1];
    out[off2 + 20] = this.shade[0];
    out[off2 + 21] = this.shade[1];
    out[off2 + 22] = this.shade[2];
    out[off2 + 23] = this.shade[3];
    out[off2 + 24] = this.unit;
    out[off2 + 25] = this.blend;
  }
  bindEntries(_s, base) {
    return [
      { binding: base, resource: this.texes[this.a].createView() },
      { binding: base + 1, resource: this.texes[this.b].createView() },
      { binding: base + 2, resource: this.lutTex.createView() }
    ];
  }
};

// render/sequence.ts
var Sequence = class {
  indexName;
  indexUnit;
  indexType;
  numericIndexValueTolerance;
  items = [];
  constructor(opts = {}) {
    this.indexName = opts.indexName ?? "time";
    this.indexUnit = opts.indexUnit ?? "s";
    this.indexType = opts.indexType ?? "numeric";
    this.numericIndexValueTolerance = opts.numericIndexValueTolerance ?? 1e-3;
  }
  get numberOfDataNodes() {
    return this.items.length;
  }
  getNthIndexValue(i) {
    return this.items[i]?.index;
  }
  getNthDataNode(i) {
    return this.items[i]?.data;
  }
  allItems() {
    return this.items;
  }
  /** Numeric indices insert in sorted position; text indices append (insertion order). */
  insertPosition(index) {
    if (this.indexType !== "numeric") return this.items.length;
    const v = parseFloat(index);
    let i = 0;
    while (i < this.items.length && parseFloat(this.items[i].index) < v) i++;
    return i;
  }
  setDataNodeAtValue(data, index) {
    const at = this.getItemNumberFromIndexValue(index, true);
    if (at >= 0) {
      this.items[at].data = data;
      return;
    }
    this.items.splice(this.insertPosition(index), 0, { index, data });
  }
  getDataNodeAtValue(index, exactMatchRequired = true) {
    const i = this.getItemNumberFromIndexValue(index, exactMatchRequired);
    return i < 0 ? void 0 : this.items[i].data;
  }
  /** -1 if not found. Non-exact numeric lookup returns the item just BEFORE the value
   *  (clamped to the ends), matching vtkMRMLSequenceNode::GetItemNumberFromIndexValue. */
  getItemNumberFromIndexValue(index, exactMatchRequired = true) {
    if (this.indexType !== "numeric") {
      const i = this.items.findIndex((it) => it.index === index);
      return i;
    }
    const v = parseFloat(index);
    const tol = this.numericIndexValueTolerance;
    for (let i = 0; i < this.items.length; i++) {
      if (Math.abs(parseFloat(this.items[i].index) - v) <= tol) return i;
    }
    if (exactMatchRequired) return -1;
    if (!this.items.length) return -1;
    if (v <= parseFloat(this.items[0].index)) return 0;
    if (v >= parseFloat(this.items[this.items.length - 1].index)) return this.items.length - 1;
    let best = 0;
    for (let i = 0; i < this.items.length; i++) {
      if (parseFloat(this.items[i].index) <= v) best = i;
      else break;
    }
    return best;
  }
};
var SequenceBrowser = class {
  sequences = [];
  selectedItemNumber = -1;
  playbackActive = false;
  playbackRateFps = 10;
  playbackLooped = true;
  playbackItemSkippingEnabled = true;
  /** Continuous position, so fractional values can drive inter-frame interpolation. */
  continuousItem = 0;
  lastTimeSec = null;
  get master() {
    return this.sequences[0]?.sequence;
  }
  get numberOfItems() {
    return this.master?.numberOfDataNodes ?? 0;
  }
  addSynchronizedSequence(sequence, apply, opts = {}) {
    const m = this.master;
    if (m && (m.indexName !== sequence.indexName || m.indexUnit !== sequence.indexUnit || m.indexType !== sequence.indexType)) {
      throw new Error(
        `sequence not compatible for browsing: index (${sequence.indexName},${sequence.indexUnit},${sequence.indexType}) != master (${m.indexName},${m.indexUnit},${m.indexType})`
      );
    }
    this.sequences.push({
      sequence,
      apply,
      playback: opts.playback ?? true,
      missingItemMode: opts.missingItemMode ?? "createFromPrevious"
    });
    if (this.selectedItemNumber < 0 && sequence.numberOfDataNodes) this.setSelectedItemNumber(0);
  }
  setSelectedItemNumber(i) {
    this.selectedItemNumber = i;
    this.continuousItem = i;
    this.updateProxies();
  }
  setSelectedItemByIndexValue(index, exactMatchRequired = false) {
    const m = this.master;
    if (!m) return;
    const i = m.getItemNumberFromIndexValue(index, exactMatchRequired);
    if (i >= 0) this.setSelectedItemNumber(i);
  }
  /** Mirrors SelectNextItem: wraps when looped, otherwise stops playback and rewinds. */
  selectNextItem(increment = 1) {
    const n = this.numberOfItems;
    if (n <= 0) return;
    let i = this.selectedItemNumber + increment;
    if (i >= n || i < 0) {
      if (this.playbackLooped) i = (i % n + n) % n;
      else {
        this.playbackActive = false;
        i = increment >= 0 ? 0 : n - 1;
      }
    }
    this.setSelectedItemNumber(i);
  }
  /** Call once per rendered frame with a monotonic clock in seconds. Returns true if the
   *  displayed position changed (i.e. the view needs a redraw / a kick). */
  tick(nowSec, continuous = false) {
    if (!this.playbackActive || this.numberOfItems <= 0) {
      this.lastTimeSec = nowSec;
      return false;
    }
    if (this.lastTimeSec === null) {
      this.lastTimeSec = nowSec;
      return false;
    }
    const elapsed = nowSec - this.lastTimeSec;
    if (continuous) {
      const n = this.numberOfItems;
      let p = this.continuousItem + elapsed * this.playbackRateFps;
      if (p >= n) {
        if (this.playbackLooped) p = p % n;
        else {
          p = n - 1;
          this.playbackActive = false;
        }
      }
      this.lastTimeSec = nowSec;
      this.continuousItem = p;
      this.selectedItemNumber = Math.floor(p);
      this.updateProxies();
      return true;
    }
    const increment = Math.floor(elapsed * this.playbackRateFps + 0.5);
    if (increment <= 0) return false;
    this.lastTimeSec = nowSec;
    this.selectNextItem(this.playbackItemSkippingEnabled ? increment : 1);
    return true;
  }
  /** Fan out the current index to every synchronized sequence (UpdateProxyNodesFromSequences). */
  updateProxies() {
    const m = this.master;
    if (!m) return;
    const indexValue = m.getNthIndexValue(this.selectedItemNumber);
    for (const s of this.sequences) {
      if (!s.playback) continue;
      let item;
      let itemNumber = this.selectedItemNumber;
      if (s.sequence === m) {
        item = m.getNthDataNode(this.selectedItemNumber);
      } else if (indexValue !== void 0) {
        const exact = s.sequence.getItemNumberFromIndexValue(indexValue, true);
        if (exact >= 0) {
          itemNumber = exact;
          item = s.sequence.getNthDataNode(exact);
        } else if (s.missingItemMode === "createFromPrevious") {
          const prev = s.sequence.getItemNumberFromIndexValue(indexValue, false);
          if (prev >= 0) {
            itemNumber = prev;
            item = s.sequence.getNthDataNode(prev);
          }
        } else if (s.missingItemMode === "ignore") continue;
      }
      s.apply(item, itemNumber);
    }
  }
};

// render/zarr.ts
var ZDT = {
  "<f4": Float32Array,
  "<f8": Float64Array,
  "<i4": Int32Array,
  "<u4": Uint32Array,
  "<i2": Int16Array,
  "<u2": Uint16Array,
  "|i1": Int8Array,
  "|u1": Uint8Array,
  "<i1": Int8Array,
  "<u1": Uint8Array
};
async function inflateDeflate(buf) {
  const ds = new DecompressionStream("deflate");
  return await new Response(new Response(buf).body.pipeThrough(ds)).arrayBuffer();
}
async function fetchZarrVolume(blobBase, z, onBytes, concurrency = 12) {
  const Ctor = ZDT[z.dtype] ?? Int16Array;
  const [nz, ny, nx] = z.shape, [cz, cy, cx2] = z.chunks, [ncz, ncy, ncx] = z.chunkGrid;
  const hashes = z.chunkHashes;
  const posBase = blobBase + z.dir + "/" + z.dataset + "/";
  const chunkUrl = (kk, jj, ii) => hashes ? blobBase + hashes[kk + "." + jj + "." + ii] : posBase + kk + "." + jj + "." + ii;
  const out = new Float32Array(nz * ny * nx);
  let lo = Infinity, hi = -Infinity;
  const jobs = [];
  for (let kk = 0; kk < ncz; kk++) for (let jj = 0; jj < ncy; jj++) for (let ii = 0; ii < ncx; ii++) jobs.push([kk, jj, ii]);
  let idx = 0;
  const worker = async () => {
    while (idx < jobs.length) {
      const [kk, jj, ii] = jobs[idx++];
      const resp = await fetch(chunkUrl(kk, jj, ii));
      let gz;
      if (resp.body && onBytes) {
        const parts = [];
        const rd = resp.body.getReader();
        let total = 0;
        for (; ; ) {
          const { done, value } = await rd.read();
          if (done) break;
          parts.push(value);
          total += value.byteLength;
          onBytes(value.byteLength);
        }
        const all = new Uint8Array(total);
        let o = 0;
        for (const p of parts) {
          all.set(p, o);
          o += p.byteLength;
        }
        gz = all.buffer;
      } else {
        gz = await resp.arrayBuffer();
        onBytes?.(gz.byteLength);
      }
      const chunk = new Ctor(await inflateDeflate(gz));
      const z0 = kk * cz, y0 = jj * cy, x0 = ii * cx2;
      const zw = Math.min(cz, nz - z0), yw = Math.min(cy, ny - y0), xw = Math.min(cx2, nx - x0);
      for (let zz = 0; zz < zw; zz++) {
        for (let yy = 0; yy < yw; yy++) {
          const src = (zz * cy + yy) * cx2;
          const dst = ((z0 + zz) * ny + (y0 + yy)) * nx + x0;
          for (let xx = 0; xx < xw; xx++) {
            const v = chunk[src + xx];
            out[dst + xx] = v;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
        }
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, jobs.length) }, worker));
  return { data: out, dims: [nx, ny, nz], range: [lo, hi] };
}

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
  fillUniforms(out, off2) {
    out[off2 + 0] = this.n;
    out[off2 + 1] = 1;
    out[off2 + 2] = this.sh;
    out[off2 + 3] = this.ka;
    out[off2 + 4] = this.kd;
    out[off2 + 5] = this.ks;
    out[off2 + 6] = this.maxR;
    out[off2 + 7] = this.active;
    out[off2 + 8] = this.light[0];
    out[off2 + 9] = this.light[1];
    out[off2 + 10] = this.light[2];
    out.set(this.spheres, off2 + 12);
    out.set(this.colors, off2 + 12 + MAX * 4);
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
  fillUniforms(out, off2) {
    out[off2 + 0] = this.center[0];
    out[off2 + 1] = this.center[1];
    out[off2 + 2] = this.center[2];
    out[off2 + 4] = this.half[0];
    out[off2 + 5] = this.half[1];
    out[off2 + 6] = this.half[2];
    out[off2 + 8] = this.color[0];
    out[off2 + 9] = this.color[1];
    out[off2 + 10] = this.color[2];
    out[off2 + 11] = this.opacity;
    out[off2 + 12] = this.bar;
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
  const moveFace = (axis, sign, box02, deltaAxis) => {
    const opp = box02.center[axis] - sign * box02.half[axis];
    let face = box02.center[axis] + sign * box02.half[axis] + deltaAxis;
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
    applyDrag(meta, box02, delta) {
      if (meta.kind === "center") {
        for (let a = 0; a < 3; a++) center[a] = box02.center[a] + delta[a];
      } else if (meta.kind === "face") {
        const [c, h] = moveFace(meta.axis, meta.sign, box02, delta[meta.axis]);
        center[meta.axis] = c;
        half[meta.axis] = h;
      } else {
        for (let a = 0; a < 3; a++) {
          const [c, h] = moveFace(a, meta.s[a], box02, delta[a]);
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

// examples/cardiac/presets.ts
var CARDIAC_PRESETS = {
  "CT-EndoVascular": {
    shade: [0.1, 0.9, 0.2, 10],
    blurb: "Myocardium opaque, contrast blood transparent above 338 HU \u2014 fly inside the chamber",
    windowLevel: [1400, 300],
    color: [
      [-3024, 0, 0, 0],
      [-77.6875, 0.54902, 0.25098, 0.14902],
      [94.9518, 0.882353, 0.603922, 0.290196],
      [179.052, 1, 0.937033, 0.954531],
      [260.439, 0.615686, 0, 0],
      [3071, 0.827451, 0.658824, 1]
    ],
    scalarOpacity: [
      [-3024, 0],
      [-140.4853515625, 0],
      [94.9518, 0.285714],
      [179.052, 0.553571],
      [260.439, 0.848214],
      [289.798675537109, 0.871428549289703],
      [338.101623535156, 0],
      // <- contrast-opacified blood vanishes here
      [2784.18041992188, 0],
      [2930.0205078125, 0.899999976158142],
      [3071, 0.875]
    ]
  },
  "CT-Cardiac3": {
    shade: [0.1, 0.9, 0.2, 10],
    blurb: "Standard cardiac CT preset \u2014 contrast blood pool opaque",
    windowLevel: [1400, 300],
    color: [
      [-3024, 0, 0, 0],
      [-86.9767, 0, 0.25098, 1],
      [45.3791, 1, 0, 0],
      [139.919, 1, 0.894893, 0.894893],
      [347.907, 1, 1, 0.25098],
      [1224.16, 1, 1, 1],
      [3071, 0.827451, 0.658824, 1]
    ],
    scalarOpacity: [
      [-3024, 0],
      [-86.9767, 0],
      [45.3791, 0.169643],
      [139.919, 0.589286],
      [347.907, 0.607143],
      [1224.16, 0.607143],
      [3071, 0.616071]
    ]
  },
  "CT-Coronary-Arteries-3": {
    shade: [0.1, 0.9, 0.2, 10],
    blurb: "Coronary/vessel emphasis, dark below 129 HU",
    windowLevel: [1e3, 300],
    color: [
      [-2048, 0, 0, 0],
      [128.643, 0, 0, 0],
      [129.982, 0.615686, 0, 0.0156863],
      [173.636, 0.909804, 0.454902, 0],
      [255.884, 0.886275, 0.886275, 0.886275],
      [584.878, 0.968627, 0.968627, 0.968627],
      [3661, 1, 1, 1]
    ],
    scalarOpacity: [
      [-2048, 0],
      [128.643, 0],
      [129.982, 0.0982143],
      [173.636, 0.669643],
      [255.884, 0.857143],
      [584.878, 0.866071],
      [3661, 1]
    ]
  },
  "CT-Chest-Contrast-Enhanced": {
    shade: [0.1, 0.9, 0.2, 10],
    blurb: "Contrast-enhanced chest \u2014 soft tissue and vessels together",
    windowLevel: [1400, 300],
    color: [
      [-3024, 0, 0, 0],
      [67.0106, 0.54902, 0.25098, 0.14902],
      [251.105, 0.882353, 0.603922, 0.290196],
      [439.291, 1, 0.937033, 0.954531],
      [3071, 0.827451, 0.658824, 1]
    ],
    scalarOpacity: [
      [-3024, 0],
      [67.0106, 0],
      [251.105, 0.446429],
      [439.291, 0.625],
      [3071, 0.616071]
    ]
  },
  "MR-Default": {
    shade: [0.2, 1, 0, 1],
    blurb: "Default MR preset (for the HVSMR-2.0 whole-heart MRI path)",
    windowLevel: [500, 250],
    color: [
      [0, 0, 0, 0],
      [20, 0.168627, 0, 0],
      [40, 0.403922, 0.145098, 0.0784314],
      [120, 0.780392, 0.607843, 0.380392],
      [220, 0.847059, 0.835294, 0.788235],
      [1024, 1, 1, 1]
    ],
    scalarOpacity: [
      [0, 0],
      [20, 0],
      [40, 0.15],
      [120, 0.3],
      [220, 0.375],
      [1024, 0.5]
    ]
  }
};
var PRESET_NAMES = Object.keys(CARDIAC_PRESETS);

// examples/cardiac/cardiac-scene.ts
async function loadScene(url) {
  const raw = await (await fetch(url)).json();
  const pageBase = globalThis.location?.href ?? "file:///";
  const sceneAbs = new URL(url, pageBase).href;
  return { nodes: raw.nodes, blobBase: new URL(raw.blobBase ?? "./blobs/", sceneAbs).href };
}
function presetClim(name) {
  const p = CARDIAC_PRESETS[name];
  return [p.color[0][0], p.color[p.color.length - 1][0]];
}
function presetLUT(name) {
  const p = CARDIAC_PRESETS[name];
  const clim = presetClim(name);
  return { lut: lutFromTransferFunctions(p.color, p.scalarOpacity, clim), clim, shade: p.shade };
}
async function buildCardiacScene(gpu2, base, format, onProgress, buildOpts = {}) {
  const dev = gpu2.device;
  const wantCine = buildOpts.only !== "cta";
  const cineScene = wantCine ? await loadScene(base + "cine.json") : { nodes: {}, blobBase: base };
  const seqNode = wantCine ? Object.values(cineScene.nodes).find((n) => n.class === "vtkMRMLSequenceNode") : null;
  const items = seqNode?.attrs.items ?? [{ index: "0", node: "" }];
  const firstVol = wantCine ? cineScene.nodes[items[0].node] : null;
  const cineIjkToRAS = firstVol?.attrs.ijkToRAS ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const z0 = firstVol?.attrs.zarr;
  const cineDims = z0 ? [z0.shape[2], z0.shape[1], z0.shape[0]] : [2, 2, 2];
  const cinePreset = presetLUT("CT-Coronary-Arteries-3");
  const cine = new CineField(dev, items.length, cineDims, cinePreset.lut, {
    clim: cinePreset.clim,
    ijkToRAS: cineIjkToRAS,
    shade: cinePreset.shade
  });
  const report = (what, bytes) => onProgress?.({ bytes, frames: cine.framesLoaded, totalFrames: items.length, what });
  if (wantCine && z0) {
    const zv = await fetchZarrVolume(cineScene.blobBase, z0, (n) => report("cine", n));
    cine.setFrameData(0, zv.data);
    report("cine", 0);
  }
  const cineReady = (async () => {
    if (!wantCine) return;
    for (let i = 1; i < items.length; i++) {
      const vn = cineScene.nodes[items[i].node];
      const zv = await fetchZarrVolume(cineScene.blobBase, vn.attrs.zarr, (n) => report("cine", n));
      cine.setFrameData(i, zv.data);
      report("cine", 0);
    }
  })();
  let cta = null;
  let ctaIjkToRAS = [];
  let ctaDims = [0, 0, 0];
  let ctaPending = null;
  const ensureCta = (onP) => {
    if (cta) return Promise.resolve();
    if (ctaPending) return ctaPending;
    ctaPending = (async () => {
      const ctaScene = await loadScene(base + "cta.json");
      const ctaVol = Object.values(ctaScene.nodes).find((n) => n.class === "vtkMRMLScalarVolumeNode");
      ctaIjkToRAS = ctaVol.attrs.ijkToRAS;
      const zv = await fetchZarrVolume(
        ctaScene.blobBase,
        ctaVol.attrs.zarr,
        (n) => onP?.({ bytes: n, frames: 0, totalFrames: 0, what: "cta" })
      );
      ctaDims = zv.dims;
      const p = presetLUT("CT-EndoVascular");
      cta = new ImageField(dev, zv.data, zv.dims, [1, 1, 1], p.lut, {
        clim: p.clim,
        ijkToRAS: ctaIjkToRAS,
        shade: p.shade
      });
    })();
    return ctaPending;
  };
  const sa = seqNode?.attrs ?? {};
  const sequence = new Sequence({
    indexName: sa.indexName ?? "frame",
    indexUnit: sa.indexUnit ?? "",
    indexType: sa.indexType ?? "numeric",
    numericIndexValueTolerance: sa.numericIndexValueTolerance ?? 1e-3
  });
  items.forEach((it, i) => sequence.setDataNodeAtValue(i, it.index));
  const browser = new SequenceBrowser();
  const brAttrs = Object.values(cineScene.nodes).find((n) => n.class === "vtkMRMLSequenceBrowserNode")?.attrs ?? {};
  browser.playbackRateFps = brAttrs.playbackRateFps ?? 10;
  browser.playbackLooped = brAttrs.playbackLooped ?? true;
  browser.addSynchronizedSequence(sequence, () => {
    cine.setFrame(browser.continuousItem, browser.playbackLooped);
  });
  if (buildOpts.only === "cta") await ensureCta((p) => onProgress?.(p));
  const scene = new SceneRenderer(gpu2, format);
  let mode = buildOpts.only === "cta" ? "cta" : "cine";
  let preset = buildOpts.only === "cta" ? "CT-EndoVascular" : "CT-Coronary-Arteries-3";
  let roi = createRoiWidget(...(mode === "cta" && cta ? cta : cine).aabb(), { coverage: 0.3 });
  let cropOn = false, roiOn = false;
  const rebuild = () => {
    const vol = mode === "cta" && cta ? cta : cine;
    scene.build(roiOn ? [vol, roi.box, roi.handles] : [vol]);
    scene.setBackground(0.05, 0.06, 0.09);
    if (cropOn) scene.setClipBox(roi.lo(), roi.hi());
    else scene.clearClip();
  };
  rebuild();
  const slice = new SliceRenderer(gpu2, format);
  const applySliceVolume = () => {
    const f = mode === "cta" && cta ? cta : cine;
    const [lo, hi] = f.aabb();
    slice.setVolume(f.patientToTexture(), lo, hi);
    slice.setTextures(f.volumeTexture());
    const wl = CARDIAC_PRESETS[preset].windowLevel ?? [1400, 300];
    slice.setWindowLevel(wl[0], wl[1]);
    slice.setOverlayOpacity(0);
  };
  applySliceVolume();
  const bounds = () => {
    const [lo, hi] = (mode === "cta" && cta ? cta : cine).aabb();
    return {
      center: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2],
      radius: Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2
    };
  };
  const b0 = bounds();
  const out = {
    scene,
    slice,
    cine,
    browser,
    get cta() {
      return cta;
    },
    cineReady,
    ensureCta,
    ctaLoaded: () => !!cta,
    center: b0.center,
    radius: b0.radius,
    cineIjkToRAS,
    get ctaDims() {
      return ctaDims;
    },
    cineDims,
    get ctaIjkToRAS() {
      return ctaIjkToRAS;
    },
    mode: () => mode,
    presetName: () => preset,
    roi,
    cropEnabled: () => cropOn,
    roiVisible: () => roiOn,
    setCropEnabled(on) {
      cropOn = on;
      if (on) scene.setClipBox(roi.lo(), roi.hi());
      else scene.clearClip();
    },
    setRoiVisible(on) {
      roiOn = on;
      rebuild();
    },
    setPreset(name) {
      if (!CARDIAC_PRESETS[name]) return;
      preset = name;
      const { lut, clim, shade } = presetLUT(name);
      const f = mode === "cta" && cta ? cta : cine;
      f.setLUT(lut);
      f.clim = clim;
      f.shade = shade;
      scene.syncUniforms();
      applySliceVolume();
    },
    setMode(m) {
      if (m === mode) return;
      mode = m;
      const [lo, hi] = (m === "cta" && cta ? cta : cine).aabb();
      roi = createRoiWidget(lo, hi, { coverage: 0.3 });
      out.roi = roi;
      rebuild();
      const b = bounds();
      out.center = b.center;
      out.radius = b.radius;
      applySliceVolume();
    }
  };
  return out;
}

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
function sliceBoundsFor(orient, rasLo2, rasHi2) {
  const { axis, sign } = NORMAL[orient];
  return sign > 0 ? [rasLo2[axis], rasHi2[axis]] : [-rasHi2[axis], -rasLo2[axis]];
}
function offset01ToMm(orient, offset01, rasLo2, rasHi2) {
  const { axis, sign } = NORMAL[orient];
  return sign * (rasLo2[axis] + offset01 * (rasHi2[axis] - rasLo2[axis]));
}
function mmToOffset01(orient, mm, rasLo2, rasHi2) {
  const { axis, sign } = NORMAL[orient];
  const ras = sign * mm;
  const span = rasHi2[axis] - rasLo2[axis];
  return span === 0 ? 0.5 : (ras - rasLo2[axis]) / span;
}
var SliceInteractor = class {
  constructor(geom) {
    this.geom = geom;
  }
  geom;
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
    const { rasLo: rasLo2, rasHi: rasHi2 } = this.geom;
    const cur = offset01ToMm(orient, offset01, rasLo2, rasHi2);
    const next = cur + deltaMm;
    const [lo, hi] = this.bounds(orient);
    if (next < lo || next > hi) return offset01;
    return mmToOffset01(orient, next, rasLo2, rasHi2);
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
  constructor(camera2, onChange) {
    this.camera = camera2;
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
    const cx2 = width / 2, cy = height / 2;
    const newAngle = Math.atan2(y - cy, x - cx2) * 180 / Math.PI;
    const oldAngle = Math.atan2(py - cy, px - cx2) * 180 / Math.PI;
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
function attachCameraControls(canvas, camera2, opts = {}) {
  const interactor = new CameraInteractor(camera2, opts.onChange);
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
    if (pointers.size >= 2) {
      const p = pinchState();
      if (pinch) {
        if (p.dist > 0 && pinch.dist > 0) camera2.dolly(p.dist / pinch.dist);
        camera2.panByDisplayDelta(p.mx - pinch.mx, pinch.my - p.my, canvas.clientWidth, canvas.clientHeight);
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
    opts.onLog?.("cameraWheel", { deltaY: e.deltaY, distance: camera2.distance });
  }, { passive: false });
  return interactor;
}

// render/endoscopy-control.ts
var norm2 = (v) => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};
var cross2 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
var dot2 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function rotate(v, k, ang) {
  const c = Math.cos(ang), s = Math.sin(ang), d = dot2(k, v);
  const kv = cross2(k, v);
  return [
    v[0] * c + kv[0] * s + k[0] * d * (1 - c),
    v[1] * c + kv[1] * s + k[1] * d * (1 - c),
    v[2] * c + kv[2] * s + k[2] * d * (1 - c)
  ];
}
function attachEndoscopyControls(canvas, camera2, opts = {}) {
  let speed = opts.speedMmPerSec ?? 8;
  const turn = (opts.turnDegPerSec ?? 60) * Math.PI / 180;
  const lookRad = opts.lookRadPerPx ?? 5e-3;
  const refUp = opts.referenceUp ?? [0, 0, 1];
  const focalDist = opts.focalDistanceMm ?? 30;
  const margin = opts.marginMm ?? 6;
  let cruise = "stopped";
  const keys = /* @__PURE__ */ new Set();
  const mods = { shift: false, ctrl: false };
  const forward = () => norm2([
    camera2.focalPoint[0] - camera2.position[0],
    camera2.focalPoint[1] - camera2.position[1],
    camera2.focalPoint[2] - camera2.position[2]
  ]);
  const basis = () => {
    const f = forward();
    const u0 = camera2.viewUp;
    const d = dot2(u0, f);
    let u = norm2([u0[0] - f[0] * d, u0[1] - f[1] * d, u0[2] - f[2] * d]);
    if (!Number.isFinite(u[0])) u = norm2(cross2(f, [0, 0, 1]));
    return { f, u, r: cross2(f, u) };
  };
  const setFrame = (f, u) => {
    const fn = norm2(f);
    const d = dot2(u, fn);
    const un = norm2([u[0] - fn[0] * d, u[1] - fn[1] * d, u[2] - fn[2] * d]);
    camera2.focalPoint = [
      camera2.position[0] + fn[0] * focalDist,
      camera2.position[1] + fn[1] * focalDist,
      camera2.position[2] + fn[2] * focalDist
    ];
    camera2.viewUp = un;
  };
  const yaw = (ang) => {
    const b = basis();
    setFrame(rotate(b.f, b.u, ang), b.u);
  };
  const pitch = (ang) => {
    const b = basis();
    setFrame(rotate(b.f, b.r, ang), rotate(b.u, b.r, ang));
  };
  const roll = (ang) => {
    const b = basis();
    setFrame(b.f, rotate(b.u, b.f, ang));
  };
  const setDirection = (dir) => {
    const f = norm2(dir);
    const u0 = camera2.viewUp;
    const d = dot2(u0, f);
    let u = [u0[0] - f[0] * d, u0[1] - f[1] * d, u0[2] - f[2] * d];
    if (Math.hypot(u[0], u[1], u[2]) < 1e-4) u = cross2(f, refUp);
    setFrame(f, u);
  };
  const setCruise = (c) => {
    if (c === cruise) return;
    cruise = c;
    opts.onState?.(cruise);
  };
  let dragging = false, lastX = 0, lastY = 0;
  const onDown = (e) => {
    if (e.button !== 0) return;
    if (e.shiftKey) return;
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  };
  const onMove = (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (dx) yaw(-dx * lookRad);
    if (dy) pitch(-dy * lookRad);
    if (dx || dy) opts.onLook?.();
    opts.onChange?.();
    e.preventDefault();
  };
  const onUp = (e) => {
    if (!dragging) return;
    dragging = false;
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
    }
  };
  const NAV_KEYS = /* @__PURE__ */ new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " ", "Escape"]);
  const onKeyDown = (e) => {
    if (!NAV_KEYS.has(e.key)) return;
    if (e.key === " ") {
      const want = e.shiftKey ? "back" : "forward";
      setCruise(cruise === want ? "stopped" : want);
      e.preventDefault();
      return;
    }
    if (e.key === "Escape") {
      setCruise("stopped");
      e.preventDefault();
      return;
    }
    keys.add(e.key);
    mods.shift = e.shiftKey;
    mods.ctrl = e.ctrlKey || e.metaKey;
    e.preventDefault();
  };
  const onKeyUp = (e) => {
    keys.delete(e.key);
    mods.shift = e.shiftKey;
    mods.ctrl = e.ctrlKey || e.metaKey;
  };
  const onBlur = () => {
    keys.clear();
    mods.shift = false;
    mods.ctrl = false;
  };
  canvas.style.touchAction = "none";
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  globalThis.addEventListener("keydown", onKeyDown);
  globalThis.addEventListener("keyup", onKeyUp);
  globalThis.addEventListener("blur", onBlur);
  return {
    cruise: () => cruise,
    setCruise,
    lookAlong: (dir) => {
      setDirection(dir);
      opts.onChange?.();
    },
    speed: () => speed,
    setSpeed: (mmPerSec) => {
      speed = Math.max(0.1, mmPerSec);
    },
    tick(dtSec) {
      let moved = false;
      const dt = Math.min(dtSec, 0.1);
      let manualStep = 0;
      if (keys.has("ArrowUp")) manualStep += speed * dt;
      if (keys.has("ArrowDown")) manualStep -= speed * dt;
      const turnAmt = turn * dt;
      if (keys.has("ArrowLeft") || keys.has("ArrowRight")) {
        const sign = keys.has("ArrowLeft") ? 1 : -1;
        if (mods.ctrl) roll(turnAmt * sign);
        else if (mods.shift) pitch(turnAmt * sign);
        else yaw(turnAmt * sign);
        opts.onLook?.();
        moved = true;
      }
      if (manualStep !== 0) {
        const f = forward();
        const dir = manualStep > 0 ? f : [-f[0], -f[1], -f[2]];
        const room = opts.clearance ? opts.clearance(dir) - margin : Infinity;
        const step = Math.max(0, Math.min(Math.abs(manualStep), room));
        if (step > 0) {
          camera2.position = [
            camera2.position[0] + dir[0] * step,
            camera2.position[1] + dir[1] * step,
            camera2.position[2] + dir[2] * step
          ];
          setDirection(f);
          moved = true;
        }
      }
      if (cruise !== "stopped") {
        const f = forward();
        const sign = cruise === "forward" ? 1 : -1;
        const dir = [f[0] * sign, f[1] * sign, f[2] * sign];
        const want = speed * dt;
        const room = opts.clearance ? opts.clearance(dir) - margin : Infinity;
        const step = Math.max(0, Math.min(want, room));
        if (step > 0) {
          camera2.position = [
            camera2.position[0] + dir[0] * step,
            camera2.position[1] + dir[1] * step,
            camera2.position[2] + dir[2] * step
          ];
          setDirection(f);
          moved = true;
        }
      }
      if (moved) opts.onChange?.();
      return moved;
    },
    detach() {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      globalThis.removeEventListener("keydown", onKeyDown);
      globalThis.removeEventListener("keyup", onKeyUp);
      globalThis.removeEventListener("blur", onBlur);
      keys.clear();
    }
  };
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
    if (h.onLeftGrab?.(u, v, w, hh)) {
      grabbed = { moved: 0 };
    } else scroll = { x: e.clientX, y: e.clientY, acc: 0 };
    canvas.setPointerCapture(e.pointerId);
  };
  const onMove = (e) => {
    if (view) {
      const dx = e.clientX - view.x, dy = e.clientY - view.y;
      const r = canvas.getBoundingClientRect();
      if (view.mode === "pan") cfg.getSlice().panByPixels(cfg.orient, dx, dy, r.width, r.height);
      else cfg.getSlice().zoomAbout(cfg.orient, Math.exp(dy * 6e-3), view.pu, view.pv, r.width, r.height);
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
function attachViewGrid(grid2, cells, onResize) {
  let maxed = null;
  const cellDiv = (cell) => grid2.querySelector(`.cell[data-cell="${cell}"]`);
  return {
    toggleMax(cell) {
      maxed = maxed === cell ? null : cell;
      for (const n of cells) cellDiv(n).classList.toggle("max", n === maxed);
      grid2.classList.toggle("has-max", maxed !== null);
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

// render/budget-controller.ts
var BudgetController = class {
  budgetPx;
  targetMs;
  minPx;
  maxPx;
  constructor(opts = {}) {
    this.targetMs = opts.targetMs ?? 16;
    this.minPx = opts.minPx ?? 3e4;
    this.maxPx = opts.maxPx ?? 8e6;
    this.budgetPx = opts.startPx ?? 35e4;
  }
  /** Nudge the budget toward hitting targetMs. Multiplicative, clamped per step (0.8–1.25×) so the
   *  loop is stable, and bounded to [minPx, maxPx]. Faster-than-target grows it; slower shrinks it. */
  update(measuredMs) {
    if (!(measuredMs > 0) || !Number.isFinite(measuredMs)) return;
    const adj = Math.max(0.35, Math.min(1.2, this.targetMs / measuredMs));
    this.budgetPx = Math.max(this.minPx, Math.min(this.maxPx, this.budgetPx * adj));
  }
  /** Resolution scale for a `w×h` view: sqrt(budget / area), clamped to [0.25, 1]. 1 when the view
   *  already fits the budget (small window); a fraction for a big/retina window under load. */
  scale(w, h) {
    const area = Math.max(1, w * h);
    return Math.max(0.25, Math.min(1, Math.sqrt(this.budgetPx / area)));
  }
};

// render/demos/accum-loop.ts
function mountAdaptiveLoop(opts) {
  const target = opts.target ?? 32;
  const idleGap = opts.idleGapMs ?? 120;
  const paced = () => Promise.race([
    new Promise((r) => requestAnimationFrame(() => r())),
    new Promise((r) => setTimeout(r, 33))
  ]);
  const sync = opts.sync ?? (() => Promise.resolve());
  let running = false, stopped = false, lastKick = -1e12, wasMoving = false;
  const step = () => {
    if (performance.now() - lastKick < idleGap) {
      opts.renderMoving();
      wasMoving = true;
      return true;
    }
    if (wasMoving) {
      wasMoving = false;
      opts.renderSettled(true);
      return true;
    }
    if (opts.count() < target) {
      opts.renderSettled(false);
      return true;
    }
    return false;
  };
  const run = async () => {
    running = true;
    stopped = false;
    while (!stopped && step()) await Promise.all([sync(), paced()]);
    running = false;
  };
  return {
    kick() {
      lastKick = performance.now();
      if (!running) run();
    },
    // run() renders the 1st frame synchronously
    stop() {
      stopped = true;
    }
  };
}
function mountAdaptive3d(opts) {
  const budget = new BudgetController({ targetMs: opts.targetMs ?? 16 });
  const DBG = typeof location !== "undefined" && new URLSearchParams(location.search).has("perf");
  let dbgN = 0, dbgMoving = 0, dbgSettled = 0, dbgLast = 0;
  const dbgTick = (kind, ms, s) => {
    if (!DBG) return;
    dbgN++;
    if (kind === "mov") dbgMoving += ms;
    else dbgSettled += ms;
    const now = performance.now();
    if (now - dbgLast > 500) {
      console.log(`[perf] mov=${dbgMoving.toFixed(0)}ms/${dbgN}f settled=${dbgSettled.toFixed(0)}ms lastScale=${s.toFixed(2)} last=${ms.toFixed(1)}ms`);
      dbgLast = now;
      dbgMoving = dbgSettled = dbgN = 0;
    }
  };
  const movingCap = opts.movingScaleCap ?? 1;
  const renderMoving = () => {
    const sc2 = opts.scene();
    if (!sc2) return;
    const { w: vw, h: vh } = opts.size();
    if (!vw || !vh) return;
    const s = Math.min(movingCap, budget.scale(vw, vh)), t0 = performance.now();
    if (s > 0.98) {
      opts.setCamera(sc2, vw, vh);
      sc2.renderToView(opts.view(), vw, vh);
    } else {
      const rw = Math.max(16, Math.round(vw * s)), rh = Math.max(16, Math.round(vh * s));
      opts.setCamera(sc2, rw, rh);
      sc2.renderUpscaled(opts.view(), rw, rh, vw, vh);
    }
    opts.gpu.device.queue.onSubmittedWorkDone().then(() => {
      const ms = performance.now() - t0;
      budget.update(ms);
      dbgTick("mov", ms, s);
    });
    opts.onFrame?.();
  };
  const renderSettled = (reset) => {
    const sc2 = opts.scene();
    if (!sc2) return;
    const { w: vw, h: vh } = opts.size();
    if (!vw || !vh) return;
    const t0 = performance.now();
    opts.setCamera(sc2, vw, vh);
    sc2.renderAccum(opts.view(), vw, vh, reset);
    if (DBG) opts.gpu.device.queue.onSubmittedWorkDone().then(() => dbgTick("set", performance.now() - t0, 1));
    opts.onFrame?.();
  };
  const loop = mountAdaptiveLoop({
    renderMoving,
    renderSettled,
    count: () => opts.scene()?.accumCount() ?? 1e9,
    target: opts.target ?? 24,
    idleGapMs: opts.idleGapMs,
    sync: () => opts.gpu.device.queue.onSubmittedWorkDone()
    // GPU-paced: no backlog, input preempts
  });
  let kickN = 0, kickLast = 0;
  const draw = () => {
    if (DBG) {
      kickN++;
      const now = performance.now();
      if (now - kickLast > 500) {
        console.log(`[perf] kicks=${kickN} in 500ms`);
        kickN = 0;
        kickLast = now;
      }
    }
    loop.kick();
  };
  return { draw, budget, renderSettled, renderMoving, loop };
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

// render/demos/widget-control.ts
var sub2 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
var dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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
function attachWidgetControls(canvas, camera2, opts) {
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
    const n = sub2(camera2.position, camera2.focalPoint);
    const denom = dot3(rd, n);
    if (Math.abs(denom) < 1e-9) return [...planePt];
    const t = dot3(sub2(planePt, ro), n) / denom;
    return [ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t];
  };
  const pick = (e) => {
    const { x, y, rw, rh } = cursorCss(e);
    const { w, h } = opts.getSize();
    const { vp } = camMatrices(camera2, w, h);
    let best = null, bestD = Infinity;
    for (const hnd of opts.getHandles()) {
      const s = project(vp, hnd.world, rw, rh);
      if (!s) continue;
      const d = Math.hypot(s.x - x, s.y - y), r = hnd.pickPx ?? 16;
      if (d < r && d < bestD) {
        bestD = d;
        best = hnd;
      }
    }
    return best;
  };
  let grabbed = null, hovered = null;
  const onDown = (e) => {
    if (e.button !== 0) return;
    const h = pick(e);
    if (!h) return;
    e.stopPropagation();
    e.preventDefault();
    grabbed = h;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = h.cursor ? h.cursor : "grabbing";
    opts.onDragStart?.(h);
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  };
  const onMove = (e) => {
    if (!grabbed) return;
    e.stopPropagation();
    const { x, y, rw, rh } = cursorCss(e);
    const { w, h } = opts.getSize();
    const { invVp } = camMatrices(camera2, w, h);
    const world = unprojectToPlane(invVp, x, y, rw, rh, grabbed.world);
    opts.onDrag(grabbed, world);
    opts.onChange?.();
  };
  const onUp = (e) => {
    if (!grabbed) return;
    e.stopPropagation();
    const g = grabbed;
    grabbed = null;
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
    }
    window.removeEventListener("pointermove", onMove, true);
    window.removeEventListener("pointerup", onUp, true);
    opts.onDragEnd?.(g);
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
    }
  };
}

// render/demos/sl-logo.ts
var SL_LOGO = "data:image/png;base64,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";

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
  const help = opts.help ?? DEFAULT_HELP;
  const helpBtn = document.createElement("button");
  helpBtn.textContent = "?";
  helpBtn.title = "Controls & key bindings";
  helpBtn.style.cssText = "position:fixed;top:12px;left:12px;z-index:74;width:32px;height:32px;padding:0;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;border-radius:50%;color:#cfe6ff;font:700 15px -apple-system,system-ui,sans-serif;";
  glass(helpBtn);
  helpBtn.onclick = openHelp;
  document.body.appendChild(helpBtn);
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
    document.body.appendChild(helpEl);
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
  document.body.appendChild(logo);
  const place = () => {
    const a = opts.anchor;
    const r = a && a.getClientRects().length ? a.getBoundingClientRect() : null;
    if (r && r.width > 2 && r.height > 2) {
      logo.style.top = Math.round(r.top + 8) + "px";
      logo.style.right = Math.round(window.innerWidth - r.right + 8) + "px";
    } else {
      logo.style.top = "10px";
      logo.style.right = "12px";
    }
  };
  place();
  requestAnimationFrame(place);
  globalThis.addEventListener("resize", place);
  if (opts.anchor && "ResizeObserver" in globalThis) new ResizeObserver(place).observe(opts.anchor);
  const pop = document.createElement("div");
  pop.style.cssText = "position:fixed;z-index:73;min-width:210px;max-width:300px;max-height:84vh;overflow-y:auto;padding:10px 12px;border-radius:12px;color:#eaf0ff;font:13px -apple-system,system-ui,sans-serif;opacity:0;pointer-events:none;transform:translateY(-6px);transition:opacity 120ms ease-out,transform 120ms ease-out;";
  glass(pop);
  document.body.appendChild(pop);
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
      const lab = document.createElement("span");
      lab.textContent = c.label;
      row.appendChild(lab);
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
    pop.style.right = Math.round(window.innerWidth - b.right) + "px";
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
  return { refresh };
}

// examples/cardiac/cardiac-browser.ts
var NAMES = ["axial", "coronal", "sagittal", "threeD"];
var SLICES = ["axial", "coronal", "sagittal"];
var $ = (id) => document.getElementById(id);
var status = (s) => {
  $("status").textContent = s;
};
var gpu = await initDevice();
var preferred = navigator.gpu.getPreferredCanvasFormat();
var srgb = preferred.endsWith("-srgb") ? preferred : preferred + "-srgb";
var cv = {};
var cx = {};
for (const n of NAMES) {
  cv[n] = $("c-" + n);
  cx[n] = cv[n].getContext("webgpu");
  cx[n].configure({
    device: gpu.device,
    format: preferred,
    viewFormats: [srgb],
    alphaMode: "opaque",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  });
}
var CINE_BYTES = 132e5;
var mb = 0;
var barFill = $("barfill");
var loadPct = $("loadpct");
var loadWrap = $("loadwrap");
var pips = $("pips");
var setBar = (frac, label) => {
  barFill.style.width = `${Math.max(2, Math.min(100, frac * 100)).toFixed(1)}%`;
  loadPct.textContent = label;
};
var setPips = (loaded, total) => {
  if (pips.childElementCount !== total) {
    pips.innerHTML = "";
    for (let i = 0; i < total; i++) pips.appendChild(document.createElement("span"));
  }
  [...pips.children].forEach((el, i) => el.classList.toggle("on", i < loaded));
};
var onPhaseLoaded = null;
status("loading cardiac data\u2026");
var DATA_BASE = new URLSearchParams(location.search).get("data") ?? "https://js2.jetstream-cloud.org:8001/swift/v1/slicerlive/cardiac/";
var DEMO = globalThis.CARDIAC_DEMO ?? new URLSearchParams(location.search).get("demo") ?? "cine";
var ENDO = DEMO === "endo";
var TOTAL_BYTES = ENDO ? 57e6 : CINE_BYTES;
var sc = await buildCardiacScene(gpu, DATA_BASE, srgb, (p) => {
  mb += p.bytes;
  if (ENDO) {
    setBar(mb / TOTAL_BYTES, `${(mb / 1e6).toFixed(0)} of ~57 MB`);
    status(`loading\u2026 ${(mb / 1e6).toFixed(1)} MB`);
    return;
  }
  if (p.what !== "cine") return;
  setBar(
    Math.max(mb / CINE_BYTES, p.frames / p.totalFrames),
    `${p.frames} of ${p.totalFrames} phases \xB7 ${(mb / 1e6).toFixed(1)} MB`
  );
  setPips(p.frames, p.totalFrames);
  status(`loading\u2026 ${(mb / 1e6).toFixed(1)} MB`);
  if (p.bytes === 0) onPhaseLoaded?.(p.frames);
}, { only: ENDO ? "cta" : "cine" });
var SEED_POS = [-33.922, -4.023, -200.459];
var SEED_DIR = [0.0376, 0.05, 0.998];
var seedRAS = SEED_POS;
var rasLo;
var rasHi;
var sliceIx;
var off = { axial: 0.5, coronal: 0.5, sagittal: 0.5 };
function resetPlanes() {
  const dims = sc.mode() === "cta" ? sc.ctaDims : sc.cineDims;
  const ijk = sc.mode() === "cta" ? sc.ctaIjkToRAS : sc.cineIjkToRAS;
  [rasLo, rasHi] = (sc.mode() === "cta" ? sc.cta : sc.cine).aabb();
  for (const o of SLICES) off[o] = slicerDefaultOffset01(o, dims, ijk, rasLo, rasHi);
  sliceIx = new SliceInteractor({ ijkToRAS: ijk, rasLo, rasHi });
}
resetPlanes();
var shown = (n) => cv[n].width > 0 && cv[n].height > 0;
var endo = null;
var flying = false;
var clearanceAhead = Infinity;
var probedDir = [0, 0, 1];
var probeInFlight = false;
var lastGoodPos = null;
var escapeChecks = 0;
var MARGIN_MM = 6;
var autoTarget = null;
var autoDir = null;
var autoBusy = false;
var autoTicks = 0;
var autoStuck = 0;
var LEAD_MM = 10;
var AIM_SMOOTH = 1.6;
var seekTarget = null;
var seekDir = null;
var seekDist = 0;
var aimDir = null;
var leadTicks = 0;
var seekBusy = false;
var seekTicks = 0;
var manualLookAt = 0;
var accN = 0;
var invalidateStrip = () => {
  accN = 0;
};
var drawPlane = (o) => {
  if (!shown(o)) return;
  sc.slice.setPlane(o, off[o]);
  sc.slice.renderToView(cx[o].getCurrentTexture().createView({ format: srgb }), cv[o].width, cv[o].height);
};
var drawSlices = () => {
  for (const o of SLICES) drawPlane(o);
};
var camera = VtkCamera.slicerDefault();
var frameCamera = () => {
  camera.focalPoint = [...sc.center];
  camera.position = [sc.center[0], sc.center[1] + sc.radius * 2.6, sc.center[2]];
  camera.viewUp = [0, 0, 1];
  camera.viewAngle = 30;
};
frameCamera();
var a3d = mountAdaptive3d({
  scene: () => sc.scene,
  view: () => cx.threeD.getCurrentTexture().createView({ format: srgb }),
  size: () => ({ w: shown("threeD") ? cv.threeD.width : 0, h: cv.threeD.height }),
  setCamera: (s, w, h) => s.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, w, h),
  gpu,
  // Exterior DVR of a 512^3 volume needs the resolution cut to stay interactive while dragging.
  // The endovascular view does not: rays terminate at the vessel wall within a few cm, so a
  // full-resolution pass costs ~6 ms even at a very fine step (measured — see setStepForView).
  // Upscaling there would only blur a frame we can afford to trace properly.
  movingScaleCap: ENDO ? 1 : 0.75,
  // In cine mode tickCine owns the canvas and the accumulator; the adaptive loop must never
  // render there — two consumers of SceneRenderer's single accumulator means flashing phases.
  onFrame: () => {
    if (sc.mode() === "cine") adaptiveFramesInCine++;
    cross3.redraw();
  }
});
var adaptiveFramesInCine = 0;
var draw3d = () => {
  if (sc.mode() !== "cine") a3d.draw();
};
var draw3dNow = () => a3d.renderSettled(true);
var converge = (n = 32) => {
  for (let i = 0; i < n; i++) a3d.renderSettled(i === 0);
};
var cross3 = mountCrosshair({
  cells: cv,
  getScene: () => sc.scene,
  getSlice: () => sc.slice,
  getCamera: () => camera,
  getOffset: (o) => off[o],
  onJump: (ras) => {
    scrollSlicesTo(ras);
    setMarker(ras);
    draw3d();
  }
});
var SLICE_AXIS = { axial: 2, coronal: 1, sagittal: 0 };
function scrollSlicesTo(ras) {
  for (const o of SLICES) {
    const a = SLICE_AXIS[o];
    off[o] = Math.max(0, Math.min(1, (ras[a] - rasLo[a]) / (rasHi[a] - rasLo[a])));
  }
  drawSlices();
}
function setMarker(ras) {
  cross3.state.set([...ras]);
  cross3.redraw();
}
function focusPoint(fallback) {
  return seekTarget ?? fallback;
}
function jumpSlicesTo(cameraPos) {
  const p = focusPoint(cameraPos);
  scrollSlicesTo(p);
  setMarker(p);
}
var resize = () => {
  const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
  for (const n of NAMES) {
    cv[n].width = Math.floor(cv[n].clientWidth * dpr);
    cv[n].height = Math.floor(cv[n].clientHeight * dpr);
  }
  accN = 0;
  drawSlices();
  draw3d();
};
globalThis.addEventListener("resize", resize);
var grid = attachViewGrid($("grid"), NAMES, resize);
attachDoubleClick(cv.threeD, () => grid.toggleMax("threeD"));
for (const o of SLICES) {
  attachSliceControls(cv[o], {
    orient: o,
    getSlice: () => sc.slice,
    step: (fwd) => {
      off[o] = sliceIx.wheel(o, off[o], fwd);
    },
    redraw: () => {
      drawPlane(o);
      cross3.redraw();
    },
    hooks: { onDoubleClick: () => {
      grid.toggleMax(o);
      return true;
    } }
  });
}
var presetSel = $("preset");
for (const n of PRESET_NAMES) {
  if (n === "MR-Default") continue;
  const o = document.createElement("option");
  o.value = n;
  o.textContent = n;
  presetSel.appendChild(o);
}
var applyPreset = (name) => {
  presetSel.value = name;
  sc.setPreset(name);
  accN = 0;
  $("blurb").textContent = CARDIAC_PRESETS[name].blurb;
  drawSlices();
  draw3d();
};
presetSel.onchange = () => applyPreset(presetSel.value);
var ctaBar = $("ctabar");
var ctaFill = $("ctafill");
var ctaText = $("ctatext");
var ctaMb = 0;
var CTA_BYTES = 57e6;
var loadCtaIfNeeded = async () => {
  if (sc.ctaLoaded()) return true;
  ctaBar.style.display = "flex";
  ctaMb = 0;
  ctaText.textContent = "loading CTA\u2026";
  await sc.ensureCta((p) => {
    ctaMb += p.bytes;
    ctaFill.style.width = `${Math.min(100, ctaMb / CTA_BYTES * 100).toFixed(1)}%`;
    ctaText.textContent = `loading CTA\u2026 ${(ctaMb / 1e6).toFixed(0)} MB`;
  });
  ctaBar.style.display = "none";
  return true;
};
var setMode = (m) => {
  if (flying) {
    endo?.detach();
    endo = null;
    flying = false;
    $("cruise").textContent = "";
  }
  sc.setMode(m);
  resetPlanes();
  frameCamera();
  accN = 0;
  for (const o of ["cta", "cine"]) $(`mode-${o}`).classList.toggle("on", o === m);
  $("transport").style.display = m === "cine" ? "flex" : "none";
  $("flyBtn").style.display = m === "cta" ? "inline-block" : "none";
  applyPreset(m === "cine" ? "CT-Coronary-Arteries-3" : "CT-EndoVascular");
  status(m === "cine" ? "4D cine \xB7 10 cardiac phases \xB7 press play" : "static CTA 512\xD7512\xD7321 \xB7 try \u201Cfly inside\u201D");
};
for (const m of ["cta", "cine"]) {
  $(`mode-${m}`).onclick = async () => {
    if (m === "cta") {
      sc.browser.playbackActive = false;
      playBtn.textContent = "\u25B6 play";
      await loadCtaIfNeeded();
    }
    setMode(m);
  };
}
function seatFlight(p) {
  camera.position = [...p.pos];
  camera.viewUp = [...p.up];
  const v = [p.fp[0] - p.pos[0], p.fp[1] - p.pos[1], p.fp[2] - p.pos[2]];
  const l = Math.hypot(v[0], v[1], v[2]);
  lastGoodPos = [...p.pos];
  aimDir = null;
  seekDir = null;
  seekTarget = null;
  if (l > 1e-6) endo?.lookAlong([v[0] / l, v[1] / l, v[2] / l]);
  jumpSlicesTo(camera.position);
  draw3d();
}
var speedEl = document.getElementById("speed");
var speedLbl = document.getElementById("speedLbl");
var DEFAULT_SPEED = 8;
var flightSpeed = () => speedEl ? Number(speedEl.value) : DEFAULT_SPEED;
var showSpeed = () => {
  if (speedLbl) speedLbl.textContent = `${flightSpeed()} mm/s`;
};
if (speedEl) {
  speedEl.value = String(DEFAULT_SPEED);
  showSpeed();
  speedEl.oninput = () => {
    showSpeed();
    endo?.setSpeed(flightSpeed());
  };
}
var startFlight = async () => {
  if (!sc.ctaLoaded()) {
    await loadCtaIfNeeded();
    setMode("cta");
  }
  applyPreset("CT-EndoVascular");
  const fromSlicer = followedPose;
  flightFromSlicer = !!fromSlicer;
  camera.position = [...seedRAS];
  camera.viewUp = [0, 1, 0];
  camera.viewAngle = 80;
  flying = true;
  lastGoodPos = [...camera.position];
  endo?.detach();
  endo = attachEndoscopyControls(cv.threeD, camera, {
    speedMmPerSec: flightSpeed(),
    marginMm: MARGIN_MM,
    referenceUp: [0, 0, 1],
    onChange: () => {
      jumpSlicesTo(camera.position);
      draw3d();
    },
    onLook: () => {
      manualLookAt = performance.now();
    },
    onState: (c) => showCruise(c),
    // Rails, forward only for now: pick(0.5,0.5) is the ray straight ahead, so it already
    // answers "how far to the wall?" without any renderer change. Sideways/backward
    // clearance needs a general probe(origin, dir) and is not wired yet.
    // Rails. The probe is fired for whatever direction we are actually travelling, so this
    // works for reverse as well as forward — pick(u,v) could only ever see what is on screen.
    clearance: (dir) => dot32(dir, probedDir) > 0.9 ? clearanceAhead : Infinity
  });
  endo.setSpeed(flightSpeed());
  endo.lookAlong(SEED_DIR);
  if (fromSlicer) seatFlight(fromSlicer);
  status(fromSlicer ? "flight started from Slicer's camera" : "flight started from the aortic seed");
  jumpSlicesTo(camera.position);
  showCruise("stopped");
  draw3d();
};
$("flyBtn").onclick = startFlight;
cv.threeD.addEventListener("pointerdown", (e) => {
  if (!flying || e.button !== 0 || !e.shiftKey) return;
  const r = cv.threeD.getBoundingClientRect();
  const u = (e.clientX - r.left) / r.width, v = (e.clientY - r.top) / r.height;
  e.preventDefault();
  sc.scene.pick(u, v).then((ras) => {
    if (ras) setAutoTarget(ras);
  });
});
var keysDown = /* @__PURE__ */ new Set();
globalThis.addEventListener("keydown", (e) => {
  keysDown.add(e.key);
  if (["ArrowLeft", "ArrowRight"].includes(e.key)) manualLookAt = performance.now();
});
globalThis.addEventListener("keyup", (e) => keysDown.delete(e.key));
var forwardDir = () => {
  const d = [
    camera.focalPoint[0] - camera.position[0],
    camera.focalPoint[1] - camera.position[1],
    camera.focalPoint[2] - camera.position[2]
  ];
  const l = Math.hypot(d[0], d[1], d[2]) || 1;
  return [d[0] / l, d[1] / l, d[2] / l];
};
var dot32 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
var showCruise = (c) => {
  const label = c === "forward" ? "\u25B6 forward" : c === "back" ? "\u25C0 back" : "\u25A0 stopped";
  $("cruise").textContent = label;
  $("cruise").className = c === "stopped" ? "" : "on";
  status(`endovascular flight \xB7 ${label} \xB7 ${flightSpeed()} mm/s \xB7 \u2191\u2193 in/out \xB7 \u2190\u2192 yaw \xB7 shift \u2190\u2192 pitch \xB7 ctrl \u2190\u2192 roll \xB7 space cruise`);
};
var scrub = $("scrub");
var fps = $("fps");
var playBtn = $("playBtn");
scrub.max = String(sc.cine.frameCount - 1);
var selectFrame = (i) => {
  sc.cine.setFrame(i, sc.browser.playbackLooped);
  sc.scene.refreshBindings();
  sc.scene.syncUniforms();
};
var shownFrame = -1;
var renderInFlight = false;
var SETTLE_TARGET = 48;
var showFrameSlices = (i) => {
  sc.slice.setTextures(sc.cine.volumeTexture());
  drawSlices();
  scrub.value = String(i);
  $("frameLbl").textContent = `${i + 1}/${sc.cine.frameCount}`;
};
scrub.oninput = () => {
  sc.browser.playbackActive = false;
  playBtn.textContent = "\u25B6 play";
  sc.browser.setSelectedItemNumber(Number(scrub.value));
};
fps.oninput = () => {
  sc.browser.playbackRateFps = Number(fps.value);
  $("fpsLbl").textContent = `${fps.value} fps`;
};
playBtn.onclick = () => {
  sc.browser.playbackActive = !sc.browser.playbackActive;
  playBtn.textContent = sc.browser.playbackActive ? "\u275A\u275A pause" : "\u25B6 play";
  status(`4D cine \xB7 phase ${sc.browser.selectedItemNumber + 1}/${sc.cine.frameCount}` + (sc.browser.playbackActive ? ` \xB7 playing at ${sc.browser.playbackRateFps} fps` : " \xB7 converging"));
};
var acc = 0;
var lastT = 0;
var lastCamMove = 0;
var camMoved = () => {
  lastCamMove = performance.now();
  accN = 0;
};
var stepMmNow = 0;
var setStep = (mm) => {
  if (Math.abs(mm - stepMmNow) < 1e-6) return;
  stepMmNow = mm;
  sc.scene.setSampleStep(mm);
  accN = 0;
};
var ENDO_STEP_MULT = 0.125;
var CTA_STEP_MULT = 0.5;
var setStepForView = () => {
  if (sc.mode() === "cine") {
    setStep(sc.cine.sampleStep() * 0.5);
    return;
  }
  const sp = sc.cta?.sampleStep();
  if (sp) setStep(sp * (flying ? ENDO_STEP_MULT : CTA_STEP_MULT));
};
var flightLastT = 0;
var tickFlight = (msNow) => {
  if (!flying || !endo) {
    flightLastT = msNow;
    return;
  }
  const dt = flightLastT ? (msNow - flightLastT) / 1e3 : 0;
  flightLastT = msNow;
  if (!probeInFlight) {
    probeInFlight = true;
    const f = forwardDir();
    const c = endo.cruise();
    const dir = c === "back" ? [-f[0], -f[1], -f[2]] : f;
    probedDir = dir;
    const from = [...camera.position];
    sc.scene.probe(from, dir).then((d) => {
      clearanceAhead = d;
      probeInFlight = false;
    }).catch(() => {
      probeInFlight = false;
    });
  }
  if (++escapeChecks % 30 === 0 && !probeInFlight) {
    const from = [...camera.position];
    const axes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    Promise.all(axes.map((d) => sc.scene.probe(from, d))).then((ds) => {
      const enclosed = ds.some((d) => Number.isFinite(d) && d < 200);
      if (enclosed) lastGoodPos = from;
      else if (lastGoodPos) {
        camera.position = [...lastGoodPos];
        endo?.setCruise("stopped");
        status("left the blood pool \u2014 returned to the last enclosed position");
        jumpSlicesTo(camera.position);
        draw3d();
      }
    }).catch(() => {
    });
  }
  if (autoTarget) steerAutopilot();
  depthSeek(dt);
  endo.tick(dt);
};
function depthSeek(dt) {
  if (!endo || !flying) return;
  const moving = endo.cruise() === "forward" || keysDown.has("ArrowUp");
  const steer = moving && !autoTarget && performance.now() - manualLookAt >= 600;
  if (!seekBusy && ++seekTicks % 8 === 0) {
    seekBusy = true;
    const eye = [...camera.position];
    const uv = [];
    for (const v of [0.32, 0.5, 0.68]) for (const u of [0.32, 0.5, 0.68]) uv.push([u, v]);
    Promise.all(uv.map(([u, v]) => sc.scene.pick(u, v))).then((hits) => {
      let bestD = -1, bestHit = null;
      hits.forEach((h) => {
        if (!h) return;
        const d = Math.hypot(h[0] - eye[0], h[1] - eye[1], h[2] - eye[2]);
        if (d > bestD) {
          bestD = d;
          bestHit = h;
        }
      });
      if (bestHit && bestD > 4) {
        seekDist = bestD;
        const v = [bestHit[0] - eye[0], bestHit[1] - eye[1], bestHit[2] - eye[2]];
        const l = Math.hypot(v[0], v[1], v[2]) || 1;
        seekDir = [v[0] / l, v[1] / l, v[2] / l];
      }
      seekBusy = false;
    }).catch(() => {
      seekBusy = false;
    });
  }
  const want = autoDir ?? seekDir;
  if (want) aimDir = aimDir ? slerpDir(aimDir, want, AIM_SMOOTH * dt) : want;
  if (aimDir) {
    const lead = Math.max(2, Math.min(LEAD_MM, seekDist - 2));
    const p = [
      camera.position[0] + aimDir[0] * lead,
      camera.position[1] + aimDir[1] * lead,
      camera.position[2] + aimDir[2] * lead
    ];
    const moved = !seekTarget || Math.hypot(p[0] - seekTarget[0], p[1] - seekTarget[1], p[2] - seekTarget[2]) > 0.2;
    seekTarget = p;
    if (moved && ++leadTicks % 3 === 0) {
      scrollSlicesTo(p);
      setMarker(p);
    }
    if (steer) endo.lookAlong(slerpDir(forwardDir(), aimDir, 0.9 * dt));
  }
}
function slerpDir(from, to, maxRad) {
  const d = Math.max(-1, Math.min(1, dot32(from, to)));
  const ang = Math.acos(d);
  if (ang < 1e-4) return to;
  const t = Math.min(1, maxRad / ang);
  const s1 = Math.sin((1 - t) * ang) / Math.sin(ang), s2 = Math.sin(t * ang) / Math.sin(ang);
  const v = [from[0] * s1 + to[0] * s2, from[1] * s1 + to[1] * s2, from[2] * s1 + to[2] * s2];
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function steerAutopilot() {
  if (!autoTarget || !endo) return;
  const pos = camera.position;
  const toT = [autoTarget[0] - pos[0], autoTarget[1] - pos[1], autoTarget[2] - pos[2]];
  const dist = Math.hypot(toT[0], toT[1], toT[2]);
  if (dist < 8) {
    endo.setCruise("stopped");
    autoTarget = null;
    autoDir = null;
    status("autopilot: arrived at the target");
    return;
  }
  const bearing = [toT[0] / dist, toT[1] / dist, toT[2] / dist];
  if (!autoBusy && ++autoTicks % 6 === 0) {
    autoBusy = true;
    const cands = [bearing];
    const f = forwardDir();
    const left = normalize3(cross32([0, 0, 1], bearing));
    const up = cross32(bearing, left);
    for (const ang of [0.35, 0.7, 1.05]) {
      for (let k = 0; k < 6; k++) {
        const th = k / 6 * Math.PI * 2;
        const off2 = [
          left[0] * Math.cos(th) * Math.sin(ang) + up[0] * Math.sin(th) * Math.sin(ang) + bearing[0] * Math.cos(ang),
          left[1] * Math.cos(th) * Math.sin(ang) + up[1] * Math.sin(th) * Math.sin(ang) + bearing[1] * Math.cos(ang),
          left[2] * Math.cos(th) * Math.sin(ang) + up[2] * Math.sin(th) * Math.sin(ang) + bearing[2] * Math.cos(ang)
        ];
        cands.push(normalize3(off2));
      }
    }
    void f;
    const from = [...pos];
    Promise.all(cands.map((d) => sc.scene.probe(from, d))).then((ds) => {
      let best = -1, bestDir = null;
      cands.forEach((d, i) => {
        const clear = Math.min(Number.isFinite(ds[i]) ? ds[i] : 200, 60) - MARGIN_MM;
        if (clear <= 2) return;
        const progress = dot32(d, bearing);
        if (progress <= 0.1) return;
        const score = progress * clear;
        if (score > best) {
          best = score;
          bestDir = d;
        }
      });
      if (bestDir) {
        autoDir = bestDir;
        autoStuck = 0;
      } else if (++autoStuck > 2) {
        endo?.setCruise("stopped");
        autoTarget = null;
        autoDir = null;
        status(`autopilot: blocked \u2014 this is as close as the blood pool allows (${dist.toFixed(0)} mm short)`);
      }
      autoBusy = false;
    }).catch(() => {
      autoBusy = false;
    });
  }
  if (autoDir) {
    endo.lookAlong(slerpDir(forwardDir(), autoDir, 0.06));
    endo.setCruise("forward");
  }
}
var cross32 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
var normalize3 = (v) => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};
var setAutoTarget = (ras) => {
  autoTarget = [...ras];
  autoDir = null;
  autoStuck = 0;
  status("autopilot: steering toward the picked target\u2026");
};
var tickCine = (msNow) => {
  requestAnimationFrame(tickCine);
  tickFlight(msNow);
  if (!shown("threeD")) {
    lastT = msNow;
    return;
  }
  setStepForView();
  if (sc.mode() !== "cine") {
    lastT = msNow;
    return;
  }
  if (sc.browser.playbackActive) {
    const dt = lastT ? (msNow - lastT) / 1e3 : 0;
    acc += dt * sc.browser.playbackRateFps;
    const inc = Math.floor(acc);
    if (inc > 0) {
      acc -= inc;
      sc.browser.selectNextItem(sc.browser.playbackItemSkippingEnabled ? inc : 1);
    }
  } else acc = 0;
  lastT = msNow;
  const cur = sc.browser.selectedItemNumber;
  if (cur !== shownFrame) {
    selectFrame(cur);
    shownFrame = cur;
    accN = 0;
    showFrameSlices(cur);
    cross3.redraw();
    status(`4D cine \xB7 phase ${cur + 1}/${sc.cine.frameCount}` + (sc.browser.playbackActive ? ` \xB7 playing at ${sc.browser.playbackRateFps} fps` : " \xB7 press play"));
  }
  if (renderInFlight) return;
  if (!sc.browser.playbackActive && accN >= SETTLE_TARGET) return;
  renderInFlight = true;
  sc.scene.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, cv.threeD.width, cv.threeD.height);
  sc.scene.renderAccum(cx.threeD.getCurrentTexture().createView({ format: srgb }), cv.threeD.width, cv.threeD.height, accN === 0);
  accN++;
  cross3.redraw();
  gpu.device.queue.onSubmittedWorkDone().then(() => {
    renderInFlight = false;
  });
};
requestAnimationFrame(tickCine);
var box0 = sc.roi.snapshot();
attachWidgetControls(cv.threeD, camera, {
  getHandles: () => sc.roiVisible() ? sc.roi.handleList().map((h) => ({ id: h.id, world: h.world, data: h.data, cursor: h.cursor })) : [],
  getSize: () => ({ w: cv.threeD.width, h: cv.threeD.height }),
  onDragStart: () => {
    box0 = sc.roi.snapshot();
  },
  onDrag: (h, world) => {
    const d = [world[0] - h.world[0], world[1] - h.world[1], world[2] - h.world[2]];
    sc.roi.applyDrag(h.data, box0, d);
    if (sc.cropEnabled()) sc.scene.setClipBox(sc.roi.lo(), sc.roi.hi());
    sc.scene.syncUniforms();
    invalidateStrip();
  },
  onHover: (h) => {
    sc.roi.setHover(h ? h.id : null);
    sc.scene.syncUniforms();
  },
  onChange: draw3d
});
var chrome = installChrome({
  anchor: cv.threeD.parentElement ?? void 0,
  controls: [
    {
      label: "Enable cropping",
      get: () => sc.cropEnabled(),
      set: (on) => {
        sc.setCropEnabled(on);
        invalidateStrip();
        draw3d();
      }
    },
    {
      label: "Display ROI",
      get: () => sc.roiVisible(),
      set: (on) => {
        sc.setRoiVisible(on);
        invalidateStrip();
        draw3d();
      }
    }
  ],
  onChange: () => {
    drawSlices();
    draw3d();
  }
});
attachCameraControls(cv.threeD, camera, {
  enabled: () => !flying,
  onChange: () => {
    camMoved();
    invalidateStrip();
    draw3d();
  }
});
var followWs = null;
var followedPose = null;
var resyncOnce = false;
var flightFromSlicer = false;
var follow = (port = 2132) => {
  followWs?.close();
  const ws = new WebSocket(`ws://localhost:${port}`);
  followWs = ws;
  ws.onopen = () => {
    ws.send(JSON.stringify({ op: "subscribe", types: ["camera"] }));
    status(`following Slicer's camera on :${port}`);
  };
  ws.onerror = () => status(`could not reach the mrson live stream on :${port}`);
  ws.onclose = () => {
    if (followWs === ws) status("mrson camera follow disconnected");
  };
  ws.onmessage = (e) => {
    let m;
    try {
      m = JSON.parse(String(e.data));
    } catch {
      return;
    }
    const isCam = m.event === "CameraModified" || m.event === "NodeAdded" && m.node?.type === "camera";
    if (!isCam) return;
    const c = m.event === "CameraModified" ? m : m.node;
    const pos = c.position;
    const fp = c.focalPoint;
    const up = c.viewUp;
    const va = c.viewAngle;
    if (!pos || !fp || !up) return;
    followedPose = { pos: [...pos], fp: [...fp], up: [...up], va };
    if (flying) {
      if (!flightFromSlicer || resyncOnce) {
        resyncOnce = false;
        flightFromSlicer = true;
        seatFlight({ pos: [...pos], fp: [...fp], up: [...up], va });
      }
      return;
    }
    camera.position = [...pos];
    camera.focalPoint = [...fp];
    camera.viewUp = [...up];
    if (va) camera.viewAngle = va;
    camMoved();
    invalidateStrip();
    draw3d();
  };
};
if (new URLSearchParams(location.search).has("follow")) {
  follow(Number(new URLSearchParams(location.search).get("follow")) || 2132);
}
globalThis.cardiac = {
  state: () => ({
    mode: sc.mode(),
    preset: sc.presetName(),
    off: { ...off },
    rasLo,
    rasHi,
    frame: sc.browser.selectedItemNumber,
    frames: sc.cine.frameCount,
    playing: sc.browser.playbackActive,
    fps: sc.browser.playbackRateFps,
    crop: sc.cropEnabled(),
    roiVisible: sc.roiVisible(),
    boundFrame: sc.cine.frame,
    accN,
    accumN: sc.scene.accumCount(),
    // the renderer's real accumulation depth (accN is cine-only)
    adaptiveFramesInCine,
    // must stay 0: two accumulator owners = flashing phases
    lastCamMove,
    renderInFlight,
    flying,
    cruise: endo ? endo.cruise() : "stopped",
    speedMmPerSec: endo ? endo.speed() : flightSpeed(),
    autoTarget: autoTarget ? [...autoTarget] : null,
    seekTarget: seekTarget ? [...seekTarget] : null,
    // the lead point: LEAD_MM ahead along aimDir
    aimDir: aimDir ? [...aimDir] : null,
    seekDist,
    clearanceAhead: Number.isFinite(clearanceAhead) ? +clearanceAhead.toFixed(2) : null,
    cameraPos: [...camera.position],
    cameraFocal: [...camera.focalPoint],
    sizes: Object.fromEntries(NAMES.map((n) => [n, [cv[n].width, cv[n].height]]))
  }),
  drawSlices,
  draw3dNow,
  converge,
  resize,
  /** Explicit-ray depth probe — lets a driver assert the crosshair lead point sits in the lumen. */
  probe: (o, d) => sc.scene.probe(o, d),
  device: () => gpu.device,
  cineAccum: () => accN,
  setOffset: (o, v) => {
    off[o] = v;
    drawPlane(o);
  },
  setMode,
  applyPreset,
  follow,
  startFlight,
  /** Re-adopt Slicer's current pose once, mid-flight. */
  resync: () => {
    resyncOnce = true;
  },
  followedPose: () => followedPose,
  setCruise: (c) => endo?.setCruise(c),
  setSpeed: (mmPerSec) => {
    if (speedEl) {
      speedEl.value = String(mmPerSec);
      showSpeed();
    }
    endo?.setSpeed(mmPerSec);
  },
  setAutoTarget,
  // Drive the camera to exact values so a view can be matched 1:1 against Slicer.
  getCamera: () => ({
    position: [...camera.position],
    focalPoint: [...camera.focalPoint],
    viewUp: [...camera.viewUp],
    viewAngle: camera.viewAngle
  }),
  setCamera: (p) => {
    if (p.position) camera.position = [...p.position];
    if (p.focalPoint) camera.focalPoint = [...p.focalPoint];
    if (p.viewUp) camera.viewUp = [...p.viewUp];
    if (p.viewAngle) camera.viewAngle = p.viewAngle;
    camMoved();
    invalidateStrip();
    draw3d();
  },
  setCrop: (on) => {
    sc.setCropEnabled(on);
    invalidateStrip();
    chrome.refresh();
    draw3dNow();
  },
  setRoi: (on) => {
    sc.setRoiVisible(on);
    invalidateStrip();
    chrome.refresh();
    draw3dNow();
  }
};
if (ENDO) {
  for (const el of ["mode-cta", "mode-cine", "flyBtn", "transport"]) $(el).style.display = "none";
  $("loadtitle").textContent = "Loading the CTA\u2026";
  $("loadsub").textContent = "512x512x321 contrast CT streaming from a public JS2 bucket";
  loadWrap.classList.add("done");
  setTimeout(() => {
    loadWrap.style.display = "none";
  }, 600);
  await startFlight();
  requestAnimationFrame(() => {
    resize();
    draw3dNow();
    setTimeout(() => {
      if (flying) endo?.setCruise("forward");
    }, 900);
  });
} else {
  setMode("cine");
  sc.browser.playbackRateFps = 10;
  fps.value = "10";
  $("fpsLbl").textContent = "10 fps";
  sc.browser.playbackActive = false;
  playBtn.textContent = "\u25B6 play";
  onPhaseLoaded = (n) => {
    sc.browser.setSelectedItemNumber(n - 1);
    status(`4D cine \xB7 loaded phase ${n}/${sc.cine.frameCount}`);
  };
  onPhaseLoaded(1);
  sc.cineReady.then(() => {
    setBar(1, `all ${sc.cine.frameCount} phases loaded`);
    setPips(sc.cine.frameCount, sc.cine.frameCount);
    onPhaseLoaded = null;
    loadWrap.classList.add("done");
    setTimeout(() => {
      loadWrap.style.display = "none";
    }, 600);
    sc.browser.setSelectedItemNumber(0);
    sc.browser.playbackActive = true;
    playBtn.textContent = "\u275A\u275A pause";
    status(`4D cine \xB7 ${sc.cine.frameCount} phases \xB7 playing at ${sc.browser.playbackRateFps} fps \xB7 rotate while it plays`);
  });
  requestAnimationFrame(() => resize());
}
