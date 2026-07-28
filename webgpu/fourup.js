// render/device.ts
async function initDevice() {
  const gpu = navigator.gpu;
  if (!gpu) throw new Error("WebGPU not available (need Chrome/Edge/Safari or Deno --unstable-webgpu)");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const want = ["float32-filterable", "timestamp-query"].filter((f) => adapter.features.has(f));
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
  constructor(gpu, format = DEFAULT_FORMAT) {
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
    this.mat = new Float32Array(uoff + CLIP_FLOATS + 4);
    this.matBuf = this.dev.createBuffer({ size: (uoff + CLIP_FLOATS + 4) * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
    let js = fract(sin(dot(v.position.xy + vec2<f32>(f32(safety) * 0.7548, f32(safety) * 0.5698), vec2<f32>(12.9898, 78.233))) * 43758.5453) - 0.5; // per-(pixel,sample) jitter \u2014 frame-invariant; temporal AA rides on the sub-pixel NDC jitter (frame.xy), which is exact identity at 0
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
  let ro = ndc_to_world(vec4<f32>(u_material.pick_cursor.x, u_material.pick_cursor.y, 0.0, 1.0));
  let rd = normalize(ndc_to_world(vec4<f32>(u_material.pick_cursor.x, u_material.pick_cursor.y, 1.0, 1.0)) - ro);
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
    this.mat[this.pickOff] = u * 2 - 1;
    this.mat[this.pickOff + 1] = 1 - v * 2;
    this.flush();
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
  params : vec4<f32>,    // win, lev, overlayOpacity, outlineMode(0/1)
  size : vec4<f32>,      // sizeX, sizeY, _, _
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var s_lin : sampler;
@group(0) @binding(2) var t_scalar : texture_3d<f32>;
@group(0) @binding(3) var t_overlay : texture_3d<f32>;

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
fn ov_at(ras : vec3<f32>) -> vec4<f32> {   // overlay at a RAS point (0 outside the volume)
  let t = (u.p2t * vec4<f32>(ras, 1.0)).xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec4<f32>(0.0); }
  return textureSampleLevel(t_overlay, s_lin, t, 0.0);
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
  let ov = textureSampleLevel(t_overlay, s_lin, tex, 0.0);
  var ovA = clamp(ov.a * u.params.z, 0.0, 1.0);
  if (u.params.w > 0.5) {   // OUTLINE mode: keep the overlay only at segment boundaries (screen-space)
    let du = u.uvec.xyz / u.size.x * 1.5;   // ~1.5 px right, in RAS
    let dv = u.vvec.xyz / u.size.y * 1.5;   // ~1.5 px up
    let n0 = ov_at(ras + du); let n1 = ov_at(ras - du); let n2 = ov_at(ras + dv); let n3 = ov_at(ras - dv);
    let e = max(max(distance(n0.rgb, ov.rgb) + abs(n0.a - ov.a), distance(n1.rgb, ov.rgb) + abs(n1.a - ov.a)),
                max(distance(n2.rgb, ov.rgb) + abs(n2.a - ov.a), distance(n3.rgb, ov.rgb) + abs(n3.a - ov.a)));
    ovA = ovA * clamp((e - 0.03) * 12.0, 0.0, 1.0);   // 0 in the interior, full at a colour/label edge
  }
  col = mix(col, ov.rgb, ovA);
  return vec4<f32>(srgb2physical(col), 1.0);
}
`
);
var BASES = {
  axial: { uDir: [-1, 0, 0], vDir: [0, 1, 0], uAxis: 0, vAxis: 1, nAxis: 2 },
  coronal: { uDir: [-1, 0, 0], vDir: [0, 0, 1], uAxis: 0, vAxis: 2, nAxis: 1 },
  sagittal: { uDir: [0, -1, 0], vDir: [0, 0, 1], uAxis: 1, vAxis: 2, nAxis: 0 }
};
var SliceRenderer = class {
  dev;
  format;
  pipeline;
  sampler;
  ubuf;
  u = new Float32Array(36);
  // p2t(16) + origin(4) + uvec(4) + vvec(4) + params(4) + size(4)
  bind;
  overlay;
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
  constructor(gpu, format = DEFAULT_FORMAT2) {
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
    this.ubuf = this.dev.createBuffer({ size: this.u.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.setWindowLevel(255, 127);
    this.setOverlayOpacity(0.55);
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
    this.bind = this.dev.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: scalar.createView() },
        { binding: 3, resource: this.overlay.createView() }
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
  setOverlayOpacity(o) {
    this.u[30] = o;
  }
  /** Overlay draw mode: false = FILL (solid coloured regions), true = OUTLINE (segment boundaries only). */
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

// render/demos/sphere-scene.ts
var N = 128;
var SPACING = [1.5, 1.5, 1.5];
var clamp01 = (v) => Math.max(0, Math.min(1, v));
function syntheticVolume() {
  const data = new Float32Array(N * N * N);
  const c = (N - 1) / 2;
  const ic = [c + 16, c, c + 12];
  for (let z = 0; z < N; z++) {
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const ro = Math.hypot(x - c, y - c, z - c);
        const ri = Math.hypot(x - ic[0], y - ic[1], z - ic[2]);
        const soft = 45 * clamp01((44 - ro) / 3);
        const dense = 210 * clamp01((20 - ri) / 3);
        data[(z * N + y) * N + x] = Math.max(soft, dense);
      }
    }
  }
  return data;
}
function orbitEye(azimuth, elevation, distance) {
  const ce = Math.cos(elevation);
  return [
    distance * ce * Math.sin(azimuth),
    -distance * ce * Math.cos(azimuth),
    distance * Math.sin(elevation)
  ];
}

// render/textures.ts
function createScalarTexture(dev, data, dims) {
  const [dx, dy, dz] = dims;
  const tex = dev.createTexture({ size: [dx, dy, dz], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  dev.queue.writeTexture({ texture: tex }, data, { bytesPerRow: dx * 4, rowsPerImage: dy }, [dx, dy, dz]);
  return tex;
}

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

// render/fields.ts
var RGBAVolumeField = class {
  kind = "rgba";
  bindingCount = 1;
  // baked rgba texture (sampler shared)
  clippable;
  tex;
  p2t;
  shade;
  unit;
  stepMm;
  box;
  constructor(tex, dims, spacing, opts = {}) {
    const center = opts.center ?? [0, 0, 0];
    this.tex = tex;
    if (opts.ijkToRAS) {
      this.p2t = patientToTextureFromIjkToRAS(opts.ijkToRAS, dims);
      this.box = volumeAABBFromIjkToRAS(opts.ijkToRAS, dims);
      this.stepMm = Math.min(...spacingFromIjkToRAS(opts.ijkToRAS));
    } else {
      this.p2t = patientToTexture(dims, spacing, center);
      this.box = volumeAABB(dims, spacing, center);
      this.stepMm = Math.min(...spacing);
    }
    this.shade = opts.shade ?? [0.3, 0.75, 0.45, 24];
    this.unit = opts.opacityUnitDistance ?? this.stepMm;
    this.clippable = opts.clippable ?? true;
  }
  uniformFloats() {
    return 24;
  }
  // mat4(16) + params(4) + shade(4)
  aabb() {
    return this.box;
  }
  sampleStep() {
    return this.stepMm;
  }
  /** Swap the baked texture in place (e.g. after re-baking an updated mask). The
   *  geometry is unchanged; the caller refreshes the SceneRenderer bind group. */
  setTexture(tex, destroyPrev = true) {
    if (destroyPrev && this.tex !== tex) this.tex.destroy();
    this.tex = tex;
  }
  get texture() {
    return this.tex;
  }
  structMembers(s) {
    return [
      `  rgba${s}_p2t : mat4x4<f32>,`,
      `  rgba${s}_params : vec4<f32>,`,
      // opacity_unit_distance, _, _, _
      `  rgba${s}_shade : vec4<f32>,`
      // ka, kd, ks, shininess
    ].join("\n");
  }
  declareBindings(s, base) {
    return `@group(0) @binding(${base}) var t_rgba${s} : texture_3d<f32>;`;
  }
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn alpha_rgba${s}(wp : vec3<f32>) -> f32 {
  let t4 = u_material.rgba${s}_p2t * vec4<f32>(transform_point_rgba${s}(wp), 1.0);
  return textureSampleLevel(t_rgba${s}, s_lin, clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).a;
}
fn sample_field_rgba${s}(wp : vec3<f32>, rd : vec3<f32>) -> vec4<f32> {
  let t4 = u_material.rgba${s}_p2t * vec4<f32>(transform_point_rgba${s}(wp), 1.0);
  let tex = t4.xyz;
  if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) { return vec4<f32>(0.0); }
  let c = textureSampleLevel(t_rgba${s}, s_lin, tex, 0.0);
  let step = u_material.scene.x;
  let unit = max(u_material.rgba${s}_params.x, 1e-3);
  let opacity = clamp(1.0 - pow(1.0 - clamp(c.a, 0.0, 1.0), step / unit), 0.0, 1.0);
  if (opacity <= 0.001) { return vec4<f32>(0.0); }
  let h = step * 2.0;   // wider central difference -> smoother normals (less shading aliasing on coarse volumes)
  let g = vec3<f32>(
    alpha_rgba${s}(wp + vec3<f32>(h,0,0)) - alpha_rgba${s}(wp - vec3<f32>(h,0,0)),
    alpha_rgba${s}(wp + vec3<f32>(0,h,0)) - alpha_rgba${s}(wp - vec3<f32>(0,h,0)),
    alpha_rgba${s}(wp + vec3<f32>(0,0,h)) - alpha_rgba${s}(wp - vec3<f32>(0,0,h))) / (2.0 * h);
  let glen = length(g);
  let ka = u_material.rgba${s}_shade.x; let kd = u_material.rgba${s}_shade.y;
  let ks = u_material.rgba${s}_shade.z; let sh = u_material.rgba${s}_shade.w;
  var lit_srgb = c.rgb * ka;
  if (glen > 1e-6) {
    var n = g / glen;
    if (dot(n, -rd) < 0.0) { n = -n; }
    let view_dir = normalize(-rd);
    let ldotn = dot(view_dir, n);
    if (ldotn > 0.0) {
      let refl = normalize(2.0 * ldotn * n - view_dir);
      let rdotv = max(0.0, dot(refl, view_dir));
      lit_srgb = c.rgb * (ka + kd * ldotn) + vec3<f32>(ks * pow(rdotv, sh));
    }
  }
  let lit = srgb2physical(clamp(lit_srgb, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * opacity, opacity);
}`
    );
  }
  fillUniforms(out, off) {
    out.set(this.p2t, off);
    out[off + 16] = this.unit;
    out[off + 20] = this.shade[0];
    out[off + 21] = this.shade[1];
    out[off + 22] = this.shade[2];
    out[off + 23] = this.shade[3];
  }
  bindEntries(_s, base) {
    return [{ binding: base, resource: this.tex.createView() }];
  }
};

// render/demos/fourup-scene.ts
function buildFourUpScene(dev) {
  const dims = [N, N, N], spacing = [SPACING[0], SPACING[1], SPACING[2]];
  const data = syntheticVolume();
  const lab = new Uint8Array(N * N * N);
  for (let i = 0; i < lab.length; i++) {
    const v = data[i];
    lab[i] = v >= 150 ? 1 : v >= 20 ? 2 : 0;
  }
  const pal = new Float32Array(256 * 4);
  const set = (i, r, g, b, a) => {
    pal[i * 4] = r;
    pal[i * 4 + 1] = g;
    pal[i * 4 + 2] = b;
    pal[i * 4 + 3] = a;
  };
  set(1, 0.95, 0.8, 0.35, 0.95);
  set(2, 0.3, 0.62, 0.72, 0.55);
  const scalarTex = createScalarTexture(dev, data, dims);
  const colorizeTex = bakeColorizeRGBA(dev, lab, dims, pal, 1.5);
  const field3d = new RGBAVolumeField(colorizeTex, dims, spacing, { opacityUnitDistance: SPACING[0], shade: [0.3, 0.78, 0.5, 28] });
  const [rasLo, rasHi] = volumeAABB(dims, spacing);
  return { scalarTex, colorizeTex, dims, spacing, p2t: patientToTexture(dims, spacing), rasLo, rasHi, win: 240, lev: 110, field3d };
}

// render/budget-controller.ts
var BudgetController = class {
  budgetPx;
  targetMs;
  minPx;
  maxPx;
  constructor(opts = {}) {
    this.targetMs = opts.targetMs ?? 16;
    this.minPx = opts.minPx ?? 15e4;
    this.maxPx = opts.maxPx ?? 8e6;
    this.budgetPx = opts.startPx ?? 12e5;
  }
  /** Nudge the budget toward hitting targetMs. Multiplicative, clamped per step (0.8–1.25×) so the
   *  loop is stable, and bounded to [minPx, maxPx]. Faster-than-target grows it; slower shrinks it. */
  update(measuredMs) {
    if (!(measuredMs > 0) || !Number.isFinite(measuredMs)) return;
    const adj = Math.max(0.6, Math.min(1.2, this.targetMs / measuredMs));
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
  const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
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
    while (!stopped && step()) await Promise.all([sync(), raf()]);
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
  const renderMoving = () => {
    const sc = opts.scene();
    if (!sc) return;
    const { w: vw, h: vh } = opts.size();
    if (!vw || !vh) return;
    const s = budget.scale(vw, vh), t0 = performance.now();
    if (s > 0.98) {
      opts.setCamera(sc, vw, vh);
      sc.renderToView(opts.view(), vw, vh);
    } else {
      const rw = Math.max(16, Math.round(vw * s)), rh = Math.max(16, Math.round(vh * s));
      opts.setCamera(sc, rw, rh);
      sc.renderUpscaled(opts.view(), rw, rh, vw, vh);
    }
    opts.gpu.device.queue.onSubmittedWorkDone().then(() => budget.update(performance.now() - t0));
    opts.onFrame?.();
  };
  const renderSettled = (reset) => {
    const sc = opts.scene();
    if (!sc) return;
    const { w: vw, h: vh } = opts.size();
    if (!vw || !vh) return;
    opts.setCamera(sc, vw, vh);
    sc.renderAccum(opts.view(), vw, vh, reset);
    opts.onFrame?.();
  };
  const loop = mountAdaptiveLoop({
    renderMoving,
    renderSettled,
    count: () => opts.scene()?.accumCount() ?? 1e9,
    target: opts.target ?? 24,
    sync: () => opts.gpu.device.queue.onSubmittedWorkDone()
    // GPU-paced: no backlog, input preempts
  });
  return { draw: () => loop.kick(), budget, renderSettled, renderMoving, loop };
}

// render/demos/fourup-browser.ts
var status = (msg, err = false) => {
  const el2 = document.getElementById("status");
  if (el2) {
    el2.textContent = msg;
    el2.style.color = err ? "#ff6b74" : "#9fb3d0";
  }
};
var el = (id) => document.getElementById(id);
async function main() {
  if (!navigator.gpu) {
    status("WebGPU not available \u2014 try Chrome/Edge 113+ or Safari 18+.", true);
    return;
  }
  status("initializing WebGPU\u2026");
  const gpu = await initDevice();
  const preferred = navigator.gpu.getPreferredCanvasFormat();
  const srgb = preferred + "-srgb";
  const names = ["axial", "coronal", "sagittal", "threeD"];
  const cv = {}, cx = {};
  for (const n of names) {
    cv[n] = el("c-" + n);
    cx[n] = cv[n].getContext("webgpu");
    cx[n].configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
  }
  status("baking segmentation\u2026");
  const sc = buildFourUpScene(gpu.device);
  const scene = new SceneRenderer(gpu, srgb);
  scene.build([sc.field3d]);
  scene.setBackground(0.05, 0.06, 0.09);
  const slice = new SliceRenderer(gpu, srgb);
  slice.setVolume(sc.p2t, sc.rasLo, sc.rasHi);
  slice.setTextures(sc.scalarTex, sc.colorizeTex);
  slice.setWindowLevel(sc.win, sc.lev);
  slice.setOverlayOpacity(0.6);
  const off = { axial: 0.5, coronal: 0.5, sagittal: 0.55 };
  let az = 0.5, elev = 0.3, dist = 430;
  const drawSlice = (n) => {
    slice.setPlane(n, off[n]);
    slice.renderToView(cx[n].getCurrentTexture().createView({ format: srgb }), cv[n].width, cv[n].height);
  };
  const a3d = mountAdaptive3d({
    scene: () => scene,
    view: () => cx.threeD.getCurrentTexture().createView({ format: srgb }),
    size: () => ({ w: cv.threeD.width, h: cv.threeD.height }),
    setCamera: (s, w, h) => s.setCamera(orbitEye(az, elev, dist), [0, 0, 0], [0, 0, 1], 28, w, h),
    gpu
  });
  const draw3d = () => a3d.draw();
  const draw3dNow = () => a3d.renderSettled(true);
  const drawAll = () => {
    drawSlice("axial");
    drawSlice("coronal");
    drawSlice("sagittal");
    draw3dNow();
    status("4-up \xB7 3 MPR + 3D ColorizeVolume \xB7 scroll a slice to scrub, drag 3D to orbit");
  };
  const resize = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const n of names) {
      const s = Math.floor(cv[n].clientWidth * dpr);
      cv[n].width = s;
      cv[n].height = s;
    }
    drawAll();
  };
  globalThis.addEventListener("resize", resize);
  for (const n of ["axial", "coronal", "sagittal"]) {
    cv[n].addEventListener("wheel", (e) => {
      e.preventDefault();
      off[n] = Math.max(0, Math.min(1, off[n] + (e.deltaY > 0 ? 0.02 : -0.02)));
      drawSlice(n);
    }, { passive: false });
  }
  let dragging = false, lx = 0, ly = 0;
  cv.threeD.addEventListener("pointerdown", (e) => {
    dragging = true;
    lx = e.clientX;
    ly = e.clientY;
    cv.threeD.setPointerCapture(e.pointerId);
  });
  cv.threeD.addEventListener("pointerup", (e) => {
    dragging = false;
    cv.threeD.releasePointerCapture(e.pointerId);
  });
  cv.threeD.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    az += (e.clientX - lx) * 8e-3;
    elev = Math.max(-1.4, Math.min(1.4, elev - (e.clientY - ly) * 8e-3));
    lx = e.clientX;
    ly = e.clientY;
    draw3d();
  });
  cv.threeD.addEventListener("wheel", (e) => {
    e.preventDefault();
    dist = Math.max(200, Math.min(1100, dist * (e.deltaY > 0 ? 1.08 : 0.93)));
    draw3d();
  }, { passive: false });
  resize();
}
main().catch((e) => status("error: " + (e?.message ?? e), true));
