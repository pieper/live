// render/device.ts
async function initDevice() {
  const gpu = navigator.gpu;
  if (!gpu) throw new Error("WebGPU not available (need Chrome/Edge/Safari or Deno --unstable-webgpu)");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const want = ["float32-filterable"].filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredFeatures: want });
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
function spacingFromIjkToRAS(ijkToRAS) {
  const col = (c) => Math.hypot(ijkToRAS[c], ijkToRAS[4 + c], ijkToRAS[8 + c]);
  return [col(0), col(1), col(2)];
}

// render/scene-renderer.ts
var DEFAULT_FORMAT = "rgba8unorm-srgb";
var SCENE_FLOATS = 16;
var SceneRenderer = class {
  dev;
  format;
  placed = [];
  pipeline;
  sampler;
  camBuf;
  matBuf;
  mat;
  bind;
  constructor(gpu, format = DEFAULT_FORMAT) {
    this.dev = gpu.device;
    this.format = format;
    this.sampler = this.dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.camBuf = this.dev.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
    this.mat = new Float32Array(uoff);
    this.matBuf = this.dev.createBuffer({ size: uoff * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: this.dev.createShaderModule({ code: this.wgsl() }), entryPoint: "vs_main" },
      fragment: { module: this.dev.createShaderModule({ code: this.wgsl() }), entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    const entries = [
      { binding: 0, resource: { buffer: this.camBuf } },
      { binding: 1, resource: { buffer: this.matBuf } },
      { binding: 2, resource: this.sampler }
    ];
    for (const p of this.placed) entries.push(...p.field.bindEntries(p.slot, p.bbase));
    this.bind = this.dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries });
    this.setBackground(0.07, 0.08, 0.12);
    const step = this.placed.length ? Math.min(...this.placed.map((p) => p.field.sampleStep())) : 1;
    this.setSampleStep(step * 0.7);
    this.recomputeBounds();
    for (const p of this.placed) p.field.fillUniforms(this.mat, p.uoff);
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
    const fns = [modFns, tpFns, fieldFns].filter((s) => s.trim()).join("\n");
    const dispatch = receivers.map((p) => `    { let c = sample_field_${p.field.kind}${p.slot}(wp, rd); sum += c; }`).join("\n");
    return (
      /* wgsl */
      `
struct Camera { inv_view_proj : mat4x4<f32>, size : vec4<f32> };
struct Material {
  bmin : vec4<f32>,
  bmax : vec4<f32>,
  scene : vec4<f32>,   // sample_step, _, _, _
  bg : vec4<f32>,
${members}
};
@group(0) @binding(0) var<uniform> u_cam : Camera;
@group(0) @binding(1) var<uniform> u_material : Material;
@group(0) @binding(2) var s_lin : sampler;
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

@fragment
fn fs_main(v : Varyings) -> @location(0) vec4<f32> {
  let size = u_cam.size.xy;
  let ndc_x = (v.position.x / size.x) * 2.0 - 1.0;
  let ndc_y = 1.0 - (v.position.y / size.y) * 2.0;
  let ro = ndc_to_world(vec4<f32>(ndc_x, ndc_y, 0.0, 1.0));
  let rd = normalize(ndc_to_world(vec4<f32>(ndc_x, ndc_y, 1.0, 1.0)) - ro);
  let bg = srgb2physical(u_material.bg.rgb);

  let inv = vec3<f32>(1.0) / rd;
  let tb = (u_material.bmin.xyz - ro) * inv;
  let tt = (u_material.bmax.xyz - ro) * inv;
  let tmn = min(tt, tb); let tmx = max(tt, tb);
  var t_near = max(max(tmn.x, tmn.y), tmn.z);
  var t_far  = min(min(tmx.x, tmx.y), tmx.z);
  if (t_far <= t_near || t_far <= 0.0) { return vec4<f32>(bg, 1.0); }

  let step = max(u_material.scene.x, 1e-3);
  t_near = max(t_near + step, 0.0);
  t_far  = t_far - step;
  if (t_far <= t_near) { return vec4<f32>(bg, 1.0); }
  let seed = ign(v.position.xy);
  var t = t_near;
  var integrated = vec4<f32>(0.0);
  var safety : i32 = 0;
  loop {
    if (t >= t_far || safety >= 5000 || integrated.a >= 0.99) { break; }
    let js = fract(sin(dot(v.position.xy + vec2<f32>(f32(safety) * 0.7548, f32(safety) * 0.5698), vec2<f32>(12.9898, 78.233))) * 43758.5453) - 0.5; // per-(pixel,sample) jitter
    let wp = ro + rd * (t + js * step);
    var sum = vec4<f32>(0.0);
${dispatch}
    if (sum.a > 0.0) { integrated = integrated + (1.0 - integrated.a) * vec4<f32>(sum.rgb, clamp(sum.a, 0.0, 1.0)); }
    t = t + step;
    safety = safety + 1;
  }
  return vec4<f32>(mix(bg, integrated.rgb, integrated.a), 1.0);
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
  /** Rebuild the bind group from the fields' current resources (e.g. after a field
   *  swapped a texture) without recompiling the pipeline. Field set/structure must be unchanged. */
  refreshBindings() {
    const entries = [
      { binding: 0, resource: { buffer: this.camBuf } },
      { binding: 1, resource: { buffer: this.matBuf } },
      { binding: 2, resource: this.sampler }
    ];
    for (const p of this.placed) entries.push(...p.field.bindEntries(p.slot, p.bbase));
    this.bind = this.dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries });
  }
  setCamera(eye, center, up, fovyDeg, width, height) {
    const view = lookAt(eye, center, up);
    const proj = perspectiveZO(fovyDeg * Math.PI / 180, width / height, 1, 1e5);
    const invVP = invert(multiply(proj, view));
    const cam = new Float32Array(20);
    cam.set(invVP, 0);
    cam[16] = width;
    cam[17] = height;
    this.dev.queue.writeBuffer(this.camBuf, 0, cam);
  }
  flush() {
    this.dev.queue.writeBuffer(this.matBuf, 0, this.mat);
  }
  renderToView(view, width, height) {
    this.flush();
    const enc = this.dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.draw(3);
    pass.end();
    this.dev.queue.submit([enc.finish()]);
  }
  async renderToRGBA(width, height) {
    this.flush();
    const target = this.dev.createTexture({ size: [width, height], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const enc = this.dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.draw(3);
    pass.end();
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
var ImageField = class {
  kind = "img";
  bindingCount = 2;
  // volume (3d) + lut (2d)
  volTex;
  lutTex;
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
  }
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
    out[off + 16] = this.clim[0];
    out[off + 17] = this.clim[1];
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
  }
  setSpheres(list) {
    this.n = Math.min(list.length, MAX);
    this.spheres.fill(0);
    this.colors.fill(0);
    for (let i = 0; i < this.n; i++) {
      const s = list[i];
      this.spheres.set([s.center[0], s.center[1], s.center[2], s.radius], i * 4);
      this.colors.set(s.color, i * 4);
    }
  }
  get count() {
    return this.n;
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
      const r = this.spheres[i * 4 + 3];
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], this.spheres[i * 4 + a] - r);
        hi[a] = Math.max(hi[a], this.spheres[i * 4 + a] + r);
      }
    }
    return [lo, hi];
  }
  structMembers(s) {
    return [
      `  fid${s}_params : vec4<f32>,`,
      // n_spheres, visible, shininess, k_ambient
      `  fid${s}_params2 : vec4<f32>,`,
      // k_diffuse, k_specular, _, _
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
  var found = false;
  for (var k = 0; k < n; k = k + 1) {
    let sp = u_material.fid${s}_spheres[k];
    let r = sp.w;
    if (r <= 0.0) { continue; }
    let depth = r - length(wp_r - sp.xyz);   // > 0 -> inside this sphere
    if (depth > best_depth) { best_depth = depth; best_center = sp.xyz; best_color = u_material.fid${s}_colors[k]; found = true; }
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
  let opacity = clamp(best_color.a, 0.0, 1.0);
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
    out[off + 8] = this.light[0];
    out[off + 9] = this.light[1];
    out[off + 10] = this.light[2];
    out.set(this.spheres, off + 12);
    out.set(this.colors, off + 12 + MAX * 4);
  }
};

// render/transform-field.ts
var TransformField = class {
  kind = "grid";
  modifier = true;
  // never composited into the ray-march sum
  bindingCount = 1;
  // displacement texture (sampler shared)
  tex;
  p2t;
  box;
  gainValue;
  stepMm;
  /** `displacement` is (dx,dy,dz,_) per voxel in RAS mm, C-order (z,y,x), dims=[X,Y,Z]. */
  constructor(dev, displacement, dims, spacing, opts = {}) {
    const center = opts.center ?? [0, 0, 0];
    this.gainValue = opts.gain ?? 1;
    this.tex = dev.createTexture({
      size: dims,
      dimension: "3d",
      format: "rgba32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    dev.queue.writeTexture({ texture: this.tex }, displacement, { bytesPerRow: dims[0] * 16, rowsPerImage: dims[1] }, dims);
    this.p2t = patientToTexture(dims, spacing, center);
    this.box = volumeAABB(dims, spacing, center);
    this.stepMm = Math.min(...spacing);
  }
  get gain() {
    return this.gainValue;
  }
  setGain(g) {
    this.gainValue = g;
  }
  uniformFloats() {
    return 20;
  }
  // mat4(16) + params(4)
  aabb() {
    return this.box;
  }
  sampleStep() {
    return this.stepMm;
  }
  structMembers(s) {
    return [
      `  grid${s}_p2t : mat4x4<f32>,`,
      `  grid${s}_params : vec4<f32>,`
      // gain, _, _, _
    ].join("\n");
  }
  declareBindings(s, base) {
    return `@group(0) @binding(${base}) var t_grid${s} : texture_3d<f32>;`;
  }
  /** Modifier fields emit the displacement lookup, not a sample_field_* function. */
  samplingWGSL(s) {
    return (
      /* wgsl */
      `
fn displacement_grid${s}(wp : vec3<f32>) -> vec3<f32> {
  let t4 = u_material.grid${s}_p2t * vec4<f32>(wp, 1.0);
  let tex = t4.xyz;
  if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) { return vec3<f32>(0.0); }
  let d = textureSampleLevel(t_grid${s}, s_lin, tex, 0.0).xyz;
  return u_material.grid${s}_params.x * d;
}`
    );
  }
  fillUniforms(out, off) {
    out.set(this.p2t, off);
    out[off + 16] = this.gainValue;
  }
  bindEntries(_s, base) {
    return [{ binding: base, resource: this.tex.createView() }];
  }
};
function solve(A, b) {
  const n = b.length;
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    [b[c], b[piv]] = [b[piv], b[c]];
    const d = A[c][c];
    if (Math.abs(d) < 1e-12) continue;
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / d;
      if (!f) continue;
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
    x[r] = Math.abs(A[r][r]) < 1e-12 ? 0 : s / A[r][r];
  }
  return x;
}
function tps3d(source, target) {
  const n = source.length;
  const U = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const build = () => {
    const M = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(n + 4).fill(0);
      for (let j = 0; j < n; j++) row[j] = U(source[i], source[j]);
      row[n] = 1;
      row[n + 1] = source[i][0];
      row[n + 2] = source[i][1];
      row[n + 3] = source[i][2];
      M.push(row);
    }
    for (let c = 0; c < 4; c++) {
      const row = new Array(n + 4).fill(0);
      for (let j = 0; j < n; j++) row[j] = c === 0 ? 1 : source[j][c - 1];
      M.push(row);
    }
    return M;
  };
  const coeffs = [];
  for (let c = 0; c < 3; c++) {
    const b = new Array(n + 4).fill(0);
    for (let i = 0; i < n; i++) b[i] = target[i][c] - source[i][c];
    coeffs.push(solve(build(), b));
  }
  return (p) => {
    const out = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const w = coeffs[c];
      let v = w[n] + w[n + 1] * p[0] + w[n + 2] * p[1] + w[n + 3] * p[2];
      for (let i = 0; i < n; i++) v += w[i] * U(p, source[i]);
      out[c] = v;
    }
    return out;
  };
}
function sampleDisplacementGrid(dims, spacing, center, f) {
  const [X, Y, Z] = dims;
  const out = new Float32Array(X * Y * Z * 4);
  const ext = [X * spacing[0], Y * spacing[1], Z * spacing[2]];
  for (let z = 0; z < Z; z++) {
    for (let y = 0; y < Y; y++) {
      for (let x = 0; x < X; x++) {
        const p = [
          center[0] - ext[0] / 2 + (x + 0.5) * spacing[0],
          center[1] - ext[1] / 2 + (y + 0.5) * spacing[1],
          center[2] - ext[2] / 2 + (z + 0.5) * spacing[2]
        ];
        const d = f(p);
        const o = ((z * Y + y) * X + x) * 4;
        out[o] = d[0];
        out[o + 1] = d[1];
        out[o + 2] = d[2];
        out[o + 3] = 0;
      }
    }
  }
  return out;
}

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
function buildLUT() {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    let r = 0, g = 0, b = 0, a = 0;
    if (i >= 15 && i < 120) {
      const t = clamp01((i - 25) / 55);
      r = 0.78;
      g = 0.4;
      b = 0.34;
      a = 0.02 + 0.11 * t;
    } else if (i >= 120) {
      const t = clamp01((i - 120) / 100);
      r = 0.8 + 0.15 * t;
      g = 0.76 + 0.16 * t;
      b = 0.66 + 0.19 * t;
      a = 0.25 + 0.65 * t;
    }
    lut[i * 4] = Math.round(r * 255);
    lut[i * 4 + 1] = Math.round(g * 255);
    lut[i * 4 + 2] = Math.round(b * 255);
    lut[i * 4 + 3] = Math.round(a * 255);
  }
  return lut;
}
function orbitEye(azimuth, elevation, distance) {
  const ce = Math.cos(elevation);
  return [
    distance * ce * Math.sin(azimuth),
    -distance * ce * Math.cos(azimuth),
    distance * Math.sin(elevation)
  ];
}

// render/demos/deform-scene.ts
var GRID_DIMS = [32, 32, 32];
var WORLD = N * SPACING[0];
var GRID_SPACING = [WORLD / GRID_DIMS[0], WORLD / GRID_DIMS[1], WORLD / GRID_DIMS[2]];
function landmarks() {
  const R = 52;
  const source = [
    [R, 0, 0],
    [-R, 0, 0],
    [0, R, 0],
    [0, -R, 0],
    [0, 0, R],
    [0, 0, -R]
  ];
  const target = [
    [R + 34, 0, 12],
    [-R, 0, 0],
    [0, R + 10, 0],
    [0, -R, 0],
    [0, 0, R - 26],
    [0, 0, -R]
  ];
  return { source, target };
}
function buildDeformScene(dev, gain = 1) {
  const { source, target } = landmarks();
  const f = tps3d(source, target);
  const disp = sampleDisplacementGrid(GRID_DIMS, GRID_SPACING, [0, 0, 0], f);
  const warp = new TransformField(dev, disp, GRID_DIMS, GRID_SPACING, { gain });
  const image = new ImageField(dev, syntheticVolume(), [N, N, N], SPACING, buildLUT(), {
    clim: [0, 255],
    shade: [0.35, 0.75, 0.35, 24]
  });
  image.transform = warp;
  const pins = [
    ...source.map((c) => ({ center: c, radius: 4, color: [0.25, 0.85, 1, 1] })),
    // cyan = source
    ...target.map((c) => ({ center: c, radius: 4, color: [1, 0.35, 0.85, 1] }))
    // magenta = target
  ];
  const fiducials = new FiducialField(pins, { shininess: 90, kSpecular: 0.6 });
  return { image, warp, fiducials };
}

// render/introspect.ts
var LOG_MAX = 500;
function installIntrospection(api) {
  const log = [];
  const hook = {
    ...api,
    ready: true,
    log,
    logEvent(kind, detail = {}) {
      log.push({ t: Math.round(performance.now()), kind, detail });
      if (log.length > LOG_MAX) log.shift();
    },
    clearLog() {
      log.length = 0;
    },
    snapshot() {
      const s = { camera: api.getCamera() };
      try {
        if (api.getPlanes) s.planes = api.getPlanes();
      } catch (e) {
        s.planesErr = String(e);
      }
      try {
        if (api.getVolume) s.volume = api.getVolume();
      } catch (e) {
        s.volumeErr = String(e);
      }
      try {
        if (api.extra) s.extra = api.extra();
      } catch (e) {
        s.extraErr = String(e);
      }
      s.logCount = log.length;
      return s;
    }
  };
  globalThis.__slicerlive = hook;
  return hook;
}

// render/demos/deform-browser.ts
var status = (msg, err = false) => {
  const el = document.getElementById("status");
  if (el) {
    el.textContent = msg;
    el.style.color = err ? "#ff6b74" : "#9fb3d0";
  }
};
async function main() {
  const canvas = document.getElementById("gpu");
  if (!navigator.gpu) {
    status("WebGPU not available \u2014 try Chrome/Edge 113+ or Safari 18+.", true);
    return;
  }
  status("initializing WebGPU\u2026");
  const gpu = await initDevice();
  const ctx = canvas.getContext("webgpu");
  const preferred = navigator.gpu.getPreferredCanvasFormat();
  const srgb = preferred + "-srgb";
  ctx.configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
  status("solving thin-plate spline\u2026");
  const sc = buildDeformScene(gpu.device, 1);
  const scene = new SceneRenderer(gpu, srgb);
  scene.build([sc.warp, sc.image, sc.fiducials]);
  scene.setBackground(0.06, 0.07, 0.1);
  let az = 0.75, el = 0.28, dist = 360;
  const draw = () => {
    const w = canvas.width, h = canvas.height;
    scene.setCamera(orbitEye(az, el, dist), [0, 0, 0], [0, 0, 1], 26, w, h);
    const t0 = performance.now();
    scene.renderToView(ctx.getCurrentTexture().createView({ format: srgb }), w, h);
    status(`Landmark deform (TPS) \xB7 gain ${sc.warp.gain.toFixed(2)} \xB7 ${w}\xD7${h} \xB7 ${(performance.now() - t0).toFixed(0)} ms/frame \xB7 drag to orbit`);
  };
  const resize = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    const size = Math.min(720, Math.floor(canvas.clientWidth * dpr));
    canvas.width = size;
    canvas.height = size;
    draw();
  };
  globalThis.addEventListener("resize", resize);
  const slider = document.getElementById("gain");
  slider?.addEventListener("input", () => {
    sc.warp.setGain(Number(slider.value) / 100);
    scene.build([sc.warp, sc.image, sc.fiducials]);
    scene.setBackground(0.06, 0.07, 0.1);
    draw();
  });
  let dragging = false, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lx = e.clientX;
    ly = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false;
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    az += (e.clientX - lx) * 8e-3;
    el = Math.max(-1.4, Math.min(1.4, el - (e.clientY - ly) * 8e-3));
    lx = e.clientX;
    ly = e.clientY;
    draw();
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    dist = Math.max(120, Math.min(900, dist * (e.deltaY > 0 ? 1.08 : 0.93)));
    draw();
  }, { passive: false });
  installIntrospection({
    getCamera: () => ({ azimuth: az, elevation: el, distance: dist, position: orbitEye(az, el, dist), focalPoint: [0, 0, 0], viewUp: [0, 0, 1], viewAngle: 26 }),
    setCamera: (p) => {
      if (p.azimuth !== void 0) az = p.azimuth;
      if (p.elevation !== void 0) el = p.elevation;
      if (p.distance !== void 0) dist = p.distance;
      draw();
    },
    extra: () => ({ gain: sc.warp.gain }),
    render: () => draw()
  });
  resize();
}
main().catch((e) => status("error: " + (e?.message ?? e), true));
