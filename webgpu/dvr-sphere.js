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
function patientToTextureCentered(dims, spacing) {
  const m = new Float32Array(16);
  for (let a = 0; a < 3; a++) {
    const s = 1 / (spacing[a] * dims[a]);
    m[a * 4 + a] = s;
    m[12 + a] = ((dims[a] - 1) / 2 + 0.5) / dims[a];
  }
  m[15] = 1;
  return m;
}

// render/volume-renderer.ts
var DEFAULT_FORMAT = "rgba8unorm-srgb";
var SHADER = (
  /* wgsl */
  `
struct Camera {
  inv_view_proj : mat4x4<f32>,
  size          : vec4<f32>,   // physical_size.x, .y, _, _
};
struct Material {
  patient_to_texture : mat4x4<f32>,
  clim               : vec4<f32>,   // lo, hi
  gradient_range     : vec4<f32>,   // gmin, gmax
  bounds_min         : vec4<f32>,
  bounds_max         : vec4<f32>,
  background         : vec4<f32>,   // sRGB rgb, a
  shade              : vec4<f32>,   // k_a, k_d, k_s, shininess
  steps              : vec4<f32>,   // sample_step, opacity_unit_distance, grad_opacity_on, sample_budget
  dither             : vec4<f32>,   // dither_scale, ox, oy, frame_seed
};

@group(0) @binding(0) var<uniform> u_cam : Camera;
@group(0) @binding(1) var<uniform> u_material : Material;
@group(0) @binding(2) var s_lin : sampler;
@group(0) @binding(3) var t_volume : texture_3d<f32>;
@group(0) @binding(4) var t_lut : texture_2d<f32>;
@group(0) @binding(5) var t_grad : texture_2d<f32>;

struct Varyings { @builtin(position) position : vec4<f32> };

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> Varyings {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var o : Varyings;
  o.position = vec4<f32>(x, y, 0.0, 1.0);
  return o;
}

fn srgb2physical(c : vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
fn ndc_to_world(ndc : vec4<f32>) -> vec3<f32> {
  let w = u_cam.inv_view_proj * ndc;
  return w.xyz / w.w;
}
fn sample_lut(value : f32) -> vec4<f32> {
  let t = clamp((value - u_material.clim.x) / max(u_material.clim.y - u_material.clim.x, 1e-6), 0.0, 1.0);
  return textureSampleLevel(t_lut, s_lin, vec2<f32>(t, 0.5), 0.0);
}
fn sample_volume_world(wp : vec3<f32>) -> vec2<f32> {
  let tex4 = u_material.patient_to_texture * vec4<f32>(wp, 1.0);
  let tex = tex4.xyz;
  if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) { return vec2<f32>(0.0, 0.0); }
  return vec2<f32>(textureSampleLevel(t_volume, s_lin, tex, 0.0).r, 1.0);
}
fn sample_volume_clamped(wp : vec3<f32>) -> f32 {
  let tex4 = u_material.patient_to_texture * vec4<f32>(wp, 1.0);
  return textureSampleLevel(t_volume, s_lin, clamp(tex4.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r;
}
fn gradient_world(wp : vec3<f32>, h : f32) -> vec3<f32> {
  let gx = sample_volume_clamped(wp + vec3<f32>(h,0,0)) - sample_volume_clamped(wp - vec3<f32>(h,0,0));
  let gy = sample_volume_clamped(wp + vec3<f32>(0,h,0)) - sample_volume_clamped(wp - vec3<f32>(0,h,0));
  let gz = sample_volume_clamped(wp + vec3<f32>(0,0,h)) - sample_volume_clamped(wp - vec3<f32>(0,0,h));
  return vec3<f32>(gx, gy, gz) / (2.0 * h);
}

@fragment
fn fs_main(v : Varyings) -> @location(0) vec4<f32> {
  let size = u_cam.size.xy;
  let ndc_x = (v.position.x / size.x) * 2.0 - 1.0;
  let ndc_y = 1.0 - (v.position.y / size.y) * 2.0;
  let ro = ndc_to_world(vec4<f32>(ndc_x, ndc_y, 0.0, 1.0));   // WebGPU near z=0
  let rd = normalize(ndc_to_world(vec4<f32>(ndc_x, ndc_y, 1.0, 1.0)) - ro);

  let bmin = u_material.bounds_min.xyz;
  let bmax = u_material.bounds_max.xyz;
  let inv = vec3<f32>(1.0) / rd;
  let tb = (bmin - ro) * inv;
  let tt = (bmax - ro) * inv;
  let tmn = min(tt, tb);
  let tmx = max(tt, tb);
  var t_near = max(max(tmn.x, tmn.y), tmn.z);
  var t_far  = min(min(tmx.x, tmx.y), tmx.z);
  let bg = srgb2physical(u_material.background.rgb);
  if (t_far <= t_near || t_far <= 0.0) { return vec4<f32>(bg, 1.0); }

  let step = max(u_material.steps.x, 1e-3);
  let unit = max(u_material.steps.y, 1e-3);
  t_near = max(t_near + step, 0.0);
  t_far  = t_far - step;
  if (t_far <= t_near) { return vec4<f32>(bg, 1.0); }

  let dpos = v.position.xy * u_material.dither.x + vec2<f32>(u_material.dither.y, u_material.dither.z);
  let seed = fract(sin(dot(vec3<f32>(dpos, 0.0), vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
  var t = t_near + seed * step;

  let k_a = u_material.shade.x; let k_d = u_material.shade.y;
  let k_s = u_material.shade.z; let sh = u_material.shade.w;
  let grad_on = u_material.steps.z;

  var integrated = vec4<f32>(0.0);
  var safety : i32 = 0;
  loop {
    if (t >= t_far || safety >= 5000 || integrated.a >= 0.99) { break; }
    let wp = ro + rd * t;
    let s = sample_volume_world(wp);
    if (s.y > 0.5) {
      let tf = sample_lut(s.x);
      let grad = gradient_world(wp, step);
      let glen = length(grad);
      var opacity = 1.0 - pow(1.0 - clamp(tf.a, 0.0, 1.0), step / unit);
      if (grad_on > 0.5) {
        let gmin = u_material.gradient_range.x;
        let gmax = max(u_material.gradient_range.y, gmin + 1e-6);
        let gn = clamp((glen - gmin) / (gmax - gmin), 0.0, 1.0);
        opacity = opacity * textureSampleLevel(t_grad, s_lin, vec2<f32>(gn, 0.5), 0.0).r;
      }
      opacity = clamp(opacity, 0.0, 1.0);
      if (opacity > 0.001) {
        var lit_srgb = tf.rgb * k_a;
        if (glen > 1e-6) {
          var n = grad / glen;
          if (dot(n, -rd) < 0.0) { n = -n; }
          let view_dir = normalize(ro - wp);
          let ldotn = dot(view_dir, n);      // headlight: light == eye
          if (ldotn > 0.0) {
            let refl = normalize(2.0 * ldotn * n - view_dir);
            let rdotv = max(0.0, dot(refl, view_dir));
            lit_srgb = tf.rgb * (k_a + k_d * ldotn) + vec3<f32>(k_s * pow(rdotv, sh));
          }
        }
        let lit = srgb2physical(clamp(lit_srgb, vec3<f32>(0.0), vec3<f32>(1.0)));
        integrated = integrated + (1.0 - integrated.a) * vec4<f32>(opacity * lit, opacity);
      }
    }
    t = t + step;
    safety = safety + 1;
  }
  let final_linear = mix(bg, integrated.rgb, integrated.a);
  return vec4<f32>(final_linear, 1.0);
}
`
);
var VolumeRenderer = class {
  dev;
  pipeline;
  sampler;
  camBuf;
  matBuf;
  volTex;
  lutTex;
  gradTex;
  bind;
  mat = new Float32Array(48);
  // 192-byte Material UBO = 48 f32
  dims = [1, 1, 1];
  format;
  constructor(gpu, format = DEFAULT_FORMAT) {
    this.dev = gpu.device;
    this.format = format;
    const module = this.dev.createShaderModule({ code: SHADER });
    this.pipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.sampler = this.dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.camBuf = this.dev.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.matBuf = this.dev.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.lutTex = this.makeLUT2D(defaultLUT());
    this.gradTex = this.makeLUT2D(new Uint8Array(256 * 4).fill(255));
    this.mat.set(identity(), 0);
    this.setClim(0, 255);
    this.setShade(0.4, 0.7, 0.2, 10);
    this.setBackground(0.08, 0.09, 0.13);
    this.mat[40] = 1;
    this.mat[41] = 1;
    this.mat[42] = 0;
    this.mat[43] = 1;
    this.mat[44] = 1;
    this.mat[45] = 0;
    this.mat[46] = 0;
    this.mat[47] = 0;
  }
  makeLUT2D(rgba) {
    const tex = this.dev.createTexture({ size: [256, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.dev.queue.writeTexture({ texture: tex }, rgba, { bytesPerRow: 256 * 4 }, [256, 1]);
    return tex;
  }
  setVolume(v) {
    const [dx, dy, dz] = v.dims;
    this.dims = v.dims;
    this.volTex = this.dev.createTexture({
      size: [dx, dy, dz],
      dimension: "3d",
      format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    this.dev.queue.writeTexture({ texture: this.volTex }, v.data.buffer, { bytesPerRow: dx * 4, rowsPerImage: dy }, [dx, dy, dz]);
    this.mat.set(patientToTextureCentered(v.dims, v.spacing), 0);
    const ext = [dx * v.spacing[0] / 2, dy * v.spacing[1] / 2, dz * v.spacing[2] / 2];
    this.setBoundsMinMax([-ext[0], -ext[1], -ext[2]], [ext[0], ext[1], ext[2]]);
    const minSp = Math.min(...v.spacing);
    this.mat[40] = minSp;
    this.mat[41] = minSp;
    this.bind = void 0;
  }
  setLUT(rgba) {
    this.lutTex = this.makeLUT2D(rgba);
    this.bind = void 0;
  }
  setClim(lo, hi) {
    this.mat[16] = lo;
    this.mat[17] = hi;
  }
  setShade(a, d, s, sh) {
    this.mat[36] = a;
    this.mat[37] = d;
    this.mat[38] = s;
    this.mat[39] = sh;
  }
  setBackground(r, g, b) {
    this.mat[32] = r;
    this.mat[33] = g;
    this.mat[34] = b;
    this.mat[35] = 1;
  }
  setSampleStep(step, unit) {
    this.mat[40] = step;
    if (unit !== void 0) this.mat[41] = unit;
  }
  setBoundsMinMax(mn, mx) {
    this.mat[24] = mn[0];
    this.mat[25] = mn[1];
    this.mat[26] = mn[2];
    this.mat[28] = mx[0];
    this.mat[29] = mx[1];
    this.mat[30] = mx[2];
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
  ensureBind() {
    if (this.bind) return;
    if (!this.volTex) throw new Error("setVolume() first");
    this.bind = this.dev.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: { buffer: this.matBuf } },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.volTex.createView() },
        { binding: 4, resource: this.lutTex.createView() },
        { binding: 5, resource: this.gradTex.createView() }
      ]
    });
  }
  /** Render into a caller-supplied view (e.g. a browser canvas texture). */
  renderToView(view, width, height) {
    this.dev.queue.writeBuffer(this.matBuf, 0, this.mat);
    this.ensureBind();
    const enc = this.dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.draw(3);
    pass.end();
    this.dev.queue.submit([enc.finish()]);
  }
  /** Render to an offscreen texture and read back tightly-packed RGBA (width*height*4). */
  async renderToRGBA(width, height) {
    this.dev.queue.writeBuffer(this.matBuf, 0, this.mat);
    this.ensureBind();
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
function defaultLUT() {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const g = i;
    lut[i * 4] = g;
    lut[i * 4 + 1] = g;
    lut[i * 4 + 2] = g;
    lut[i * 4 + 3] = i;
  }
  return lut;
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

// render/demos/dvr-sphere-browser.ts
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
  const r = new VolumeRenderer(gpu, srgb);
  status("building volume\u2026");
  r.setVolume({ data: syntheticVolume(), dims: [N, N, N], spacing: SPACING });
  r.setLUT(buildLUT());
  r.setClim(0, 255);
  r.setShade(0.35, 0.75, 0.35, 24);
  r.setBackground(0.07, 0.08, 0.12);
  let az = 0.35, el = 0.32, dist = 340;
  const draw = () => {
    const w = canvas.width, h = canvas.height;
    r.setCamera(orbitEye(az, el, dist), [0, 0, 0], [0, 0, 1], 26, w, h);
    const t0 = performance.now();
    r.renderToView(ctx.getCurrentTexture().createView({ format: srgb }), w, h);
    status(`WebGPU LiveRenderer \xB7 ${w}\xD7${h} \xB7 ${(performance.now() - t0).toFixed(0)} ms/frame \xB7 drag to orbit, scroll to zoom`);
  };
  const resize = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    const size = Math.min(720, Math.floor(canvas.clientWidth * dpr));
    canvas.width = size;
    canvas.height = size;
    draw();
  };
  globalThis.addEventListener("resize", resize);
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
  resize();
}
main().catch((e) => status("error: " + (e?.message ?? e), true));
