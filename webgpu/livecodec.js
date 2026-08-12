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

// examples/livecodec/wgpu-net.js
var U = GPUBufferUsage;
var f32tof16 = (() => {
  const b = new ArrayBuffer(4), f = new Float32Array(b), i = new Uint32Array(b);
  return (x) => {
    f[0] = x;
    const bits = i[0];
    const s = bits >>> 16 & 32768;
    let e = bits >>> 23 & 255;
    let m = bits & 8388607;
    if (e === 255) return s | 31744 | (m ? 512 : 0);
    e = e - 127 + 15;
    if (e >= 31) return s | 31744;
    if (e <= 0) {
      if (e < -10) return s;
      m |= 8388608;
      return s | m >>> 14 - e;
    }
    return s | e << 10 | m >>> 13;
  };
})();
var f16tof32 = (h) => {
  const s = h & 32768 ? -1 : 1;
  const e = h >> 10 & 31;
  const m = h & 1023;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
};
var toF16 = (a) => {
  const u = new Uint16Array(a.length);
  for (let i = 0; i < a.length; i++) u[i] = f32tof16(a[i]);
  return u;
};
var _f16lut = null;
var fromF16 = (u) => {
  if (!_f16lut) {
    _f16lut = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) _f16lut[i] = f16tof32(i);
  }
  const a = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) a[i] = _f16lut[u[i]];
  return a;
};
function genConv(TM, TN, silu = false, kb = null) {
  const AS = 64 * TM, BS = 64 * TN, ACC = TM * TN;
  let af = "", bf = "", fma = "", wr = "";
  for (let i = 0; i < TM; i++) af += `af[${i}u]=As[(ar+${i}u)*4u+k4];`;
  for (let j = 0; j < TN; j++) bf += `bf[${j}u]=Bs[(br+${j}u)*4u+k4];`;
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) fma += `acc[${i * TN + j}u]+=f32(dot(af[${i}u],bf[${j}u]));`;
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) {
    wr += silu ? `{let gm=rowBase+ar+${i}u;let gn=colBase+br+${j}u;if(gm<d.Co&&gn<Nvox){let v=acc[${i * TN + j}u]+f32(bias[gm]);outp[gm*Nvox+gn]=f16(v/(1.0+exp(-v)));}}` : `{let gm=rowBase+ar+${i}u;let gn=colBase+br+${j}u;if(gm<d.Co&&gn<Nvox){outp[gm*Nvox+gn]=f16(acc[${i * TN + j}u])+bias[gm];}}`;
  }
  let gB;
  if (kb) {
    const { KD, KH, KW, pz, py, px } = kb, taps = KD * KH * KW;
    const zCheck = KD === 1 && pz === 0 ? "let iz=i32(oz);" : `let iz=i32(oz)+i32(kz)-${pz}; if(iz<0||iz>=i32(d.ID)){return f16(0);}`;
    gB = `fn gB(gk:u32,oz:u32,oy:u32,ox:u32)->f16{ let ci=gk/${taps}u; let tap=gk%${taps}u;
  let kz=tap/${KH * KW}u; let ky=(tap/${KW}u)%${KH}u; let kx=tap%${KW}u;
  ${zCheck}
  let iy=i32(oy)+i32(ky)-${py}; let ix=i32(ox)+i32(kx)-${px};
  if(iy<0||iy>=i32(d.IH)||ix<0||ix>=i32(d.IW)){return f16(0);}
  return inp[((ci*d.ID+u32(iz))*d.IH+u32(iy))*d.IW+u32(ix)]; }`;
  } else {
    gB = `fn gB(gk:u32,oz:u32,oy:u32,ox:u32)->f16{ let taps=d.KD*d.KH*d.KW; let ci=gk/taps; let tap=gk%taps;
  let kz=tap/(d.KH*d.KW); let ky=(tap/d.KW)%d.KH; let kx=tap%d.KW;
  let iz=i32(oz*d.S)+i32(kz)-i32(d.pz); let iy=i32(oy*d.S)+i32(ky)-i32(d.py); let ix=i32(ox*d.S)+i32(kx)-i32(d.px);
  if(iz<0||iz>=i32(d.ID)||iy<0||iy>=i32(d.IH)||ix<0||ix>=i32(d.IW)){return f16(0);}
  return inp[((ci*d.ID+u32(iz))*d.IH+u32(iy))*d.IW+u32(ix)]; }`;
  }
  const ktot = kb ? `d.Ci*${kb.KD * kb.KH * kb.KW}u` : "d.Ci*d.KD*d.KH*d.KW";
  return `struct D{Ci:u32,Co:u32,OD:u32,OH:u32,OW:u32,ID:u32,IH:u32,IW:u32,KD:u32,KH:u32,KW:u32,S:u32,pz:u32,py:u32,px:u32,aux:u32};
@group(0)@binding(0) var<storage,read> inp:array<f16>; @group(0)@binding(1) var<storage,read> wgt:array<f16>;
@group(0)@binding(2) var<storage,read> bias:array<f16>; @group(0)@binding(3) var<storage,read_write> outp:array<f16>;
@group(0)@binding(4) var<uniform> d:D;
var<workgroup> As:array<vec4<f16>,${AS}>; var<workgroup> Bs:array<vec4<f16>,${BS}>;
${gB}
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let Nvox=d.OD*d.OH*d.OW; let Ktot=${ktot}; let Ktiles=(Ktot+15u)/16u;
  let tid=lid.y*16u+lid.x; let rowBase=wid.y*${16 * TM}u; let ntile=wid.z*d.aux+wid.x; let colBase=ntile*${16 * TN}u;
  let OHW=d.OH*d.OW; let ar=lid.y*${TM}u; let br=lid.x*${TN}u;
  var acc:array<f32,${ACC}>; for(var i=0u;i<${ACC}u;i++){acc[i]=0.0;}
  for(var kk:u32=0u;kk<Ktiles;kk++){
    for(var e=tid;e<${AS}u;e+=256u){ let m=e/4u;let k4=e%4u;let gm=rowBase+m;let base=kk*16u+k4*4u; var v=vec4<f16>(0.0,0.0,0.0,0.0);
      if(gm<d.Co){let o=gm*Ktot+base; if(base<Ktot){v.x=wgt[o];} if(base+1u<Ktot){v.y=wgt[o+1u];} if(base+2u<Ktot){v.z=wgt[o+2u];} if(base+3u<Ktot){v.w=wgt[o+3u];}} As[e]=v; }
    for(var e=tid;e<${BS}u;e+=256u){ let n=e/4u;let k4=e%4u;let gn=colBase+n;let base=kk*16u+k4*4u; var v=vec4<f16>(0.0,0.0,0.0,0.0);
      if(gn<Nvox){let oz=gn/OHW;let rem=gn%OHW;let oy=rem/d.OW;let ox=rem%d.OW; v.x=gB(base,oz,oy,ox);v.y=gB(base+1u,oz,oy,ox);v.z=gB(base+2u,oz,oy,ox);v.w=gB(base+3u,oz,oy,ox);} Bs[e]=v; }
    workgroupBarrier();
    for(var k4=0u;k4<4u;k4++){ var af:array<vec4<f16>,${TM}>; var bf:array<vec4<f16>,${TN}>; ${af} ${bf} ${fma} }
    workgroupBarrier();
  }
  ${wr}
}`;
}
var GN_STATS = `struct P{Cg:u32,D:u32,HW:u32,G:u32,ps:u32};
@group(0)@binding(0) var<storage,read> x:array<f16>;
@group(0)@binding(1) var<storage,read_write> st:array<f32>; @group(0)@binding(2) var<uniform> p:P;
var<workgroup> ss:array<f32,256>; var<workgroup> sq:array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let w=wid.x; var g=w; var d0=0u; var N=p.Cg*p.D*p.HW;
  if(p.ps==1u){ g=w/p.D; d0=w%p.D; N=p.Cg*p.HW; }
  var s=0.0; var q=0.0;
  for(var i=lid.x;i<N;i+=256u){ var idx=g*N+i;
    if(p.ps==1u){ let cc=i/p.HW; let sp=i%p.HW; idx=((g*p.Cg+cc)*p.D+d0)*p.HW+sp; }
    let v=f32(x[idx]); s+=v; q+=v*v; }
  ss[lid.x]=s; sq[lid.x]=q; workgroupBarrier();
  for(var t=128u;t>0u;t>>=1u){ if(lid.x<t){ss[lid.x]+=ss[lid.x+t];sq[lid.x]+=sq[lid.x+t];} workgroupBarrier(); }
  if(lid.x==0u){ let m=ss[0]/f32(N); let vr=sq[0]/f32(N)-m*m; st[w*2u]=m; st[w*2u+1u]=inverseSqrt(max(vr,0.0)+1e-5); } }`;
var GN_APPLY = `struct P{C:u32,Cg:u32,D:u32,HW:u32,ps:u32,gx:u32};
@group(0)@binding(0) var<storage,read> x:array<f16>; @group(0)@binding(1) var<storage,read> st:array<f32>;
@group(0)@binding(2) var<storage,read> sc:array<f16>; @group(0)@binding(3) var<storage,read> bi:array<f16>;
@group(0)@binding(4) var<storage,read_write> y:array<f16>; @group(0)@binding(5) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let idx=(wid.y*p.gx+wid.x)*256u+lid.x; if(idx>=p.C*p.D*p.HW){return;}
  let c=idx/(p.D*p.HW); let g=c/p.Cg; var w=g;
  if(p.ps==1u){ w=g*p.D+(idx/p.HW)%p.D; }
  y[idx]=f16((f32(x[idx])-st[w*2u])*st[w*2u+1u]*f32(sc[c])+f32(bi[c])); }`;
var GN_APPLY_SILU = GN_APPLY.replace(
  "y[idx]=f16((f32(x[idx])-st[w*2u])*st[w*2u+1u]*f32(sc[c])+f32(bi[c])); }",
  "let v=(f32(x[idx])-st[w*2u])*st[w*2u+1u]*f32(sc[c])+f32(bi[c]); y[idx]=f16(v/(1.0+exp(-v))); }"
);
var SILU = `struct P{n:u32,gx:u32}; @group(0)@binding(0) var<storage,read> a:array<f16>;
@group(0)@binding(1) var<storage,read_write> y:array<f16>; @group(0)@binding(2) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let i=(wid.y*p.gx+wid.x)*256u+lid.x; if(i>=p.n){return;} let v=f32(a[i]); y[i]=f16(v/(1.0+exp(-v))); }`;
var ADD = `struct P{n:u32,gx:u32}; @group(0)@binding(0) var<storage,read> a:array<f16>;
@group(0)@binding(1) var<storage,read> b:array<f16>; @group(0)@binding(2) var<storage,read_write> y:array<f16>;
@group(0)@binding(3) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let i=(wid.y*p.gx+wid.x)*256u+lid.x; if(i>=p.n){return;} y[i]=a[i]+b[i]; }`;
var RESIZE = `struct D{C:u32,OD:u32,OH:u32,OW:u32,ID:u32,IH:u32,IW:u32,gx:u32};
@group(0)@binding(0) var<storage,read> x:array<f16>; @group(0)@binding(1) var<storage,read_write> y:array<f16>;
@group(0)@binding(2) var<uniform> d:D;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let idx=(wid.y*d.gx+wid.x)*256u+lid.x; let Nout=d.C*d.OD*d.OH*d.OW; if(idx>=Nout){return;}
  let OHW=d.OH*d.OW; let c=idx/(d.OD*OHW); let r=idx%(d.OD*OHW); let oz=r/OHW; let rr=r%OHW; let oy=rr/d.OW; let ox=rr%d.OW;
  let iz=(oz*d.ID)/d.OD; let iy=(oy*d.IH)/d.OH; let ix=(ox*d.IW)/d.OW;   // nearest/floor asymmetric
  y[idx]=x[((c*d.ID+iz)*d.IH+iy)*d.IW+ix]; }`;
var SWAPAB = `struct P{A:u32,B:u32,S:u32,gx:u32};
@group(0)@binding(0) var<storage,read> x:array<f16>; @group(0)@binding(1) var<storage,read_write> y:array<f16>;
@group(0)@binding(2) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let idx=(wid.y*p.gx+wid.x)*256u+lid.x; if(idx>=p.A*p.B*p.S){return;}
  let a=idx/(p.B*p.S); let b=(idx/p.S)%p.B; let s=idx%p.S;
  y[(b*p.A+a)*p.S+s]=x[idx]; }`;
var Net = class {
  constructor(dev, R, dtype = "f16") {
    this.dev = dev;
    this.R = R;
    this.dtype = dtype;
    this.esz = dtype === "f16" ? 2 : 4;
    this.ext = {};
    this.inBuf = {};
    this.convTM = 4;
    this.convTN = 4;
  }
  async load(graphUrl, weightsUrl) {
    const dev = this.dev;
    this.graph = await (await fetch(graphUrl)).json();
    const wblob = new Uint16Array(await (await fetch(weightsUrl)).arrayBuffer());
    this.prod = (s) => s.reduce((a, b) => a * b, 1);
    this.mk = (b) => dev.createBuffer({ size: Math.max(16, b), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.W = {};
    for (const [n, w] of Object.entries(this.graph.weights)) {
      const b = this.mk(w.numel * this.esz);
      const src = wblob.subarray(w.offset, w.offset + w.numel);
      dev.queue.writeBuffer(b, 0, this.dtype === "f16" ? src : fromF16(src));
      this.W[n] = b;
    }
    this._plan();
    return this;
  }
  bytesOf(n) {
    return this.prod(this.graph.tensors[n]) * this.esz;
  }
  _plan() {
    const g = this.graph, R = this.R;
    const inSet = new Set(g.inputs.map((i) => i.name));
    const cons = {};
    for (const nd of g.nodes) for (const i of nd.in) (cons[i] = cons[i] || []).push(nd);
    const skip = /* @__PURE__ */ new Set(), fuseOut = {};
    for (const nd of g.nodes) if (nd.op === "Conv" || nd.op === "GroupNorm") {
      const c = cons[nd.out[0]];
      if (c && c.length === 1 && c[0].op === "Silu") {
        skip.add(c[0]);
        fuseOut[nd.out[0]] = c[0].out[0];
      }
    }
    const ops = [];
    for (const nd of g.nodes) {
      if (skip.has(nd)) continue;
      const fo = nd.op === "Conv" || nd.op === "GroupNorm" ? fuseOut[nd.out[0]] : void 0;
      ops.push({ nd, out: fo || nd.out[0], fused: !!fo });
    }
    const lastUse = {};
    ops.forEach((o, i) => {
      for (const t of o.nd.in) lastUse[t] = i;
    });
    g.outputs.forEach((o) => lastUse[o.name] = 1e18);
    const T = {}, pool = [];
    this.zeroBias = {};
    const acquire = (b) => {
      let bi = -1;
      for (let i = 0; i < pool.length; i++) if (pool[i].size >= b && (bi < 0 || pool[i].size < pool[bi].size)) bi = i;
      return bi >= 0 ? pool.splice(bi, 1)[0] : this.mk(b);
    };
    const zB = (C) => this.zeroBias[C] = this.zeroBias[C] || this.mk(C * this.esz);
    this.recs = [];
    ops.forEach((o, i) => {
      const nd = o.nd, tsr = g.tensors;
      if (!inSet.has(o.out)) T[o.out] = acquire(this.bytesOf(o.out));
      const r = { op: nd.op, ins: nd.in, out: o.out };
      if (nd.op === "Conv") {
        const os = tsr[o.out], is = tsr[nd.in[0]];
        const Nvox = os[2] * os[3] * os[4];
        r.bias = nd.bias ? nd.in[2] : null;
        r.Co = nd.Co;
        r.silu = o.fused;
        r.kb = { KD: nd.KD, KH: nd.KH, KW: nd.KW, pz: nd.padZ, py: nd.padY, px: nd.padX };
        r.cs = [nd.Ci, nd.Co, os[2], os[3], os[4], is[2], is[3], is[4], nd.KD, nd.KH, nd.KW, nd.S, nd.padZ, nd.padY, nd.padX];
        this._convDispatch(r);
      } else if (nd.op === "GroupNorm") {
        const s = tsr[o.out];
        const C = s[1], D = s[2], HW = s[3] * s[4];
        const G = nd.G, Cg = C / G, ps = nd.perSlice ? 1 : 0, nInst = ps ? G * D : G;
        r.stats = this.mk(nInst * 2 * 4);
        r.u1 = R.uni([Cg, D, HW, G, ps]);
        r.statsWg = [nInst, 1, 1];
        const gd = R.grid(C * D * HW);
        r.u2 = R.uni([C, Cg, D, HW, ps, gd.gx]);
        r.wg = gd.wg;
        r.applyK = o.fused ? R.GN_APPLY_SILU : R.GN_APPLY;
      } else if (nd.op === "Silu" || nd.op === "Add") {
        const n = this.prod(tsr[o.out]);
        const gd = R.grid(n);
        r.u = R.uni([n, gd.gx]);
        r.wg = gd.wg;
      } else if (nd.op === "Concat") {
        r.parts = nd.in.map((x) => this.bytesOf(x));
      } else if (nd.op === "Resize") {
        const os = tsr[o.out], is = tsr[nd.in[0]];
        const Nout = os[1] * os[2] * os[3] * os[4];
        const gd = R.grid(Nout);
        r.u = R.uni([os[1], os[2], os[3], os[4], is[2], is[3], is[4], gd.gx]);
        r.wg = gd.wg;
      } else if (nd.op === "SwapAB") {
        const S = this.prod(tsr[o.out]) / (nd.A * nd.B);
        const gd = R.grid(nd.A * nd.B * S);
        r.u = R.uni([nd.A, nd.B, S, gd.gx]);
        r.wg = gd.wg;
      } else throw new Error("unsupported op " + nd.op);
      r.inB = nd.in.map((n) => inSet.has(n) ? { inp: n } : this.W[n] || T[n]);
      r.outB = T[o.out];
      if (r.bias) r.biasB = this.W[r.bias];
      this.recs.push(r);
      for (const t of nd.in) if (lastUse[t] === i && !(t in this.W) && !inSet.has(t) && T[t]) {
        pool.push(T[t]);
        T[t] = null;
      }
    });
    this.T = T;
    this.zB = zB;
  }
  _convDispatch(r) {
    const cs = r.cs, Nvox = cs[2] * cs[3] * cs[4];
    const TM = Math.min(this.convTM, Math.max(1, Math.ceil(cs[1] / 16)));
    const tileR = 16 * TM, tileC = 16 * this.convTN;
    const gx = Math.min(65535, Math.ceil(Nvox / tileC));
    r.u = this.R.uni([...cs, gx]);
    r.wg = [gx, Math.ceil(cs[1] / tileR), Math.ceil(Nvox / tileC / gx)];
    r.src = genConv(TM, this.convTN, r.silu, r.kb);
  }
  setInputBuffer(name, buf) {
    this.ext[name] = buf;
  }
  setInputData(name, f32) {
    if (!this.inBuf[name]) this.inBuf[name] = this.mk(this.bytesOf(name));
    this.dev.queue.writeBuffer(this.inBuf[name], 0, this.dtype === "f16" ? toF16(f32) : f32);
    this.ext[name] = this.inBuf[name];
  }
  run() {
    const B = (x) => x && x.inp ? this.ext[x.inp] : x, R = this.R, enc = this.dev.createCommandEncoder();
    for (const r of this.recs) {
      const i = r.inB, out = r.outB;
      if (r.op === "Conv") R.pass(enc, r.src, [B(i[0]), B(i[1]), r.bias ? r.biasB : this.zB(r.Co), out, r.u], r.wg);
      else if (r.op === "GroupNorm") {
        R.pass(enc, R.GN_STATS, [B(i[0]), r.stats, r.u1], r.statsWg);
        R.pass(enc, r.applyK, [B(i[0]), r.stats, B(i[1]), B(i[2]), out, r.u2], r.wg);
      } else if (r.op === "Silu") R.pass(enc, R.SILU, [B(i[0]), out, r.u], r.wg);
      else if (r.op === "Add") R.pass(enc, R.ADD, [B(i[0]), B(i[1]), out, r.u], r.wg);
      else if (r.op === "Concat") {
        let off = 0;
        i.forEach((x, k) => {
          enc.copyBufferToBuffer(B(x), 0, out, off, r.parts[k]);
          off += r.parts[k];
        });
      } else if (r.op === "Resize") R.pass(enc, R.RESIZE, [B(i[0]), out, r.u], r.wg);
      else if (r.op === "SwapAB") R.pass(enc, R.SWAPAB, [B(i[0]), out, r.u], r.wg);
    }
    this.dev.queue.submit([enc.finish()]);
  }
  outBufFor(name) {
    for (const r of this.recs) if (r.out === name) return r.outB;
    return this.ext[name];
  }
  async read(name) {
    const buf = this.outBufFor(name), n = this.prod(this.graph.tensors[name]);
    const rb = this.dev.createBuffer({ size: n * this.esz, usage: U.MAP_READ | U.COPY_DST });
    const e = this.dev.createCommandEncoder();
    e.copyBufferToBuffer(buf, 0, rb, 0, n * this.esz);
    this.dev.queue.submit([e.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = this.dtype === "f16" ? fromF16(new Uint16Array(rb.getMappedRange().slice(0))) : new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return out;
  }
  // ---- per-GPU conv autotuning (nnLive scheme: verify against the reference tiling, then time) ----
  setConvConfig(TM, TN) {
    this.convTM = TM;
    this.convTN = TN;
    for (const r of this.recs) if (r.op === "Conv") this._convDispatch(r);
  }
  async _verifyConfigs(configs) {
    const dev = this.dev, R = this.R;
    const Ci = 8, Co = 20, OD = 9, OH = 9, OW = 9, K = 3, pad = 1;
    const Nin = Ci * OD * OH * OW, Nw = Co * Ci * K * K * K, Nout = Co * OD * OH * OW, Nvox = OD * OH * OW;
    const rnd = (n, seed) => {
      const a = new Float32Array(n);
      let s = seed >>> 0;
      for (let i = 0; i < n; i++) {
        s = s * 1103515245 + 12345 & 2147483647;
        a[i] = s / 2147483647 - 0.5;
      }
      return a;
    };
    const mkb = (f) => {
      const b = this.mk(f.length * this.esz);
      dev.queue.writeBuffer(b, 0, this.dtype === "f16" ? toF16(f) : f);
      return b;
    };
    const inB = mkb(rnd(Nin, 1)), wB = mkb(rnd(Nw, 2)), biasB = mkb(rnd(Co, 3)), outB = this.mk(Nout * this.esz);
    const readOut = async () => {
      const rb = dev.createBuffer({ size: Nout * this.esz, usage: U.MAP_READ | U.COPY_DST });
      const e = dev.createCommandEncoder();
      e.copyBufferToBuffer(outB, 0, rb, 0, Nout * this.esz);
      dev.queue.submit([e.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const o = this.dtype === "f16" ? fromF16(new Uint16Array(rb.getMappedRange().slice(0))) : new Float32Array(rb.getMappedRange().slice(0));
      rb.unmap();
      rb.destroy();
      return o;
    };
    const runCfg = (TM, TN) => {
      const tileC = 16 * TN, tileR = 16 * TM, gx = Math.min(65535, Math.ceil(Nvox / tileC));
      const u = R.uni([Ci, Co, OD, OH, OW, OD, OH, OW, K, K, K, 1, pad, pad, pad, gx]);
      const wg = [gx, Math.ceil(Co / tileR), Math.ceil(Nvox / tileC / gx)];
      const e = dev.createCommandEncoder();
      R.pass(e, genConv(TM, TN), [inB, wB, biasB, outB, u], wg);
      dev.queue.submit([e.finish()]);
    };
    runCfg(4, 4);
    const ref = await readOut();
    const ok = [];
    for (const [TM, TN] of configs) {
      try {
        runCfg(TM, TN);
        const o = await readOut();
        let md = 0;
        for (let i = 0; i < Nout; i++) {
          const dd = Math.abs(o[i] - ref[i]);
          if (dd > md) md = dd;
        }
        if (md < 0.05) ok.push([TM, TN]);
      } catch (e) {
      }
    }
    return ok.length ? ok : [[4, 4]];
  }
  // pick the fastest verified conv tiling for THIS gpu by timing the real forward. Inputs must be set.
  async autotuneConv(candidates = [[4, 4], [2, 8], [8, 2], [2, 4], [4, 2]], reps = 3) {
    const verified = await this._verifyConfigs(candidates);
    let best = [4, 4], bestMs = Infinity;
    for (const [TM, TN] of verified) {
      this.setConvConfig(TM, TN);
      this.run();
      await this.dev.queue.onSubmittedWorkDone();
      const t = performance.now();
      for (let i = 0; i < reps; i++) this.run();
      await this.dev.queue.onSubmittedWorkDone();
      const ms = (performance.now() - t) / reps;
      if (ms < bestMs - 0.5) {
        bestMs = ms;
        best = [TM, TN];
      }
    }
    this.setConvConfig(best[0], best[1]);
    return { TM: best[0], TN: best[1], ms: Math.round(bestMs), verified: verified.length, tried: candidates.length };
  }
};
function makeRunner(dev, dtype = "f16") {
  const pc = /* @__PURE__ */ new Map();
  const xform = (src) => dtype === "f16" ? "enable f16;\n" + src : src.replace(/f16/g, "f32");
  const pipe = (src) => {
    if (pc.has(src)) return pc.get(src);
    const m = dev.createShaderModule({ code: xform(src) });
    const p = dev.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } });
    pc.set(src, p);
    return p;
  };
  const uni = (arr) => {
    const b = dev.createBuffer({ size: Math.max(16, arr.length * 4), usage: U.UNIFORM | U.COPY_DST });
    dev.queue.writeBuffer(b, 0, new Uint32Array(arr));
    return b;
  };
  const pass = (enc, src, bufs, wg) => {
    const p = pipe(src);
    const bg = dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const cp = enc.beginComputePass();
    cp.setPipeline(p);
    cp.setBindGroup(0, bg);
    cp.dispatchWorkgroups(wg[0], wg[1] || 1, wg[2] || 1);
    cp.end();
  };
  const grid = (n) => {
    const nWG = Math.ceil(n / 256), gx = Math.min(65535, nWG);
    return { gx, wg: [gx, Math.ceil(nWG / gx), 1] };
  };
  return { pipe, uni, pass, grid, GN_STATS, GN_APPLY, GN_APPLY_SILU, SILU, ADD, RESIZE, SWAPAB };
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
  size : vec4<f32>,      // sizeX, sizeY, _, _
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var s_lin : sampler;
@group(0) @binding(2) var t_scalar : texture_3d<f32>;
@group(0) @binding(3) var t_overlay : texture_3d<f32>;
@group(0) @binding(4) var s_nn : sampler;   // NEAREST \u2014 labelmap overlay is per-voxel crisp (matches Slicer)

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
  return textureSampleLevel(t_overlay, s_nn, t, 0.0);
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
  let ov = textureSampleLevel(t_overlay, s_nn, tex, 0.0);
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
        { binding: 3, resource: this.overlay.createView() },
        { binding: 4, resource: this.nnSampler }
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
  const pinchState = () => {
    const [a, b] = [...pointers.values()];
    return { dist: Math.hypot(b.x - a.x, b.y - a.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
  };
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
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
    if (!pointers.has(e.pointerId)) return;
    const { x, y } = local(e);
    pointers.set(e.pointerId, { x, y });
    if (pointers.size >= 2) {
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

// render/demos/view-grid.ts
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
  { title: "Slice views", rows: [
    ["Wheel / Left-drag", "Scroll through slices"],
    ["Right-drag / \u2318-wheel", "Zoom this slice"],
    ["Middle / Shift+Left-drag", "Pan"],
    ["Double-click", "Maximize / restore"],
    ["R", "Reset pan/zoom"],
    ["Shift + move", "Jump the other views to the point under the cursor"]
  ] }
];
function glass(el2, extra = "") {
  el2.style.cssText += ";background:linear-gradient(135deg,rgba(58,64,88,.55),rgba(20,24,38,.66));backdrop-filter:blur(20px) saturate(1.6);-webkit-backdrop-filter:blur(20px) saturate(1.6);border:1px solid rgba(255,255,255,.2);box-shadow:0 18px 50px rgba(0,0,0,.55);" + extra;
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
  const rows = [];
  if (controls.length) {
    const head = document.createElement("div");
    head.textContent = "Visualization";
    head.style.cssText = "font:700 10px -apple-system,system-ui,sans-serif;letter-spacing:1.1px;text-transform:uppercase;color:#9fe9ff;margin:0 0 8px;";
    pop.appendChild(head);
    for (const c of controls) {
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
  } else if (opts.about === false && !opts.segments) {
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

// examples/livecodec/livecodec-scene.ts
var DEFAULT_BUCKET = "https://js2.jetstream-cloud.org:8001/livecodec-demo/";
var bucketParam = typeof location !== "undefined" ? new URLSearchParams(location.search).get("bucket") : null;
var BUCKET = bucketParam ? bucketParam.endsWith("/") ? bucketParam : bucketParam + "/" : DEFAULT_BUCKET;
var simBps = null;
function setSimulatedBandwidth(bitsPerSec) {
  simBps = bitsPerSec;
}
var LinkPacer = class {
  t0 = 0;
  bytes = 0;
  async admit(n) {
    if (simBps == null) return;
    if (!this.t0) this.t0 = performance.now();
    this.bytes += n;
    const due = this.t0 + this.bytes * 8 / simBps * 1e3;
    const wait = due - performance.now();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  }
};
var BandwidthMeter = class {
  stats = [];
  begin(name) {
    const s = { name, bytes: 0, t0: performance.now(), t1: performance.now() };
    this.stats.push(s);
    return {
      at: (cumulative) => {
        s.bytes = cumulative;
        s.t1 = performance.now();
      },
      add: (n) => {
        s.bytes += n;
        s.t1 = performance.now();
      }
    };
  }
  summary() {
    const iv = this.stats.map((s) => [s.t0, s.t1]).sort((a, b) => a[0] - b[0]);
    let seconds = 0, end = -Infinity;
    for (const [a, b] of iv) {
      seconds += Math.max(0, b - Math.max(a, end));
      end = Math.max(end, b);
    }
    seconds /= 1e3;
    const bytes = this.stats.reduce((t, s) => t + s.bytes, 0);
    return { bytes, seconds, mbps: seconds > 0 ? bytes * 8 / seconds / 1e6 : 0, streams: this.stats };
  }
};
async function streamFetch(url, onBytes, pacer) {
  const resp = await fetch(url, { cache: "no-store" });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  if (!resp.body) {
    const buf = new Uint8Array(await resp.arrayBuffer());
    await pacer?.admit(buf.byteLength);
    onBytes?.(buf.byteLength);
    return buf;
  }
  const parts = [];
  const rd = resp.body.getReader();
  let total = 0;
  for (; ; ) {
    const { done, value } = await rd.read();
    if (done) break;
    await pacer?.admit(value.byteLength);
    parts.push(value);
    total += value.byteLength;
    onBytes?.(total);
  }
  const all = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    all.set(p, o);
    o += p.byteLength;
  }
  return all;
}
async function gunzip(gz) {
  const ds = new DecompressionStream("gzip");
  const buf = await new Response(new Response(gz).body.pipeThrough(ds)).arrayBuffer();
  return new Uint8Array(buf);
}
function latentShapes(meta) {
  const f = meta.latent.fine, c = meta.latent.coarse;
  const s = {
    C: f[1],
    Df: f[2],
    Hf: f[3],
    Wf: f[4],
    Dc: c[2],
    Hc: c[3],
    Wc: c[4],
    chunks: meta.latent.chunks,
    chunkZ: meta.chunk_z,
    H: f[3] * 8,
    W: f[4] * 8
  };
  if (s.Df !== 2 * s.Dc || s.Hf !== 2 * s.Hc || s.Wf !== 2 * s.Wc) {
    throw new Error(`latent shapes not 2x: fine [${f}] vs coarse [${c}]`);
  }
  return s;
}
function dequantCoarseUp(codes, chunk, s, dec) {
  const { C, Dc, Hc, Wc, Df, Hf, Wf } = s;
  const src = chunk * C * Dc * Hc * Wc;
  const out = new Float32Array(C * Df * Hf * Wf);
  let o = 0;
  for (let c = 0; c < C; c++) {
    const off = dec.offset[c], inv = 1 / dec.half[c];
    const cb = src + c * Dc * Hc * Wc;
    for (let z = 0; z < Df; z++) {
      const zb = cb + (z >> 1) * Hc * Wc;
      for (let y = 0; y < Hf; y++) {
        const yb = zb + (y >> 1) * Wc;
        for (let x = 0; x < Wf; x++) out[o++] = (codes[yb + (x >> 1)] - off) * inv;
      }
    }
  }
  return out;
}
function dequantFine(codes, chunk, s, dec) {
  const { C, Df, Hf, Wf } = s;
  const per = Df * Hf * Wf;
  const src = chunk * C * per;
  const out = new Float32Array(C * per);
  let o = 0;
  for (let c = 0; c < C; c++) {
    const off = dec.offset[c], inv = 1 / dec.half[c];
    const cb = src + c * per;
    for (let i = 0; i < per; i++) out[o++] = (codes[cb + i] - off) * inv;
  }
  return out;
}
function mapOutputToHU(out, vol, z0, Z, s, dec) {
  const sliceSize = s.H * s.W;
  const zw = Math.min(s.chunkZ, Z - z0);
  const scale2 = (dec.hu_max - dec.hu_min) / 2;
  const n = zw * sliceSize, base = z0 * sliceSize;
  for (let i = 0; i < n; i++) vol[base + i] = (out[i] + 1) * scale2 + dec.hu_min;
}
function dcGridDims(shape) {
  const [Z, Y, X] = shape;
  return [Math.floor(Z / Math.min(64, Z)), Math.floor(Y / 64), Math.floor(X / 64)];
}
function applyDcCorrection(vol, shape, grid) {
  const [Z, Y, X] = shape;
  const [zb, yb, xb] = dcGridDims(shape);
  if (zb < 1 || yb < 1 || xb < 1 || zb * yb * xb !== grid.length) return false;
  const axis = (n, g) => {
    const i0 = new Int32Array(n), i1 = new Int32Array(n), w = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const f = Math.min(Math.max((i + 0.5) * g / n - 0.5, 0), g - 1);
      i0[i] = Math.floor(f);
      i1[i] = Math.min(i0[i] + 1, g - 1);
      w[i] = f - i0[i];
    }
    return { i0, i1, w };
  };
  const ax = axis(X, xb), ay = axis(Y, yb), az = axis(Z, zb);
  const planes = [];
  for (let gz = 0; gz < zb; gz++) {
    const p = new Float32Array(Y * X);
    const zoff = gz * yb * xb;
    for (let y = 0; y < Y; y++) {
      const r0 = zoff + ay.i0[y] * xb, r1 = zoff + ay.i1[y] * xb, wy = ay.w[y];
      for (let x = 0; x < X; x++) {
        const wx = ax.w[x];
        const a = grid[r0 + ax.i0[x]] * (1 - wx) + grid[r0 + ax.i1[x]] * wx;
        const b = grid[r1 + ax.i0[x]] * (1 - wx) + grid[r1 + ax.i1[x]] * wx;
        p[y * X + x] = (a * (1 - wy) + b * wy) * 4;
      }
    }
    planes.push(p);
  }
  const ss = Y * X;
  for (let z = 0; z < Z; z++) {
    const pa = planes[az.i0[z]], pb = planes[az.i1[z]], wz = az.w[z], base = z * ss;
    if (wz < 1e-6) {
      for (let i = 0; i < ss; i++) vol[base + i] -= pa[i];
    } else {
      for (let i = 0; i < ss; i++) vol[base + i] -= pa[i] + (pb[i] - pa[i]) * wz;
    }
  }
  return true;
}
var WIN = 400;
var LEV = 40;
var VR_CLIM = [-1024, 2e3];
function vrLUT() {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let a = Math.max(0, (t - 0.42) / 0.58);
    a *= a;
    lut[i * 4] = lut[i * 4 + 1] = lut[i * 4 + 2] = Math.round(t * 255);
    lut[i * 4 + 3] = Math.round(Math.min(0.85, a * 1.3) * 255);
  }
  return lut;
}
function ijkToRASFromSpacing(shape, spacing) {
  const [Z, Y, X] = shape;
  const sx = spacing[2], sy = spacing[1], sz = spacing[0];
  return [
    -sx,
    0,
    0,
    sx * (X - 1) / 2,
    0,
    -sy,
    0,
    sy * (Y - 1) / 2,
    0,
    0,
    sz,
    -sz * (Z - 1) / 2,
    0,
    0,
    0,
    1
  ];
}
function makeLiveCodecScene(gpu, format, shape, spacing) {
  const dev = gpu.device;
  const [Z, Y, X] = shape;
  const dims = [X, Y, Z];
  const ijkToRAS = ijkToRASFromSpacing(shape, spacing);
  const lut = vrLUT();
  const mkRow = () => {
    const vol = new Float32Array(X * Y * Z).fill(-1024);
    const field = new ImageField(dev, vol, dims, [1, 1, 1], lut, {
      clim: VR_CLIM,
      ijkToRAS,
      shade: [0.25, 0.7, 0.45, 20]
    });
    const scene = new SceneRenderer(gpu, format);
    scene.build([field]);
    scene.setBackground(0.05, 0.06, 0.09);
    return { vol, field, scene };
  };
  const rows = { neural: mkRow(), htj2k: mkRow() };
  const [rasLo, rasHi] = rows.neural.field.aabb();
  const slice = new SliceRenderer(gpu, format);
  slice.setVolume(rows.neural.field.patientToTexture(), rasLo, rasHi);
  slice.setWindowLevel(WIN, LEV);
  return {
    shape,
    dims,
    ijkToRAS,
    rasLo,
    rasHi,
    center: [(rasLo[0] + rasHi[0]) / 2, (rasLo[1] + rasHi[1]) / 2, (rasLo[2] + rasHi[2]) / 2],
    radius: Math.hypot(rasHi[0] - rasLo[0], rasHi[1] - rasLo[1], rasHi[2] - rasLo[2]) / 2,
    win: WIN,
    lev: LEV,
    slice,
    rows,
    bindRowSlice(key) {
      slice.setTextures(rows[key].field.volumeTexture());
    },
    writeSlab(key, z0, z1) {
      z0 = Math.max(0, z0);
      z1 = Math.min(Z, z1);
      if (z1 <= z0) return;
      const row = rows[key];
      dev.queue.writeTexture(
        { texture: row.field.volumeTexture(), origin: [0, 0, z0] },
        row.vol.subarray(z0 * X * Y, z1 * X * Y),
        { bytesPerRow: X * 4, rowsPerImage: Y },
        [X, Y, z1 - z0]
      );
    },
    destroy() {
    }
  };
}
async function loadScans() {
  const r = await fetch(BUCKET + "scans.json", { cache: "no-cache" });
  if (!r.ok) throw new Error(`scans.json HTTP ${r.status}`);
  return await r.json();
}
async function loadOodScans() {
  const r = await fetch(BUCKET + "ood-scans.json", { cache: "no-cache" });
  if (!r.ok) throw new Error(`ood-scans.json HTTP ${r.status}`);
  return await r.json();
}
async function loadVersions() {
  try {
    const r = await fetch(BUCKET + "versions.json", { cache: "no-cache" });
    if (!r.ok) return [];
    const v = await r.json();
    return Array.isArray(v) ? v.filter((e) => e && typeof e.tag === "string") : [];
  } catch {
    return [];
  }
}
async function loadScanMeta(neuralBase) {
  const r = await fetch(neuralBase + "meta.json");
  if (!r.ok) throw new Error(`meta.json HTTP ${r.status} at ${neuralBase}`);
  return await r.json();
}
async function loadDecoderMeta(modelBase) {
  const r = await fetch(modelBase + "decoder.json");
  if (!r.ok) throw new Error(`decoder.json HTTP ${r.status} at ${modelBase}`);
  return await r.json();
}

// examples/livecodec/livecodec-browser.ts
var ORIENTS = ["axial", "sagittal", "coronal"];
var PARAMS = new URLSearchParams(location.search);
var el = (id) => document.getElementById(id);
var status = (msg, err = false) => {
  const s = el("status");
  if (s) {
    s.textContent = msg;
    s.style.color = err ? "#ff6b74" : "#9fb3d0";
  }
};
var fmtBytes = (b) => b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} KB`;
var fmtMB = (b) => (b / 1e6).toFixed(b >= 1e6 ? 1 : 2);
var race = {
  neural: { t0: 0, stage: "waiting", note: "", got: 0, expected: 0, tFirst: null, tFinal: null, error: null },
  htj2k: { t0: 0, stage: "waiting", note: "", got: 0, expected: 0, tFirst: null, tFinal: null, error: null }
};
var elapsed = (k) => (performance.now() - race[k].t0) / 1e3;
function updateBars() {
  for (const k of ["neural", "htj2k"]) {
    const r = race[k];
    const fill = el(`fill-${k}`), ptext = el(`ptext-${k}`), times = el(`times-${k}`);
    if (!fill || !ptext || !times) continue;
    if (r.error) {
      fill.style.width = "100%";
      fill.className = "fill err";
      ptext.textContent = r.error;
      continue;
    }
    const frac = r.tFinal != null ? 1 : r.expected > 0 ? Math.min(1, r.got / r.expected) : 0;
    fill.style.width = `${(frac * 100).toFixed(1)}%`;
    fill.className = "fill" + (r.tFinal != null ? " done" : "");
    const t = r.tFinal ?? (r.t0 ? elapsed(k) : 0);
    ptext.textContent = r.t0 === 0 ? "waiting\u2026" : `${r.stage} \xB7 ${fmtMB(r.got)} / ${fmtMB(r.expected)} MB \xB7 ${t.toFixed(1)} s${r.note ? ` \xB7 ${r.note}` : ""}`;
    times.textContent = (r.tFirst != null ? `first ${r.tFirst.toFixed(1)} s` : "") + (r.tFinal != null ? ` \xB7 final ${r.tFinal.toFixed(1)} s` : "");
  }
}
async function main() {
  if (!navigator.gpu) {
    status("WebGPU not available \u2014 try Chrome/Edge 113+ or Safari 18+.", true);
    return;
  }
  const gpu = await initDevice();
  const preferred = navigator.gpu.getPreferredCanvasFormat();
  const srgb = preferred + "-srgb";
  status("loading scan list\u2026");
  const versions = await loadVersions();
  const version = versions.find((v) => v.tag === (PARAMS.get("ver") ?? "")) ?? null;
  const scans = version ? await loadOodScans() : await loadScans();
  const wanted = PARAMS.get("scan") ?? "";
  const scan = scans.find((s) => s.id === wanted) ?? scans[Math.floor(Math.random() * scans.length)];
  const norm2 = (u) => u.endsWith("/") ? u : u + "/";
  const dataOverride = PARAMS.get("data");
  const neuralBase = dataOverride ? norm2(dataOverride) : version ? `${BUCKET}versions/${version.tag}/${scan.id}/` : `${BUCKET}scans/${scan.id}/`;
  const htj2kBase = dataOverride ? norm2(dataOverride) : version ? `${BUCKET}ood/${scan.id}/` : `${BUCKET}scans/${scan.id}/`;
  const modelOverride = PARAMS.get("model");
  const modelBase = modelOverride ? norm2(modelOverride) : version ? `${BUCKET}versions/${version.tag}/model/` : BUCKET + "model/";
  const [Z, Y, X] = scan.shape;
  el("info").textContent = `scan ${scan.id}${scan.heldout ? " (held-out)" : ""}${scan.source ? ` \xB7 ${scan.source}` : ""}${version ? ` \xB7 ${version.tag}` : ""} \xB7 ${Z}\xD7${Y}\xD7${X} @ ${scan.spacing.map((s) => s.toFixed(2)).join("/")} mm \xB7 raw ${fmtBytes(scan.bytes.raw)}`;
  el("rand").addEventListener("click", () => {
    const others = scans.filter((s) => s.id !== scan.id);
    const pick = others[Math.floor(Math.random() * others.length)] ?? scan;
    const p = new URLSearchParams(location.search);
    p.set("scan", pick.id);
    location.search = p.toString();
  });
  const verSel = el("ver");
  if (verSel && versions.length > 0) {
    const wrap = el("verwrap");
    if (wrap) wrap.style.display = "";
    const fmtSteps = (s) => s >= 1e3 ? `${Math.round(s / 1e3)}k` : String(s);
    const fmtParams = (p) => typeof p === "number" ? p >= 1e6 ? `${(p / 1e6).toFixed(1)}M` : `${Math.round(p / 1e3)}k` : p;
    verSel.add(new Option("v3 \xB7 31 vols (baseline)", ""));
    for (const v of versions) {
      verSel.add(new Option(`${v.tag} \xB7 ${fmtSteps(v.steps)} steps \xB7 ${fmtParams(v.params)}`, v.tag));
    }
    verSel.value = version?.tag ?? "";
    verSel.addEventListener("change", () => {
      const p = new URLSearchParams(location.search);
      if (verSel.value) p.set("ver", verSel.value);
      else p.delete("ver");
      p.set("scan", scan.id);
      location.search = p.toString();
    });
  }
  const scanSel = el("scan");
  if (scanSel) {
    for (const s of scans) {
      const hint = s.heldout ? " (held-out)" : s.source ? ` (${s.source})` : "";
      scanSel.add(new Option(`${s.id.slice(0, 8)} \xB7 ${s.shape.join("\xD7")}${hint}`, s.id));
    }
    scanSel.value = scan.id;
    scanSel.addEventListener("change", () => {
      const p = new URLSearchParams(location.search);
      p.set("scan", scanSel.value);
      location.search = p.toString();
    });
  }
  const netSel = el("net");
  const netParam = PARAMS.get("net") ?? netSel?.value ?? "25";
  if (netSel) {
    netSel.value = netParam;
    netSel.addEventListener("change", () => {
      const p = new URLSearchParams(location.search);
      p.set("net", netSel.value);
      p.set("scan", scan.id);
      location.search = p.toString();
    });
  }
  setSimulatedBandwidth(netParam === "off" ? null : Number(netParam) * 1e6);
  const pacers = { neural: new LinkPacer(), htj2k: new LinkPacer() };
  const meters = { neural: new BandwidthMeter(), htj2k: new BandwidthMeter() };
  const reportIfDone = () => {
    if (race.neural.tFinal == null || race.htj2k.tFinal == null) return;
    const ns = meters.neural.summary(), hs = meters.htj2k.summary();
    for (const [k, m] of [["neural", ns], ["htj2k", hs]]) {
      const t = el(`times-${k}`);
      if (t) t.textContent += ` \xB7 avg ${m.mbps.toFixed(1)} Mbps`;
    }
    const delta = Math.abs(ns.mbps - hs.mbps) / Math.max(ns.mbps, hs.mbps);
    const target = netParam === "off" ? "" : ` \xB7 target ${netParam} Mbps`;
    const verdict = delta <= 0.15 ? "delivery fair \u2713" : `\u26A0 unequal delivery`;
    status(`measured: neural ${ns.mbps.toFixed(1)} Mbps \xB7 HTJ2K ${hs.mbps.toFixed(1)} Mbps \xB7 \u0394${(delta * 100).toFixed(0)}%${target} \u2014 ${verdict}`);
    console.table([...ns.streams, ...hs.streams].map((x) => ({
      stream: x.name,
      MB: (x.bytes / 1e6).toFixed(2),
      s: ((x.t1 - x.t0) / 1e3).toFixed(2),
      Mbps: (x.bytes * 8 / Math.max(1, x.t1 - x.t0) / 1e3).toFixed(1)
    })));
  };
  status(`loading ${scan.id} meta\u2026`);
  const [meta, dec] = await Promise.all([loadScanMeta(neuralBase), loadDecoderMeta(modelBase)]);
  const nb = version ? meta.bytes : scan.bytes;
  const bytes = {
    raw: scan.bytes.raw,
    htj2k: scan.bytes.htj2k,
    coarse: nb.coarse ?? 0,
    fine: nb.fine ?? 0,
    dc: nb.dc ?? 0,
    residual: nb.residual
  };
  el("name-neural").textContent = `LiveCodec neural \u2014 coarse ${fmtBytes(bytes.coarse)} \u2192 fine ${fmtBytes(bytes.fine + bytes.dc)}` + (bytes.residual ? ` \u2192 lossless ${fmtBytes(bytes.residual)}` : "");
  el("name-htj2k").textContent = `HTJ2K${bytes.residual ? " lossless" : ""} \u2014 ${fmtBytes(bytes.htj2k)}`;
  const sc = makeLiveCodecScene(gpu, srgb, scan.shape, scan.spacing);
  const keys = ["neural", "htj2k"];
  const cellNames = [...ORIENTS, "threeD"];
  const cv = {};
  const cx = {};
  for (const k of keys) {
    for (const c of cellNames) {
      const id = `c-${k}-${c}`;
      cv[id] = document.getElementById(id);
      cx[id] = cv[id].getContext("webgpu");
      cx[id].configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
    }
  }
  const off = {
    axial: slicerDefaultOffset01("axial", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
    coronal: slicerDefaultOffset01("coronal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi),
    sagittal: slicerDefaultOffset01("sagittal", sc.dims, sc.ijkToRAS, sc.rasLo, sc.rasHi)
  };
  const sliceIx = new SliceInteractor({ ijkToRAS: sc.ijkToRAS, rasLo: sc.rasLo, rasHi: sc.rasHi });
  const camera = framedCamera(sc.center, sc.radius);
  const viewKey = `lcview:${scan.id}`;
  try {
    const saved = JSON.parse(sessionStorage.getItem(viewKey) ?? "null");
    if (saved) {
      Object.assign(off, saved.off ?? {});
      sc.slice.setViewState(saved.slice ?? {});
      if (saved.camera) {
        camera.position = saved.camera.position ?? camera.position;
        camera.focalPoint = saved.camera.focalPoint ?? camera.focalPoint;
        camera.viewUp = saved.camera.viewUp ?? camera.viewUp;
        camera.viewAngle = saved.camera.viewAngle ?? camera.viewAngle;
      }
    }
  } catch {
  }
  addEventListener("beforeunload", () => {
    sessionStorage.setItem(viewKey, JSON.stringify({
      off,
      slice: sc.slice.getViewState(),
      camera: {
        position: camera.position,
        focalPoint: camera.focalPoint,
        viewUp: camera.viewUp,
        viewAngle: camera.viewAngle
      }
    }));
  });
  const drawSlice = (k, o) => {
    const c = cv[`c-${k}-${o}`];
    if (!c || !c.width) return;
    sc.bindRowSlice(k);
    sc.slice.setPlane(o, off[o]);
    sc.slice.renderToView(cx[`c-${k}-${o}`].getCurrentTexture().createView({ format: srgb }), c.width, c.height);
  };
  let fast3d = false;
  let settle3dTimer = 0;
  const draw3dCell = (k) => {
    const c = cv[`c-${k}-threeD`];
    if (!c || !c.width) return;
    const scene = sc.rows[k].scene;
    const view = cx[`c-${k}-threeD`].getCurrentTexture().createView({ format: srgb });
    if (fast3d) {
      const rw = Math.max(16, Math.round(c.width * 0.5)), rh = Math.max(16, Math.round(c.height * 0.5));
      scene.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, rw, rh);
      scene.renderUpscaled(view, rw, rh, c.width, c.height);
    } else {
      scene.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, c.width, c.height);
      scene.renderToView(view, c.width, c.height);
    }
  };
  const touch3d = () => {
    fast3d = true;
    clearTimeout(settle3dTimer);
    settle3dTimer = setTimeout(() => {
      fast3d = false;
      drawAll3d();
    }, 350);
  };
  const drawAll3d = () => {
    for (const k of keys) draw3dCell(k);
  };
  const drawSlices = () => {
    for (const k of keys) for (const o of ORIENTS) drawSlice(k, o);
  };
  const drawAll = () => {
    drawSlices();
    drawAll3d();
  };
  let drawRaf = 0;
  const requestDraw = () => {
    if (drawRaf) return;
    drawRaf = requestAnimationFrame(() => {
      drawRaf = 0;
      drawAll();
    });
  };
  for (const k of keys) {
    for (const o of ORIENTS) {
      attachSliceControls(cv[`c-${k}-${o}`], {
        orient: o,
        getSlice: () => sc.slice,
        step: (fwd) => {
          off[o] = sliceIx.wheel(o, off[o], fwd);
        },
        redraw: () => {
          for (const kk of keys) drawSlice(kk, o);
        },
        hooks: { onDoubleClick: () => {
          toggleMax(`c-${k}-${o}`);
          return true;
        } }
      });
    }
    attachCameraControls(cv[`c-${k}-threeD`], camera, { onChange: () => {
      touch3d();
      drawAll3d();
    } });
    attachDoubleClick(cv[`c-${k}-threeD`], () => toggleMax(`c-${k}-threeD`));
  }
  let maxed = null;
  const toggleMax = (id) => {
    maxed = maxed === id ? null : id;
    const rowsEl = el("rows");
    rowsEl.classList.toggle("maxmode", !!maxed);
    for (const k of keys) {
      for (const c of cellNames) {
        const cell = cv[`c-${k}-${c}`].parentElement;
        cell.classList.toggle("max", maxed === `c-${k}-${c}`);
      }
    }
    for (const r of rowsEl.querySelectorAll(".mrow")) {
      r.classList.toggle("hasmax", !!maxed && !!r.querySelector(".cell.max"));
    }
    resize();
  };
  installChrome({ controls: [], anchor: cv["c-htj2k-threeD"].parentElement ?? void 0 });
  const resize = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    for (const c of Object.values(cv)) {
      c.width = Math.floor(c.clientWidth * dpr);
      c.height = Math.floor(c.clientHeight * dpr);
    }
    drawAll();
  };
  globalThis.addEventListener("resize", resize);
  resize();
  const runNeural = async () => {
    const r = race.neural;
    try {
      const dtype = gpu.features.has("shader-f16") ? "f16" : "f32";
      const tNet = performance.now();
      const netP = new Net(gpu.device, makeRunner(gpu.device, dtype), dtype).load(modelBase + "decoder25.graph.json", modelBase + "decoder25.weights.bin").then((n) => {
        console.log(`livecodec: wgsl decoder (${dtype}) ready in ${((performance.now() - tNet) / 1e3).toFixed(1)} s`);
        return n;
      });
      netP.catch(() => {
      });
      r.stage = "coarse";
      r.expected = bytes.coarse;
      r.got = 0;
      const mCoarse = meters.neural.begin("coarse.gz");
      const coarseGz = await streamFetch(neuralBase + "coarse.gz", (n) => {
        r.got = n;
        mCoarse.at(n);
      }, pacers.neural);
      const coarseCodes = await gunzip(coarseGz);
      r.note = "loading decoder";
      const net = await netP;
      const sh = latentShapes(meta);
      const vol = sc.rows.neural.vol;
      const zfZero = new Float32Array(sh.C * sh.Df * sh.Hf * sh.Wf);
      const decodeChunk = async (zf, ch) => {
        net.setInputData("zf", zf);
        net.setInputData("zc_up", dequantCoarseUp(coarseCodes, ch, sh, dec));
        net.run();
        const out = await net.read("volume");
        const z0 = ch * sh.chunkZ;
        mapOutputToHU(out, vol, z0, Z, sh, dec);
        sc.writeSlab("neural", z0, Math.min(Z, z0 + sh.chunkZ));
      };
      const tDec = performance.now();
      for (let ch = 0; ch < sh.chunks; ch++) {
        r.note = `decode ${ch + 1}/${sh.chunks}`;
        await decodeChunk(zfZero, ch);
        if (ch === 0) r.tFirst = elapsed("neural");
        requestDraw();
      }
      console.log(`livecodec: coarse decode ${sh.chunks} chunks in ${((performance.now() - tDec) / 1e3).toFixed(1)} s`);
      r.note = "";
      requestDraw();
      r.stage = "fine+dc";
      r.expected = bytes.fine + bytes.dc;
      r.got = 0;
      let fGot = 0, dGot = 0;
      const [fineGz, dcGz] = await Promise.all([
        streamFetch(neuralBase + "fine.gz", /* @__PURE__ */ ((m) => (n) => {
          fGot = n;
          r.got = fGot + dGot;
          m.at(n);
        })(meters.neural.begin("fine.gz")), pacers.neural),
        streamFetch(neuralBase + "dc.gz", /* @__PURE__ */ ((m) => (n) => {
          dGot = n;
          r.got = fGot + dGot;
          m.at(n);
        })(meters.neural.begin("dc.gz")), pacers.neural)
      ]);
      const fineCodes = await gunzip(fineGz);
      const dcBytes = await gunzip(dcGz);
      const dcGrid = new Int8Array(dcBytes.buffer, dcBytes.byteOffset, dcBytes.byteLength);
      for (let ch = 0; ch < sh.chunks; ch++) {
        r.note = `refine ${ch + 1}/${sh.chunks}`;
        await decodeChunk(dequantFine(fineCodes, ch, sh, dec), ch);
        requestDraw();
      }
      r.note = "dc correction";
      if (!applyDcCorrection(vol, scan.shape, dcGrid)) {
        console.warn(`dc grid size ${dcGrid.length} does not match the volume shape \u2014 skipping DC correction`);
      }
      sc.writeSlab("neural", 0, Z);
      requestDraw();
      if (bytes.residual) {
        r.stage = "residual";
        r.expected = bytes.residual;
        r.got = 0;
        r.note = "";
        const factory = globalThis.Module;
        if (!factory) throw new Error("openjph script did not load");
        const rDecoder = new (await factory()).HTJ2KDecoder();
        const idxResp = await fetch(neuralBase + "residual-index.json", { cache: "no-store" });
        if (!idxResp.ok) throw new Error(`residual-index.json HTTP ${idxResp.status}`);
        const ridx = await idxResp.json();
        const total = ridx.length ? ridx[ridx.length - 1].offset + ridx[ridx.length - 1].bytes : 0;
        const sliceSize = X * Y;
        const rbuf = new Uint8Array(total);
        let received = 0, next = 0, flushed = 0;
        const applySlice = (e) => {
          const enc = rDecoder.getEncodedBuffer(e.bytes);
          enc.set(rbuf.subarray(e.offset, e.offset + e.bytes));
          rDecoder.decode();
          const out = rDecoder.getDecodedBuffer();
          const u16 = new Uint16Array(out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength));
          const b = e.z * sliceSize;
          for (let i = 0; i < sliceSize; i++) vol[b + i] += u16[i] - 4096;
        };
        const resp = await fetch(neuralBase + "residual.bin", { cache: "no-store" });
        if (!resp.ok || !resp.body) throw new Error(`residual.bin HTTP ${resp.status}`);
        const mRes = meters.neural.begin("residual.bin");
        const rrd = resp.body.getReader();
        for (; ; ) {
          const { done, value } = await rrd.read();
          if (done) break;
          await pacers.neural.admit(value.byteLength);
          mRes.add(value.byteLength);
          rbuf.set(value, received);
          received += value.byteLength;
          r.got = received;
          while (next < ridx.length && ridx[next].offset + ridx[next].bytes <= received) {
            applySlice(ridx[next]);
            next++;
            r.note = `${next}/${ridx.length} slices`;
            if (next - flushed >= 32) {
              sc.writeSlab("neural", flushed, next);
              flushed = next;
              requestDraw();
            }
          }
        }
        while (next < ridx.length && ridx[next].offset + ridx[next].bytes <= received) {
          applySlice(ridx[next]);
          next++;
        }
        sc.writeSlab("neural", flushed, Z);
        requestDraw();
        r.note = "lossless";
      }
      r.tFinal = elapsed("neural");
      reportIfDone();
    } catch (e) {
      r.error = "neural: " + (e?.message ?? String(e));
      console.error(e);
    }
  };
  const runHTJ2KProgressive = async (idx, decoder, vol, r) => {
    const sliceSize = X * Y;
    const slices = idx.slices, nS = slices.length, R = idx.rounds;
    const schedSlice = [];
    const schedEnd = [];
    const roundEnd = [];
    for (let rnd = 0; rnd < R; rnd++) {
      for (let si = 0; si < nS; si++) {
        const p = slices[si].parts[rnd];
        if (p) {
          schedSlice.push(si);
          schedEnd.push(p[0] + p[1]);
        }
      }
      roundEnd.push(schedSlice.length);
    }
    const total = schedEnd.length ? schedEnd[schedEnd.length - 1] : 0;
    r.stage = "slices";
    r.expected = total;
    r.got = 0;
    const buf = new Uint8Array(total);
    const arrived = new Uint8Array(nS);
    const applied = new Uint8Array(nS);
    const decodePrefix = (si) => {
      const s = slices[si], k = arrived[si];
      let len = 0;
      for (let i = 0; i < k; i++) len += s.parts[i][1];
      const enc = decoder.getEncodedBuffer(len);
      let o = 0;
      for (let i = 0; i < k; i++) {
        const [off2, n] = s.parts[i];
        enc.set(buf.subarray(off2, off2 + n), o);
        o += n;
      }
      const level = s.parts.length - k;
      decoder.decodeSubResolution(level);
      const out = decoder.getDecodedBuffer();
      const u16 = new Uint16Array(out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength));
      const b = s.z * sliceSize;
      if (level === 0) {
        const n = Math.min(sliceSize, u16.length);
        for (let i = 0; i < n; i++) vol[b + i] = u16[i] - 1024;
      } else {
        const w = Math.max(1, Math.ceil(X / (1 << level)));
        const h = Math.max(1, Math.ceil(Y / (1 << level)));
        for (let y = 0; y < Y; y++) {
          const srow = Math.min(h - 1, y * h / Y | 0) * w;
          const drow = b + y * X;
          for (let x = 0; x < X; x++) {
            vol[drow + x] = u16[srow + Math.min(w - 1, x * w / X | 0)] - 1024;
          }
        }
      }
      applied[si] = k;
    };
    const applyPass = async () => {
      let minZ = Z, maxZ = -1, n = 0;
      for (let si = 0; si < nS; si++) {
        if (arrived[si] <= applied[si]) continue;
        decodePrefix(si);
        const z = slices[si].z;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
        if (++n % 32 === 0) await new Promise((res) => setTimeout(res));
      }
      if (maxZ < 0) return;
      sc.writeSlab("htj2k", minZ, maxZ + 1);
      let full = R;
      for (let si = 0; si < nS; si++) if (applied[si] < full) full = applied[si];
      if (full >= 1 && r.tFirst == null) r.tFirst = elapsed("htj2k");
      const shown = Math.max(1, full);
      const px = Math.max(1, Math.ceil(X / (1 << R - shown)));
      r.note = `round ${shown}/${R} \xB7 ${px}px${full < 1 ? " \u2026" : ""}`;
      requestDraw();
    };
    const resp = await fetch(htj2kBase + "slices.bin", { cache: "no-store" });
    if (!resp.ok || !resp.body) throw new Error(`slices.bin HTTP ${resp.status}`);
    const mSl = meters.htj2k.begin("slices.bin");
    const rd = resp.body.getReader();
    let received = 0, cursor = 0, nextRound = 0, lastPassAt = 0;
    for (; ; ) {
      const { done, value } = await rd.read();
      if (done) break;
      await pacers.htj2k.admit(value.byteLength);
      mSl.add(value.byteLength);
      buf.set(value, received);
      received += value.byteLength;
      r.got = received;
      while (cursor < schedEnd.length && schedEnd[cursor] <= received) arrived[schedSlice[cursor++]]++;
      let roundDone = false;
      while (nextRound < R && cursor >= roundEnd[nextRound]) {
        nextRound++;
        roundDone = true;
      }
      if (roundDone || received - lastPassAt >= 2e6) {
        lastPassAt = received;
        await applyPass();
      }
    }
    while (cursor < schedEnd.length && schedEnd[cursor] <= received) arrived[schedSlice[cursor++]]++;
    await applyPass();
    r.note = "lossless";
    r.tFinal = elapsed("htj2k");
    reportIfDone();
  };
  const runHTJ2K = async () => {
    const r = race.htj2k;
    try {
      const factory = globalThis.Module;
      if (!factory) throw new Error("openjph script did not load");
      const openjphP = factory();
      const idxResp = await fetch(htj2kBase + "index.json", { cache: "no-store" });
      if (!idxResp.ok) throw new Error(`index.json HTTP ${idxResp.status}`);
      const rawIdx = await idxResp.json();
      const openjph = await openjphP;
      const decoder = new openjph.HTJ2KDecoder();
      const vol = sc.rows.htj2k.vol;
      const sliceSize = X * Y;
      if (!Array.isArray(rawIdx) && rawIdx.layout === "res-progressive") {
        await runHTJ2KProgressive(rawIdx, decoder, vol, r);
        return;
      }
      const idx = rawIdx;
      const total = idx.length ? idx[idx.length - 1].offset + idx[idx.length - 1].bytes : 0;
      r.stage = "slices";
      r.expected = total;
      r.got = 0;
      const buf = new Uint8Array(total);
      let received = 0, next = 0, flushed = 0;
      const decodeSlice = (e) => {
        const sub2 = buf.subarray(e.offset, e.offset + e.bytes);
        const enc = decoder.getEncodedBuffer(e.bytes);
        enc.set(sub2);
        decoder.decode();
        const out = decoder.getDecodedBuffer();
        const u16 = new Uint16Array(out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength));
        const n = Math.min(sliceSize, u16.length), b = e.z * sliceSize;
        for (let i = 0; i < n; i++) vol[b + i] = u16[i] - 1024;
      };
      const flush = () => {
        if (next <= flushed) return;
        sc.writeSlab("htj2k", flushed, next);
        flushed = next;
        if (r.tFirst == null) r.tFirst = elapsed("htj2k");
        requestDraw();
      };
      const resp = await fetch(htj2kBase + "slices.bin", { cache: "no-store" });
      if (!resp.ok || !resp.body) throw new Error(`slices.bin HTTP ${resp.status}`);
      const mSl = meters.htj2k.begin("slices.bin");
      const rd = resp.body.getReader();
      for (; ; ) {
        const { done, value } = await rd.read();
        if (done) break;
        await pacers.htj2k.admit(value.byteLength);
        mSl.add(value.byteLength);
        buf.set(value, received);
        received += value.byteLength;
        r.got = received;
        while (next < idx.length && idx[next].offset + idx[next].bytes <= received) {
          decodeSlice(idx[next]);
          next++;
          r.note = `${next}/${idx.length} slices`;
          if (next - flushed >= 32) flush();
        }
      }
      while (next < idx.length && idx[next].offset + idx[next].bytes <= received) {
        decodeSlice(idx[next]);
        next++;
      }
      flush();
      r.note = "";
      r.tFinal = elapsed("htj2k");
      reportIfDone();
    } catch (e) {
      r.error = "htj2k: " + (e?.message ?? String(e));
      console.error(e);
    }
  };
  globalThis.__lcDbg = {
    ready: () => true,
    scan: () => scan.id,
    ver: () => version?.tag ?? null,
    bases: () => ({ neuralBase, htj2kBase, modelBase }),
    dims: () => sc.dims,
    offsets: () => ({ ...off }),
    race: () => JSON.parse(JSON.stringify(race)),
    camera: () => ({ position: [...camera.position], focalPoint: [...camera.focalPoint] }),
    volSample: (k, z, y, x) => sc.rows[k].vol[(z * Y + y) * X + x]
  };
  const start = performance.now();
  race.neural.t0 = start;
  race.htj2k.t0 = start;
  status(`racing on ${scan.id} \u2014 scroll a slice, drag a 3D to orbit (linked), double-click to maximize`);
  const barTimer = setInterval(() => {
    updateBars();
    if ((race.neural.tFinal != null || race.neural.error) && (race.htj2k.tFinal != null || race.htj2k.error)) {
      clearInterval(barTimer);
      updateBars();
    }
  }, 100);
  await Promise.all([runNeural(), runHTJ2K()]);
  updateBars();
  drawAll();
}
main().catch((e) => {
  status("error: " + (e?.message ?? e), true);
  console.error(e);
});
