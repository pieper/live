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

// render/budget-controller.ts
var BudgetController = class {
  budgetPx;
  /** Mutable so a demo can expose it: a viewer who would rather have detail than frame rate raises
   *  the target frame time, and the loop then keeps a bigger fraction of the native resolution while
   *  interacting instead of downsampling into aliasing. */
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
  const dbgTick = (kind, ms, s, k = 1) => {
    if (!DBG) return;
    dbgN++;
    if (kind === "mov") dbgMoving += ms;
    else dbgSettled += ms;
    const now = performance.now();
    if (now - dbgLast > 500) {
      console.log(`[perf] mov=${dbgMoving.toFixed(0)}ms/${dbgN}f settled=${dbgSettled.toFixed(0)}ms lastScale=${s.toFixed(2)} samples=${k} last=${ms.toFixed(1)}ms`);
      dbgLast = now;
      dbgMoving = dbgSettled = dbgN = 0;
    }
  };
  const movingCap = opts.movingScaleCap ?? 1;
  const maxMovingSamples = Math.max(1, Math.round(opts.maxMovingSamples ?? 1));
  let sampleMs = 0;
  const renderMoving = () => {
    const sc = opts.scene();
    if (!sc) return;
    const { w: vw, h: vh } = opts.size();
    if (!vw || !vh) return;
    const s = Math.min(movingCap, budget.scale(vw, vh)), t0 = performance.now();
    const k = maxMovingSamples > 1 && s > 0.98 && sampleMs > 0 ? Math.max(1, Math.min(maxMovingSamples, Math.floor(budget.targetMs / sampleMs))) : 1;
    if (s > 0.98) {
      opts.setCamera(sc, vw, vh);
      if (k > 1) {
        const view = opts.view();
        sc.renderAccum(view, vw, vh, true);
        for (let i = 1; i < k; i++) {
          opts.setCamera(sc, vw, vh);
          sc.renderAccum(view, vw, vh, false);
        }
      } else {
        sc.renderToView(opts.view(), vw, vh);
      }
    } else {
      const rw = Math.max(16, Math.round(vw * s)), rh = Math.max(16, Math.round(vh * s));
      opts.setCamera(sc, rw, rh);
      sc.renderUpscaled(opts.view(), rw, rh, vw, vh);
    }
    opts.gpu.device.queue.onSubmittedWorkDone().then(() => {
      const ms = performance.now() - t0;
      sampleMs = ms / k;
      budget.update(sampleMs);
      dbgTick("mov", ms, s, k);
    });
    opts.onFrame?.();
  };
  const renderSettled = (reset) => {
    const sc = opts.scene();
    if (!sc) return;
    const { w: vw, h: vh } = opts.size();
    if (!vw || !vh) return;
    const t0 = performance.now();
    opts.setCamera(sc, vw, vh);
    sc.renderAccum(opts.view(), vw, vh, reset);
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
function spacingFromIjkToRAS(ijkToRAS) {
  const col = (c) => Math.hypot(ijkToRAS[c], ijkToRAS[4 + c], ijkToRAS[8 + c]);
  return [col(0), col(1), col(2)];
}

// render/scene-renderer.ts
var DEFAULT_FORMAT = "rgba8unorm-srgb";
var SCENE_FLOATS = 16;
var CLIP_FLOATS = 36;
var MESH_WGSL = (
  /* wgsl */
  `
struct MU { view_proj : mat4x4<f32>, eye : vec4<f32>, color : vec4<f32> };
@group(0) @binding(0) var<uniform> mu : MU;
struct VO { @builtin(position) pos : vec4<f32>, @location(0) wp : vec3<f32> };
@vertex fn vs_mesh(@location(0) p : vec3<f32>) -> VO { var o : VO; o.pos = mu.view_proj * vec4<f32>(p, 1.0); o.wp = p; return o; }
struct FO { @location(0) col : vec4<f32>, @location(1) depth : vec4<f32> };
@fragment fn fs_mesh(i : VO) -> FO {
  let n = normalize(cross(dpdx(i.wp), dpdy(i.wp)));       // flat face normal (no normals on the wire)
  let l = normalize(mu.eye.xyz - i.wp);                   // headlight
  let lam = 0.25 + 0.75 * abs(dot(n, l));
  let a = mu.color.a;
  var o : FO;
  o.col = vec4<f32>(mu.color.rgb * lam * a, a);           // premultiplied
  o.depth = vec4<f32>(distance(mu.eye.xyz, i.wp), 0.0, 0.0, 1.0);
  return o;
}`
);
var SceneRenderer = class _SceneRenderer {
  // ── surface meshes (models): rasterised before each trace into colour+depth targets the march composites ──
  meshPipeline;
  gpuMeshes = [];
  meshTargetsBySize = /* @__PURE__ */ new Map();
  viewProj = new Float32Array(16);
  eyePos = [0, 0, 0];
  /** Replace the surface meshes (world/RAS float32 xyz + uint32 triangles, colour, opacity). */
  setMeshes(meshes) {
    for (const m of this.gpuMeshes) {
      m.vbuf.destroy();
      m.ibuf.destroy();
      m.ubuf.destroy();
    }
    this.gpuMeshes = meshes.filter((m) => m.indices.length >= 3).map((m) => {
      const vbuf = this.dev.createBuffer({ size: Math.ceil(m.positions.byteLength / 4) * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      this.dev.queue.writeBuffer(vbuf, 0, m.positions);
      const ibuf = this.dev.createBuffer({ size: Math.ceil(m.indices.byteLength / 4) * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      this.dev.queue.writeBuffer(ibuf, 0, m.indices);
      const ubuf = this.dev.createBuffer({ size: 24 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      return { vbuf, ibuf, count: m.indices.length, ubuf, color: m.color, opacity: m.opacity };
    });
  }
  hasMeshes() {
    return this.gpuMeshes.length > 0;
  }
  ensureMeshPipeline() {
    if (this.meshPipeline) return;
    const mod = this.dev.createShaderModule({ code: MESH_WGSL });
    this.meshPipeline = this.dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: mod, entryPoint: "vs_mesh", buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }] },
      fragment: { module: mod, entryPoint: "fs_mesh", targets: [{ format: "rgba16float" }, { format: "r32float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });
  }
  /** Colour/depth targets (+ the group-1 bind group of the trace pipeline) for a given trace size. */
  meshTargets(w, h) {
    const key = w + "x" + h;
    let t = this.meshTargetsBySize.get(key);
    if (!t) {
      if (this.meshTargetsBySize.size > 4) {
        for (const old of this.meshTargetsBySize.values()) {
          old.col.destroy();
          old.depth.destroy();
          old.z.destroy();
        }
        this.meshTargetsBySize.clear();
      }
      t = {
        w,
        h,
        col: this.dev.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
        depth: this.dev.createTexture({ size: [w, h], format: "r32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
        z: this.dev.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT })
      };
      this.meshTargetsBySize.set(key, t);
    }
    if (!t.bind) t.bind = this.dev.createBindGroup({ layout: this.pipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: t.col.createView() }, { binding: 1, resource: t.depth.createView() }] });
    return t;
  }
  /** Rasterise the meshes for this frame's trace size; returns the bind group the trace pass needs. */
  meshPass(enc, w, h) {
    const t = this.meshTargets(w, h);
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: t.col.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } },
        { view: t.depth.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 1e30, g: 0, b: 0, a: 1 } }
      ],
      depthStencilAttachment: { view: t.z.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" }
    });
    if (this.gpuMeshes.length) {
      this.ensureMeshPipeline();
      pass.setPipeline(this.meshPipeline);
      for (const m of this.gpuMeshes) {
        const u = new Float32Array(24);
        u.set(this.viewProj, 0);
        u[16] = this.eyePos[0];
        u[17] = this.eyePos[1];
        u[18] = this.eyePos[2];
        u[19] = 1;
        u[20] = m.color[0];
        u[21] = m.color[1];
        u[22] = m.color[2];
        u[23] = m.opacity;
        this.dev.queue.writeBuffer(m.ubuf, 0, u);
        pass.setBindGroup(0, this.dev.createBindGroup({ layout: this.meshPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: m.ubuf } }] }));
        pass.setVertexBuffer(0, m.vbuf);
        pass.setIndexBuffer(m.ibuf, "uint32");
        pass.drawIndexed(m.count);
      }
    }
    pass.end();
    return t.bind;
  }
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
    this.dev.queue.writeBuffer(this.resolveBgBuf, 0, this.mat.subarray(12, 16));
    const enc = this.dev.createCommandEncoder();
    const mb = this.meshPass(enc, renderW, renderH);
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.lowView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.setBindGroup(1, mb);
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
    const mb = this.meshPass(enc, this.traceW, this.traceH);
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.traceView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.setBindGroup(1, mb);
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
  /** ROLLING accumulation for scenes whose content keeps changing (an animation): each frame blends
   *  in with weight 1/min(n, accumWindow) — an exponential window of ~accumWindow frames instead of
   *  the running mean — so static content still converges toward jittered temporal AA while moving
   *  content keeps a short trail rather than smearing. Infinity (default) = the running mean. */
  accumWindow = Infinity;
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
    this.dev.queue.writeBuffer(this.accumUniformBuf, 0, new Float32Array([this.mat[12], this.mat[13], this.mat[14], 1 / Math.min(n, this.accumWindow)]));
    const prev = this.accumPing, next = 1 - this.accumPing;
    const enc = this.dev.createCommandEncoder();
    const mb = this.meshPass(enc, width, height);
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.traceView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.pipeline);
    tp.setBindGroup(0, this.bind);
    tp.setBindGroup(1, mb);
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
    for (const t of this.meshTargetsBySize.values()) t.bind = void 0;
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
    const args = (p) => p.field.intervalSampling ? `wp, rd, s_here - last_${p.field.kind}${p.slot}` : "wp, rd";
    const advance = (p) => p.field.intervalSampling ? ` last_${p.field.kind}${p.slot} = s_here;` : "";
    const sampleInto = (p, ghost) => {
      const call = `sample_field_${p.field.kind}${p.slot}(${args(p)})`;
      return ghost ? `let c = ${call}; if (c.a > g_op) { g_op = c.a; g_col = c.rgb / max(c.a, 1e-4); }` : `let c = ${call}; sum += c;`;
    };
    const skipBranch = (p, clip, ghost = false) => {
      const nm = `${p.field.kind}${p.slot}`;
      const smp = sampleInto(p, ghost);
      return `    if (t >= resume_${nm}) {
      let d_${nm} = max(skip_${nm}(wp) - step, 0.0);
      if (d_${nm} > 0.0) { resume_${nm} = t + d_${nm}; }
      else { ${clip ? clipGuard(p, smp) : smp}${advance(p)} }
    }
    if (t < resume_${nm}) { jump_t = min(jump_t, resume_${nm}); } else { all_defer = false; }`;
    };
    const plainBranch = (p, clip, ghost = false) => {
      const smp = sampleInto(p, ghost);
      return `    { ${clip ? clipGuard(p, smp) : smp}${advance(p)} all_defer = false; }`;
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
    const intervalInit = receivers.filter((p) => p.field.intervalSampling).map((p) => `  var last_${p.field.kind}${p.slot} : f32 = max(t_near - step, 0.0);`).join("\n");
    const dispatch = normalReceivers.map(
      (p) => canSkip.has(p.field) ? skipBranch(p, true) : plainBranch(p, true)
    ).join("\n");
    const ghostDispatch = ghostFields.map(
      (p) => ghostCanSkip.has(p.field) ? skipBranch(p, false, true) : plainBranch(p, false, true)
    ).join("\n");
    const hasGhost = ghostFields.length > 0;
    const pickDispatch = normalReceivers.map(
      (p) => `    ${clipGuard(p, `{ let c = sample_field_${p.field.kind}${p.slot}(wp, rd${p.field.intervalSampling ? ", step" : ""}); sum += c; }`)}`
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
// Rasterised surface meshes (models): nearest-surface colour (premultiplied) + its distance along the
// ray, produced by the mesh pass before each trace. The march composites the surface at that depth,
// so volumes in front occlude it and it occludes what is behind \u2014 the depth-composite seam.
@group(1) @binding(0) var t_mesh_col : texture_2d<f32>;
@group(1) @binding(1) var t_mesh_depth : texture_2d<f32>;
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

  let mpix = vec2<i32>(v.position.xy);
  let mesh_c = textureLoad(t_mesh_col, mpix, 0);          // premultiplied surface colour (0 = no mesh)
  let mesh_t = textureLoad(t_mesh_depth, mpix, 0).r;      // distance along the ray (1e30 = none)
  var mesh_done = mesh_c.a <= 0.0;

  let inv = vec3<f32>(1.0) / rd;
  let tb = (u_material.bmin.xyz - ro) * inv;
  let tt = (u_material.bmax.xyz - ro) * inv;
  let tmn = min(tt, tb); let tmx = max(tt, tb);
  var t_near = max(max(tmn.x, tmn.y), tmn.z);
  var t_far  = min(min(tmx.x, tmx.y), tmx.z);
  if (t_far <= t_near || t_far <= 0.0) { return mesh_c; }

  let step = max(u_material.scene.x, 1e-3);
  t_near = max(t_near + step, 0.0);
  t_far  = t_far - step;
  if (t_far <= t_near) { return mesh_c; }
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
${intervalInit}
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
    if (!mesh_done && t + 0.5 * step >= mesh_t) {         // the ray reaches the surface: composite it here
      integrated = integrated + (1.0 - integrated.a) * mesh_c;
      mesh_done = true;
    }
    let s_here = t + js * step;   // ray distance of this (jittered) sample
    let wp = ro + rd * s_here;
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
  if (!mesh_done) { integrated = integrated + (1.0 - integrated.a) * mesh_c; }   // surface beyond the slab
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
    return this.placed.some((p) => p.field.usesSampler ?? p.field.bindingCount > 0);
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
    this.viewProj = multiply(proj, view);
    this.eyePos = eye;
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
    this.viewProj = multiply(proj, view);
    this.eyePos = eye;
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
      const mb = this.meshPass(enc, width, height);
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bind);
      pass.setBindGroup(1, mb);
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
    this.meshPass(enc, width, height);
    const smt = this.meshTargets(width, height);
    const streamMb = this.dev.createBindGroup({ layout: this.streamPipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: smt.col.createView() }, { binding: 1, resource: smt.depth.createView() }] });
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    tp.setPipeline(this.streamPipeline);
    tp.setBindGroup(0, this.streamBind);
    tp.setBindGroup(1, streamMb);
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

// algorithms/geom.ts
function transpose42(m) {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) o[c * 4 + r] = m[r * 4 + c];
  return o;
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

// algorithms/effects/paint.ts
var PAINT_WGSL = (
  /* wgsl */
  `
struct U {
  ijkToRAS : mat4x4<f32>,   // column-major (transpose of the row-major host matrix)
  dims     : vec4<u32>,
  params   : vec4<f32>,     // x=radiusMm, y=id, z=mode(0 add/1 remove), w=pointCount
};
@group(0) @binding(0) var t_label : texture_storage_3d<r32uint, write>;
@group(0) @binding(1) var<uniform> u : U;
@group(0) @binding(2) var<storage, read> pts : array<vec4<f32>>;   // xyz = RAS sample points

fn seg_dist(p : vec3<f32>, a : vec3<f32>, b : vec3<f32>) -> f32 {
  let ab = b - a;
  let denom = max(dot(ab, ab), 1e-8);
  let t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
  return length(p - (a + t * ab));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let p = (u.ijkToRAS * vec4<f32>(vec3<f32>(gid), 1.0)).xyz;
  let n = u32(u.params.w);
  if (n == 0u) { return; }
  var dmin = 1e30;
  if (n == 1u) {
    dmin = length(p - pts[0].xyz);
  } else {
    for (var i = 0u; i < n - 1u; i = i + 1u) {
      dmin = min(dmin, seg_dist(p, pts[i].xyz, pts[i + 1u].xyz));
    }
  }
  if (dmin <= u.params.x) {
    let id = select(u32(u.params.y), 0u, u.params.z > 0.5);   // remove \u2192 0
    textureStore(t_label, vec3<i32>(gid), vec4<u32>(id, 0u, 0u, 0u));
  }
}`
);
var PaintEffect = class {
  constructor(seg) {
    this.seg = seg;
    const dev = seg.device;
    this.dev = dev;
    this.pipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: PAINT_WGSL }), entryPoint: "main" } });
    this.uni = dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  dev;
  pipe;
  uni;
  ptsBuf;
  ptsCap = 0;
  /** Rasterize a stroke (RAS polyline + spherical brush) into the master, interpolating between
   *  samples, then mark the segmentation dirty (one redraw). A single point = a sphere dab. */
  stampStroke(points, opts) {
    if (points.length === 0) return;
    const dev = this.dev, dims = this.seg.dims;
    const need = points.length * 4 * 4;
    if (!this.ptsBuf || this.ptsCap < points.length) {
      this.ptsBuf?.destroy();
      this.ptsCap = Math.max(points.length, 64);
      this.ptsBuf = dev.createBuffer({ size: this.ptsCap * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const pd = new Float32Array(points.length * 4);
    for (let i = 0; i < points.length; i++) {
      pd[i * 4] = points[i][0];
      pd[i * 4 + 1] = points[i][1];
      pd[i * 4 + 2] = points[i][2];
    }
    dev.queue.writeBuffer(this.ptsBuf, 0, pd, 0, points.length * 4);
    const ab = new ArrayBuffer(96);
    const f = new Float32Array(ab), uu = new Uint32Array(ab);
    f.set(transpose42(this.seg.ijkToRAS), 0);
    uu[16] = dims[0];
    uu[17] = dims[1];
    uu[18] = dims[2];
    uu[19] = 0;
    f[20] = opts.radiusMm;
    f[21] = opts.id ?? 1;
    f[22] = opts.mode === "remove" ? 1 : 0;
    f[23] = points.length;
    dev.queue.writeBuffer(this.uni, 0, ab);
    const bind = dev.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seg.masterTexture().createView() },
      { binding: 1, resource: { buffer: this.uni } },
      { binding: 2, resource: { buffer: this.ptsBuf, size: need } }
    ] });
    const [gx, gy, gz] = [Math.ceil(dims[0] / 4), Math.ceil(dims[1] / 4), Math.ceil(dims[2] / 4)];
    const enc = dev.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pipe);
    p.setBindGroup(0, bind);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc.finish()]);
    this.seg.markDirty();
  }
  /** Incremental segment: weld a capsule from `prev` to `next` (one pointer move). */
  extend(prev, next, opts) {
    this.stampStroke([prev, next], opts);
  }
  destroy() {
    this.uni.destroy();
    this.ptsBuf?.destroy();
  }
};

// algorithms/effects/scissors.ts
var SCISSORS_WGSL = (
  /* wgsl */
  `
struct U {
  ijkToRAS : mat4x4<f32>,    // column-major
  uAxis    : vec4<f32>,      // in-plane basis (xyz)
  vAxis    : vec4<f32>,
  dims     : vec4<u32>,
  params   : vec4<f32>,      // x=vertexCount, y=id, z=op(0 fillInside/1 eraseInside/2 eraseOutside)
};
@group(0) @binding(0) var t_label : texture_storage_3d<r32uint, write>;
@group(0) @binding(1) var<uniform> u : U;
@group(0) @binding(2) var<storage, read> poly : array<vec2<f32>>;   // contour projected to (u,v)

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let p = (u.ijkToRAS * vec4<f32>(vec3<f32>(gid), 1.0)).xyz;
  let q = vec2<f32>(dot(p, u.uAxis.xyz), dot(p, u.vAxis.xyz));   // project the voxel onto the view plane
  let n = u32(u.params.x);
  if (n < 3u) { return; }
  // even-odd (crossing-number) point-in-polygon on the projected contour.
  var inside = false;
  var j = n - 1u;
  for (var i = 0u; i < n; i = i + 1u) {
    let a = poly[i]; let b = poly[j];
    if ((a.y > q.y) != (b.y > q.y)) {
      let xcross = (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x;
      if (q.x < xcross) { inside = !inside; }
    }
    j = i;
  }
  let op = u.params.z;
  let affected = select(inside, !inside, op > 1.5);        // eraseOutside acts where NOT inside
  if (affected) {
    let val = select(0u, u32(u.params.y), op < 0.5);       // fillInside \u2192 id; erase \u2192 0
    textureStore(t_label, vec3<i32>(gid), vec4<u32>(val, 0u, 0u, 0u));
  }
}`
);
var ScissorsEffect = class {
  constructor(seg) {
    this.seg = seg;
    const dev = seg.device;
    this.dev = dev;
    this.pipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: SCISSORS_WGSL }), entryPoint: "main" } });
    this.uni = dev.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  dev;
  pipe;
  uni;
  polyBuf;
  polyCap = 0;
  /** Carve the labelmap with a closed RAS contour on the (u,v) view plane, then mark dirty. */
  apply(contourRAS, opts) {
    if (contourRAS.length < 3) return;
    const dev = this.dev, dims = this.seg.dims;
    const [u, v] = [opts.u, opts.v];
    const proj = new Float32Array(contourRAS.length * 2);
    for (let i = 0; i < contourRAS.length; i++) {
      const p2 = contourRAS[i];
      proj[i * 2] = p2[0] * u[0] + p2[1] * u[1] + p2[2] * u[2];
      proj[i * 2 + 1] = p2[0] * v[0] + p2[1] * v[1] + p2[2] * v[2];
    }
    const need = contourRAS.length * 8;
    if (!this.polyBuf || this.polyCap < contourRAS.length) {
      this.polyBuf?.destroy();
      this.polyCap = Math.max(contourRAS.length, 64);
      this.polyBuf = dev.createBuffer({ size: this.polyCap * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    dev.queue.writeBuffer(this.polyBuf, 0, proj);
    const op = opts.operation ?? "eraseInside";
    const ab = new ArrayBuffer(128);
    const f = new Float32Array(ab), uu = new Uint32Array(ab);
    f.set(transpose42(this.seg.ijkToRAS), 0);
    f[16] = u[0];
    f[17] = u[1];
    f[18] = u[2];
    f[19] = 0;
    f[20] = v[0];
    f[21] = v[1];
    f[22] = v[2];
    f[23] = 0;
    uu[24] = dims[0];
    uu[25] = dims[1];
    uu[26] = dims[2];
    uu[27] = 0;
    f[28] = contourRAS.length;
    f[29] = opts.id ?? 1;
    f[30] = op === "fillInside" ? 0 : op === "eraseInside" ? 1 : 2;
    f[31] = 0;
    dev.queue.writeBuffer(this.uni, 0, ab);
    const bind = dev.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seg.masterTexture().createView() },
      { binding: 1, resource: { buffer: this.uni } },
      { binding: 2, resource: { buffer: this.polyBuf, size: need } }
    ] });
    const [gx, gy, gz] = [Math.ceil(dims[0] / 4), Math.ceil(dims[1] / 4), Math.ceil(dims[2] / 4)];
    const enc = dev.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pipe);
    p.setBindGroup(0, bind);
    p.dispatchWorkgroups(gx, gy, gz);
    p.end();
    dev.queue.submit([enc.finish()]);
    this.seg.markDirty();
  }
  destroy() {
    this.uni.destroy();
    this.polyBuf?.destroy();
  }
};

// algorithms/effects/growcut.ts
var INIT_WGSL = (
  /* wgsl */
  `
@group(0) @binding(0) var t_label : texture_3d<u32>;
@group(0) @binding(1) var t_dst : texture_storage_3d<rg32float, write>;
struct U { dims : vec4<u32>, params : vec4<f32> };
@group(0) @binding(2) var<uniform> u : U;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let lbl = textureLoad(t_label, vec3<i32>(gid), 0).r;
  textureStore(t_dst, vec3<i32>(gid), vec4<f32>(f32(lbl), select(0.0, 1.0, lbl != 0u), 0.0, 0.0));   // seed \u2192 strength 1
}`
);
var ITER_WGSL = (
  /* wgsl */
  `
@group(0) @binding(0) var t_img : texture_3d<f32>;
@group(0) @binding(1) var t_src : texture_3d<f32>;
@group(0) @binding(2) var t_dst : texture_storage_3d<rg32float, write>;
struct U { dims : vec4<u32>, params : vec4<f32> };   // params.x = edgeHi (t1), params.y = 1/(t1-t0)
@group(0) @binding(3) var<uniform> u : U;
fn img(c : vec3<i32>) -> f32 { let d = vec3<i32>(u.dims.xyz); return textureLoad(t_img, clamp(c, vec3<i32>(0), d - vec3<i32>(1)), 0).r; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let st = textureLoad(t_src, c, 0);   // (label, strength)
  var bestLabel = st.r;
  var bestStr = st.g;
  let myI = img(c);
  let t1 = u.params.x; let invSpan = u.params.y;
  let offs = array<vec3<i32>, 6>(vec3<i32>(1,0,0), vec3<i32>(-1,0,0), vec3<i32>(0,1,0), vec3<i32>(0,-1,0), vec3<i32>(0,0,1), vec3<i32>(0,0,-1));
  for (var i = 0; i < 6; i = i + 1) {
    let nc = c + offs[i];
    if (any(nc < vec3<i32>(0)) || any(nc >= vec3<i32>(u.dims.xyz))) { continue; }
    let ns = textureLoad(t_src, nc, 0);
    if (ns.g <= 0.0) { continue; }
    // THRESHOLDED similarity: g=1 below the noise floor t0 (no decay inside a region), \u21920 at the edge
    // contrast t1. Lets a label flood homogeneous tissue and stall at a boundary \u2014 robust to noise,
    // unlike the raw 1\u2212d/range which decays every step and can't cross a large homogeneous region.
    let g = clamp((t1 - abs(myI - img(nc))) * invSpan, 0.0, 1.0);
    let attack = ns.g * g;
    if (attack > bestStr) { bestStr = attack; bestLabel = ns.r; }
  }
  textureStore(t_dst, c, vec4<f32>(bestLabel, bestStr, 0.0, 0.0));
}`
);
var FINAL_WGSL = (
  /* wgsl */
  `
@group(0) @binding(0) var t_src : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<r32uint, write>;
struct U { dims : vec4<u32>, params : vec4<f32> };
@group(0) @binding(2) var<uniform> u : U;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (any(gid >= u.dims.xyz)) { return; }
  let lbl = u32(textureLoad(t_src, vec3<i32>(gid), 0).r + 0.5);
  textureStore(t_out, vec3<i32>(gid), vec4<u32>(lbl, 0u, 0u, 0u));
}`
);
var GrowCutEffect = class {
  /** `imageTex` is the r32float intensity volume aligned to the segmentation grid (same dims/ijkToRAS). */
  constructor(seg, imageTex) {
    this.seg = seg;
    this.imageTex = imageTex;
    const dev = seg.device;
    this.dev = dev;
    const mk = (code) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } });
    this.initPipe = mk(INIT_WGSL);
    this.iterPipe = mk(ITER_WGSL);
    this.finalPipe = mk(FINAL_WGSL);
    const state = () => dev.createTexture({ size: seg.dims, dimension: "3d", format: "rg32float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.a = state();
    this.b = state();
    this.uni = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const [dx, dy, dz] = seg.dims;
    this.g = [Math.ceil(dx / 4), Math.ceil(dy / 4), Math.ceil(dz / 4)];
  }
  dev;
  initPipe;
  iterPipe;
  finalPipe;
  a;
  b;
  uni;
  g;
  writeUni(t1, invSpan) {
    const ab = new ArrayBuffer(32);
    const u = new Uint32Array(ab), f = new Float32Array(ab);
    u[0] = this.seg.dims[0];
    u[1] = this.seg.dims[1];
    u[2] = this.seg.dims[2];
    u[3] = 0;
    f[4] = t1;
    f[5] = invSpan;
    f[6] = 0;
    f[7] = 0;
    this.dev.queue.writeBuffer(this.uni, 0, ab);
  }
  pass(pipe, entries) {
    const enc = this.dev.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(pipe);
    p.setBindGroup(0, this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries }));
    p.dispatchWorkgroups(...this.g);
    p.end();
    this.dev.queue.submit([enc.finish()]);
  }
  /** Grow the current sparse seed labelmap to fill the volume, writing the dense result back to the
   *  master. Returns the iterations actually run (early-stops at convergence). */
  async grow(opts = {}) {
    const dims = this.seg.dims;
    const range = opts.intensityRange ?? await this.imageRange();
    const t0 = (opts.edgeLo ?? 0.15) * range, t1 = (opts.edgeHi ?? 0.5) * range;
    this.writeUni(t1, 1 / Math.max(t1 - t0, 1e-6));
    const maxIter = opts.iterations ?? Math.ceil(Math.max(...dims) * 1.2);
    const checkEvery = opts.checkEvery ?? 16;
    this.pass(this.initPipe, [
      { binding: 0, resource: this.seg.masterTexture().createView() },
      { binding: 1, resource: this.a.createView() },
      { binding: 2, resource: { buffer: this.uni } }
    ]);
    let src = this.a, dst = this.b, ran = 0;
    for (let i = 0; i < maxIter; i++) {
      this.pass(this.iterPipe, [
        { binding: 0, resource: this.imageTex.createView() },
        { binding: 1, resource: src.createView() },
        { binding: 2, resource: dst.createView() },
        { binding: 3, resource: { buffer: this.uni } }
      ]);
      [src, dst] = [dst, src];
      ran++;
      if (checkEvery > 0 && (i + 1) % checkEvery === 0) {
        const filled = await this.filledCount(src);
        if (filled === this.lastFilled) break;
        this.lastFilled = filled;
      }
    }
    this.lastFilled = -1;
    this.pass(this.finalPipe, [
      { binding: 0, resource: src.createView() },
      { binding: 1, resource: this.seg.masterTexture().createView() },
      { binding: 2, resource: { buffer: this.uni } }
    ]);
    this.seg.markDirty();
    return ran;
  }
  lastFilled = -1;
  /** Count labelled voxels in a state texture (readback of the label channel) — for convergence + tests. */
  async filledCount(state) {
    const [dx, dy, dz] = this.seg.dims;
    const bpr = Math.ceil(dx * 8 / 256) * 256;
    const rowF = bpr / 4;
    const buf = this.dev.createBuffer({ size: bpr * dy * dz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: state }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(buf.getMappedRange());
    let n = 0;
    for (let z = 0; z < dz; z++) for (let y = 0; y < dy; y++) for (let x = 0; x < dx; x++) if (f[(z * dy + y) * rowF + x * 2] > 0.5) n++;
    buf.unmap();
    buf.destroy();
    return n;
  }
  /** Image intensity range (max−min) for the default similarity scale. */
  async imageRange() {
    const [dx, dy, dz] = this.seg.dims;
    const bpr = Math.ceil(dx * 4 / 256) * 256;
    const rowF = bpr / 4;
    const buf = this.dev.createBuffer({ size: bpr * dy * dz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.imageTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(buf.getMappedRange());
    let mn = Infinity, mx = -Infinity;
    for (let z = 0; z < dz; z++) for (let y = 0; y < dy; y++) for (let x = 0; x < dx; x++) {
      const v = f[(z * dy + y) * rowF + x];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    buf.unmap();
    buf.destroy();
    return mx - mn;
  }
  destroy() {
    this.a.destroy();
    this.b.destroy();
    this.uni.destroy();
  }
};

// algorithms/seg-edit-driver.ts
var SegEditDriver = class _SegEditDriver {
  constructor(seg, opts = {}) {
    this.seg = seg;
    this.opts = opts;
    this.paint = new PaintEffect(seg);
  }
  paint;
  scissors;
  growcut;
  growcutImg;
  // the image the current growcut was built for
  labels = /* @__PURE__ */ new Map();
  // Slicer segment id → r32uint master label
  nextLabel = 1;
  active;
  /** Normalize any of the three carriers → the bare SegEdit payload (or null). */
  static unwrap(op) {
    const o = op;
    if (o && typeof o === "object") {
      if (o.edit && typeof o.edit === "object") return o.edit;
      if (o.cmd === "segEdit" && o.args && typeof o.args === "object") return o.args;
      if (typeof o.kind === "string") return o;
    }
    return null;
  }
  image() {
    const t = this.opts.imageTex;
    return typeof t === "function" ? t() : t;
  }
  labelFor(segmentId) {
    if (!segmentId) return 1;
    if (this.opts.labelForSegment) return this.opts.labelForSegment(segmentId);
    let id = this.labels.get(segmentId);
    if (id === void 0) {
      id = this.nextLabel++;
      this.labels.set(segmentId, id);
    }
    return id;
  }
  radiusFor(e) {
    return (e.brush?.diameterMm ?? this.opts.defaultDiameterMm ?? 6) / 2;
  }
  modeFor(e) {
    if (e.mode === "remove" || (e.effect ?? "").toLowerCase().startsWith("erase")) return "remove";
    return "add";
  }
  strokeOpts(e) {
    return { radiusMm: this.radiusFor(e), id: this.labelFor(e.segmentId), mode: this.modeFor(e) };
  }
  /** Apply one COMMITTED edit (all its points at once) — the replay/live path. `stroke`/`scissors`
   *  submit synchronously; `seeds` (grow-from-seeds) awaits the CA to converge, so this is async. */
  // deno-lint-ignore require-await
  async applyEdit(op) {
    const e = _SegEditDriver.unwrap(op);
    if (!e) return;
    switch (e.kind) {
      case "stroke": {
        const s = e;
        if (s.points?.length) this.paint.stampStroke(s.points, this.strokeOpts(s));
        return;
      }
      case "scissors":
        return this.applyScissors(e);
      case "seeds":
        return this.applySeeds(e);
      default:
        this.opts.onUnhandled?.(e.kind);
        return;
    }
  }
  applyScissors(e) {
    if (!e.contour?.length || !e.u || !e.v) return;
    this.scissors ??= new ScissorsEffect(this.seg);
    this.scissors.apply(e.contour, {
      u: e.u,
      v: e.v,
      operation: e.operation ?? "eraseInside",
      id: this.labelFor(e.segmentId)
    });
  }
  /** Grow-from-seeds: stamp the sparse scribbles into the master (as seeds, one label each), then run
   *  the intensity-guided CA to flood the volume. Needs the source image (opts.imageTex). */
  async applySeeds(e) {
    if (!e.scribbles?.length) return;
    const img = this.image();
    if (!img) {
      this.opts.onUnhandled?.("seeds(no image)");
      return;
    }
    for (const sc of e.scribbles) {
      if (!sc.points?.length) continue;
      const label = sc.label ?? this.labelFor(sc.segmentId ?? e.segmentId);
      const radiusMm = (sc.brush?.diameterMm ?? this.opts.defaultDiameterMm ?? 6) / 2;
      this.paint.stampStroke(sc.points, { radiusMm, id: label, mode: "add" });
    }
    if (!this.growcut || this.growcutImg !== img) {
      this.growcut?.destroy();
      this.growcut = new GrowCutEffect(this.seg, img);
      this.growcutImg = img;
    }
    await this.growcut.grow({
      edgeLo: e.edgeLo,
      edgeHi: e.edgeHi,
      intensityRange: e.intensityRange,
      iterations: e.iterations
    });
  }
  // ── Incremental live path: begin / addPoint / end, as pointer samples arrive (real-time apply,
  //    no wait for mouse-up). A stroke is a pointer-drag stream, exactly like a camera drag. ──
  /** Start an incremental stroke; `meta` carries the same fields a full edit would (minus points). */
  beginStroke(meta = {}) {
    this.active = { opts: this.strokeOpts({ kind: "stroke", points: [], ...meta }), last: void 0 };
  }
  /** Add one sampled point — welds a capsule from the previous sample (first point = a dab). */
  addPoint(p) {
    if (!this.active) return;
    this.paint.stampStroke(this.active.last ? [this.active.last, p] : [p], this.active.opts);
    this.active.last = p;
  }
  endStroke() {
    this.active = void 0;
  }
  destroy() {
    this.paint.destroy();
    this.scissors?.destroy();
    this.growcut?.destroy();
  }
};

// render/bake.ts
var INIT_WGSL2 = (
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
    this.initPipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: INIT_WGSL2 }), entryPoint: "main" } });
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

// render/sdf-bake.ts
var INIT_WGSL3 = (
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
var FINAL_WGSL2 = (
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
  // .r = opacity, .g = shading mode, .a = CRISP presence (1 inside a real segment, 0 background) \u2014
  // the FULLBLUR carries .a unblurred so the shader can tell a genuine in-segment voxel from one that
  // merely caught BLED colour/opacity outside any segment, and gate the shell to the real edge.
  //
  // .b is the BAND-LIMITED LABEL FIELD, and in outer mode it is what the surface IS. It starts as the
  // crisp 0/1 presence and FULLBLUR low-passes it, so the shader reads p = G_\u03C3 * 1(label\u22600) and takes
  // the surface to be its p = 0.5 level set. A labelmap is a SAMPLED indicator function: its voxel
  // staircase is ALIASING, not data, and the faithful surface is that indicator reconstructed with a
  // kernel at the grid's Nyquist limit \u2014 which is what the smoothing control now sets. Thresholding a
  // BOUNDED field at 0.5 also cannot do what blurring the unbounded SIGNED DISTANCE did (fuse two
  // vertebrae across a disc space): in the middle of a 4-voxel gap p \u2248 2\xB7\u03A6(-2) \u2248 0.05, nowhere near
  // 0.5, so the gap stays open at every \u03C3 in range.
  //
  // Interface ("all") mode keeps the blurred DISTANCE here instead: its field is UNSIGNED (distance to
  // the nearest label change), so a union indicator says nothing about the internal interfaces that
  // mode exists to show, and its normal still comes from the smoothed distance.
  let bchan = select(select(0.0, 1.0, ins), sdf, u.params.y > 0.5);
  textureStore(t_attr, c, vec4<f32>(pal.a, mode, bchan, select(0.0, 1.0, ins)));
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
var JfaSdfBaker = class _JfaSdfBaker {
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
    this.scratch = mk("rgba16float", GPUTextureUsage.COPY_SRC);
    this.uni = dev.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.palBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.modeBuf = dev.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const mod = (code) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } });
    this.initPipe = mod(INIT_WGSL3);
    this.jfaPipe = mod(JFA_WGSL);
    this.finalPipe = mod(FINAL_WGSL2);
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
  // ONE blur scratch, shared by the colour, distance and attribute blurs. They are the same format
  // and size and blurStage runs them one at a time, so a second copy was 8 B per voxel of pure
  // duplication — 12% of this pipeline, which is the difference between fitting a device and losing
  // it. Named `scratch` rather than per-target so it cannot drift back into two.
  scratch;
  // rgba16float blur ping-pong (shared)
  lastSeed = 0;
  // seed buffer the last sweep finalized from
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
  /** GPU bytes this baker holds, and what it WOULD hold for a given label grid.
   *
   *  The JFA is the memory-hungry part of the pipeline by a wide margin: the seed ping-pong is two
   *  rgba32float volumes (32 B/voxel between them) and there are four more rgba16float volumes
   *  (sdf, attr, and a scratch for each). On the PADDED grid that is 68 B per voxel including the
   *  r32uint master the caller owns — so a 250³ grid is 1.06 GB, per segmentation. Two of them side
   *  by side is how a comparison view walks off the end of a laptop's VRAM: the device is lost and
   *  the canvas goes white, with no error a user can act on. Callers size their grid against this
   *  (see render/gpu-budget.ts) instead of finding out the hard way. */
  static bytesPerVoxel() {
    return 2 * 16 + 8 + 8 + 8 + 4;
  }
  /** Bytes for a label grid of `dims` once padded by `pad` voxels on every side. */
  static estimateBytes(dims, pad = 2) {
    const n = (dims[0] + 2 * pad) * (dims[1] + 2 * pad) * (dims[2] + 2 * pad);
    return n * _JfaSdfBaker.bytesPerVoxel();
  }
  /** Bytes this instance actually holds. */
  byteSize() {
    const n = this.dims[0] * this.dims[1] * this.dims[2];
    return n * _JfaSdfBaker.bytesPerVoxel();
  }
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
  /** Distance-blur σ, in VOXELS, applied to the signed distance on every bake/refine (0 = none).
   *  The zero level set barely moves, so the silhouette stays where the labelmap put it; what the
   *  blur smooths is the GRADIENT, i.e. the shading normal — which is what turns a voxel-faceted
   *  "golf ball" into a surface. Takes effect on the next bake(); the caller decides when that is,
   *  since a sweep is ~20 compute passes. */
  setSmoothing(sigmaVoxels) {
    this.smoothSigma = Math.max(0, sigmaVoxels);
  }
  smoothing() {
    return this.smoothSigma;
  }
  /** σ for the COLOUR-seam blur, which is NOT the geometry σ and must not follow it up.
   *
   *  The seam blur exists for one narrow job: the region colour is flooded per voxel, so the border
   *  between two labels is a voxel staircase in COLOUR even when the surface is smooth. A fraction of
   *  a voxel of blur hides that. Tying it to the geometry σ (it used to be min(1, σ)) meant that
   *  asking for a smoother SURFACE also dissolved the label boundaries — and a boundary is the entire
   *  signal in a comparison view. It showed up worst on the SPINEPS disc layer: a near-white disc
   *  between two saturated vertebra colours, blurred a full voxel, reads as a milky smear over both
   *  bones instead of a disc. Capped well below a voxel, the seam stays a seam. */
  seamSigma() {
    return Math.min(0.4, this.smoothSigma);
  }
  /** FAST bake for LIVE editing: plain JFA (approximate) + the attr blur that band-limits the label
   *  field (no colour-seam blur). Cheap, so it keeps up with an in-progress stroke; the colour seams
   *  stay voxel-jagged until the edit settles and refine() runs. */
  bake() {
    this.sweep([], this.smoothSigma, 0);
  }
  /** REFINE for a STATIC labelmap (run once the edit settles): JFA+2 extra passes → a near-exact
   *  Voronoi/SDF (fixes the small JFA mistakes near close/overlapping segments) and a colour-seam blur
   *  so neighbouring-label boundaries are smooth, not a voxel staircase. The attr blur stays at the
   *  caller's σ — it is the geometry, not a quality knob. Higher quality lives in the resident
   *  texture, so camera renders stay cheap. */
  refine() {
    this.sweep([2, 1], this.smoothSigma, this.seamSigma());
  }
  /** REGION-LIMITED refine: re-flood ONLY `regionIjk` (padded-grid coords) after a labelmap edit
   *  confined to it — a per-vertebra visibility flip re-bakes a few % of the grid instead of the
   *  whole volume, which is what makes level stepping feel instant. The seed ping-pong pair is
   *  first made consistent with a full-texture copy, so region passes can ping-pong while JFA
   *  taps read valid exterior seeds (surfaces just outside the region flood in correctly).
   *  Exterior distances that referenced a surface REMOVED inside the region go stale, but only
   *  ≫band away from any visible shell — invisible, and the next full sweep cleans them. */
  refineRegion(regionIjk, maxStep = Infinity) {
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
    const steps = [...this.steps.filter((v) => v <= maxStep), 2, 1];
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
    if (this.seamSigma() > 0) this.blurStage(this.colBlurPipe, this.seamSigma(), this.sdfTex, this.scratch, region);
    if (this.smoothSigma > 0) this.blurStage(this.fullBlurPipe, this.smoothSigma, this.attrTex, this.scratch, region);
  }
  /** ATTR-ONLY rebake for palette/opacity changes (per-segment visibility): the distance field
   *  doesn't move, so re-run ONLY the finalize (from the last sweep's seed) with its sdf writes
   *  routed to the scratch texture (discarded — the blurred resident sdfTex stays pristine) and
   *  re-blur the attributes. ~4 passes instead of the ~20-pass init+JFA+blur sweep, which is
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
      { binding: 2, resource: this.scratch.createView() },
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
    if (this.smoothSigma > 0) this.blurStage(this.fullBlurPipe, this.smoothSigma, this.attrTex, this.scratch, region);
  }
  /** Re-apply the SMOOTHING only: the distance field does not depend on σ, so re-running the JFA
   *  for a smoothing change is both wasteful and WRONG — the fast bake's approximate flood differs
   *  from the refined one by a fraction of a voxel, so the surface visibly shifted back and forth
   *  as the control moved. This re-derives the attributes and the colour from the EXISTING seeds
   *  (so the distance is bit-identical) and re-blurs them at the new σ. */
  resmooth() {
    const dev = this.dev;
    this.writeUni(0, [0, 0, 0]);
    const enc = dev.createCommandEncoder();
    const bf = dev.createBindGroup({ layout: this.finalPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.seed[this.lastSeed].createView() },
      { binding: 1, resource: this.labelTex.createView() },
      { binding: 2, resource: this.sdfTex.createView() },
      // re-derive colour + the SHARP distance
      { binding: 3, resource: { buffer: this.uni } },
      { binding: 4, resource: { buffer: this.palBuf } },
      { binding: 5, resource: this.attrTex.createView() },
      { binding: 6, resource: { buffer: this.modeBuf } }
    ] });
    const p = enc.beginComputePass();
    p.setPipeline(this.finalPipe);
    p.setBindGroup(0, bf);
    p.dispatchWorkgroups(this.g[0], this.g[1], this.g[2]);
    p.end();
    dev.queue.submit([enc.finish()]);
    const seam = this.seamSigma();
    if (seam > 0) this.blurStage(this.colBlurPipe, seam, this.sdfTex, this.scratch);
    if (this.smoothSigma > 0) this.blurStage(this.fullBlurPipe, this.smoothSigma, this.attrTex, this.scratch);
  }
  /** Blur the attribute seams of the CURRENT attr texture in place (no re-finalize) — the
   *  cheapest possible settle after a run of rebakeAttr(false) visibility steps. */
  blurAttrOnly() {
    if (this.smoothSigma > 0) this.blurStage(this.fullBlurPipe, this.smoothSigma, this.attrTex, this.scratch);
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
    if (colorSigma > 0) this.blurStage(this.colBlurPipe, colorSigma, this.sdfTex, this.scratch);
    if (distSigma > 0) this.blurStage(this.fullBlurPipe, distSigma, this.attrTex, this.scratch);
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
    this.scratch.destroy();
    this.uni.destroy();
    this.palBuf.destroy();
    this.modeBuf.destroy();
  }
};

// render/fields.ts
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
  voxelMm = 1;
  providesSkip;
  /** The band-limited SOLID path: sdf mode, outer boundary, with the attr texture that carries the
   *  band-limited label field. This is the only variant that uses the interval crossing test. */
  outerSolid;
  /** Interval sampling (Field): the shader is handed the ray distance back to THIS field's previous
   *  sample, so consecutive intervals tile the ray exactly — what makes the crossing test immune to
   *  the per-step ray jitter and to an empty-space leap. */
  intervalSampling;
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
    this.voxelMm = voxelMm;
    this.mode = opts.mode ?? "iso";
    this.providesSkip = this.mode === "sdf";
    this.colorFromTex = opts.colorFromTexture ?? false;
    this.interfaceMode = this.mode === "sdf" && (opts.interfaceMode ?? false);
    this.attrTex = this.mode === "sdf" ? opts.attrTexture : void 0;
    this.bindingCount = this.attrTex ? 2 : 1;
    this.outerSolid = this.mode === "sdf" && !this.interfaceMode && !!this.attrTex;
    this.intervalSampling = this.outerSolid;
  }
  /** Field-level opacity (multiplies every segment's per-label opacity in the shader). Live global
   *  segmentation opacity — the caller does scene.syncUniforms() + redraw to apply. */
  setOpacity(o) {
    this.opacity = Math.max(0, Math.min(1, o));
  }
  /** Render the sdf field as VOXEL CUBES instead of its smooth iso-surface (sdf mode only).
   *  What "no smoothing" should mean for a labelmap: one cube per voxel, flat-shaded per face. */
  setVoxelMode(on) {
    this.voxelMode = on;
  }
  voxelMode = false;
  /** Reconstruction width σ (VOXELS) of the band-limited label field the sdf surface is the 0.5
   *  level set of. The field itself is baked by JfaSdfBaker (attr.b); the renderer only needs σ to
   *  keep the sphere-trace skip from leaping over a surface that has moved off the sharp one.
   *  Caller does scene.syncUniforms(). */
  setSmoothingVox(sigmaVoxels) {
    this.sigmaVox = Math.max(0, sigmaVoxels);
  }
  sigmaVox = 0;
  uniformFloats() {
    return 36;
  }
  // mat4(16) + color(4) + shade(4) + params(4) + bmin(4) + bmax(4)
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
  /** Empty-space skip for "sdf" mode: the texture .a is a TRUE distance-to-surface (mm), so the ray can
   *  leap |sdf| minus the shell band toward the surface — sphere tracing. A one-voxel safety margin
   *  absorbs the JFA distance approximation so the leap never overshoots a thin shell (image unchanged,
   *  just far fewer march steps). Self-contained (no dependency on the sampling fns' emission order). */
  skipWGSL(s) {
    return (
      /* wgsl */
      `
fn skip_seg${s}(wp : vec3<f32>) -> f32 {
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) {
    // OUTSIDE the SDF grid: leap only to the seg's AABB (a contract-valid lower bound), never past it.
    let cen = (u_material.seg${s}_bmin.xyz + u_material.seg${s}_bmax.xyz) * 0.5;
    let ext = (u_material.seg${s}_bmax.xyz - u_material.seg${s}_bmin.xyz) * 0.5;
    let q = abs(transform_point_seg${s}(wp) - cen) - ext;
    return max(0.0, length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0));
  }
  let sd = textureSampleLevel(t_seg${s}, s_lin, t, 0.0).a;                      // SIGNED distance to the labelmap surface (mm)
  // An UNSEEDED voxel \u2014 one the jump flood never reached a seed from \u2014 reports 1e3 mm. That is not a
  // distance, it is "unknown", and leaping on it sails through whatever surface is really there,
  // which is how a partial re-flood used to come back as a flat cut across the shell. No real SDF
  // grid here spans 500 mm, so treat anything beyond that as unknown and march instead of leaping.
  if (abs(sd) > 500.0) { return 0.0; }
  // params.w = \u03C3 in mm. The rendered surface is the band-limited field's 0.5 level set, which sits
  // within ~\u03C3 of this sharp one, so the leap has to stop that much short of it or a \u03C3-wide feature
  // can be jumped clean over.
  var leap = max(0.0, abs(sd) - u_material.seg${s}_params.x - u_material.seg${s}_params.y - u_material.seg${s}_params.w);
${this.clippable && this.outerSolid ? `  // INSIDE the solid, a clip plane is a surface too (the cut face). Leaping on |sd| alone would sail
  // past it \u2014 the ray enters through the clip, sees a big interior distance, and jumps to the far
  // wall, so the cap never gets sampled. Clamp the leap to the nearest plane while inside.
  if (sd < 0.0) {
    let ccount = u32(u_material.clip_count.x);
    for (var ci = 0u; ci < ccount; ci = ci + 1u) {
      let cp = u_material.clip_planes[ci];
      leap = min(leap, abs(dot(wp, cp.xyz) + cp.w));
    }
  }` : ``}
  return leap;
}`
    );
  }
  structMembers(s) {
    return [
      `  seg${s}_p2t : mat4x4<f32>,`,
      `  seg${s}_color : vec4<f32>,`,
      // rgb, opacity
      `  seg${s}_shade : vec4<f32>,`,
      // ka, kd, ks, shininess
      `  seg${s}_params : vec4<f32>,`,
      // band_mm, voxel_mm, voxel_cubes, smoothing_mm
      `  seg${s}_bmin : vec4<f32>,`,
      // aabb min (RAS) — for a contract-valid skip outside the texture
      `  seg${s}_bmax : vec4<f32>,`
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
}` : ""}${this.interfaceMode && this.attrTex ? `
fn bdist_seg${s}(wp : vec3<f32>) -> f32 {   // SMOOTHED interface distance from attr.b (interface mode)
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  return textureSampleLevel(t_attr${s}, s_lin, t, 0.0).b;
}
fn pres_seg${s}(wp : vec3<f32>) -> f32 {   // CRISP in-segment presence (attr.a), linear-sampled for ~1-voxel AA
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return 0.0; }
  return textureSampleLevel(t_attr${s}, s_lin, t, 0.0).a;
}` : ""}${this.outerSolid ? `
fn pfield_seg${s}(wp : vec3<f32>) -> f32 {
  // THE BAND-LIMITED LABEL FIELD: p = G_\u03C3 * 1(label != 0), carried in attr.b (JfaSdfBaker finalize
  // writes the 0/1 indicator there, the attr blur low-passes it at the smoothing \u03C3). Clamped to the
  // texture edge so a gradient tap just outside the padded grid reads real background instead of a
  // cull sentinel. The surface is the p = 0.5 level set of this, and nothing else.
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  return textureSampleLevel(t_attr${s}, s_lin, t, 0.0).b;
}` : ""}${this.interfaceMode ? `
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
// VOXEL CUBES (params.z): the labelmap AS SEGMENTED, one cube per voxel, flat-shaded per face.
//
// This exists because reading the distance field with LINEAR sampling cannot produce it. The zero
// level set of a Euclidean distance field is a smooth surface, and its gradient \u2014 the shading
// normal \u2014 has RIDGES along the medial lines where the distances from neighbouring source voxels
// meet. Turning the blur off therefore does not reveal voxels; it reveals those ridges, which is
// what "no smoothing" used to look like.
//
// So this path never interpolates: it reads the same texture with textureLoad at the voxel the
// sample falls in. That voxel is solid iff ITS OWN CENTRE is inside (.a < 0), the surface is the
// cube faces of the solid voxels, and the normal is the face normal \u2014 constant across each face,
// which is what makes a face read as a face. A face whose neighbour is also solid is interior and
// draws nothing, so the result is the outside of the voxel set, not a lattice.
//
// It is still the same ray-march through the same field: no geometry is built, and the opacity
// uses the same shell profile as the smooth path, measured to the cube face instead of to the
// iso-surface, so the opacity controls behave identically at every smoothing setting.
fn solid_seg${s}(ix : vec3<i32>) -> bool {
  // "Is this voxel part of the segmentation?" \u2014 which channel answers that depends on the field.
  // The signed (outer) field says it with the sign of the distance; the multi-material INTERFACE
  // field's distance is UNSIGNED (it measures to the nearest label change, not to the outside),
  // so there it is the crisp per-voxel presence in the attr texture that knows. Reading the sign
  // of an unsigned field is how voxel mode first came out completely empty on every nested
  // segmentation \u2014 which is every segmentation with a tumour inside an organ.
${this.interfaceMode && this.attrTex ? `  return textureLoad(t_attr${s}, ix, 0).a > 0.5;` : `  return textureLoad(t_seg${s}, ix, 0).a < 0.0;`}
}
fn voxels_seg${s}(wp : vec3<f32>, rd : vec3<f32>, band : f32, step : f32, seg_op : f32, op0 : f32) -> vec4<f32> {
  let t4 = u_material.seg${s}_p2t * vec4<f32>(transform_point_seg${s}(wp), 1.0);
  let t = t4.xyz;
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return vec4<f32>(0.0); }
  let dimsf = vec3<f32>(textureDimensions(t_seg${s}));
  let ci = floor(t * dimsf - vec3<f32>(0.5) + vec3<f32>(0.5));   // the voxel this sample is in
  let idx = vec3<i32>(ci);
  let me = textureLoad(t_seg${s}, idx, 0);
  if (!solid_seg${s}(idx)) { return vec4<f32>(0.0); }  // this voxel's centre is outside the segmentation
  // A face perpendicular to texture axis k has texture-space normal e_k; a normal maps from
  // texture to world by the TRANSPOSE of the world\u2192texture linear part (row k of p2t).
  let M = u_material.seg${s}_p2t;
  var rows = array<vec3<f32>, 3>(
    normalize(vec3<f32>(M[0][0], M[1][0], M[2][0])),
    normalize(vec3<f32>(M[0][1], M[1][1], M[2][1])),
    normalize(vec3<f32>(M[0][2], M[1][2], M[2][2])));
  var offs = array<vec3<i32>, 6>(
    vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0), vec3<i32>(0, 1, 0),
    vec3<i32>(0, -1, 0), vec3<i32>(0, 0, 1), vec3<i32>(0, 0, -1));
  // Which faces of this cube are EXPOSED (their neighbour is outside)? Of those, shade the one
  // the ray meets most squarely. Picking the geometrically nearest face instead leaves the
  // surface stippled: a ray crossing near a cube's edge is often nearest an INTERIOR face, and
  // that sample then draws nothing even though the cube is visible there.
  let lim = vec3<i32>(dimsf) - vec3<i32>(1);
  var bestDot = -2.0;
  var n = vec3<f32>(0.0);
  var exposed = false;
  var open = array<bool, 6>(false, false, false, false, false, false);
  for (var f = 0; f < 6; f = f + 1) {
    if (solid_seg${s}(clamp(idx + offs[f], vec3<i32>(0), lim))) { continue; }   // interior face
    open[f] = true;
    exposed = true;
    let sgn = select(-1.0, 1.0, (f & 1) == 0);
    let nw = rows[f >> 1] * sgn;
    let dd = -dot(nw, rd);
    if (dd > bestDot) { bestDot = dd; n = nw; }
  }
  if (!exposed) { return vec4<f32>(0.0); }             // fully buried voxel \u2014 the shell is one cube thick
  // Which face did the ray actually come IN through? That \u2014 not the face most square to the
  // camera \u2014 is what makes a cube read as a cube: the step faces catch the light differently
  // from the faces around them, which is the whole visual signal that this is voxels. Slab
  // entry test in voxel space; if that face turns out to be interior (possible when the march
  // steps past a thin feature), fall back to the exposed face the ray meets most squarely.
  let rv = vec3<f32>(
    dot(vec3<f32>(M[0][0], M[1][0], M[2][0]), rd),
    dot(vec3<f32>(M[0][1], M[1][1], M[2][1]), rd),
    dot(vec3<f32>(M[0][2], M[1][2], M[2][2]), rd)) * dimsf;
  let r = t * dimsf - vec3<f32>(0.5) - ci;             // position within the voxel, [-0.5, 0.5]
  var tIn = -1e9;
  var face = -1;
  if (abs(rv.x) > 1e-6) { let tk = (-0.5 * sign(rv.x) - r.x) / rv.x; if (tk > tIn) { tIn = tk; face = select(1, 0, rv.x < 0.0); } }
  if (abs(rv.y) > 1e-6) { let tk = (-0.5 * sign(rv.y) - r.y) / rv.y; if (tk > tIn) { tIn = tk; face = select(3, 2, rv.y < 0.0); } }
  if (abs(rv.z) > 1e-6) { let tk = (-0.5 * sign(rv.z) - r.z) / rv.z; if (tk > tIn) { tIn = tk; face = select(5, 4, rv.z < 0.0); } }
  if (face >= 0 && open[face]) {
    n = rows[face >> 1] * select(-1.0, 1.0, (face & 1) == 0);
  }
${this.attrTex ? `  let at = textureLoad(t_attr${s}, idx, 0);
  let vop = at.r;                                      // per-segment opacity, this voxel's own
  let vcol = me.rgb / max(vop, 1e-3);                  // un-premultiply` : `  let vop = seg_op;
  let vcol = me.rgb;`}
  let T = clamp(op0 * vop, 0.0, 1.0);
  if (T <= 0.0) { return vec4<f32>(0.0); }
  // Optical depth calibrated so crossing ONE voxel accumulates exactly T \u2014 the cube-sized
  // equivalent of the shell profile the smooth path uses, so the opacity controls mean the same
  // thing at every smoothing setting.
  let vox = max(u_material.seg${s}_params.y, 1e-4);
  let rate = max(abs(dot(rd, n)), 1e-3);
  let tau = -log(1.0 - min(T, 0.9999)) * (step * rate) / vox;
  let op = 1.0 - exp(-tau);
  if (op <= 0.0004) { return vec4<f32>(0.0); }
  let ka = u_material.seg${s}_shade.x; let kd = u_material.seg${s}_shade.y;
  let ldn = max(dot(-rd, n), 0.0);
  // Faces need to differ in brightness by WHICH WAY THEY FACE, not by how square they are to the
  // camera. A head-lit cube surface viewed down one of its own axes has every visible face at the
  // same angle to the light, and with ka+kd \u2248 1 they all clip to the same flat colour \u2014 which is
  // exactly what "voxel mode" looked like before this: a flat silhouette with stepped edges.
  // A fixed overhead sky term (superior-facing brightest, inferior darkest, sides between) makes
  // each face of each cube read separately from any viewpoint, which is the whole point of the
  // mode. No specular: a highlight across flat faces reads as curvature.
  let sky = 0.55 + 0.45 * clamp(0.5 + 0.5 * n.z, 0.0, 1.0);
  var lit = vcol * sky * (ka + kd * ldn);
  lit = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * op, op);
}
// Surface contribution at wp: a crisp Phong surface. Weighted by (1-mode) so it morphs smoothly
// into the volume contribution across a blurred surface\u2194volume boundary.
fn surface_seg${s}(wp : vec3<f32>, rd : vec3<f32>, ds : f32, sdf : f32, band : f32, step : f32, seg_op : f32, op0 : f32) -> vec4<f32> {
  if (u_material.seg${s}_params.z > 0.5) { return voxels_seg${s}(wp, rd, band, step, seg_op, op0); }
${this.outerSolid ? `  // A BAND-LIMITED SOLID, found by an INTERVAL CROSSING TEST.
  //
  // A labelmap is a SAMPLED indicator function, and its voxel steps are above the grid's Nyquist
  // limit \u2014 they are aliasing, not anatomy. So the faithful surface is the 0.5 level set of that
  // indicator RECONSTRUCTED with a kernel at the limit: p = G_\u03C3 * 1(label != 0), which the baker
  // low-passes into attr.b and the smoothing control sets the \u03C3 of. Reconstructing with plain
  // trilinear interpolation instead (what \u03C3\u21920 gives) reproduces the ALIASED surface \u2014 flat voxel
  // faces and chamfered edges, which is the chunk. Band-limited, the surface still passes through
  // the voxel faces but carries no energy above Nyquist, which is the smooth.
  //
  // This MOVES the geometry with \u03C3, which earlier versions refused to do because smoothing the
  // signed DISTANCE closed the disc space between two vertebrae and fused them. A BOUNDED
  // indicator cannot: at the centre of a 4-voxel gap p \u2248 2\xB7\u03A6(-2) \u2248 0.05, nowhere near 0.5. What it
  // does cost is the curvature term, \u2248 \u03C3\xB2/2R inward on a convex ridge of radius R \u2014 a quarter of a
  // voxel at \u03C3=1 on a 2-voxel vertebral process. Bounded and predictable, which is the trade a
  // smoothing control is supposed to offer.
  //
  // WHY A CROSSING TEST AND NOT A SHELL BAND. A band wants the PERPENDICULAR distance to the
  // surface, and the only way to get that from a scalar field is (0.5 - p)/|grad p| \u2014 which falls
  // apart exactly where |grad p| collapses: concave creases where two surfaces meet, and thin
  // features. There the estimated distance diverges, no sample falls inside the band, and an
  // OPAQUE segment renders dusty, with background showing through it. (Measured on a sphere with a
  // plate through it: 5-7% of interior pixels leaking, up to 31% on a single pixel; the shell-band
  // design leaked 8-15%.) The test p > 0.5 has no such failure: it is well defined everywhere, so
  // asking whether THIS interval contains the crossing cannot miss a surface, at any \u03C3, curvature or
  // incidence. Field.intervalSampling hands us the exact distance back to this field's previous
  // sample, so consecutive intervals tile the ray with no gap and no overlap \u2014 which is what makes
  // the test exact under the per-step ray jitter, and across an empty-space leap.
  let p1 = pfield_seg${s}(wp);
  let back = wp - rd * max(ds, 1.0e-4);
  let p0 = pfield_seg${s}(back);
  let in1 = p1 > 0.5;
  let in0 = p0 > 0.5;
  var capNormal = vec3<f32>(0.0);
  var prevClipped = false;` : ``}${this.outerSolid && this.clippable ? `
  // SOLID: where a clip plane cuts the body, the body has a FACE there. The ray enters the kept
  // half-space already inside the solid \u2014 no surface crossing \u2014 so the cut would be a hole into a
  // bubble unless we treat that entry as a face of its own, with the plane's normal. The previous
  // sample is the test: if it was on the discarded side, this interval crossed the plane.
  let ccount = u32(u_material.clip_count.x);
  for (var ci = 0u; ci < ccount; ci = ci + 1u) {
    let cp = u_material.clip_planes[ci];
    if (dot(back, cp.xyz) + cp.w < 0.0) { prevClipped = true; capNormal = -cp.xyz; }
  }` : ``}${this.outerSolid ? `
  let surfaceCross = in1 != in0;
  if (!(surfaceCross || (in1 && prevClipped))) { return vec4<f32>(0.0); }
  // Where in the interval: linear in p, which is exact to first order and sub-voxel in practice.
  let u = select(1.0, clamp((0.5 - p0) / (p1 - p0), 0.0, 1.0), surfaceCross);
  let wc = back + (wp - back) * u;
  let T = clamp(op0 * seg_op, 0.0, 1.0);
  if (T <= 0.0) { return vec4<f32>(0.0); }
  // Normal from a FULL-VOXEL central difference of p. attr.b is read trilinearly, so the field is
  // only C0 and its gradient jumps at every voxel-face plane; differencing inside a single cell
  // (h = half a voxel) reports that cell's own gradient and the jumps show up as a rectangular
  // lattice of dark creases over the surface \u2014 a maze, worst at small \u03C3 where the field is
  // steepest. A full-voxel difference straddles the discontinuity and averages it out. Geometry
  // and normal now come from the SAME field, which is what stops a stepped silhouette from being
  // lit like a smooth one.
  let hg = max(u_material.seg${s}_params.y, step);
  let gp = vec3<f32>(
    pfield_seg${s}(wc + vec3<f32>(hg,0,0)) - pfield_seg${s}(wc - vec3<f32>(hg,0,0)),
    pfield_seg${s}(wc + vec3<f32>(0,hg,0)) - pfield_seg${s}(wc - vec3<f32>(0,hg,0)),
    pfield_seg${s}(wc + vec3<f32>(0,0,hg)) - pfield_seg${s}(wc - vec3<f32>(0,0,hg))) / (2.0 * hg);
  let glen = length(gp);
  // p increases INWARD, so the outward normal is -grad p. On a cap (entered through a clip plane
  // with no crossing) the face is the plane, so its normal is the plane's.
  var n = select(capNormal, -gp / max(glen, 1.0e-6), surfaceCross && glen > 1.0e-5);
  if (dot(n, n) < 0.25) { n = -rd; }     // degenerate: face the camera rather than drop the sample
  if (dot(n, -rd) < 0.0) { n = -n; }
  // ONE crossing = exactly the target opacity T, by construction \u2014 no band, no profile integral,
  // no dependence on the sample step or the incidence angle. Front and back faces of a translucent
  // segment still compose to 1-(1-T)\xB2, as a pair of polydata surfaces would.
  let op = T;
  let ka = u_material.seg${s}_shade.x; let kd = u_material.seg${s}_shade.y;
  let ks = u_material.seg${s}_shade.z; let sh = u_material.seg${s}_shade.w;
  let ldn = max(dot(-rd, n), 0.0);
  let refl = normalize(2.0 * ldn * n + rd);
  let rdv = max(dot(refl, -rd), 0.0);
  let col = col_seg${s}(wc);
  var lit = col * ka + col * (kd * ldn) + vec3<f32>(ks * pow(rdv, max(sh, 1.0)));
  lit = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * op, op);` : `
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
  if (glen < 1.0e-5) { return vec4<f32>(0.0); }
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
  let rate = max(abs(dot(rd, ${this.interfaceMode ? "n" : "g"})), 1.0e-3);
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
  return vec4<f32>(lit * op, op);`}
}
fn sample_field_seg${s}(wp : vec3<f32>, rd : vec3<f32>${this.intervalSampling ? ", ds : f32" : ""}) -> vec4<f32> {
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
    acc += (1.0 - mode) * surface_seg${s}(wp, rd, ${this.intervalSampling ? "ds" : "step"}, sdf, band, step, seg_op, op0);
  }
  return acc;` : `  return surface_seg${s}(wp, rd, step, sdf, band, step, 1.0, op0);`}
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
    out[off + 25] = this.voxelMm;
    out[off + 26] = this.voxelMode ? 1 : 0;
    out[off + 27] = this.sigmaVox * this.voxelMm;
    out[off + 28] = this.box[0][0];
    out[off + 29] = this.box[0][1];
    out[off + 30] = this.box[0][2];
    out[off + 32] = this.box[1][0];
    out[off + 33] = this.box[1][1];
    out[off + 34] = this.box[1][2];
  }
  bindEntries(_s, base) {
    const e = [{ binding: base, resource: this.tex.createView() }];
    if (this.attrTex) e.push({ binding: base + 1, resource: this.attrTex.createView() });
    return e;
  }
};

// render/label-surface-bake.ts
var DEFAULT_LABEL_SURFACE_BAKE = {
  sigmaVoxels: 0.85,
  outer: "twicing",
  keepDeepest: true,
  fillGaps: true,
  thinRule: true
};
var LABEL_SURFACE_BLOCK = 4;
var U_STRUCT = (
  /* wgsl */
  `
struct U {
  ijkToRAS : mat4x4<f32>,   // padded grid -> RAS (column-major)
  dims : vec4<u32>,          // padded grid dims; w = pad
  params : vec4<f32>,        // x = jump-flood step (voxels)
  origin : vec4<i32>,        // dispatch origin (always 0 here)
  ldims : vec4<u32>,         // label grid dims; w = flags (1 keep deepest, 2 thin rule)
  sigma : vec4<f32>,         // xyz = \u03C3 per axis (voxels), w = \u03C3 (mm)
  gap : vec4<f32>,           // xyz = widest gap filled per axis (voxels), w = outer mode (0 twicing, 1 blur, 2 mask, 3 sharp)
  spacing : vec4<f32>,       // xyz = spacing (mm), w = band (mm): distances are clamped to it
};`
);
var U_BYTES = 176;
var LABELS_WGSL = (
  /* wgsl */
  `
fn rawLabel(c : vec3<i32>) -> u32 {
  let lc = c - vec3<i32>(i32(u.dims.w));
  if (any(lc < vec3<i32>(0)) || any(lc >= vec3<i32>(u.ldims.xyz))) { return 0u; }
  return textureLoad(t_label, lc, 0).r;
}
fn shownLabel(c : vec3<i32>) -> u32 {
  let l = rawLabel(c);
  if (l == 0u || l > 255u) { return 0u; }
  return select(0u, l, textureLoad(t_pal, vec2<i32>(i32(l), 0), 0).a > 0.0);
}`
);
var GRID_GUARD = (
  /* wgsl */
  `
  let c = vec3<i32>(gid);
  if (any(c >= vec3<i32>(u.dims.xyz))) { return; }`
);
var LABEL_SURFACE_MAX_DIM = 1024;
var SEED_PACK_WGSL = (
  /* wgsl */
  `
const NO_SEED = 0xffffffffu;
fn seedPack(lo : vec3<i32>, axis : i32) -> u32 {
  let l = vec3<u32>(max(lo, vec3<i32>(0)));
  return l.x | (l.y << 10u) | (l.z << 20u) | (u32(axis) << 30u);
}
// Grid position of a seed: half a voxel along its axis from its lower voxel.
fn seedPos(s : u32) -> vec3<f32> {
  var p = vec3<f32>(f32(s & 1023u), f32((s >> 10u) & 1023u), f32((s >> 20u) & 1023u));
  let a = min(s >> 30u, 2u);
  p[a] = p[a] + 0.5;
  return p;
}
// Squared distance (mm\xB2) from grid voxel c to seed s.
fn seedDist2(c : vec3<i32>, s : u32) -> f32 {
  let d = (u.ijkToRAS * vec4<f32>(seedPos(s) - vec3<f32>(c), 0.0)).xyz;
  return dot(d, d);
}`
);
var SHARE_PACK_WGSL = (
  /* wgsl */
  `
fn sharePack(own : i32, other : i32, share : f32, near : bool) -> u32 {
  return u32(own) | (u32(other) << 8u) | (u32(round(clamp(share, 0.0, 1.0) * 32767.0)) << 16u) | select(0u, 0x80000000u, near);
}
fn shareUnpack(p : u32) -> vec4<f32> {
  return vec4<f32>(f32(p & 255u), f32((p >> 8u) & 255u), f32((p >> 16u) & 32767u) / 32767.0, f32(p >> 31u));
}`
);
var MASK_WGSL = (
  /* wgsl */
  `${U_STRUCT}
@group(0) @binding(0) var t_label : texture_3d<u32>;
@group(0) @binding(1) var t_pal : texture_2d<f32>;
@group(0) @binding(2) var t_out : texture_storage_3d<r32uint, write>;
@group(0) @binding(3) var<uniform> u : U;
${LABELS_WGSL}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {${GRID_GUARD}
  var inside = shownLabel(c) != 0u;
  for (var axis = 0; axis < 3; axis = axis + 1) {
    if (inside) { break; }
    let maxGap = i32(u.gap[axis] + 0.5);
    if (maxGap <= 0) { continue; }
    var e = vec3<i32>(0);
    e[axis] = 1;
    var plus = 0;
    var minus = 0;
    for (var i = 1; i <= 8; i = i + 1) {
      if (i > maxGap) { break; }
      if (plus == 0 && shownLabel(c + e * i) != 0u) { plus = i; }
      if (minus == 0 && shownLabel(c - e * i) != 0u) { minus = i; }
    }
    inside = plus > 0 && minus > 0 && plus + minus - 1 <= maxGap;
  }
  textureStore(t_out, c, vec4<u32>(select(0u, 1u, inside), 0u, 0u, 0u));
}`
);
var SEED_WGSL = (
  /* wgsl */
  `${U_STRUCT}
@group(0) @binding(0) var t_mask : texture_3d<u32>;
@group(0) @binding(1) var t_seed_out : texture_storage_3d<r32uint, write>;
@group(0) @binding(2) var<uniform> u : U;
${SEED_PACK_WGSL}
fn insideAt(c : vec3<i32>) -> bool {
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(u.dims.xyz))) { return false; }
  return textureLoad(t_mask, c, 0).r != 0u;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {${GRID_GUARD}
  let inside = insideAt(c);
  var best = 1.0e30;
  var seed = NO_SEED;
  for (var axis = 0; axis < 3; axis = axis + 1) {
    var e = vec3<i32>(0);
    e[axis] = 1;
    let fd = 0.5 * u.spacing[axis];
    let pd = insideAt(c + e) != inside;
    let md = insideAt(c - e) != inside;
    if (fd < best && (pd || md)) {
      best = fd;
      seed = seedPack(select(c - e, c, pd), axis);   // the face to +e is c's; the face to -e is (c - e)'s
    }
  }
  textureStore(t_seed_out, c, vec4<u32>(seed, 0u, 0u, 0u));
}`
);
var JFA_WGSL2 = (
  /* wgsl */
  `${U_STRUCT}
@group(0) @binding(0) var t_in : texture_3d<u32>;
@group(0) @binding(1) var t_out : texture_storage_3d<r32uint, write>;
@group(0) @binding(2) var<uniform> u : U;
${SEED_PACK_WGSL}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {${GRID_GUARD}
  let step = i32(u.params.x);
  let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
  var best = textureLoad(t_in, c, 0).r;
  var bestD = select(1.0e30, seedDist2(c, best), best != NO_SEED);
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        if (dx == 0 && dy == 0 && dz == 0) { continue; }
        let sd = textureLoad(t_in, clamp(c + vec3<i32>(dx, dy, dz) * step, vec3<i32>(0), dmax), 0).r;
        if (sd != NO_SEED) {
          let d = seedDist2(c, sd);
          if (d < bestD) { bestD = d; best = sd; }
        }
      }
    }
  }
  textureStore(t_out, c, vec4<u32>(best, 0u, 0u, 0u));
}`
);
var FINAL_WGSL3 = (
  /* wgsl */
  `${U_STRUCT}
@group(0) @binding(0) var t_seed : texture_3d<u32>;
@group(0) @binding(1) var t_mask : texture_3d<u32>;
@group(0) @binding(2) var t_dist : texture_storage_3d<r32float, write>;
@group(0) @binding(3) var t_ind : texture_storage_3d<r32float, write>;
@group(0) @binding(4) var<uniform> u : U;
${SEED_PACK_WGSL}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {${GRID_GUARD}
  let s = textureLoad(t_seed, c, 0).r;
  let band = u.spacing.w;
  let d = select(band, min(sqrt(seedDist2(c, s)), band), s != NO_SEED);
  let inside = textureLoad(t_mask, c, 0).r != 0u;
  textureStore(t_dist, c, vec4<f32>(select(d, -d, inside), 0.0, 0.0, 0.0));
  textureStore(t_ind, c, vec4<f32>(select(0.0, 1.0, inside), 0.0, 0.0, 0.0));
}`
);
var BLUR_WGSL3 = (
  /* wgsl */
  `
struct BU { dims : vec4<u32>, axis_r : vec4<u32>, w : array<vec4<f32>, 4> };
@group(0) @binding(0) var t_in : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> b : BU;
fn wt(i : u32) -> f32 { return b.w[i >> 2u][i & 3u]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = vec3<i32>(gid);
  if (any(c >= vec3<i32>(b.dims.xyz))) { return; }
  let dmax = vec3<i32>(b.dims.xyz) - vec3<i32>(1);
  var av = vec3<i32>(0);
  av[b.axis_r.x] = 1;
  var sum = wt(0u) * textureLoad(t_in, c, 0).r;
  let R = i32(b.axis_r.y);
  for (var i = 1; i <= R; i = i + 1) {
    sum = sum + wt(u32(i)) * (textureLoad(t_in, clamp(c + av * i, vec3<i32>(0), dmax), 0).r
                            + textureLoad(t_in, clamp(c - av * i, vec3<i32>(0), dmax), 0).r);
  }
  textureStore(t_out, c, vec4<f32>(sum, 0.0, 0.0, 0.0));
}`
);
var BU_BYTES = 96;
var COMBINE_WGSL = (
  /* wgsl */
  `${U_STRUCT}
@group(0) @binding(0) var t_dist : texture_3d<f32>;
@group(0) @binding(1) var t_g1 : texture_3d<f32>;
@group(0) @binding(2) var t_g2 : texture_3d<f32>;
@group(0) @binding(3) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<uniform> u : U;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {${GRID_GUARD}
  let d = textureLoad(t_dist, c, 0).r;
  let mode = u32(u.gap.w + 0.5);
  var s = d;
  if (mode == 0u) {
    // Twicing: G smooths away the staircase but moves curved surfaces inwards (\u03C3\xB2/2 \xD7 curvature) and
    // removes thin structures; 2\xB7G\xB7f \u2212 G\xB7G\xB7f cancels that to first order.
    s = 2.0 * textureLoad(t_g1, c, 0).r - textureLoad(t_g2, c, 0).r;
  } else if (mode == 1u) {
    s = textureLoad(t_g1, c, 0).r;
  } else if (mode == 2u) {
    // The blurred inside/outside mask, crossing one half on the surface. Scaled to about a distance in
    // mm (the mask's slope across the edge is about 1/(\u03C3\xB7sqrt(2\u03C0))) so normals and the bisection see a
    // field of the same size as the others; only its sign and direction are used.
    let minSp = min(u.spacing.x, min(u.spacing.y, u.spacing.z));
    s = (0.5 - textureLoad(t_g1, c, 0).r) * max(2.5 * u.sigma.w, 1.25 * minSp);
  }
  if (mode <= 1u && (u.ldims.w & 1u) != 0u && d < 0.0 && s > 0.5 * d) {
    // The deepest voxel of a structure (a local minimum of the distance) stays inside, so a structure
    // painted in one thick slice is not smoothed out of existence.
    let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
    var deepest = true;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
      for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
          if (deepest) {
            let n = clamp(c + vec3<i32>(dx, dy, dz), vec3<i32>(0), dmax);
            deepest = textureLoad(t_dist, n, 0).r >= d - 1.0e-3;
          }
        }
      }
    }
    if (deepest) { s = 0.5 * d; }
  }
  textureStore(t_out, c, vec4<f32>(s, d, 0.0, 0.0));
}`
);
var SEGMENT_WGSL = (
  /* wgsl */
  `${U_STRUCT}
@group(0) @binding(0) var t_label : texture_3d<u32>;
@group(0) @binding(1) var t_pal : texture_2d<f32>;
@group(0) @binding(2) var t_out : texture_storage_3d<r32uint, write>;
@group(0) @binding(3) var<uniform> u : U;
@group(0) @binding(4) var t_seed : texture_3d<u32>;
${LABELS_WGSL}
${SEED_PACK_WGSL}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {${GRID_GUARD}
  var l = shownLabel(c);
  if (l == 0u) {
    var nd = 1.0e30;
    for (var dz = -2; dz <= 2; dz = dz + 1) {
      for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
          let o = vec3<i32>(dx, dy, dz);
          let nl = shownLabel(c + o);
          if (nl != 0u) {
            let dd = length(vec3<f32>(o) * u.spacing.xyz);
            if (dd < nd) { nd = dd; l = nl; }
          }
        }
      }
    }
  }
  if (l == 0u) {
    let sd = textureLoad(t_seed, c, 0).r;
    if (sd != NO_SEED) {
      let lo = vec3<i32>(i32(sd & 1023u), i32((sd >> 10u) & 1023u), i32((sd >> 20u) & 1023u));
      var e = vec3<i32>(0);
      e[min(sd >> 30u, 2u)] = 1;
      l = shownLabel(lo);
      if (l == 0u) { l = shownLabel(lo + e); }
    }
  }
  textureStore(t_out, c, vec4<u32>(l, 0u, 0u, 0u));
}`
);
var INTERFACE_WGSL = (
  /* wgsl */
  `${U_STRUCT}
@group(0) @binding(0) var t_seg : texture_3d<u32>;
@group(0) @binding(1) var t_out : texture_storage_3d<r32uint, write>;
@group(0) @binding(2) var<uniform> u : U;
${SHARE_PACK_WGSL}
fn segAt(c : vec3<i32>) -> i32 {
  return i32(textureLoad(t_seg, clamp(c, vec3<i32>(0), vec3<i32>(u.dims.xyz) - vec3<i32>(1)), 0).r);
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {${GRID_GUARD}
  let own = segAt(c);
  var first = own;
  var near = false;
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let l = segAt(c + vec3<i32>(dx, dy, dz));
        if (l != 0) {
          if (first == 0) { first = l; } else if (l != first) { near = true; }
        }
      }
    }
  }
  if (!near || own == 0) {
    textureStore(t_out, c, vec4<u32>(sharePack(own, 0, 1.0, near), 0u, 0u, 0u));
    return;
  }
  var ownShare = 0.0;
  var ol = array<i32, 4>(0, 0, 0, 0);
  var os = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  let sg = max(min(u.sigma.xyz, vec3<f32>(1.0)), vec3<f32>(1.0e-3));
  for (var dz = -2; dz <= 2; dz = dz + 1) {
    for (var dy = -2; dy <= 2; dy = dy + 1) {
      for (var dx = -2; dx <= 2; dx = dx + 1) {
        let o = vec3<i32>(dx, dy, dz);
        let l = segAt(c + o);
        if (l == 0) { continue; }
        let q = vec3<f32>(o) / sg;
        let w = exp(-0.5 * dot(q, q));
        if (l == own) { ownShare = ownShare + w; continue; }
        for (var k = 0; k < 4; k = k + 1) {
          if (ol[k] == l || ol[k] == 0) { ol[k] = l; os[k] = os[k] + w; break; }
        }
      }
    }
  }
  var other = 0;
  var oshare = -1.0;
  for (var k = 0; k < 4; k = k + 1) {
    if (ol[k] > 0 && os[k] > oshare) { other = ol[k]; oshare = os[k]; }
  }
  let share = ownShare / max(ownShare + max(oshare, 0.0), 1.0e-6);
  textureStore(t_out, c, vec4<u32>(sharePack(own, other, share, true), 0u, 0u, 0u));
}`
);
var THIN_WGSL = (
  /* wgsl */
  `${U_STRUCT}
@group(0) @binding(0) var t_label : texture_3d<u32>;
@group(0) @binding(1) var t_pal : texture_2d<f32>;
@group(0) @binding(2) var t_seg : texture_3d<u32>;
@group(0) @binding(3) var t_in : texture_3d<u32>;
@group(0) @binding(4) var t_out : texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var<uniform> u : U;
${LABELS_WGSL}
${SHARE_PACK_WGSL}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {${GRID_GUARD}
  var v = shareUnpack(textureLoad(t_in, c, 0).r);
  if ((u.ldims.w & 2u) != 0u && v.a > 0.5 && v.b < 0.5) {
    let own = textureLoad(t_seg, c, 0).r;
    if (own != 0u && shownLabel(c) == own) {
      let dmax = vec3<i32>(u.dims.xyz) - vec3<i32>(1);
      var sticks = false;
      for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
          for (var dx = -1; dx <= 1; dx = dx + 1) {
            if (!sticks) {
              let n = clamp(c + vec3<i32>(dx, dy, dz), vec3<i32>(0), dmax);
              sticks = textureLoad(t_seg, n, 0).r == own && shareUnpack(textureLoad(t_in, n, 0).r).b >= v.b + 0.35;
            }
          }
        }
      }
      if (!sticks) { v.b = 0.75; }
    }
  }
  textureStore(t_out, c, v);
}`
);
var BLOCK_WGSL = (
  /* wgsl */
  `${U_STRUCT}
@group(0) @binding(0) var t_if : texture_3d<f32>;
@group(0) @binding(1) var t_out : texture_storage_3d<r32uint, write>;
@group(0) @binding(2) var<uniform> u : U;
const BS = ${LABEL_SURFACE_BLOCK};
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let b = vec3<i32>(gid);
  let dimsI = vec3<i32>(u.dims.xyz);
  if (any(b >= (dimsI + vec3<i32>(BS - 1)) / BS)) { return; }
  let first = b * BS;
  var found = false;
  for (var dz = -1; dz <= BS; dz = dz + 1) {
    for (var dy = -1; dy <= BS; dy = dy + 1) {
      for (var dx = -1; dx <= BS; dx = dx + 1) {
        let v = first + vec3<i32>(dx, dy, dz);
        if (!found && all(v >= vec3<i32>(0)) && all(v < dimsI)) { found = textureLoad(t_if, v, 0).a > 0.5; }
      }
    }
  }
  textureStore(t_out, b, vec4<u32>(select(0u, 1u, found), 0u, 0u, 0u));
}`
);
var LabelSurfaceBaker = class {
  constructor(dev, labelTex, labelDims, labelIjkToRAS, opts = {}, pad = 3) {
    this.labelTex = labelTex;
    this.dev = dev;
    this.opts = { ...DEFAULT_LABEL_SURFACE_BAKE, ...opts };
    this.pad = pad;
    this.labelDims = [labelDims[0], labelDims[1], labelDims[2]];
    this.dims = [labelDims[0] + 2 * pad, labelDims[1] + 2 * pad, labelDims[2] + 2 * pad];
    const m = labelIjkToRAS.slice();
    for (let r = 0; r < 3; r++) m[r * 4 + 3] -= pad * (m[r * 4] + m[r * 4 + 1] + m[r * 4 + 2]);
    this.ijkToRAS = m;
    const col = (c) => Math.hypot(m[c], m[4 + c], m[8 + c]);
    this.spacing = [col(0), col(1), col(2)];
    this.blockDims = this.dims.map((d) => Math.ceil(d / LABEL_SURFACE_BLOCK));
    if (this.dims.some((d) => d > LABEL_SURFACE_MAX_DIM)) {
      throw new Error(`LabelSurfaceBaker: padded grid ${this.dims.join("\xD7")} exceeds ${LABEL_SURFACE_MAX_DIM} voxels on a side`);
    }
    const S = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const mk = (format, label, size = this.dims, extra = 0) => dev.createTexture({ label, size, dimension: "3d", format, usage: S | extra });
    this.mask = mk("r32uint", "lsb mask, then segment");
    this.seed = [mk("r32uint", "lsb seed 0, then raw shares"), mk("r32uint", "lsb seed 1")];
    this.dist = mk("r32float", "lsb distance");
    this.g1 = mk("r32float", "lsb G\xB7f");
    this.g2 = mk("r32float", "lsb indicator, then G\xB7G\xB7f");
    this.scratch = mk("r32float", "lsb blur scratch");
    this.surf = mk("rgba16float", "lsb surface", this.dims, GPUTextureUsage.COPY_SRC);
    this.ifB = mk("rgba16float", "lsb shares", this.dims, GPUTextureUsage.COPY_SRC);
    this.block = mk("r32uint", "lsb wall blocks", this.blockDims);
    this.palTex = dev.createTexture({ label: "lsb palette", size: [256, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const maxDim = Math.max(...this.dims);
    const steps = [];
    for (let s = 1 << Math.floor(Math.log2(maxDim - 1)); s >= 1; s >>= 1) steps.push(s);
    this.steps = [...steps, 2, 1];
    const ub = (size) => dev.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uni = ub(U_BYTES);
    this.jfaUni = this.steps.map(() => ub(U_BYTES));
    this.blurUni = [ub(BU_BYTES), ub(BU_BYTES), ub(BU_BYTES)];
    const pipe = (code, label) => dev.createComputePipeline({ label, layout: "auto", compute: { module: dev.createShaderModule({ label, code }), entryPoint: "main" } });
    this.pipes = {
      mask: pipe(MASK_WGSL, "lsb mask"),
      seed: pipe(SEED_WGSL, "lsb seed"),
      jfa: pipe(JFA_WGSL2, "lsb jfa"),
      final: pipe(FINAL_WGSL3, "lsb final"),
      blur: pipe(BLUR_WGSL3, "lsb blur"),
      combine: pipe(COMBINE_WGSL, "lsb combine"),
      segment: pipe(SEGMENT_WGSL, "lsb segment"),
      interface: pipe(INTERFACE_WGSL, "lsb interface"),
      thin: pipe(THIN_WGSL, "lsb thin"),
      block: pipe(BLOCK_WGSL, "lsb block")
    };
  }
  /** The PADDED grid every texture here lives on, and its voxel → RAS matrix (row-major). */
  dims;
  ijkToRAS;
  pad;
  labelDims;
  spacing;
  dev;
  opts;
  palette = new Uint8Array(256 * 4);
  palTex;
  mask;
  // inside/outside, then (after the distance) the segment of each voxel
  seed;
  // the flood's ping-pong; seed[0] then holds the raw shares
  dist;
  g1;
  g2;
  // the indicator, then G·G·f
  scratch;
  surf;
  ifB;
  block;
  blockDims;
  uni;
  jfaUni;
  blurUni;
  steps;
  pipes;
  /** GPU bytes per padded-grid voxel, counting the r32uint master the EditableSegmentation owns, as
   *  JfaSdfBaker.bytesPerVoxel() does (52), so the two compare directly. */
  static bytesPerVoxel() {
    return 4 + 2 * 4 + 4 * 4 + 8 + 8 + 4;
  }
  options() {
    return { ...this.opts };
  }
  /** Takes effect on the next bake(). */
  setOptions(o) {
    this.opts = { ...this.opts, ...o };
  }
  /** σ in mm (of the largest spacing — of the source image, if resampled — as Slicer's SmoothingFactor is). */
  sigmaMm() {
    return Math.max(0, this.opts.sigmaVoxels) * Math.max(...this.opts.sourceSpacingMm ?? this.spacing);
  }
  /** How far the rendered surface can be from the sharp one — the margin the empty-space skip keeps. */
  skipMarginMm() {
    return Math.max(...this.spacing) + 3.5 * this.sigmaMm();
  }
  /** Palette entry (rgb 0..1, opacity 0..1). Returns true if the label was shown before and not now,
   *  or the other way round — that changes the surface and needs a bake(); a colour or an opacity
   *  change alone needs only flushPalette(). */
  setLabel(id, rgb, opacity) {
    if (id < 1 || id > 255) return false;
    const o = id * 4, wasShown = this.palette[o + 3] > 0;
    if (rgb) for (let i = 0; i < 3; i++) this.palette[o + i] = Math.round(Math.max(0, Math.min(1, rgb[i])) * 255);
    if (opacity !== null) {
      const a = Math.max(0, Math.min(1, opacity));
      this.palette[o + 3] = a > 0 ? Math.max(1, Math.round(a * 255)) : 0;
    }
    return wasShown !== this.palette[o + 3] > 0;
  }
  labelOpacity(id) {
    return this.palette[id * 4 + 3] / 255;
  }
  flushPalette() {
    this.dev.queue.writeTexture({ texture: this.palTex }, this.palette, { bytesPerRow: 256 * 4 }, [256, 1]);
  }
  surfaceTexture() {
    return this.surf;
  }
  sharesTexture() {
    return this.ifB;
  }
  blockTexture() {
    return this.block;
  }
  paletteTexture() {
    return this.palTex;
  }
  writeU(buf, step, mode) {
    const ab = new ArrayBuffer(U_BYTES);
    const f = new Float32Array(ab), u = new Uint32Array(ab);
    f.set(transpose4(this.ijkToRAS), 0);
    u[16] = this.dims[0];
    u[17] = this.dims[1];
    u[18] = this.dims[2];
    u[19] = this.pad;
    f[20] = step;
    u[28] = this.labelDims[0];
    u[29] = this.labelDims[1];
    u[30] = this.labelDims[2];
    u[31] = (this.opts.keepDeepest ? 1 : 0) | (this.opts.thinRule ? 2 : 0);
    const smm = this.sigmaMm();
    const src = this.opts.sourceSpacingMm ?? this.spacing;
    for (let a = 0; a < 3; a++) {
      f[32 + a] = smm / this.spacing[a];
      const gapMm = Math.min(4, Math.floor(smm / src[a] + 1e-6)) * src[a];
      f[36 + a] = this.opts.fillGaps ? Math.min(8, Math.floor(gapMm / this.spacing[a] + 1e-6)) : 0;
      f[40 + a] = this.spacing[a];
    }
    f[35] = smm;
    f[39] = mode;
    f[43] = Math.min(400, 64 * Math.max(...this.spacing));
    this.dev.queue.writeBuffer(buf, 0, ab);
  }
  pass(enc, name, resources, groups) {
    const pipe = this.pipes[name];
    const entries = resources.map((r, i) => r instanceof GPUBuffer ? { binding: i, resource: { buffer: r } } : { binding: i, resource: r.createView() });
    const bg = this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const p = enc.beginComputePass({ label: `lsb ${name}` });
    p.setPipeline(pipe);
    p.setBindGroup(0, bg);
    p.dispatchWorkgroups(Math.ceil(groups[0] / 4), Math.ceil(groups[1] / 4), Math.ceil(groups[2] / 4));
    p.end();
  }
  /** Separable Gaussian at the current σ (per axis), src -> dst (x), dst -> scratch (y), scratch -> dst (z). */
  blur(enc, src, dst) {
    const chain = [[src, dst], [dst, this.scratch], [this.scratch, dst]];
    chain.forEach(([a, b], axis) => this.pass(enc, "blur", [a, b, this.blurUni[axis]], this.dims));
  }
  writeBlurUniforms() {
    const smm = this.sigmaMm();
    for (let axis = 0; axis < 3; axis++) {
      const { radius, w } = gaussHalfKernel2(Math.max(smm / this.spacing[axis], 1e-3));
      const ab = new ArrayBuffer(BU_BYTES);
      const u = new Uint32Array(ab), f = new Float32Array(ab);
      u[0] = this.dims[0];
      u[1] = this.dims[1];
      u[2] = this.dims[2];
      u[4] = axis;
      u[5] = radius;
      f.set(w, 8);
      this.dev.queue.writeBuffer(this.blurUni[axis], 0, ab);
    }
  }
  /** The whole pipeline: mask → face seeds → jump flood → distance → smoothing → surface →
   *  segments → shares → thin rule → wall blocks. Uploads the palette first. Resolves when the GPU
   *  has finished it.
   *
   *  With `onProgress` the four stages go to the queue as four submits, and each one reports as
   *  the GPU FINISHES it — progress of the work itself, not of its encoding, which is instant. The
   *  message names the stage now running; `frac` is the share of the passes already done. Without
   *  a listener it is one submit, as before. */
  bake(onProgress) {
    this.flushPalette();
    const smm = this.sigmaMm();
    const outer = this.opts.outer;
    const smoothed = smm > 0;
    const mode = outer === "mask" ? 2 : !smoothed ? 3 : outer === "blur" ? 1 : 0;
    this.writeU(this.uni, 0, mode);
    this.steps.forEach((s, i) => this.writeU(this.jfaUni[i], s, mode));
    if (smoothed) this.writeBlurUniforms();
    const blurs = !smoothed ? 0 : outer === "twicing" ? 6 : 3;
    const stagePasses = [2, this.steps.length + 1, blurs + 1, 4];
    const total = stagePasses.reduce((a, b) => a + b, 0);
    const STAGES = ["finding segment boundaries", "measuring distances", smoothed ? "smoothing surfaces" : "shaping surfaces", "separating segments"];
    let enc = this.dev.createCommandEncoder({ label: "label surface bake" });
    let stage = 0, passesDone = 0;
    onProgress?.({ msg: STAGES[0], frac: 0 });
    const endStage = () => {
      if (!onProgress) return;
      this.dev.queue.submit([enc.finish()]);
      passesDone += stagePasses[stage++];
      const p = stage < STAGES.length ? { msg: STAGES[stage], frac: passesDone / total } : { msg: "surfaces ready", frac: 1 };
      this.dev.queue.onSubmittedWorkDone().then(() => onProgress(p));
      enc = this.dev.createCommandEncoder({ label: "label surface bake" });
    };
    const D = this.dims;
    this.pass(enc, "mask", [this.labelTex, this.palTex, this.mask, this.uni], D);
    this.pass(enc, "seed", [this.mask, this.seed[0], this.uni], D);
    endStage();
    let src = 0;
    this.steps.forEach((_, i) => {
      this.pass(enc, "jfa", [this.seed[src], this.seed[src ^ 1], this.jfaUni[i]], D);
      src ^= 1;
    });
    this.pass(enc, "final", [this.seed[src], this.mask, this.dist, this.g2, this.uni], D);
    endStage();
    let g1 = this.dist, g2 = this.dist;
    if (outer === "mask") {
      g1 = this.g2;
      if (smoothed) {
        this.blur(enc, this.g2, this.g1);
        g1 = this.g1;
      }
    } else if (smoothed) {
      this.blur(enc, this.dist, this.g1);
      g1 = this.g1;
      if (outer === "twicing") {
        this.blur(enc, this.g1, this.g2);
        g2 = this.g2;
      }
    }
    this.pass(enc, "combine", [this.dist, g1, g2, this.surf, this.uni], D);
    endStage();
    this.pass(enc, "segment", [this.labelTex, this.palTex, this.mask, this.uni, this.seed[src]], D);
    this.pass(enc, "interface", [this.mask, this.seed[0], this.uni], D);
    this.pass(enc, "thin", [this.labelTex, this.palTex, this.mask, this.seed[0], this.ifB, this.uni], D);
    this.pass(enc, "block", [this.ifB, this.block, this.uni], this.blockDims);
    if (onProgress) endStage();
    else this.dev.queue.submit([enc.finish()]);
    return this.dev.queue.onSubmittedWorkDone();
  }
  /** Read the surface texture back: s = the surface function (negative inside), d = the sharp signed
   *  distance, both mm, x-fastest on the padded grid. For tests. */
  async readSurface() {
    const [dx, dy, dz] = this.dims;
    const bpr = Math.ceil(dx * 8 / 256) * 256, rowU16 = bpr / 2;
    const buf = this.dev.createBuffer({ size: bpr * dy * dz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.surf }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: dy }, [dx, dy, dz]);
    this.dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const h = new Uint16Array(buf.getMappedRange());
    const h2f = (x) => {
      const sg = x & 32768 ? -1 : 1, e = (x & 31744) >> 10, fr = x & 1023;
      if (e === 0) return sg * 2 ** -14 * (fr / 1024);
      if (e === 31) return fr ? NaN : sg * Infinity;
      return sg * 2 ** (e - 15) * (1 + fr / 1024);
    };
    const s = new Float32Array(dx * dy * dz), d = new Float32Array(dx * dy * dz);
    for (let z = 0; z < dz; z++) for (let y = 0; y < dy; y++) for (let x = 0; x < dx; x++) {
      const i = (z * dy + y) * dx + x, j = (z * dy + y) * rowU16 + x * 4;
      s[i] = h2f(h[j]);
      d[i] = h2f(h[j + 1]);
    }
    buf.unmap();
    buf.destroy();
    return { s, d };
  }
  destroy() {
    for (const t of [this.mask, ...this.seed, this.dist, this.g1, this.g2, this.scratch, this.surf, this.ifB, this.block, this.palTex]) t.destroy();
    for (const b of [this.uni, ...this.jfaUni, ...this.blurUni]) b.destroy();
  }
};

// render/label-surface-field.ts
var LabelSurfaceField = class {
  constructor(baker, opts = {}) {
    this.baker = baker;
    this.p2t = patientToTextureFromIjkToRAS(baker.ijkToRAS, baker.dims);
    this.box = volumeAABBFromIjkToRAS(baker.ijkToRAS, baker.dims);
    this.minSp = Math.min(...baker.spacing);
    this.maxSp = Math.max(...baker.spacing);
    this.opacity = opts.opacity ?? 1;
    this.shade = opts.shade ?? [0.2, 0.85, 0.3, 32];
    this.walls = opts.walls ?? true;
    this.bothFaces = opts.bothFaces ?? true;
    this.seamPx = opts.seamPx ?? 1;
    this.clippable = opts.clippable ?? false;
  }
  kind = "lsf";
  bindingCount = 4;
  // surface (3d, linear), shares (3d), wall blocks (3d u32), palette (2d)
  clippable;
  providesSkip = true;
  intervalSampling = true;
  p2t;
  box;
  minSp;
  maxSp;
  opacity;
  shade;
  walls;
  bothFaces;
  seamPx;
  setOpacity(o) {
    this.opacity = Math.max(0, Math.min(1, o));
  }
  setWalls(on) {
    this.walls = on;
  }
  setBothFaces(on) {
    this.bothFaces = on;
  }
  setSeamPx(px) {
    this.seamPx = Math.max(0, px);
  }
  uniformFloats() {
    return 40;
  }
  // p2t(16) + dims(4) + opts(4) + shade(4) + skip(4) + bmin(4) + bmax(4)
  aabb() {
    return this.box;
  }
  // Half a voxel, as SegmentField (the renderer steps at 0.7 of this), so the two cost the same per
  // step. Smoothing can leave a one-voxel sheet a fraction of a voxel thick; the shader probes inside
  // a step that could hold one (sample_field), rather than every step being that fine.
  sampleStep() {
    return 0.5 * this.minSp;
  }
  structMembers(s) {
    return [
      `  lsf${s}_p2t : mat4x4<f32>,`,
      `  lsf${s}_dims : vec4<f32>,`,
      // padded grid dims, min spacing (mm)
      `  lsf${s}_opts : vec4<f32>,`,
      // opacity, walls, both faces, seam px
      `  lsf${s}_shade : vec4<f32>,`,
      // ka, kd, ks, shininess
      `  lsf${s}_skip : vec4<f32>,`,
      // margin (mm), max spacing, block size, min spacing
      `  lsf${s}_bmin : vec4<f32>,`,
      `  lsf${s}_bmax : vec4<f32>,`
    ].join("\n");
  }
  declareBindings(s, base) {
    return [
      `@group(0) @binding(${base}) var t_lsd${s} : texture_3d<f32>;`,
      `@group(0) @binding(${base + 1}) var t_lsl${s} : texture_3d<f32>;`,
      `@group(0) @binding(${base + 2}) var t_lsb${s} : texture_3d<u32>;`,
      `@group(0) @binding(${base + 3}) var t_lsp${s} : texture_2d<f32>;`
    ].join("\n");
  }
  skipWGSL(s) {
    return (
      /* wgsl */
      `
fn skip_lsf${s}(wp : vec3<f32>) -> f32 {
  let t = lsf${s}_tex(wp);
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) {
    // Outside the grid: leap only to its box.
    let cen = (u_material.lsf${s}_bmin.xyz + u_material.lsf${s}_bmax.xyz) * 0.5;
    let ext = (u_material.lsf${s}_bmax.xyz - u_material.lsf${s}_bmin.xyz) * 0.5;
    let q = abs(transform_point_lsf${s}(wp) - cen) - ext;
    return max(0.0, length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0));
  }
  // The SHARP distance (.g): smoothing moves the rendered surface by up to the margin from it.
  let sharp = textureSampleLevel(t_lsd${s}, s_lin, t, 0.0).g;
  if (abs(sharp) > 500.0) { return 0.0; }
  var leap = max(0.0, abs(sharp) - u_material.lsf${s}_skip.x);
  if (sharp < 0.0 && u_material.lsf${s}_opts.y > 0.5 && leap > 0.0) {
    // Inside: the next thing may be a wall. A block that holds no voxel near a wall (in it or one voxel
    // around it) holds no wall, so the ray may leap to the edge of the block it is in.
    let gp = t * u_material.lsf${s}_dims.xyz - vec3<f32>(0.5);
    let nv = clamp(vec3<i32>(floor(gp + vec3<f32>(0.5))), vec3<i32>(0), vec3<i32>(u_material.lsf${s}_dims.xyz) - vec3<i32>(1));
    let b = nv / ${LABEL_SURFACE_BLOCK};
    if (textureLoad(t_lsb${s}, b, 0).r != 0u) { return 0.0; }
    let lo = vec3<f32>(b * ${LABEL_SURFACE_BLOCK}) - vec3<f32>(0.5);
    let dv = min(gp - lo, lo + vec3<f32>(${LABEL_SURFACE_BLOCK}.0) - gp);
    leap = min(leap, max(0.0, min(dv.x, min(dv.y, dv.z))) * u_material.lsf${s}_skip.w);
  }${this.clippable ? `
  if (sharp < 0.0) {
    // Inside, a clip plane is a face too (the cap): do not leap past one.
    let ccount = u32(u_material.clip_count.x);
    for (var ci = 0u; ci < ccount; ci = ci + 1u) {
      let cp = u_material.clip_planes[ci];
      leap = min(leap, abs(dot(wp, cp.xyz) + cp.w));
    }
  }` : ""}
  return leap;
}`
    );
  }
  samplingWGSL(s) {
    const P = `lsf${s}`;
    return (
      /* wgsl */
      `
fn ${P}_tex(wp : vec3<f32>) -> vec3<f32> {
  return (u_material.${P}_p2t * vec4<f32>(transform_point_${P}(wp), 1.0)).xyz;
}
fn ${P}_gp(wp : vec3<f32>) -> vec3<f32> { return ${P}_tex(wp) * u_material.${P}_dims.xyz - vec3<f32>(0.5); }
// The surface function (negative inside). Outside the grid: outside the surface.
fn ${P}_surf(wp : vec3<f32>) -> f32 {
  let t = ${P}_tex(wp);
  if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) { return 1.0e3; }
  return textureSampleLevel(t_lsd${s}, s_lin, t, 0.0).r;
}
// At a texture coordinate, clamped to the grid edge \u2014 for gradient taps (the padded edge is background).
fn ${P}_surft(t : vec3<f32>) -> f32 {
  return textureSampleLevel(t_lsd${s}, s_lin, clamp(t, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r;
}
// A gradient taken along the GRID axes (per voxel) to a world direction. Normals are differenced over
// one voxel per axis: on an anisotropic grid (5 mm slices) a tap a fraction of a slice apart reads the
// slope of one trilinear cell, which jumps at every slice plane \u2014 bands across the surface, one per
// slice. A one-voxel span averages that jump out. (On an isotropic grid this is what it always was.)
fn ${P}_gridToWorld(g : vec3<f32>) -> vec3<f32> {
  let m = u_material.${P}_p2t;
  let p3 = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
  return transpose(p3) * (u_material.${P}_dims.xyz * g);
}
fn ${P}_pal(l : i32) -> vec4<f32> { return textureLoad(t_lsp${s}, vec2<i32>(clamp(l, 0, 255), 0), 0); }

// Up to four segments near a point and their shares (label 0 = empty slot).
struct LsfShares${s} { l : array<i32, 4>, w : array<f32, 4> };
fn ${P}_add(sh : ptr<function, LsfShares${s}>, l : i32, a : f32) {
  for (var k = 0; k < 4; k = k + 1) {
    if ((*sh).l[k] == l || (*sh).l[k] == 0) { (*sh).l[k] = l; (*sh).w[k] = (*sh).w[k] + a; return; }
  }
}
// The shares of the segments at a grid position: each of the 8 voxels around it gives its trilinear
// weight to its own segment and, near a wall, part of it to the other segment, in proportion of the
// shares baked there. Labels are never interpolated, so there is always a definite segment at a point.
fn ${P}_shares(gp : vec3<f32>) -> LsfShares${s} {
  var sh : LsfShares${s};
  for (var k = 0; k < 4; k = k + 1) { sh.l[k] = 0; sh.w[k] = 0.0; }
  let dimsI = vec3<i32>(u_material.${P}_dims.xyz);
  let base = floor(gp);
  let f = gp - base;
  var total = 0.0;
  for (var c = 0; c < 8; c = c + 1) {
    let o = vec3<i32>(c & 1, (c >> 1u) & 1, c >> 2u);
    let v = clamp(vec3<i32>(base) + o, vec3<i32>(0), dimsI - vec3<i32>(1));
    let e = textureLoad(t_lsl${s}, v, 0);
    let own = i32(e.r + 0.5);
    if (own == 0) { continue; }
    let aw = mix(vec3<f32>(1.0) - f, f, vec3<f32>(o));
    let w = max(aw.x * aw.y * aw.z, 1.0e-4);
    total = total + w;
    let other = i32(e.g + 0.5);
    if (other == 0) { ${P}_add(&sh, own, w); }
    else { ${P}_add(&sh, own, w * e.b); ${P}_add(&sh, other, w * (1.0 - e.b)); }
  }
  if (total > 0.0) { for (var k = 0; k < 4; k = k + 1) { sh.w[k] = sh.w[k] / total; } }
  return sh;
}
fn ${P}_shareOf(sh : LsfShares${s}, l : i32) -> f32 {
  var r = 0.0;
  for (var k = 0; k < 4; k = k + 1) { if (l > 0 && sh.l[k] == l) { r = sh.w[k]; } }
  return r;
}
// The segment with the largest share, not counting one (0 if none).
fn ${P}_largest(sh : LsfShares${s}, excl : i32) -> i32 {
  var l = 0;
  var w = -1.0;
  for (var k = 0; k < 4; k = k + 1) {
    if (sh.l[k] > 0 && sh.l[k] != excl && sh.w[k] > w) { l = sh.l[k]; w = sh.w[k]; }
  }
  return l;
}
// The segment at a point; in a block with no wall it is that of the nearest voxel (one texel read).
fn ${P}_label(wp : vec3<f32>) -> i32 {
  let gp = ${P}_gp(wp);
  let nv = clamp(vec3<i32>(floor(gp + vec3<f32>(0.5))), vec3<i32>(0), vec3<i32>(u_material.${P}_dims.xyz) - vec3<i32>(1));
  if (textureLoad(t_lsb${s}, nv / ${LABEL_SURFACE_BLOCK}, 0).r == 0u) { return i32(textureLoad(t_lsl${s}, nv, 0).r + 0.5); }
  return ${P}_largest(${P}_shares(gp), 0);
}
fn ${P}_diffG(gp : vec3<f32>, la : i32, lb : i32) -> f32 {
  let sh = ${P}_shares(gp);
  return ${P}_shareOf(sh, la) - ${P}_shareOf(sh, lb);
}
fn ${P}_diff(wp : vec3<f32>, la : i32, lb : i32) -> f32 { return ${P}_diffG(${P}_gp(wp), la, lb); }
fn ${P}_over(acc : vec4<f32>, c : vec4<f32>) -> vec4<f32> { return acc + (1.0 - acc.a) * c; }
// Lit exactly as SegmentField lights its surfaces, so the two renderers differ only in geometry and colour.
fn ${P}_lit(col : vec3<f32>, nIn : vec3<f32>, rd : vec3<f32>, a : f32) -> vec4<f32> {
  var n = nIn;
  if (dot(n, -rd) < 0.0) { n = -n; }
  let ka = u_material.${P}_shade.x; let kd = u_material.${P}_shade.y;
  let ks = u_material.${P}_shade.z; let shn = u_material.${P}_shade.w;
  let ldn = max(dot(-rd, n), 0.0);
  let refl = normalize(2.0 * ldn * n + rd);
  let rdv = max(dot(refl, -rd), 0.0);
  var lit = col * ka + col * (kd * ldn) + vec3<f32>(ks * pow(rdv, max(shn, 1.0)));
  lit = srgb2physical(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(lit * a, a);
}
// The colour of a point of the outer surface: its own segment's, turning to the neighbour's across
// about seamPx pixels at the seam (the share difference changes by about one per voxel), weighted by
// the two opacities so a nearly transparent neighbour hardly tints an opaque segment's edge.
fn ${P}_faceColor(wc : vec3<f32>, l : i32, sh : LsfShares${s}) -> vec3<f32> {
  let pc = ${P}_pal(l);
  var col = pc.rgb;
  let nb = ${P}_largest(sh, l);
  if (nb > 0) {
    let diff = ${P}_shareOf(sh, l) - ${P}_shareOf(sh, nb);
    let seam = u_material.${P}_opts.w;
    var wgt = select(0.0, 1.0, diff >= 0.0);
    if (seam > 0.0) {
      let pxVox = distance(u_cam.eye.xyz, wc) / max(u_cam.size.z, 1.0e-6) / max(u_material.${P}_dims.w, 1.0e-6);
      wgt = clamp(0.5 + 0.5 * diff / max(pxVox * seam, 1.0e-3), 0.0, 1.0);
    }
    let nc = ${P}_pal(nb);
    let nw = (1.0 - wgt) * nc.a;
    col = mix(col, nc.rgb, nw / max(wgt * pc.a + nw, 1.0e-6));
  }
  return col;
}
// A face of the outer surface at wc, of segment l \u2014 or, given a normal (a clip cap), a face with that normal.
fn ${P}_surfaceFace(wc : vec3<f32>, rd : vec3<f32>, l : i32, nGiven : vec3<f32>) -> vec4<f32> {
  if (l <= 0) { return vec4<f32>(0.0); }
  let a = ${P}_pal(l).a * u_material.${P}_opts.x;
  if (a <= 0.0) { return vec4<f32>(0.0); }
  var n = nGiven;
  if (dot(n, n) < 0.25) {
    // \xB1 half a voxel along each grid axis (one voxel in all; not more: a structure may be one voxel thick).
    let t = ${P}_tex(wc);
    let hd = vec3<f32>(0.5) / u_material.${P}_dims.xyz;
    let gi = vec3<f32>(
      ${P}_surft(t + vec3<f32>(hd.x, 0.0, 0.0)) - ${P}_surft(t - vec3<f32>(hd.x, 0.0, 0.0)),
      ${P}_surft(t + vec3<f32>(0.0, hd.y, 0.0)) - ${P}_surft(t - vec3<f32>(0.0, hd.y, 0.0)),
      ${P}_surft(t + vec3<f32>(0.0, 0.0, hd.z)) - ${P}_surft(t - vec3<f32>(0.0, 0.0, hd.z)));
    let g = ${P}_gridToWorld(gi);
    n = select(-rd, normalize(g), length(g) > 1.0e-6);
  }
  return ${P}_lit(${P}_faceColor(wc, l, ${P}_shares(${P}_gp(wc))), n, rd, a);
}
// The wall where the ray passes from segment la (at pa) into lb (at pb): where their shares are
// equal. Two faces: la's (the segment the ray leaves), then lb's.
fn ${P}_wall(pa : vec3<f32>, pb : vec3<f32>, la : i32, lb : i32, rd : vec3<f32>) -> vec4<f32> {
  let ca = ${P}_pal(la);
  let cb = ${P}_pal(lb);
  let aA = ca.a * u_material.${P}_opts.x;
  let aB = cb.a * u_material.${P}_opts.x;
  let omitA = u_material.${P}_opts.z < 0.5 && aA < 0.999 && aB >= 0.999;
  if ((aA <= 0.0 || omitA) && aB <= 0.0) { return vec4<f32>(0.0); }
  // Bisect on the share difference (positive at pa, negative at pb), then interpolate the last bracket.
  var a = 0.0;
  var b = 1.0;
  var da = 1.0e-4;
  var db = -1.0e-4;
  for (var i = 0; i < 8; i = i + 1) {
    // i = 0, 1: the two ends; then the bisection.
    let m = select(0.5 * (a + b), f32(i), i < 2);
    let dm = ${P}_diff(mix(pa, pb, m), la, lb);
    if (i == 0) { da = max(dm, 1.0e-4); } else if (i == 1) { db = min(dm, -1.0e-4); }
    else if (dm > 0.0) { a = m; da = dm; } else { b = m; db = dm; }
  }
  let pw = mix(pa, pb, a + (b - a) * clamp(da / max(da - db, 1.0e-6), 0.0, 1.0));
  // The normal: the gradient of the share difference, from four points at the vertices of a tetrahedron
  // 0.7 voxel out along each grid axis.
  let gpw = ${P}_gp(pw);
  var tet = array<vec3<f32>, 4>(vec3<f32>(1.0, -1.0, -1.0), vec3<f32>(-1.0, -1.0, 1.0), vec3<f32>(-1.0, 1.0, -1.0), vec3<f32>(1.0, 1.0, 1.0));
  var gg = vec3<f32>(0.0);
  for (var v = 0; v < 4; v = v + 1) { gg = gg + tet[v] * ${P}_diffG(gpw + 0.7 * tet[v], lb, la); }
  let g = ${P}_gridToWorld(gg);
  let n = select(-rd, normalize(g), length(g) > 1.0e-6);
  var acc = vec4<f32>(0.0);
  if (aA > 0.0 && !omitA) { acc = ${P}_over(acc, ${P}_lit(ca.rgb, n, rd, aA)); }
  if (aB > 0.0) { acc = ${P}_over(acc, ${P}_lit(cb.rgb, n, rd, aB)); }
  return acc;
}
// The outer surface between pa and pb, which are on opposite sides of it: bisect, then interpolate
// the last bracket.
fn ${P}_cross(pa : vec3<f32>, pb : vec3<f32>, sa0 : f32, sb0 : f32) -> vec3<f32> {
  let inA = sa0 < 0.0;
  var a = 0.0;
  var b = 1.0;
  var sa = sa0;
  var sb = sb0;
  for (var i = 0; i < 6; i = i + 1) {
    let m = 0.5 * (a + b);
    let sm = ${P}_surf(mix(pa, pb, m));
    if ((sm < 0.0) == inA) { a = m; sa = sm; } else { b = m; sb = sm; }
  }
  let den = sa - sb;
  let fr = select(0.5, clamp(sa / den, 0.0, 1.0), abs(den) > 1.0e-9);
  return mix(pa, pb, a + (b - a) * fr);
}
// What a sample's stretch of ray crosses, in ray order: outer faces (kind 1: at p0, of segment l0),
// walls (kind 2: from p0 in l0 to p1 in l1) and clip caps (kind 3: at p0, of l0, normal p1). Collected first and shaded in ONE loop, so the
// shading code is in the shader once \u2014 inlined at several call sites it was large enough to slow
// every step of the march, not just the ones that hit something.
struct LsfEvents${s} { n : i32, kind : array<i32, 4>, p0 : array<vec3<f32>, 4>, p1 : array<vec3<f32>, 4>, l0 : array<i32, 4>, l1 : array<i32, 4> };
fn ${P}_push(ev : ptr<function, LsfEvents${s}>, kind : i32, p0 : vec3<f32>, p1 : vec3<f32>, l0 : i32, l1 : i32) {
  let i = (*ev).n;
  if (i < 4) { (*ev).kind[i] = kind; (*ev).p0[i] = p0; (*ev).p1[i] = p1; (*ev).l0[i] = l0; (*ev).l1[i] = l1; (*ev).n = i + 1; }
}
// The events on the stretch pa -> pb, which crosses the outer surface at most once.
fn ${P}_events(pa : vec3<f32>, pb : vec3<f32>, sa : f32, sb : f32, ev : ptr<function, LsfEvents${s}>) {
  let inA = sa < 0.0;
  let inB = sb < 0.0;
  if (!inA && !inB) { return; }
  let walls = u_material.${P}_opts.y > 0.5;
  var la = 0;
  var lb = 0;
  if (walls && inA) { la = ${P}_label(pa); }
  if (walls && inB) { lb = ${P}_label(pb); }
  if (inA != inB) {
    let wc = ${P}_cross(pa, pb, sa, sb);
    let lc = ${P}_label(wc);
    if (inB) {
      // Into the surface, then perhaps through a wall before pb.
      ${P}_push(ev, 1, wc, wc, lc, 0);
      if (walls && lc > 0 && lb > 0 && lb != lc) { ${P}_push(ev, 2, wc, pb, lc, lb); }
    } else if (walls && la > 0 && lc > 0 && la != lc) {
      // Through a wall, then out of the segment the ray is in.
      ${P}_push(ev, 2, pa, wc, la, lc);
      ${P}_push(ev, 1, wc, wc, lc, 0);
    } else {
      ${P}_push(ev, 1, wc, wc, select(lc, la, la > 0), 0);
    }
    return;
  }
  // Inside at both ends: a wall if the segment changed \u2014 or two, if a segment thinner than the step
  // lies between (the middle is in another segment).
  if (!walls || la == 0 || lb == 0) { return; }
  if (la != lb) { ${P}_push(ev, 2, pa, pb, la, lb); return; }
  let pm = 0.5 * (pa + pb);
  let lm = ${P}_label(pm);
  if (lm != 0 && lm != la) { ${P}_push(ev, 2, pa, pm, la, lm); ${P}_push(ev, 2, pm, pb, lm, la); }
}
fn sample_field_${P}(wp : vec3<f32>, rd : vec3<f32>, ds : f32) -> vec4<f32> {
  if (u_material.${P}_opts.x <= 0.0) { return vec4<f32>(0.0); }
  var L = max(ds, 1.0e-4);
  var back = wp - rd * L;
  var cap = vec3<f32>(0.0);${this.clippable ? `
  // The stretch may start on the cut-away side of a clip plane (wp is on the kept side, or this would
  // not be called): start it at the plane, and if the plane is inside a segment, that is a face (cap).
  let ccount = u32(u_material.clip_count.x);
  var fc = -1.0;
  for (var ci = 0u; ci < ccount; ci = ci + 1u) {
    let cp = u_material.clip_planes[ci];
    let db = dot(back, cp.xyz) + cp.w;
    if (db < 0.0) {
      let f = -db / max(dot(wp, cp.xyz) + cp.w - db, 1.0e-9);
      if (f > fc) { fc = f; cap = -cp.xyz; }
    }
  }
  if (fc >= 0.0) { let f = min(fc, 1.0); back = mix(back, wp, f); L = max(L * (1.0 - f), 1.0e-4); }` : ""}
  let s1 = ${P}_surf(wp);
  let s0 = ${P}_surf(back);
  var pts = array<vec3<f32>, 3>(back, wp, wp);
  var ss = array<f32, 3>(s0, s1, s1);
  var stretches = 1;
  if (s0 >= 0.0 && s1 >= 0.0) {
    // Outside at both ends. A structure thinner than the step can still lie between, but only if the
    // two distances to the surface add up to less than the stretch; then look at three points inside it.
    stretches = 0;
    if (s0 + s1 < 1.5 * L) {
      for (var k = 1; k <= 3; k = k + 1) {
        let pk = back + rd * (L * f32(k) * 0.25);
        let sk = ${P}_surf(pk);
        if (sk < 0.0) { pts[1] = pk; ss[1] = sk; stretches = 2; break; }
      }
    }
    if (stretches == 0) { return vec4<f32>(0.0); }
  }
  var ev : LsfEvents${s};
  ev.n = 0;${this.clippable ? `
  if (dot(cap, cap) > 0.25 && s0 < 0.0) { ${P}_push(&ev, 3, back, cap, ${P}_label(back), 0); }` : ""}
  for (var j = 0; j < stretches; j = j + 1) { ${P}_events(pts[j], pts[j + 1], ss[j], ss[j + 1], &ev); }
  var acc = vec4<f32>(0.0);
  for (var i = 0; i < ev.n; i = i + 1) {
    var c = vec4<f32>(0.0);
    if (ev.kind[i] == 2) { c = ${P}_wall(ev.p0[i], ev.p1[i], ev.l0[i], ev.l1[i], rd); }
    else { c = ${P}_surfaceFace(ev.p0[i], rd, ev.l0[i], select(vec3<f32>(0.0), ev.p1[i], ev.kind[i] == 3)); }
    acc = ${P}_over(acc, c);
  }
  return acc;
}`
    );
  }
  fillUniforms(out, off) {
    out.set(this.p2t, off);
    const d = this.baker.dims;
    out[off + 16] = d[0];
    out[off + 17] = d[1];
    out[off + 18] = d[2];
    out[off + 19] = this.minSp;
    out[off + 20] = this.opacity;
    out[off + 21] = this.walls ? 1 : 0;
    out[off + 22] = this.bothFaces ? 1 : 0;
    out[off + 23] = this.seamPx;
    out[off + 24] = this.shade[0];
    out[off + 25] = this.shade[1];
    out[off + 26] = this.shade[2];
    out[off + 27] = this.shade[3];
    out[off + 28] = this.baker.skipMarginMm();
    out[off + 29] = this.maxSp;
    out[off + 30] = LABEL_SURFACE_BLOCK;
    out[off + 31] = this.minSp;
    out[off + 32] = this.box[0][0];
    out[off + 33] = this.box[0][1];
    out[off + 34] = this.box[0][2];
    out[off + 36] = this.box[1][0];
    out[off + 37] = this.box[1][1];
    out[off + 38] = this.box[1][2];
  }
  bindEntries(_s, base) {
    return [
      { binding: base, resource: this.baker.surfaceTexture().createView() },
      { binding: base + 1, resource: this.baker.sharesTexture().createView() },
      { binding: base + 2, resource: this.baker.blockTexture().createView() },
      { binding: base + 3, resource: this.baker.paletteTexture().createView() }
    ];
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
var SegmentationLogic = class _SegmentationLogic {
  constructor(device, seg, opts = {}) {
    this.seg = seg;
    this.device = device;
    let mode = opts.renderMode ?? "label";
    if (mode === "label" && seg.dims.some((d) => d + 6 > LABEL_SURFACE_MAX_DIM)) {
      console.warn(`SegmentationLogic: ${seg.dims.join("\xD7")} is too large for the label renderer; using sdf`);
      mode = "sdf";
    }
    this.renderMode = mode;
    this.sigma = opts.sigmaVoxels ?? 1;
    this.bandMm = opts.bandMm;
    this.opacity = opts.opacity ?? 1;
    this.refineDelayMs = opts.refineDelayMs ?? 180;
    this.boundaryMode = opts.boundaryMode ?? "outer";
    this.clippable = opts.clippable ?? false;
    this.regionReflood = opts.regionReflood ?? false;
    this.lsRender = { ...this.lsRender, ...opts.labelRender };
    this.onProgress = opts.onProgress;
    if (this.renderMode === "label") {
      this.lsb = new LabelSurfaceBaker(
        device,
        seg.masterTexture(),
        seg.dims,
        seg.ijkToRAS,
        { ...opts.labelBake, sigmaVoxels: this.sigma, sourceSpacingMm: opts.sourceSpacingMm }
      );
      this.setLabelColor(1, opts.color ?? [0.3, 0.85, 0.55]);
      this.labelsDirty = true;
      this.scheduleLabelBake();
      this.unsubDirty = seg.onDirty(() => {
        this.labelsDirty = true;
        this.scheduleLabelBake();
      });
      return;
    }
    this.setLabelColor(1, opts.color ?? [0.3, 0.85, 0.55]);
    if (this.renderMode === "sdf") {
      this.sdf = new JfaSdfBaker(device, seg.masterTexture(), seg.dims, seg.ijkToRAS, this.sigma, 2, this.boundaryMode);
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
  lsb;
  // label path
  lsf;
  lsRender = { walls: true, bothFaces: true, seamPx: 1 };
  visibilityDirty = false;
  // label: a segment was shown or hidden since the last bake
  labelsDirty = false;
  // label: the labelmap was edited since the last bake
  bakePending = false;
  // label: a bake is queued for the end of this task
  /** label mode: GPU time of the last bake, ms (to queue completion; 0 until the first one lands). */
  lastBakeMs = 0;
  /** label mode: progress of each bake as the GPU works through it (undefined = no reports). */
  onProgress;
  bakeDone = Promise.resolve();
  // GPU completion of the latest bake
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
  device;
  redrawCbs = [];
  unsubDirty;
  refineTimer;
  refineDelayMs;
  // quiescence before the settle-refine (sdf mode; capability-tuned)
  boundaryMode;
  regionReflood;
  notify() {
    for (const cb of this.redrawCbs) cb();
  }
  /** label: bake at the end of this task (coalesces a burst of edits into one bake). */
  scheduleLabelBake() {
    if (this.bakePending) return;
    this.bakePending = true;
    queueMicrotask(() => {
      this.bakePending = false;
      if (this.labelsDirty) {
        this.bakeLabels();
        this.notify();
      }
    });
  }
  /** label: one full bake (palette included), timed to queue completion. */
  bakeLabels() {
    const t0 = performance.now();
    this.bakeDone = this.lsb.bake(this.onProgress);
    this.visibilityDirty = false;
    this.labelsDirty = false;
    this.bakeDone.then(() => {
      this.lastBakeMs = performance.now() - t0;
    });
  }
  /** Resolves when the GPU has finished the latest bake — including one still queued for the end
   *  of this task — i.e. when field() shows the current labelmap. For a loader that wants to say
   *  "ready" when the surfaces are actually there, not when their work was merely submitted. */
  async ready() {
    await Promise.resolve();
    if (!this.lsb) return this.device.queue.onSubmittedWorkDone();
    return this.bakeDone;
  }
  /** Assign a display colour to a label id (0..255). Keeps the current opacity (defaults to 1 =
   *  opaque). Takes effect on the next rebake (label mode: refineNow(), a palette write). */
  setLabelColor(id, rgb) {
    if (id < 1 || id > 255) return;
    if (this.lsb) {
      const first = this.lsb.labelOpacity(id) === 0;
      if (this.lsb.setLabel(id, rgb, first ? 1 : null)) this.visibilityDirty = true;
      return;
    }
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
    if (this.lsb) {
      if (this.lsb.setLabel(id, null, opacity)) this.visibilityDirty = true;
      return;
    }
    this.palette[id * 4 + 3] = Math.max(0, Math.min(1, opacity));
  }
  /** Per-segment shading (sdf mode): "surface" = crisp SDF shell (surface model), "volume" = DVR fill
   *  of the interior (translucent cloud). Rebake/refine to apply. The label renderer draws every
   *  segment as a surface and ignores this. */
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
    if (this.lsb) {
      if (this.visibilityDirty || this.labelsDirty) this.bakeLabels();
      else this.lsb.flushPalette();
      this.notify();
      return;
    }
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
  /** RAS bbox → padded-SDF-grid ijk bbox with an M-voxel HALO.
   *
   *  The halo is not just room for the shell band and the blur radii. A region re-flood reuses the
   *  previous flood's seeds outside the box, and a seed OUTSIDE the box that pointed at a surface
   *  just DELETED inside it is stale — a voxel in the box that taps it inherits a phantom surface
   *  that is no longer there. Re-initialising over a wide halo rebuilds those seeds, so the halo has
   *  to be at least as wide as the spacing to the next real structure (~15-20 mm between vertebrae);
   *  at 8 voxels it was not, and one arrow step left a sliver of the hidden level on screen. The box
   *  is still a few per cent of the grid, so this costs essentially nothing (measured: a level step
   *  stays ~14 ms against ~390 ms for a full rebake). */
  static HALO = 32;
  // voxels of re-initialised ring around a region patch
  /** Largest JFA step a region patch may use: capped so a tap (reach = 2·step - 1) stays inside the
   *  halo this patch re-seeded, and can never read a seed left over from an older bake. */
  static haloMaxStep() {
    return 1 << Math.floor(Math.log2((_SegmentationLogic.HALO + 1) / 2));
  }
  regionToIjk(regionRAS, M = _SegmentationLogic.HALO) {
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
    if (this.lsb) {
      this.bakeLabels();
      this.notify();
      return;
    }
    if (!this.sdf) {
      this.rebake();
      return;
    }
    const region = this.regionToIjk(regionRAS);
    const core = this.regionToIjk(regionRAS, 0);
    const reach = 2 * _SegmentationLogic.haloMaxStep();
    if (!this.regionReflood || [0, 1, 2].some((d) => core.hi[d] - core.lo[d] > reach)) {
      this.refineNow();
      return;
    }
    if (this.refineTimer !== void 0) {
      clearTimeout(this.refineTimer);
      this.refineTimer = void 0;
    }
    this.sdf.setPalette(this.palette);
    this.sdf.setModePalette(this.modePalette);
    this.sdf.refineRegion(region, _SegmentationLogic.haloMaxStep());
    for (const cb of this.redrawCbs) cb();
  }
  refreshOpacity(regionRAS) {
    if (this.lsb) {
      this.refineNow();
      return;
    }
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
    this.sdf.rebakeAttr(true);
    for (const cb of this.redrawCbs) cb();
  }
  /** The field to hand to the SceneRenderer once; edits update its textures in place. label mode: a
   *  LabelSurfaceField; otherwise a SegmentField whose colour comes from the texture (per-label) and
   *  whose uniform supplies the global opacity. */
  field() {
    if (this.lsb) {
      if (!this.lsf) this.lsf = new LabelSurfaceField(this.lsb, { ...this.lsRender, opacity: this.opacity, clippable: this.clippable });
      return this.lsf;
    }
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
      this.segField.setVoxelMode(this.renderMode === "sdf" && this.sigma <= 0);
      this.segField.setSmoothingVox(this.sigma);
    }
    return this.segField;
  }
  /** Live GLOBAL segmentation opacity (0..1) — the field-level multiplier over every segment's own
   *  opacity. Caller does scene.syncUniforms() + redraw. */
  setGlobalOpacity(o) {
    this.opacity = o;
    this.segField?.setOpacity(o);
    this.lsf?.setOpacity(o);
  }
  /** label mode: the bake options (σ, outer surface, deepest voxel, gap fill, thin rule). */
  bakeOptions() {
    return this.lsb?.options();
  }
  /** label mode: change bake options and re-bake. */
  setBakeOptions(o) {
    if (!this.lsb) return;
    if (o.sigmaVoxels !== void 0) this.sigma = Math.max(0, o.sigmaVoxels);
    this.lsb.setOptions(o);
    this.bakeLabels();
    this.notify();
  }
  /** label mode: walls / both faces / seam width (uniforms; the caller syncs the scene on redraw). */
  renderOptions() {
    return { ...this.lsRender };
  }
  setRenderOptions(o) {
    this.lsRender = { ...this.lsRender, ...o };
    if (this.lsf) {
      this.lsf.setWalls(this.lsRender.walls);
      this.lsf.setBothFaces(this.lsRender.bothFaces);
      this.lsf.setSeamPx(this.lsRender.seamPx);
    }
    this.notify();
  }
  /** GPU bytes the render path holds, counting the master labelmap. */
  byteSize() {
    if (this.lsb) {
      const [x2, y2, z2] = this.lsb.dims;
      return x2 * y2 * z2 * LabelSurfaceBaker.bytesPerVoxel();
    }
    if (this.sdf) return this.sdf.byteSize();
    const [x, y, z] = this.seg.dims;
    return x * y * z * (8 + 4);
  }
  /** Surface smoothing, σ in VOXELS (0 = none) — the RECONSTRUCTION WIDTH of the band-limited label
   *  field. In sdf/outer mode the rendered surface is the 0.5 level set of p = G_σ * 1(label != 0):
   *  σ ≈ 0.5 is the grid's Nyquist limit, the most faithful rendition of the labels that is not
   *  aliased; below that it approaches the trilinear (voxel-faceted) reconstruction and at 0 it
   *  switches to literal cubes; above it the surface is deliberately modelled, trading ≈ σ²/2R of
   *  inward motion on convex features for smoothness. In interface ("all") mode σ still only blurs
   *  the shading distance. In surface (Gaussian presence) mode it is the presence-field Gaussian.
   *
   *  Re-bakes immediately (fast path, so a slider stays responsive) and schedules the settle
   *  refine, exactly as an edit does. */
  setSmoothing(sigmaVoxels) {
    if (this.lsb) {
      this.setBakeOptions({ sigmaVoxels });
      return;
    }
    this.sigma = Math.max(0, sigmaVoxels);
    this.sdf?.setSmoothing(this.sigma);
    this.segField?.setSmoothingVox(this.sigma);
    this.segField?.setVoxelMode(this.renderMode === "sdf" && this.sigma <= 0);
    if (this.sdf) this.sdf.resmooth();
    else this.baker.bakeInto(this.presenceTex, this.palette, this.sigma);
    for (const cb of this.redrawCbs) cb();
  }
  smoothing() {
    return this.sigma;
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
    this.lsb?.destroy();
  }
};

// algorithms/demos/algorithms-scene.ts
var LABEL_COLORS = [
  [0.3, 0.85, 0.55],
  [0.35, 0.65, 0.95],
  [0.95, 0.6, 0.3],
  [0.9, 0.35, 0.45],
  [0.7, 0.45, 0.95],
  [0.35, 0.85, 0.9],
  [0.95, 0.85, 0.35],
  [0.95, 0.5, 0.8],
  [0.55, 0.8, 0.35],
  [0.5, 0.55, 0.9],
  [0.9, 0.7, 0.5],
  [0.8, 0.8, 0.85]
];
function voxelLine(a, b) {
  const n = Math.max(1, Math.ceil(Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]), Math.abs(b[2] - a[2]))));
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const v = [Math.round(a[0] + (b[0] - a[0]) * t), Math.round(a[1] + (b[1] - a[1]) * t), Math.round(a[2] + (b[2] - a[2]) * t)];
    const last = out[out.length - 1];
    if (!last || last[0] !== v[0] || last[1] !== v[1] || last[2] !== v[2]) out.push(v);
  }
  return out;
}
function buildAlgorithmsScene(gpu, format, opts = {}) {
  const dims = [96, 96, 96];
  const sp = 2;
  const ijkToRAS = [sp, 0, 0, -96, 0, sp, 0, -96, 0, 0, sp, -96, 0, 0, 0, 1];
  const seg = new EditableSegmentation(gpu.device, dims, { ijkToRAS });
  const paint = new PaintEffect(seg);
  const keyToId = /* @__PURE__ */ new Map();
  const labelColors = [];
  const labelLook = /* @__PURE__ */ new Map();
  let nextId = 1;
  const applyLook = (id) => {
    const lk = labelLook.get(id);
    logic.setLabelOpacity(id, lk.op);
    logic.setLabelShading(id, lk.shading);
    surface?.setLabelOpacity(id, lk.op);
  };
  const allocId = (key) => {
    let id = keyToId.get(key);
    if (id !== void 0) return id;
    id = nextId++;
    const rgb = LABEL_COLORS[(id - 1) % LABEL_COLORS.length];
    keyToId.set(key, id);
    labelColors.push([id, rgb]);
    labelLook.set(id, { op: 1, shading: "surface" });
    logic.setLabelColor(id, rgb);
    surface?.setLabelColor(id, rgb);
    applyLook(id);
    return id;
  };
  const scene = new SceneRenderer(gpu, format);
  let sigma = 1;
  let boundary = "outer";
  let surface;
  let sceneNew;
  const redrawCbs = [];
  let mode = opts.compare ? "sdf" : "label";
  let logic;
  let allOpacity = 1;
  const makeLogic = () => {
    logic = new SegmentationLogic(gpu.device, seg, { renderMode: mode, opacity: 1, sigmaVoxels: sigma, boundaryMode: boundary, refineDelayMs: opts.refineDelayMs });
    for (const [id, rgb] of labelColors) {
      logic.setLabelColor(id, rgb);
      applyLook(id);
    }
    logic.onRedraw(() => {
      scene.syncUniforms();
      for (const cb of redrawCbs) cb();
    });
    scene.build([logic.field()]);
    scene.setBackground(0.05, 0.06, 0.09);
  };
  makeLogic();
  if (opts.compare) {
    surface = new SegmentationLogic(gpu.device, seg, { renderMode: "label", opacity: 1, sigmaVoxels: sigma });
    sceneNew = new SceneRenderer(gpu, format);
    sceneNew.build([surface.field()]);
    sceneNew.setBackground(0.05, 0.06, 0.09);
    const sn = sceneNew;
    surface.onRedraw(() => {
      sn.syncUniforms();
      for (const cb of redrawCbs) cb();
    });
  }
  const driver = new SegEditDriver(seg, { labelForSegment: (segId) => allocId(segId) });
  allocId("seed");
  const [nx, ny, nz] = dims;
  const lab = new Uint8Array(nx * ny * nz);
  const c = [48, 48, 48], rv = 15;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const dx = x - c[0], dy = y - c[1], dz = z - c[2];
    if (dx * dx + dy * dy + dz * dz <= rv * rv) lab[(z * ny + y) * nx + x] = 1;
  }
  seg.loadLabelmap(lab);
  const center = [0, 0, 0];
  const radius = Math.hypot(96, 96, 96);
  let pokeN = 0;
  let voxN = 0;
  const toRAS = (v) => [
    ijkToRAS[0] * v[0] + ijkToRAS[1] * v[1] + ijkToRAS[2] * v[2] + ijkToRAS[3],
    ijkToRAS[4] * v[0] + ijkToRAS[5] * v[1] + ijkToRAS[6] * v[2] + ijkToRAS[7],
    ijkToRAS[8] * v[0] + ijkToRAS[9] * v[1] + ijkToRAS[10] * v[2] + ijkToRAS[11]
  ];
  const paintVoxels = (ijk, key) => {
    const id = allocId(key ?? `voxels_${voxN++}`);
    if (ijk.length) paint.stampStroke(ijk.map(toRAS), { radiusMm: 0.2 * sp, id, mode: "add" });
    return id;
  };
  const writeBox = (id, lo, size, keep = () => true) => {
    const data = new Uint32Array(size[0] * size[1] * size[2]);
    for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) for (let x = 0; x < size[0]; x++) {
      if (keep(x, y, z)) data[(z * size[1] + y) * size[0] + x] = id;
    }
    seg.writeLabelRegion(data, lo, size);
  };
  return {
    scene,
    seg,
    paint,
    driver,
    center,
    radius,
    dims,
    sceneNew,
    surface,
    paintVoxels,
    paintThinSet() {
      writeBox(allocId(`thin_sheet_${voxN}`), [8, 8, 20], [20, 20, 1]);
      writeBox(allocId(`thin_rod_${voxN}`), [84, 50, 30], [2, 2, 30]);
      writeBox(allocId(`thin_gap1_${voxN}`), [10, 70, 70], [14, 14, 7], (_x, _y, z) => z !== 3);
      writeBox(allocId(`thin_gap2_${voxN}`), [30, 70, 70], [14, 14, 8], (_x, _y, z) => z !== 3 && z !== 4);
      writeBox(allocId(`thin_inner_sheet_${voxN}`), [48, 38, 38], [1, 21, 21]);
      seg.markDirty();
      paintVoxels(voxelLine([70, 10, 10], [88, 28, 22]), `thin_line_${voxN}`);
      paintVoxels([[80, 80, 20]], `thin_dot_${voxN}`);
      paintVoxels(voxelLine([36, 44, 40], [60, 52, 56]), `thin_inner_line_${voxN}`);
      voxN++;
    },
    poke(centerRAS, radiusMm) {
      paint.stampStroke([centerRAS], { radiusMm, id: allocId(`poke_${pokeN++}`), mode: "add" });
    },
    onRedraw(cb) {
      redrawCbs.push(cb);
    },
    setRenderMode(m) {
      if (m === mode) return;
      logic.destroy();
      mode = m;
      makeLogic();
      for (const cb of redrawCbs) cb();
    },
    renderMode: () => mode,
    refine() {
      logic.refineNow();
    },
    setAllOpacity(op) {
      allOpacity = op;
      for (const [id] of labelColors) {
        labelLook.get(id).op = op;
        applyLook(id);
      }
      logic.refineNow();
      surface?.refineNow();
    },
    allOpacity: () => allOpacity,
    randomizeLook(o) {
      const shading = o?.shading ?? true;
      for (const [id] of labelColors) {
        const lk = { op: 0.3 + Math.random() * 0.7, shading: shading && Math.random() < 0.5 ? "volume" : "surface" };
        labelLook.set(id, lk);
        applyLook(id);
      }
      logic.refineNow();
      surface?.refineNow();
    },
    resetLook() {
      allOpacity = 1;
      for (const [id] of labelColors) {
        labelLook.set(id, { op: 1, shading: "surface" });
        applyLook(id);
      }
      logic.refineNow();
      surface?.refineNow();
    },
    setSmoothing(s) {
      sigma = Math.max(0, s);
      logic.setSmoothing(sigma);
      surface?.setSmoothing(sigma);
    },
    smoothing: () => sigma,
    setBoundaryMode(m) {
      if (m === boundary) return;
      boundary = m;
      if (mode !== "sdf") return;
      logic.destroy();
      makeLogic();
      for (const cb of redrawCbs) cb();
    },
    boundaryMode: () => boundary
  };
}

// logic/seg-budget.ts
var SegBudget = class _SegBudget {
  constructor(tier, refineMsAt64) {
    this.tier = tier;
    this.refineMsAt64 = refineMsAt64;
  }
  /** Measure the device by timing an SDF refine at `probeDim`³ (default 64), then classify. Cheap:
   *  one warm bake + a few refines behind a single GPU sync. */
  static async probe(device, probeDim = 64) {
    const D = probeDim;
    const dims = [D, D, D];
    const s = 2;
    const ijkToRAS = [s, 0, 0, -D, 0, s, 0, -D, 0, 0, s, -D, 0, 0, 0, 1];
    const seg = new EditableSegmentation(device, dims, { ijkToRAS });
    const lab = new Uint8Array(D * D * D);
    const c = D / 2, r = D * 0.35;
    for (let z = 0; z < D; z++) for (let y = 0; y < D; y++) for (let x = 0; x < D; x++) {
      const dx = x - c, dy = y - c, dz = z - c;
      if (dx * dx + dy * dy + dz * dz <= r * r) lab[(z * D + y) * D + x] = 1;
    }
    const baker = new JfaSdfBaker(device, seg.masterTexture(), dims, ijkToRAS);
    const pal = new Float32Array(256 * 4);
    pal[4] = 1;
    pal[5] = 1;
    pal[6] = 1;
    pal[7] = 1;
    baker.setPalette(pal);
    const mode = new Float32Array(256 * 4);
    baker.setModePalette(mode);
    let refineMs = 8;
    try {
      device.pushErrorScope?.("validation");
      seg.loadLabelmap(lab);
      baker.bake();
      await device.queue.onSubmittedWorkDone();
      const N = 3, t0 = performance.now();
      for (let i = 0; i < N; i++) baker.refine();
      await device.queue.onSubmittedWorkDone();
      refineMs = (performance.now() - t0) / N;
      await device.popErrorScope?.();
    } catch {
    } finally {
      baker.destroy();
      seg.destroy();
    }
    const tier = refineMs < 3.5 ? "high" : refineMs < 14 ? "mid" : "low";
    return new _SegBudget(tier, refineMs);
  }
  /** A fixed mid-tier budget without probing (SSR/headless/opt-out). */
  static fixed(tier = "mid") {
    return new _SegBudget(tier, tier === "high" ? 2 : tier === "mid" ? 8 : 20);
  }
  /** SDF grid cap per axis for large volumes (SEGRoulette / microCT). */
  sdfMaxDim() {
    return this.tier === "high" ? 384 : this.tier === "mid" ? 256 : 128;
  }
  /** TOTAL SDF voxel budget — the memory ceiling for the isotropic resample (~64 B/voxel across the
   *  JFA seed + sdf/attr/scratch textures, so mid ≈ 256³ ≈ 1.1 GB). Thick-slice anisotropy is upsampled
   *  toward isotropic (no per-slice gaps), and this caps how far, so a weaker device coarsens gracefully
   *  instead of OOMing (the caller can then suggest remote rendering). NOT sdfMaxDim³ — high stays ~2 GB. */
  sdfMaxVoxels() {
    return this.tier === "high" ? 320 ** 3 : this.tier === "mid" ? 256 ** 3 : 128 ** 3;
  }
  /** Debounce before the settle-refine fires (ms). Fast → near-immediate (dynamic); slow → patient. */
  refineDelayMs() {
    return this.tier === "high" ? 40 : this.tier === "mid" ? 150 : 320;
  }
  /** High-end only: refine live during a stroke rather than only on settle. */
  refineDuringStroke() {
    return this.tier === "high";
  }
  /** Reserve the exact-EDT refinement tier (when built) for capable devices. */
  useEdt() {
    return this.tier !== "low";
  }
  summary() {
    return `${this.tier} (${this.refineMsAt64.toFixed(1)} ms/refine@64\xB3 \u2192 sdf\u2264${this.sdfMaxDim()}, refine@${this.refineDelayMs()}ms)`;
  }
};

// algorithms/demos/algorithms-browser.ts
var status = (msg, err = false) => {
  const el = document.getElementById("status-text");
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
  globalThis.__gpuErr = [];
  gpu.device.addEventListener("uncapturederror", (e) => globalThis.__gpuErr.push(String(e.error?.message ?? e.error)));
  const ctx = canvas.getContext("webgpu");
  const preferred = navigator.gpu.getPreferredCanvasFormat();
  const srgb = preferred + "-srgb";
  ctx.configure({ device: gpu.device, format: preferred, viewFormats: [srgb], alphaMode: "opaque" });
  status("probing device capability\u2026");
  const budget = await SegBudget.probe(gpu.device);
  const a = buildAlgorithmsScene(gpu, srgb, { refineDelayMs: budget.refineDelayMs() });
  const camera = framedCamera(a.center, a.radius, 2.8);
  const a3d = mountAdaptive3d({
    scene: () => a.scene,
    view: () => ctx.getCurrentTexture().createView({ format: srgb }),
    size: () => ({ w: canvas.width, h: canvas.height }),
    setCamera: (s, w, h) => s.setCamera(camera.position, camera.focalPoint, camera.viewUp, camera.viewAngle, w, h),
    gpu,
    onFrame: () => {
    }
  });
  const draw = () => a3d.draw();
  const drawNow = () => a3d.renderSettled(true);
  a.onRedraw(() => drawNow());
  const resize = () => {
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    const size = Math.min(760, Math.floor(canvas.clientWidth * dpr));
    canvas.width = size;
    canvas.height = size;
    drawNow();
  };
  globalThis.addEventListener("resize", resize);
  attachCameraControls(canvas, camera, { onChange: draw });
  let n = 0;
  const poke = () => {
    const R = 70;
    const c = [(Math.random() * 2 - 1) * R, (Math.random() * 2 - 1) * R, (Math.random() * 2 - 1) * R];
    a.poke(c, 16 + Math.random() * 12);
    status(`poked ${++n} sphere${n === 1 ? "" : "s"} through the shared buffer \xB7 surface re-rendered in place`);
  };
  document.getElementById("poke")?.addEventListener("click", poke);
  let painting = false, strokeN = 0;
  const paintStroke = async () => {
    if (painting) return;
    painting = true;
    const R = 55, N = 26;
    const cx = (Math.random() * 2 - 1) * 30, cy = (Math.random() * 2 - 1) * 30, cz = (Math.random() * 2 - 1) * 30;
    a.driver.beginStroke({ segmentId: `stroke_${++strokeN}`, effect: "Paint", brush: { shape: "sphere", diameterMm: 14 } });
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1) * Math.PI * 1.5;
      a.driver.addPoint([cx + R * Math.cos(t), cy + R * Math.sin(t) * 0.6, cz + i / N * 40 - 20]);
      status(`painting stroke from the SegEdit stream\u2026 sample ${i + 1}/${N}`);
      await new Promise((r) => requestAnimationFrame(() => r(null)));
    }
    a.driver.endStroke();
    status(`painted a stroke via SegEditDriver (${N} samples, interpolated) \xB7 orbit to inspect`);
    painting = false;
  };
  document.getElementById("paint")?.addEventListener("click", paintStroke);
  const modeBtn = document.getElementById("mode");
  const MODES = { label: ["Render: Label", "label surfaces (Slicer #9570)"], sdf: ["Render: SDF", "SDF (crisp, terrace-free)"], surface: ["Render: Gaussian", "Gaussian (gradient-opacity)"] };
  const NEXT = { label: "sdf", sdf: "surface", surface: "label" };
  const syncModeBtn = () => {
    if (modeBtn) modeBtn.textContent = MODES[a.renderMode()][0];
  };
  syncModeBtn();
  modeBtn?.addEventListener("click", () => {
    a.setRenderMode(NEXT[a.renderMode()]);
    syncModeBtn();
    drawNow();
    status(`render path: ${MODES[a.renderMode()][1]}`);
  });
  document.getElementById("rand")?.addEventListener("click", () => {
    a.randomizeLook();
    drawNow();
    status("randomized per-segment opacity + surface/volume shading");
  });
  document.getElementById("opaque")?.addEventListener("click", () => {
    a.resetLook();
    drawNow();
    status("all segments opaque surfaces");
  });
  document.getElementById("reset")?.addEventListener("click", () => location.reload());
  resize();
  status(`surface-mode segmentation \xB7 ${budget.tier}-tier device \xB7 drag to orbit \xB7 scroll/pinch to zoom \xB7 Poke to edit`);
  globalThis.__algoDbg = {
    dist: () => camera.distance,
    err: () => (globalThis.__gpuErr || []).length,
    tier: () => budget.tier
  };
}
main();
