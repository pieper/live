// Self-contained help overlay for the SlicerLive deployed demo. A "?" button
// in the upper-left opens a glass modal listing all the mouse / trackpad /
// keyboard bindings actually wired in slicerlive.js. No bundle rebuild
// required — this script self-installs on DOMContentLoaded.
//
// Audit source: viewer/slicerlive.js on the `main` branch of github.com/pieper/SlicerLive.
// Bindings catalogued from the interactor manipulator setup (lines ~195–204)
// + the slice/3D pointerdown / wheel / dblclick / shift-gesture handlers.
//
// If a binding changes in the source, update the SECTIONS array below.

(function () {
  if (window.__slHelpInstalled) return;
  window.__slHelpInstalled = true;

  // -------- Binding cheat sheet (source of truth: viewer/slicerlive.js@main) --------
  const SECTIONS = [
    {
      title: '3D view',
      rows: [
        ['Left-drag',                 'Rotate'],
        ['Right-drag',                'Zoom (focal point fixed)'],
        ['Middle-drag',               'Pan'],
        ['Shift + Left-drag',         'Pan'],
        ['Wheel / two-finger scroll', 'Zoom (focal point fixed)'],
        ['Trackpad pinch',            'Zoom'],
        ['R',                         'Reset camera'],
        ['Double-click',              'Maximize / restore this view'],
        ['Right-double-click',        'Reset all views to fit the volume'],
      ],
    },
    {
      title: 'Slice views (axial / sagittal / coronal)',
      rows: [
        ['Left-drag',                 'Scroll through slices'],
        ['Middle-drag',               'Pan'],
        ['Right-drag',                'Zoom this slice'],
        ['Wheel / two-finger scroll', 'Slice offset (next / previous slice)'],
        ['Double-click',              'Maximize / restore this slice'],
        ['Right-double-click',        'Reset all views to fit the volume'],
      ],
    },
    {
      title: 'Linked navigation (works over any view)',
      rows: [
        ['Shift + drag',              'Jump the OTHER slice views to point under the cursor'],
        ['Shift + Right-drag',        'Synced zoom across all 4 views'],
      ],
    },
    {
      title: 'SEGRoulette / Display panel',
      rows: [
        ['Spin button',               'Pick a new random IDC case'],
        ['Details',                   'Citation, license, link to OHIF + IDC portal'],
        ['SlicerLive mark (top-right)', 'Hover or click to toggle volume rendering, segments, fill/outline'],
      ],
    },
  ];

  // -------- Styles + DOM ------------------------------------------------
  function makeButton() {
    const b = document.createElement('button');
    b.id = 'sl-help-btn';
    b.title = 'Help & key bindings';
    b.setAttribute('aria-label', 'Open help');
    b.style.cssText =
      'position:fixed; top:14px; left:14px; z-index:74;' +
      ' width:36px; height:36px; padding:0; cursor:pointer;' +
      ' display:inline-flex; align-items:center; justify-content:center;' +
      ' border:1px solid rgba(255,255,255,0.16); border-radius:50%; background:rgba(20,23,36,0.62);' +
      ' backdrop-filter:blur(14px) saturate(1.4); -webkit-backdrop-filter:blur(14px) saturate(1.4);' +
      ' color:#cfe6ff; font:700 16px -apple-system,system-ui,sans-serif; letter-spacing:0.4px;' +
      ' box-shadow:0 6px 20px rgba(0,0,0,0.45);' +
      ' transition:background 100ms ease-out, transform 100ms ease-out;';
    b.textContent = '?';
    b.onmouseenter = () => { b.style.background = 'rgba(30,34,52,0.82)'; b.style.transform = 'scale(1.06)'; };
    b.onmouseleave = () => { b.style.background = 'rgba(20,23,36,0.62)'; b.style.transform = 'scale(1)'; };
    b.onclick = (ev) => { ev.stopPropagation(); openHelp(); };
    return b;
  }

  let helpOverlay = null;

  function openHelp() {
    if (helpOverlay) return;
    helpOverlay = document.createElement('div');
    helpOverlay.style.cssText =
      'position:fixed; inset:0; z-index:96; display:flex; align-items:center; justify-content:center;' +
      ' background:rgba(6,8,14,0.55); opacity:0; transition:opacity 140ms ease-out;' +
      ' font:13px/1.5 -apple-system,system-ui,sans-serif; color:#e8eeff;';
    helpOverlay.addEventListener('mousedown', (e) => { if (e.target === helpOverlay) closeHelp(); });

    const panel = document.createElement('div');
    panel.style.cssText =
      'max-width:min(720px,92vw); max-height:88vh; overflow-y:auto;' +
      ' padding:26px 30px 22px; border-radius:18px; color:#eaf0ff;' +
      ' background:linear-gradient(135deg, rgba(58,64,88,0.58), rgba(20,24,38,0.66));' +
      ' backdrop-filter:blur(26px) saturate(1.7); -webkit-backdrop-filter:blur(26px) saturate(1.7);' +
      ' border:1px solid rgba(255,255,255,0.22);' +
      ' box-shadow:0 24px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.2);' +
      ' transform:translateY(8px) scale(0.98); opacity:0;' +
      ' transition:opacity 160ms ease-out, transform 160ms cubic-bezier(.2,.7,.2,1);';

    // Header
    const head = document.createElement('div');
    head.style.cssText = 'display:flex; align-items:baseline; gap:14px; margin-bottom:6px;';
    const title = document.createElement('div');
    title.textContent = 'SlicerLive — controls';
    title.style.cssText = 'font:800 22px -apple-system,system-ui,sans-serif; letter-spacing:0.2px;';
    const sub = document.createElement('div');
    sub.textContent = 'Mouse, trackpad, and keyboard';
    sub.style.cssText = 'font:600 12px -apple-system,system-ui,sans-serif; color:rgba(232,238,255,0.55); margin-left:auto;';
    head.appendChild(title); head.appendChild(sub);
    panel.appendChild(head);

    // Sections
    for (const sec of SECTIONS) {
      const block = document.createElement('div');
      block.style.cssText = 'margin-top:18px; padding:14px 16px; border-radius:12px;' +
        ' background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);';
      const h = document.createElement('div');
      h.textContent = sec.title;
      h.style.cssText = 'font:700 11px -apple-system,system-ui,sans-serif; letter-spacing:1.2px; text-transform:uppercase; color:#9fe9ff; margin-bottom:10px;';
      block.appendChild(h);
      const table = document.createElement('div');
      table.style.cssText = 'display:grid; grid-template-columns:max-content 1fr; gap:6px 18px; align-items:baseline;';
      for (const [keys, desc] of sec.rows) {
        const k = document.createElement('div');
        k.textContent = keys;
        k.style.cssText = 'font:600 12.5px ui-monospace,Menlo,monospace; color:#fff5d6; white-space:nowrap;';
        const d = document.createElement('div');
        d.textContent = desc;
        d.style.cssText = 'font:13px -apple-system,system-ui,sans-serif; color:rgba(232,238,255,0.85);';
        table.appendChild(k); table.appendChild(d);
      }
      block.appendChild(table);
      panel.appendChild(block);
    }

    // Footer
    const foot = document.createElement('div');
    foot.style.cssText = 'margin-top:18px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.10);' +
      ' display:flex; align-items:center; gap:12px; font-size:12px; color:rgba(232,238,255,0.55);';
    foot.innerHTML = '<span>Press <b style="color:#fff5d6">esc</b> or click outside to dismiss.</span>';
    const close = document.createElement('button');
    close.textContent = 'Close';
    close.style.cssText = 'margin-left:auto; cursor:pointer; border:1px solid rgba(255,255,255,0.18); border-radius:9px;' +
      ' padding:7px 16px; font:600 12px -apple-system,system-ui,sans-serif; color:#e8eeff;' +
      ' background:rgba(255,255,255,0.06);';
    close.onclick = closeHelp;
    foot.appendChild(close);
    panel.appendChild(foot);

    helpOverlay.appendChild(panel);
    document.body.appendChild(helpOverlay);
    requestAnimationFrame(() => {
      if (!helpOverlay) return;
      helpOverlay.style.opacity = '1';
      panel.style.opacity = '1';
      panel.style.transform = 'translateY(0) scale(1)';
    });

    // ESC to close
    document.addEventListener('keydown', onHelpKey, true);
  }

  function closeHelp() {
    if (!helpOverlay) return;
    helpOverlay.style.opacity = '0';
    const el = helpOverlay;
    helpOverlay = null;
    setTimeout(() => { if (el && el.parentNode) el.parentNode.removeChild(el); }, 160);
    document.removeEventListener('keydown', onHelpKey, true);
  }

  function onHelpKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); closeHelp(); }
  }

  // Defer install until the SEGRoulette bar / SlicerLive logo are present so
  // the "?" button visibly belongs to the same UI layer (its position fixed
  // at top-left next to the SEGRoulette bar feels natural).
  function install() {
    if (document.getElementById('sl-help-btn')) return;
    document.body.appendChild(makeButton());
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', install);
  } else {
    install();
  }
})();
