"""The F-K Filter's polygon editor: an anywidget drawn over the F-K domain plot (UI only - used by app_marimo.py).

Click on the plot to add points - as many as wanted - and double-click to finish the polygon (only then is it applied);
afterwards drag a point to move it, click near an edge to insert a point, right-click a point to delete it. Clicking
an empty spot starts another polygon. 🗑 Remove polygon / ✕ Remove all delete them. Every finished edit (polygon
closed, drag released, point inserted / deleted, polygon removed) is sent to Python at once (`polys`, one list of
[k, f] per polygon, in the plot's data units; `rev` counts the edits) - the app then re-runs the F-K Filter with it.
"""
from __future__ import annotations

import base64

import anywidget
import traitlets

_ESM = r"""
// the polygon being drawn lives here, outside render(): when the app redraws the page and the editor is drawn again,
// the points clicked so far are kept
const DRAFT = { open: null };

function render({ model, el }) {
  const box = document.createElement("div");
  box.style.cssText = "display:flex;flex-direction:column;gap:6px;width:100%";
  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;align-items:center";
  const mk = (txt, title, color) => {
    const b = document.createElement("button");
    b.textContent = txt; b.title = title;
    b.style.cssText = `padding:3px 10px;border-radius:6px;border:1px solid ${color};color:${color};` +
                      "background:white;cursor:pointer;font-size:13px";
    bar.appendChild(b); return b;
  };
  const bUndo = mk("↶ Undo point", "Take back the last point of the polygon being drawn", "#4a5263");
  const bDel = mk("🗑 Remove polygon", "Delete the selected polygon (the highlighted one) - or the one being drawn", "#e34948");
  const bAll = mk("✕ Remove all", "Delete every polygon (no F-K rejection)", "#6b7280");
  const info = document.createElement("div");
  info.style.cssText = "font-size:12px;opacity:.8;line-height:1.35";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;height:auto;cursor:crosshair;border-radius:6px;touch-action:none";
  box.appendChild(bar); box.appendChild(info); box.appendChild(canvas); el.appendChild(box);
  const ctx = canvas.getContext("2d");
  const img = new Image();

  // polys: [{pts: [[k, f], ...], closed}] - the open one (at most one) is being drawn and is not sent to Python
  let polys = [], active = -1, drag = null;
  const drawing = () => polys.some((p) => !p.closed);
  const load = () => {
    polys = (model.get("polys") || []).map((p) => ({ pts: p.map((q) => q.slice()), closed: true }));
    if (DRAFT.open) polys.push(DRAFT.open);            // (the same object: points added to it stay in DRAFT)
    active = polys.length - 1;
  };
  load();
  const A = () => model.get("axes") || {};
  const toPx = (p) => { const a = A(), W = canvas.width, H = canvas.height;
    return [(a.x0 + (p[0] - a.kmin) / (a.kmax - a.kmin) * (a.x1 - a.x0)) * W,
            (a.y1 - (p[1] - a.fmin) / (a.fmax - a.fmin) * (a.y1 - a.y0)) * H]; };
  const toData = (x, y) => { const a = A(), W = canvas.width, H = canvas.height;
    return [a.kmin + (x / W - a.x0) / (a.x1 - a.x0) * (a.kmax - a.kmin),
            a.fmin + (a.y1 - y / H) / (a.y1 - a.y0) * (a.fmax - a.fmin)]; };
  const inPlot = (x, y) => { const a = A(), u = x / canvas.width, v = y / canvas.height;
    return u >= a.x0 && u <= a.x1 && v >= a.y0 && v <= a.y1; };
  const evXY = (e) => { const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height]; };
  const scale = () => canvas.width / Math.max(canvas.getBoundingClientRect().width, 1);
  const tol = () => 9 * scale();

  function draw() {
    if (!img.complete || !img.naturalWidth) return;
    const u = scale();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const a = A();
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.x0 * canvas.width, a.y0 * canvas.height, (a.x1 - a.x0) * canvas.width, (a.y1 - a.y0) * canvas.height);
    ctx.clip();
    polys.forEach((poly, i) => {
      if (!poly.pts.length) return;
      const shapes = [poly.pts.map(toPx)];
      if (model.get("mirror") && poly.closed) shapes.push(poly.pts.map((p) => toPx([-p[0], p[1]])));
      shapes.forEach((pts, j) => {
        ctx.beginPath();
        pts.forEach(([x, y], n) => (n ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        if (poly.closed) ctx.closePath();
        ctx.setLineDash(j ? [6 * u, 4 * u] : (poly.closed ? [] : [4 * u, 3 * u]));
        ctx.lineWidth = (i === active ? 2.4 : 1.6) * u;
        ctx.strokeStyle = poly.closed ? "#ff4d4d" : "#ffd23f";
        ctx.fillStyle = i === active ? "rgba(255,77,77,.25)" : "rgba(255,77,77,.12)";
        if (poly.closed) ctx.fill();
        ctx.stroke();
      });
      ctx.setLineDash([]);
      poly.pts.map(toPx).forEach(([x, y], n) => {
        ctx.beginPath();
        ctx.arc(x, y, (i === active ? 5.5 : 4.5) * u, 0, 2 * Math.PI);
        ctx.fillStyle = !poly.closed ? (n === 0 ? "#ffd23f" : "white") : (i === active ? "white" : "#ffd0d0");
        ctx.fill(); ctx.lineWidth = 1.8 * u; ctx.strokeStyle = poly.closed ? "#e34948" : "#b8860b"; ctx.stroke();
      });
    });
    ctx.restore();
    const open = polys.find((p) => !p.closed);
    const used = polys.filter((p) => p.closed && p.pts.length >= 3).length;
    info.innerHTML = open
      ? `<b>Drawing:</b> ${open.pts.length} point(s) - click to add more, <b>double-click</b> to finish the polygon` +
        (open.pts.length < 3 ? ` (at least 3 points)` : "") + ` · Esc = cancel`
      : `<b>Click</b> to start a polygon (as many points as you like, <b>double-click</b> to finish) · ` +
        `<b>drag</b> a point to move it · <b>right-click</b> a point to delete it · ${used} polygon(s) in use`;
    bUndo.disabled = !open || !open.pts.length;
  }

  function commit() {               // the finished polygons -> Python (the open one, if any, is not sent)
    model.set("polys", polys.filter((p) => p.closed && p.pts.length >= 3)
      .map((p) => p.pts.map(([k, f]) => [+k.toFixed(4), +Math.max(f, 0).toFixed(3)])));
    model.set("rev", (model.get("rev") || 0) + 1);
    model.save_changes();
  }

  function hitPoint(x, y) {
    let best = null, bd = tol();
    polys.forEach((poly, i) => poly.pts.forEach((p, j) => {
      const [px, py] = toPx(p), d = Math.hypot(px - x, py - y);
      if (d < bd) { bd = d; best = [i, j]; }
    }));
    return best;
  }

  function edgeOf(poly, x, y) {          // the edge of a finished polygon closest to (x, y), if within the tolerance
    if (!poly.closed || poly.pts.length < 3) return -1;
    let best = -1, bd = tol();
    const n = poly.pts.length;
    for (let j = 0; j < n; j++) {
      const [ax, ay] = toPx(poly.pts[j]), [bx, by] = toPx(poly.pts[(j + 1) % n]);
      const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy || 1;
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / L));
      const d = Math.hypot(ax + t * dx - x, ay + t * dy - y);
      if (d < bd) { bd = d; best = j; }
    }
    return best;
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    const [x, y] = evXY(e);
    let open = polys.findIndex((p) => !p.closed);
    if (open < 0) {
      const hit = hitPoint(x, y);
      if (hit) { active = hit[0]; drag = hit; canvas.setPointerCapture(e.pointerId); draw(); return; }
      for (let i = 0; i < polys.length; i++) {          // near an edge of a finished polygon: insert a point there
        const edge = edgeOf(polys[i], x, y);
        if (edge >= 0) { polys[i].pts.splice(edge + 1, 0, toData(x, y)); active = i; draw(); commit(); return; }
      }
      if (!inPlot(x, y)) return;
      DRAFT.open = { pts: [], closed: false };         // start a new polygon
      polys.push(DRAFT.open);
      open = polys.length - 1;
    }
    if (!inPlot(x, y)) return;
    const pts = polys[open].pts, last = pts[pts.length - 1];
    if (last) { const [lx, ly] = toPx(last); if (Math.hypot(lx - x, ly - y) < tol()) return; }  // (2nd click of a double-click)
    pts.push(toData(x, y));
    active = open;
    draw();
  });
  canvas.addEventListener("dblclick", (e) => {
    const open = polys.findIndex((p) => !p.closed);
    if (open < 0) return;
    e.preventDefault();
    if (polys[open].pts.length >= 3) { polys[open].closed = true; DRAFT.open = null; active = open; draw(); commit(); }
    else draw();
  });
  canvas.addEventListener("pointermove", (e) => {
    const [x, y] = evXY(e);
    if (!drag) { canvas.style.cursor = !drawing() && hitPoint(x, y) ? "move" : "crosshair"; return; }
    const a = A();
    const cx = Math.min(Math.max(x, a.x0 * canvas.width), a.x1 * canvas.width);
    const cy = Math.min(Math.max(y, a.y0 * canvas.height), a.y1 * canvas.height);
    polys[drag[0]].pts[drag[1]] = toData(cx, cy);
    draw();
  });
  const endDrag = () => { if (drag) { drag = null; commit(); } };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("contextmenu", (e) => {
    const [x, y] = evXY(e), hit = hitPoint(x, y);
    if (!hit) return;
    e.preventDefault();
    const poly = polys[hit[0]];
    poly.pts.splice(hit[1], 1);
    if (poly.closed && poly.pts.length < 3) polys.splice(hit[0], 1);
    active = Math.min(hit[0], polys.length - 1);
    draw();
    if (poly.closed) commit();
  });
  canvas.tabIndex = 0;
  canvas.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { DRAFT.open = null; polys = polys.filter((p) => p.closed); active = polys.length - 1; draw(); }
  });

  bUndo.onclick = () => { const p = polys.find((q) => !q.closed); if (p) { p.pts.pop(); draw(); } };
  bDel.onclick = () => {
    const open = polys.findIndex((p) => !p.closed);
    const i = open >= 0 ? open : active;
    if (i < 0 || i >= polys.length) return;
    const was = polys[i].closed;
    if (!was) DRAFT.open = null;
    polys.splice(i, 1); active = polys.length - 1; draw();
    if (was) commit();
  };
  bAll.onclick = () => { DRAFT.open = null; polys = []; active = -1; draw(); commit(); };

  const setImage = () => { img.src = model.get("image") || ""; };
  img.onload = () => { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; draw(); };
  model.on("change:image", setImage);
  model.on("change:axes", draw);
  model.on("change:mirror", draw);
  // the app puts back what it filtered with - never over a polygon being drawn or a point being dragged
  model.on("change:polys", () => {
    if (drag) return;
    load();                                            // (keeps DRAFT.open, the polygon being drawn)
    draw();
  });
  setImage();
}
export default { render };
"""


class FkEditor(anywidget.AnyWidget):
    _esm = _ESM
    image = traitlets.Unicode("").tag(sync=True)      # the F-K plot as a data: URL
    axes = traitlets.Dict({}).tag(sync=True)          # plot area (figure fractions, y from the top) and its data limits
    polys = traitlets.List([]).tag(sync=True)         # [[[k, f], ...], ...] one list of corners per polygon
    mirror = traitlets.Bool(True).tag(sync=True)      # also draw each polygon at -k (dashed)
    rev = traitlets.Int(0).tag(sync=True)             # edits made in the browser

    def show(self, png: bytes, figure, polys, mirror: bool) -> None:
        """Put a new F-K plot (and the polygons it was filtered with) in the editor."""
        ax = figure.axes[0]
        pos = ax.get_position()
        (k0, k1), (f0, f1) = ax.get_xlim(), ax.get_ylim()
        self.axes = {"x0": pos.x0, "x1": pos.x1, "y0": 1 - pos.y1, "y1": 1 - pos.y0,
                     "kmin": k0, "kmax": k1, "fmin": f0, "fmax": f1}
        self.mirror = bool(mirror)
        self.polys = [[[float(k), float(f)] for k, f in p] for p in polys]
        self.image = "data:image/png;base64," + base64.b64encode(png).decode()
