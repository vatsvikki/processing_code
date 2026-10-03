"""The F-K Filter's polygon editor: an anywidget drawn over the F-K domain plot (UI only - used by app_marimo.py).

Click on the plot to add a point (near an edge: the point is inserted into that edge), drag a point to move it,
double-click (or right-click) a point to delete it. ＋ New polygon starts another reject zone, 🗑 Remove polygon deletes
the selected one. Every finished edit (click / drag released / remove) is sent to Python at once (`polys`, one list of
[k, f] per polygon, in the plot's data units; `rev` counts the edits) - the app then re-runs the F-K Filter with it.
"""
from __future__ import annotations

import base64

import anywidget
import traitlets

_ESM = r"""
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
  const bNew = mk("＋ New polygon", "Start another reject zone: the next clicks add its points", "#2b9a66");
  const bDel = mk("🗑 Remove polygon", "Delete the selected polygon (the highlighted one)", "#e34948");
  const bAll = mk("✕ Remove all", "Delete every polygon (no F-K rejection)", "#6b7280");
  const info = document.createElement("span");
  info.style.cssText = "font-size:12px;opacity:.75";
  bar.appendChild(info);
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;height:auto;cursor:crosshair;border-radius:6px;touch-action:none";
  box.appendChild(bar); box.appendChild(canvas); el.appendChild(box);
  const ctx = canvas.getContext("2d");
  const img = new Image();

  let polys = [], active = 0, drag = null;
  const load = () => {
    polys = JSON.parse(JSON.stringify(model.get("polys") || []));
    if (!polys.length) polys = [[]];
    active = Math.min(active, polys.length - 1);
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
  const tol = () => 9 * canvas.width / Math.max(canvas.getBoundingClientRect().width, 1);

  function draw() {
    if (!img.complete || !img.naturalWidth) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const a = A();
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.x0 * canvas.width, a.y0 * canvas.height, (a.x1 - a.x0) * canvas.width, (a.y1 - a.y0) * canvas.height);
    ctx.clip();
    polys.forEach((poly, i) => {
      if (!poly.length) return;
      const shapes = [poly.map(toPx)];
      if (model.get("mirror")) shapes.push(poly.map((p) => toPx([-p[0], p[1]])));
      shapes.forEach((pts, j) => {
        ctx.beginPath();
        pts.forEach(([x, y], n) => (n ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        if (pts.length >= 3) ctx.closePath();
        ctx.setLineDash(j ? [6, 4] : []);
        ctx.lineWidth = (i === active ? 2.5 : 1.6) * canvas.width / 900;
        ctx.strokeStyle = "#ff4d4d";
        ctx.fillStyle = i === active ? "rgba(255,77,77,.25)" : "rgba(255,77,77,.12)";
        if (pts.length >= 3) ctx.fill();
        ctx.stroke();
      });
      ctx.setLineDash([]);
      poly.map(toPx).forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, (i === active ? 6 : 4.5) * canvas.width / 900, 0, 2 * Math.PI);
        ctx.fillStyle = i === active ? "white" : "#ffd0d0";
        ctx.fill(); ctx.lineWidth = 2 * canvas.width / 900; ctx.strokeStyle = "#e34948"; ctx.stroke();
      });
    });
    ctx.restore();
    const n = polys.filter((p) => p.length >= 3).length;
    info.textContent = `Click = add a point · drag a point = move it · double-click a point = delete it · ` +
      `${n} polygon(s) in use` + (polys[active] && polys[active].length && polys[active].length < 3 ?
      ` · selected polygon needs ${3 - polys[active].length} more point(s)` : "");
  }

  function commit() {
    model.set("polys", polys.filter((p) => p.length).map((p) => p.map(([k, f]) => [+k.toFixed(4), +Math.max(f, 0).toFixed(3)])));
    model.set("rev", (model.get("rev") || 0) + 1);
    model.save_changes();
  }

  function hitPoint(x, y) {
    let best = null, bd = tol();
    polys.forEach((poly, i) => poly.forEach((p, j) => {
      const [px, py] = toPx(p), d = Math.hypot(px - x, py - y);
      if (d < bd) { bd = d; best = [i, j]; }
    }));
    return best;
  }

  function edgeOf(poly, x, y) {          // the edge of the polygon closest to (x, y), if within the tolerance
    if (poly.length < 2) return -1;
    let best = -1, bd = tol();
    const n = poly.length;
    for (let j = 0; j < (n >= 3 ? n : n - 1); j++) {
      const [ax, ay] = toPx(poly[j]), [bx, by] = toPx(poly[(j + 1) % n]);
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
    const hit = hitPoint(x, y);
    if (hit) { active = hit[0]; drag = hit; canvas.setPointerCapture(e.pointerId); draw(); return; }
    if (!inPlot(x, y)) return;
    const poly = polys[active] || (polys[active] = []);
    const edge = edgeOf(poly, x, y);
    if (edge >= 0) poly.splice(edge + 1, 0, toData(x, y)); else poly.push(toData(x, y));
    draw(); commit();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) { const [x, y] = evXY(e); canvas.style.cursor = hitPoint(x, y) ? "move" : "crosshair"; return; }
    const [x, y] = evXY(e), a = A();
    const cx = Math.min(Math.max(x, a.x0 * canvas.width), a.x1 * canvas.width);
    const cy = Math.min(Math.max(y, a.y0 * canvas.height), a.y1 * canvas.height);
    polys[drag[0]][drag[1]] = toData(cx, cy);
    draw();
  });
  const endDrag = () => { if (drag) { drag = null; commit(); } };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  const remove = (e) => {
    const [x, y] = evXY(e), hit = hitPoint(x, y);
    if (!hit) return;
    e.preventDefault();
    polys[hit[0]].splice(hit[1], 1);
    active = hit[0]; draw(); commit();
  };
  canvas.addEventListener("dblclick", remove);
  canvas.addEventListener("contextmenu", remove);

  bNew.onclick = () => { if (!polys.length || polys[polys.length - 1].length) polys.push([]); active = polys.length - 1; draw(); };
  bDel.onclick = () => { polys.splice(active, 1); if (!polys.length) polys = [[]]; active = Math.max(0, active - 1); draw(); commit(); };
  bAll.onclick = () => { polys = [[]]; active = 0; draw(); commit(); };

  const setImage = () => { img.src = model.get("image") || ""; };
  img.onload = () => { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; draw(); };
  model.on("change:image", setImage);
  model.on("change:axes", draw);
  model.on("change:mirror", draw);
  model.on("change:polys", () => { if (!drag) { load(); draw(); } });
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
