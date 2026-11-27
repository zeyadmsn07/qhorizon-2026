/* Q//HORIZON 2026 dashboard: vanilla JS + inline SVG, no dependencies. */
(function () {
  "use strict";
  var D = window.QH_DATA;
  var BLUE = "#6E8EEB", PURPLE = "#B07CE8", INK = "#0F172A", GRAYC = "#94A3B8";

  /* ---------- helpers ---------- */
  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function f3(x) { return Number(x).toFixed(3); }
  function f4(x) { return Number(x).toFixed(4); }
  function sg(x, d) { d = d || 3; return (x >= 0 ? "+" : "\u2212") + Math.abs(x).toFixed(d); }
  function col(fam) { return fam === "classical" ? BLUE : fam === "quantum" ? PURPLE : GRAYC; }
  function byName(arr, name) { return arr.filter(function (r) { return r.name === name; })[0]; }
  function setHTML(id, h) { $(id).innerHTML = h; }
  function hasWeights(p) { return p != null && Math.round(p) > 0; }
  function fmtWeights(p) { return hasWeights(p) ? String(Math.round(p)) : "\u2014"; }

  function syncRovingTabs(container, attr, activeVal) {
    Array.prototype.forEach.call(container.querySelectorAll("[role='tab']"), function (b) {
      var on = b.getAttribute(attr) === activeVal;
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.setAttribute("tabindex", on ? "0" : "-1");
    });
  }
  function wireTabKeys(container, attr, onSelect) {
    container.addEventListener("keydown", function (e) {
      var keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
      if (keys.indexOf(e.key) < 0) return;
      var items = Array.prototype.slice.call(container.querySelectorAll("[role='tab']"));
      var idx = items.indexOf(document.activeElement);
      if (idx < 0) return;
      e.preventDefault();
      var next = e.key === "Home" ? 0 : e.key === "End" ? items.length - 1
        : (idx + (e.key === "ArrowRight" ? 1 : -1) + items.length) % items.length;
      items[next].focus();
      onSelect(items[next].getAttribute(attr));
    });
  }

  var HEAD = "Q1-VQC (Bipartite_SignLinear_41p)", MLP = "Tiny-MLP (41p)", QK = "Q2-QuantumKernel (8q)", SVM = "RBF-SVM", RF = "RandomForest", LR = "LogReg";

  /* ---------- tooltip ---------- */
  var tip = $("tip");
  function tipHTML(t) {
    var h = "<b>" + esc(t.title) + "</b>";
    (t.lines || []).forEach(function (l) { h += "<span class='k'>" + esc(l[0]) + ":</span> " + esc(l[1]) + "<br>"; });
    return h;
  }
  function showTip(t, x, y) {
    tip.innerHTML = tipHTML(t); tip.style.opacity = 1;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var nx = Math.min(x + 14, window.innerWidth - w - 8), ny = y - h - 12;
    if (ny < 8) ny = y + 18;
    tip.style.left = Math.max(8, nx) + "px"; tip.style.top = ny + "px";
  }
  function hideTip() { tip.style.opacity = 0; }
  function wireTips(root) {
    var els = root.querySelectorAll("[data-tip]");
    Array.prototype.forEach.call(els, function (el) {
      var t = JSON.parse(el.getAttribute("data-tip"));
      el.addEventListener("mousemove", function (e) { showTip(t, e.clientX, e.clientY); });
      el.addEventListener("mouseleave", hideTip);
      el.addEventListener("focus", function () { var r = el.getBoundingClientRect(); showTip(t, r.left + r.width / 2, r.top); });
      el.addEventListener("blur", hideTip);
    });
  }
  function tipAttr(t) { return esc(JSON.stringify(t)); }

  /* ---------- chart: Cleveland dot plot ---------- */
  function dotPlot(el, rows, o) {
    var W = 760, labelW = 200, x0 = labelW + 10, x1 = 620, rowH = 36, top = 6;
    var H = top + rows.length * rowH + 40;
    function X(v) { return x0 + (Math.min(Math.max(v, o.min), o.max) - o.min) / (o.max - o.min) * (x1 - x0); }
    var s = "<svg viewBox='0 0 " + W + " " + H + "' role='img' aria-label='" + esc(o.aria || "Dot plot") + "'>";
    o.ticks.forEach(function (t) {
      s += "<line class='grid' x1='" + X(t) + "' x2='" + X(t) + "' y1='" + top + "' y2='" + (H - 30) + "'/>";
      s += "<text x='" + X(t) + "' y='" + (H - 12) + "' text-anchor='middle'>" + (o.tickFmt ? o.tickFmt(t) : t) + "</text>";
    });
    rows.forEach(function (r, i) {
      var y = top + i * rowH + rowH / 2;
      s += "<rect class='rowbg' x='0' y='" + (y - rowH / 2) + "' width='" + W + "' height='" + rowH + "' fill='transparent'/>";
      s += "<line x1='" + x0 + "' x2='" + x1 + "' y1='" + y + "' y2='" + y + "' stroke='#E2E8F0' stroke-dasharray='2 4'/>";
      var lo = r.lo, hi = r.hi;
      if (lo != null && hi != null) {
        var xl = X(lo), xh = X(hi);
        s += "<line x1='" + xl + "' x2='" + xh + "' y1='" + y + "' y2='" + y + "' stroke='" + col(r.family) + "' stroke-width='3' stroke-linecap='round' opacity='.45'/>";
        if (lo < o.min) s += "<path d='M" + (x0 + 6) + " " + (y - 5) + " L" + x0 + " " + y + " L" + (x0 + 6) + " " + (y + 5) + "' fill='none' stroke='" + col(r.family) + "' stroke-width='2'/>";
        if (hi > o.max) s += "<path d='M" + (x1 - 6) + " " + (y - 5) + " L" + x1 + " " + y + " L" + (x1 - 6) + " " + (y + 5) + "' fill='none' stroke='" + col(r.family) + "' stroke-width='2'/>";
      }
      s += "<text class='lab' x='" + (labelW) + "' y='" + (y + 4) + "' text-anchor='end'>" + esc(r.label) + "</text>";
      s += "<circle class='dot' tabindex='0' role='img' aria-label='" + esc(r.label + ": " + f3(r.mean)) + "' cx='" + X(r.mean) + "' cy='" + y + "' r='8' fill='" + col(r.family) + "' stroke='#fff' stroke-width='2' data-tip='" + tipAttr(r.tip) + "'/>";
      s += "<text x='" + (W - 4) + "' y='" + (y + 4) + "' text-anchor='end' class='lab' style='font-variant-numeric:tabular-nums;font-weight:700'>" + f3(r.mean) + "</text>";
    });
    if (o.xLabel) s += "<text x='" + ((x0 + x1) / 2) + "' y='" + (H + 2) + "' text-anchor='middle'>" + esc(o.xLabel) + "</text>";
    s += "</svg>";
    el.innerHTML = s; wireTips(el);
  }

  /* ---------- chart: line plot (categorical x) ---------- */
  var MARK = { circle: function (x, y, c) { return "<circle cx='" + x + "' cy='" + y + "' r='6' fill='" + c + "' stroke='#fff' stroke-width='2'"; },
               square: function (x, y, c) { return "<rect x='" + (x - 6) + "' y='" + (y - 6) + "' width='12' height='12' rx='2' fill='" + c + "' stroke='#fff' stroke-width='2'"; },
               diamond: function (x, y, c) { return "<path d='M" + x + " " + (y - 8) + " L" + (x + 8) + " " + y + " L" + x + " " + (y + 8) + " L" + (x - 8) + " " + y + " Z' fill='" + c + "' stroke='#fff' stroke-width='2'"; },
               triangle: function (x, y, c) { return "<path d='M" + x + " " + (y - 8) + " L" + (x + 8) + " " + (y + 6) + " L" + (x - 8) + " " + (y + 6) + " Z' fill='" + c + "' stroke='#fff' stroke-width='2'"; } };
  function linePlot(el, series, o) {
    var W = 760, H = 360, l = 58, r = 190, t = 16, b = 52;
    var xs = o.xs;
    function X(i) { return l + (xs.length === 1 ? 0 : i / (xs.length - 1)) * (W - l - r); }
    function Y(v) { return t + (1 - (v - o.yMin) / (o.yMax - o.yMin)) * (H - t - b); }
    var s = "<svg viewBox='0 0 " + W + " " + H + "' role='img' aria-label='" + esc(o.aria || "Line chart") + "'>";
    o.yTicks.forEach(function (v) {
      s += "<line class='grid' x1='" + l + "' x2='" + (W - r) + "' y1='" + Y(v) + "' y2='" + Y(v) + "'/><text x='" + (l - 8) + "' y='" + (Y(v) + 4) + "' text-anchor='end'>" + v.toFixed(o.yDec || 2) + "</text>";
    });
    xs.forEach(function (v, i) { s += "<text x='" + X(i) + "' y='" + (H - b + 22) + "' text-anchor='middle'>" + esc(o.xFmt ? o.xFmt(v) : v) + "</text>"; });
    s += "<text x='" + ((l + W - r) / 2) + "' y='" + (H - 6) + "' text-anchor='middle'>" + esc(o.xLabel) + "</text>";
    s += "<text transform='translate(14 " + ((t + H - b) / 2) + ") rotate(-90)' text-anchor='middle'>" + esc(o.yLabel) + "</text>";
    var ends = [];
    series.forEach(function (se) {
      var c = se.color, pts = se.points;
      var d = pts.map(function (p, i) { return (i ? "L" : "M") + X(p.i) + " " + Y(p.y); }).join(" ");
      s += "<path d='" + d + "' fill='none' stroke='" + c + "' stroke-width='2.5' " + (se.dash ? "stroke-dasharray='" + se.dash + "'" : "") + "/>";
      pts.forEach(function (p) {
        if (p.err) s += "<line x1='" + X(p.i) + "' x2='" + X(p.i) + "' y1='" + Y(Math.max(p.y - p.err, o.yMin)) + "' y2='" + Y(Math.min(p.y + p.err, o.yMax)) + "' stroke='" + c + "' stroke-width='2' opacity='.5'/>";
        s += MARK[se.marker || "circle"](X(p.i), Y(p.y), c) + " class='dot' tabindex='0' role='img' aria-label='" + esc(se.label + ", " + p.x + ": " + f3(p.y)) + "' data-tip='" + tipAttr(p.tip) + "'/>";
      });
      var last = pts[pts.length - 1];
      ends.push({ y: Y(last.y), label: se.label, color: c });
    });
    ends.sort(function (a, b2) { return a.y - b2.y; });
    for (var i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
    ends.forEach(function (e) { s += "<text class='lab' x='" + (W - r + 14) + "' y='" + (e.y + 4) + "' style='font-weight:600'>" + esc(e.label) + "</text>"; });
    s += "</svg>";
    el.innerHTML = s; wireTips(el);
  }

  function legendHTML(items) {
    return items.map(function (i) { return "<span><i style='background:" + i.c + "'></i>" + esc(i.t) + "</span>"; }).join("");
  }

  /* ---------- SECTION 01: hero + mission ---------- */
  var M = D.meta;
  $("hero-signals").textContent = M.n_signals.toLocaleString("en-US");
  $("hero-weights").textContent = M.weights + " vs. " + M.weights;
  $("hero-circuits").textContent = M.circuit_ratio + "\u00d7";
  var feats = ["IP_Mean", "IP_Std", "IP_Kurtosis", "IP_Skewness", "DM_Mean", "DM_Std", "DM_Kurtosis", "DM_Skewness"];
  $("qmap").innerHTML = feats.map(function (f, i) { return "<span class='chip'>" + f + "</span><span class='arrow'>&rarr;</span><span class='chip q'>qubit " + i + "</span>"; }).join("");
  $("bal-p").textContent = M.pulsar_pct.toFixed(2) + "%";
  $("bal-n").textContent = M.noise_pct.toFixed(2) + "% interference";
  var dummy = byName(D.htru2, "Dummy");
  setHTML("trap", "<b>Why not use accuracy?</b> Predicting &ldquo;not a pulsar&rdquo; for every signal is already <b>" + (100 - M.pulsar_pct).toFixed(1) + "% accurate</b> and finds <b>zero</b> pulsars. We therefore score with <b>PR-AUC</b>, where that strategy earns only <b>" + f3(dummy.mean) + "</b>. Higher is better; 1.0 is perfect.");

  /* ---------- SECTION 02/03: dataset toggle + Concentric Shell sample-size toggle + dot plot ---------- */
  var SHELL_NS = D.shell_by_n ? Object.keys(D.shell_by_n).map(Number).sort(function (a, b) { return a - b; }) : [500];
  var state = {
    ds: "htru2",
    zoom: true,
    shellN: SHELL_NS.indexOf(500) >= 0 ? 500 : (SHELL_NS[SHELL_NS.length - 1] || 500),
    study: "gold_a"
  };

  function getShellRows() {
    return (D.shell_by_n && D.shell_by_n[String(state.shellN)]) || D.shell;
  }
  function buildShellNButtons(el) {
    el.innerHTML = SHELL_NS.map(function (n) {
      return "<button type='button' class='tab' data-n='" + n + "' aria-pressed='false'>N = " + n + "</button>";
    }).join("");
    el.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-n]");
      if (b) { state.shellN = Number(b.getAttribute("data-n")); renderDataset(); }
    });
  }
  function syncShellNButtons() {
    var isShell = state.ds === "shell";
    $("shell-n-ds").hidden = !isShell;
    $("shell-n-gl").hidden = !isShell;
    $("zoom").hidden = isShell;
    Array.prototype.forEach.call(document.querySelectorAll("#shell-n-ds-btns button[data-n], #shell-n-gl button[data-n]"), function (b) {
      var on = Number(b.getAttribute("data-n")) === state.shellN;
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }
  buildShellNButtons($("shell-n-ds-btns"));
  buildShellNButtons($("shell-n-gl"));

  function dsContent() {
    var mlp = byName(D.htru2, MLP), vqc = byName(D.htru2, HEAD), qk = byName(D.htru2, QK), svm = byName(D.htru2, SVM), rf = byName(D.htru2, RF), lr = byName(D.htru2, LR);
    if (state.ds === "htru2") return {
      kicker: "Real astronomical survey &middot; " + M.n_signals.toLocaleString("en-US") + " signals",
      hero: f3(mlp.mean) + " <span style='color:var(--faint)'>vs</span> " + f3(vqc.mean),
      sub: "Tiny-MLP vs. Q1-VQC, both with " + M.weights + " weights (PR-AUC, N = 500)",
      desc: "Real pulsars are mostly separable by a straight line, so even plain Logistic Regression reaches " + f3(lr.mean) + ". Classical models have the edge.",
      what: "Classical Tiny-MLP scored <b>" + f3(mlp.mean) + "</b>, ahead of the Quantum Kernel (<b>" + f3(qk.mean) + "</b>) and the Quantum VQC (<b>" + f3(vqc.mean) + "</b>). Both quantum models still beat Random Forest (<b>" + f3(rf.mean) + "</b>).",
      so: "Pulsars and interference separate fairly cleanly along straight boundaries, so extra quantum complexity does not buy a standalone win. The gap is small (about " + f3(mlp.mean - vqc.mean) + ") but consistent across seeds. <b>No quantum advantage is claimed.</b>",
      glLede: "Each marker is a model\u2019s mean PR-AUC over 10 seeds at 500 training samples (higher is better). Hover over or focus a marker for details.",
      glWhat: "Tiny-MLP (<b>" + f3(mlp.mean) + "</b>) and RBF-SVM (<b>" + f3(svm.mean) + "</b>) lead. Quantum Kernel (<b>" + f3(qk.mean) + "</b>) and VQC (<b>" + f3(vqc.mean) + "</b>) follow, then Random Forest (<b>" + f3(rf.mean) + "</b>). All 10 seeds trained without collapse.",
      glTitle: "Classical Models Still Lead on Straight Boundaries",
      glSo: "Differences are around 0.01 to 0.015 PR-AUC: real but small. On data that is close to linearly separable, matched classical models are the safer default, and quantum models remain competitive."
    };
    var srows = getShellRows();
    var smlp = byName(srows, MLP), svqc = byName(srows, "Q1-VQC (Headline)"), sqk = byName(srows, QK), ssvm = byName(srows, SVM), slr = byName(srows, LR);
    var vqcDelta = svqc.mean - smlp.mean;
    return {
      kicker: "Synthetic curved benchmark &middot; 8 dimensions &middot; N = " + state.shellN,
      hero: sg(vqcDelta),
      sub: "Q1-VQC " + f3(svqc.mean) + " vs. Tiny-MLP " + f3(smlp.mean) + ", both with " + M.weights + " weights (PR-AUC, N = " + state.shellN + ")",
      desc: "Points are arranged in concentric shells, so no straight line can separate them: Logistic Regression collapses to " + f3(slr.mean) + ". Switch N above to inspect how geometry scales from 100 to 500 samples.",
      what: "At <b>N = " + state.shellN + "</b>, the 41-weight Quantum VQC reached <b>" + f3(svqc.mean) + "</b>, while the 41-weight neural network scored <b>" + f3(smlp.mean) + "</b> (<b>" + sg(vqcDelta) + "</b>). The Quantum Kernel scored <b>" + f3(sqk.mean) + "</b> against RBF-SVM <b>" + f3(ssvm.mean) + "</b>.",
      so: "With the same weight budget, a compact circuit learned the curved shell structure better than the matched neural net. Data is synthetic, so this illustrates <b>geometry</b>, not real-sky performance.",
      glLede: "Each marker is a model\u2019s mean PR-AUC over 10 seeds at " + state.shellN + " training samples (higher is better). Hover over or focus a marker for details.",
      glWhat: "At <b>N = " + state.shellN + "</b>, kernel methods score <b>" + f3(sqk.mean) + "</b> (Q2) and <b>" + f3(ssvm.mean) + "</b> (RBF-SVM). Q1-VQC (<b>" + f3(svqc.mean) + "</b>) sits well above Tiny-MLP (<b>" + f3(smlp.mean) + "</b>), while Logistic Regression stays near chance (<b>" + f3(slr.mean) + "</b>).",
      glTitle: "Compact Quantum Circuits Gain an Edge on Curved Boundaries",
      glSo: "Curvature changes the outcome: straight-line models fail, and the circuit\u2019s rotational structure becomes an advantage over an equal-size neural network. Whiskers are 95% bootstrap intervals."
    };
  }
  function renderDataset() {
    var c = dsContent();
    setHTML("ds-kicker", c.kicker); setHTML("ds-hero", c.hero); $("ds-hero-sub").textContent = c.sub; $("ds-desc").textContent = c.desc;
    setHTML("ds-what", c.what); setHTML("ds-so", c.so);
    $("gl-lede").textContent = c.glLede; setHTML("gl-what", c.glWhat); $("gl-title").textContent = c.glTitle; setHTML("gl-so", c.glSo);
    syncRovingTabs($("ds-tabs"), "data-ds", state.ds);
    syncShellNButtons();
    renderDot();
  }
  function renderDot() {
    var z = $("zoom"), rows;
    if (state.ds === "htru2") {
      z.setAttribute("aria-pressed", state.zoom ? "true" : "false");
      z.textContent = state.zoom ? "Zoomed axis 0.89\u20130.93" : "Full axis 0\u20131";
      setHTML("gl-whisker", "Line = mean &plusmn; 1 std over 10 seeds");
      rows = D.htru2.filter(function (r) { return state.zoom ? ["Dummy", "Q1-VQC (Ring L=2, Fragile v1)"].indexOf(r.name) < 0 : true; })
        .sort(function (a, b) { return b.mean - a.mean; }).map(function (r) {
          return { label: r.label, family: r.family, mean: r.mean, lo: r.mean - r.std, hi: r.mean + r.std,
            tip: { title: r.name, lines: [["PR-AUC", f4(r.mean) + " \u00b1 " + f4(r.std)], ["MCC", r.mcc != null ? f3(r.mcc) : "n/a"], ["F1", r.f1 != null ? f3(r.f1) : "n/a"], ["Weights", fmtWeights(r.params)], ["Collapsed runs", r.collapse + "/" + r.n]] } };
        });
      dotPlot($("dotplot"), rows, state.zoom ? { min: 0.89, max: 0.93, ticks: [0.89, 0.90, 0.91, 0.92, 0.93], tickFmt: function (t) { return t.toFixed(2); }, xLabel: "Average PR-AUC (axis zoomed to show small gaps)", aria: "HTRU2 PR-AUC dot plot" }
        : { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], tickFmt: function (t) { return t.toFixed(2); }, xLabel: "Average PR-AUC (full axis)", aria: "HTRU2 PR-AUC dot plot, full axis" });
    } else {
      setHTML("gl-whisker", "Line = 95% bootstrap interval");
      rows = getShellRows().slice().sort(function (a, b) { return b.mean - a.mean; }).map(function (r) {
        return { label: r.label, family: r.family, mean: r.mean, lo: r.lo, hi: r.hi,
          tip: { title: r.name === "Q1-VQC (Headline)" ? "Q1-VQC (41 weights)" : r.name, lines: [["Training samples", String(state.shellN)], ["PR-AUC", f4(r.mean)], ["95% interval", "[" + f3(r.lo) + ", " + f3(r.hi) + "]"]] } };
      });
      dotPlot($("dotplot"), rows, { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], tickFmt: function (t) { return t.toFixed(2); }, xLabel: "Average PR-AUC (full axis, N = " + state.shellN + ")", aria: "Concentric Shell-8D PR-AUC dot plot at N = " + state.shellN });
    }
  }
  $("ds-tabs").addEventListener("click", function (e) {
    var b = e.target.closest(".tab");
    if (b) { state.ds = b.getAttribute("data-ds"); renderDataset(); }
  });
  wireTabKeys($("ds-tabs"), "data-ds", function (ds) { state.ds = ds; renderDataset(); });
  $("zoom").addEventListener("click", function () { state.zoom = !state.zoom; renderDot(); });
  /* ---------- SECTION 04: ablation explorer (2 synchronized switchers: cards + pill tabs) ---------- */
  function verdict(id) { return D.verdicts.filter(function (v) { return v.id === id; })[0]; }
  function hw(prefix) { return D.hardware.filter(function (h) { return h.model.indexOf(prefix) === 0; })[0]; }
  var GA_LABEL = { none: "No entanglement", ring: "Ring wiring", bipartite: "Bipartite wiring" };

  var STUDIES = [
    { id: "gold_a", tab: "1 \u00b7 Wiring", title: "Experiment 1 \u00b7 Entanglement wiring and symmetry fix", short: "Entanglement wiring", q: "Does denser wiring help, or just add depth?", grp: "Circuit design", render: function () {
        var rows = D.gold_a.map(function (r) { var m = r.name.match(/VQC_(\w+)_L(\d)/); return { w: m[1], L: +m[2], r: r }; })
          .sort(function (a, b) { return ["none", "ring", "bipartite"].indexOf(a.w) - ["none", "ring", "bipartite"].indexOf(b.w) || a.L - b.L; })
          .map(function (o) { return { label: GA_LABEL[o.w] + " \u00b7 " + o.L + " layer" + (o.L > 1 ? "s" : ""), family: "quantum", mean: o.r.mean, lo: o.r.mean - o.r.std, hi: o.r.mean + o.r.std,
            tip: { title: o.r.name, lines: [["PR-AUC", f4(o.r.mean) + " \u00b1 " + f4(o.r.std)], ["Weights", fmtWeights(o.r.params)], ["Collapsed runs", o.r.collapse + "/" + o.r.n]] } }; });
        dotPlot($("ab-chart"), rows, { min: 0.86, max: 0.91, ticks: [0.86, 0.87, 0.88, 0.89, 0.90, 0.91], tickFmt: function (t) { return t.toFixed(2); }, xLabel: "Average PR-AUC (zoomed)", aria: "Gold A entanglement ablation" });
        var bi = byName(D.gold_a, "VQC_bipartite_L2"), ri = byName(D.gold_a, "VQC_ring_L2"), no = byName(D.gold_a, "VQC_none_L2");
        var hb = hw("Q1-VQC (8q Bipartite"), hr = hw("Q1-VQC (8q Ring");
        var h2b = verdict("H2b"), h2a = verdict("H2a");
        return { legend: legendHTML([{ c: PURPLE, t: "Quantum VQC variants" }]), note: "Line = mean \u00b1 1 std over 10 seeds. Axis is zoomed.",
          what: "At 2 layers, bipartite wiring scored <b>" + f3(bi.mean) + "</b> versus ring <b>" + f3(ri.mean) + "</b> and no entanglement <b>" + f3(no.mean) + "</b>. The headline bipartite circuit needs <b>" + hb.two_qubit_gates + "</b> two-qubit gates (depth " + hb.depth + ") against <b>" + hr.two_qubit_gates + "</b> (depth " + hr.depth + ") for ring.",
          so: "A lighter layout trains better rather than worse: bipartite beats ring by <b>" + sg(h2b.delta, 4) + "</b> (" + h2b.verdict + "). Against no entanglement the gain (" + sg(h2a.delta, 4) + ") is " + h2a.verdict.toLowerCase() + ", because pulsar data is mostly linear. A \u03c0/2 phase shift plus reading all 8 qubits removed training collapses." };
      } },
    { id: "gold_b", tab: "2 \u00b7 Encoding", title: "Experiment 2 \u00b7 Qubit compression and encoding", short: "Qubit compression", q: "Can 8 features fit on fewer qubits?", grp: "Circuit design", render: function () {
        var names = { "8q_Angle_L2": "8 qubits \u00b7 angle encoding", "8q_Angle_Reupload_L2": "8 qubits \u00b7 angle + re-upload", "3q_Amplitude_L5": "3 qubits \u00b7 amplitude (norm lost)", "4q_AmplitudePlusNorm_L4": "4 qubits \u00b7 amplitude + norm qubit" };
        var rows = D.gold_b.slice().sort(function (a, b) { return b.mean - a.mean; }).map(function (r) {
          return { label: names[r.name] || r.name, family: "quantum", mean: r.mean, lo: r.mean - r.std, hi: r.mean + r.std,
            tip: { title: r.name, lines: [["PR-AUC", f4(r.mean) + " \u00b1 " + f4(r.std)], ["Weights", fmtWeights(r.params)], ["Collapsed runs", r.collapse + "/" + r.n]] } };
        });
        dotPlot($("ab-chart"), rows, { min: 0.55, max: 0.95, ticks: [0.55, 0.65, 0.75, 0.85, 0.95], tickFmt: function (t) { return t.toFixed(2); }, xLabel: "Average PR-AUC", aria: "Gold B encoding ablation" });
        var a3 = byName(D.gold_b, "3q_Amplitude_L5"), a4 = byName(D.gold_b, "4q_AmplitudePlusNorm_L4"), a8 = byName(D.gold_b, "8q_Angle_L2");
        return { legend: legendHTML([{ c: PURPLE, t: "Quantum encodings" }]), note: "Line = mean \u00b1 1 std over 10 seeds.",
          what: "Squeezing 8 features into 3 qubits dropped PR-AUC to <b>" + f3(a3.mean) + "</b> (all " + a3.collapse + "/" + a3.n + " runs collapsed). Adding one qubit that stores the signal magnitude lifted it to <b>" + f3(a4.mean) + "</b>, a gain of <b>" + sg(a4.mean - a3.mean) + "</b>.",
          so: "Standard amplitude encoding rescales every signal to length 1, which erases how strong it was. Keep the magnitude, or give each feature its own qubit (<b>" + f3(a8.mean) + "</b>), and the information survives." };
      } },
    { id: "gold_c", tab: "3 \u00b7 Low data", title: "Experiment 3 \u00b7 Low-data sweep", short: "Low-data sweep", q: "Do quantum models win with few samples?", grp: "Stress tests", render: function () {
        var xs = [100, 200, 500, 1000], spec = [
          [SVM, "RBF-SVM", BLUE, "", "square"], [MLP, "Tiny-MLP (41p)", BLUE, "7 5", "circle"],
          [QK, "Q2-QuantumKernel", PURPLE, "", "square"], [HEAD, "Q1-VQC (41p)", PURPLE, "7 5", "circle"]];
        var series = spec.map(function (sp) {
          var g = D.gold_c[sp[0]];
          return { label: sp[1], color: sp[2], dash: sp[3], marker: sp[4], points: g.points.map(function (p) {
            return { i: xs.indexOf(p.n), x: p.n, y: p.mean, err: p.std, tip: { title: sp[1], lines: [["Training samples", String(p.n)], ["PR-AUC", f4(p.mean) + " \u00b1 " + f4(p.std)]] } }; }) };
        });
        linePlot($("ab-chart"), series, { xs: xs, yMin: 0.84, yMax: 0.94, yTicks: [0.84, 0.86, 0.88, 0.90, 0.92, 0.94], xLabel: "Training samples (N)", yLabel: "Average PR-AUC", aria: "Gold C sample sweep" });
        var q100 = D.gold_c[QK].points[0].mean, s100 = D.gold_c[SVM].points[0].mean, v100 = D.gold_c[HEAD].points[0].mean, m100 = D.gold_c[MLP].points[0].mean;
        var q1k = D.gold_c[QK].points[3].mean, s1k = D.gold_c[SVM].points[3].mean;
        return { legend: legendHTML([{ c: BLUE, t: "Classical (solid RBF-SVM, dashed MLP)" }, { c: PURPLE, t: "Quantum (solid kernel, dashed VQC)" }]), note: "Vertical bars = \u00b1 1 std over 10 seeds. N = 100 to 1,000.",
          what: "At <b>N = 100</b>, Tiny-MLP scored <b>" + f3(m100) + "</b> and RBF-SVM <b>" + f3(s100) + "</b>, while the Quantum VQC reached <b>" + f3(v100) + "</b> and the Quantum Kernel only <b>" + f3(q100) + "</b>. By <b>N = 1,000</b> the kernel nearly catches up (<b>" + f3(q1k) + "</b> vs. <b>" + f3(s1k) + "</b>).",
          so: "On HTRU2, small samples do not favor standalone quantum models: the quantum kernel trails RBF-SVM by <b>" + f3(s100 - q100) + "</b> at N = 100 (hypothesis H4a, " + verdict("H4a").verdict + "). The VQC is the more stable quantum option when data is scarce." };
      } },
    { id: "gold_d", tab: "4 \u00b7 Shot noise", title: "Experiment 4 \u00b7 Measurement shot noise", short: "Shot-noise stability", q: "Is accuracy stable with few measurements?", grp: "Stress tests", render: function () {
        var rows = D.gold_d.slice().reverse().map(function (r) {
          return { label: r.label, family: "quantum", mean: r.mean, lo: r.mean - r.std, hi: r.mean + r.std, tip: { title: r.label, lines: [["PR-AUC", f4(r.mean) + " \u00b1 " + f4(r.std)], ["Seeds", String(r.n)]] } };
        });
        dotPlot($("ab-chart"), rows, { min: 0.895, max: 0.915, ticks: [0.895, 0.9, 0.905, 0.91, 0.915], tickFmt: function (t) { return t.toFixed(3); }, xLabel: "Average PR-AUC (strongly zoomed: total range is 0.02)", aria: "Gold D shot noise" });
        var g = function (s) { return D.gold_d.filter(function (r) { return r.shots === s; })[0].mean; };
        return { legend: legendHTML([{ c: PURPLE, t: "Q1-VQC (41p) measured with N shots" }]), note: "Axis spans only 0.02 PR-AUC. Line = mean \u00b1 1 std over 10 seeds.",
          what: "Measuring each circuit just <b>2,000</b> times gave <b>" + f3(g("2000")) + "</b> and <b>500</b> times gave <b>" + f3(g("500")) + "</b>, versus <b>" + f3(g("analytic")) + "</b> with perfect (exact) readout. Only <b>100</b> shots dipped, to <b>" + f3(g("100")) + "</b>.",
          so: "From 500 shots upward the score is within about 0.002 of exact simulation, so realistic measurement budgets are enough. 100 shots shows mild sampling noise (" + sg(g("100") - g("analytic"), 4) + "). Hardware gate errors are not modeled here." };
      } },
    { id: "bonus_pqk", tab: "5 \u00b7 Kernels", title: "Experiment 5 \u00b7 Projected quantum kernels", short: "Projected kernels", q: "Can kernel cost drop without losing accuracy?", grp: "Cost and combination", render: function () {
        var xs = D.pqk.map(function (p) { return p.alpha; });
        var series = [
          { label: "PQK (projected)", color: PURPLE, dash: "", marker: "circle", points: D.pqk.map(function (p, i) { return { i: i, x: p.alpha, y: p.pqk, err: p.pqk_std, tip: { title: "Projected Quantum Kernel", lines: [["Angle scale \u03b1", String(p.alpha)], ["PR-AUC", f4(p.pqk) + " \u00b1 " + f4(p.pqk_std)], ["Circuits", M.pqk_circuits.toLocaleString("en-US")]] } }; }) },
          { label: "Fidelity kernel", color: PURPLE, dash: "7 5", marker: "square", points: D.pqk.map(function (p, i) { return { i: i, x: p.alpha, y: p.fqk, err: p.fqk_std, tip: { title: "Fidelity Quantum Kernel", lines: [["Angle scale \u03b1", String(p.alpha)], ["PR-AUC", f4(p.fqk) + " \u00b1 " + f4(p.fqk_std)], ["Circuits", M.fqk_circuits.toLocaleString("en-US")]] } }; }) }
        ];
        linePlot($("ab-chart"), series, { xs: xs, yMin: 0.45, yMax: 0.95, yTicks: [0.5, 0.6, 0.7, 0.8, 0.9], xLabel: "Input angle scale \u03b1 (larger = bigger rotation angles)", yLabel: "Average PR-AUC", xFmt: function (v) { return "\u03b1 = " + v; }, aria: "PQK versus fidelity kernel" });
        var last = D.pqk[D.pqk.length - 1], first = D.pqk[0];
        return { legend: legendHTML([{ c: PURPLE, t: "Both are quantum kernels (solid = PQK, dashed = fidelity kernel)" }]), note: "Vertical bars = \u00b1 1 std over 10 seeds.",
          what: "The projected kernel needs <b>" + M.pqk_circuits.toLocaleString("en-US") + "</b> circuits instead of <b>" + M.fqk_circuits.toLocaleString("en-US") + "</b>, which is <b>" + M.circuit_ratio + "\u00d7 fewer</b>. At large angles (\u03b1 = " + last.alpha + ") the fidelity kernel falls to <b>" + f3(last.fqk) + "</b> while PQK holds <b>" + f3(last.pqk) + "</b>.",
          so: "Measuring small local pieces of the circuit avoids the loss of contrast (kernel concentration) that affects the full-overlap kernel, and it makes hardware cost grow linearly with data. At gentle angles (\u03b1 = " + first.alpha + ") PQK is also ahead: <b>" + f3(first.pqk) + "</b> vs. <b>" + f3(first.fqk) + "</b>." };
      } },
    { id: "bonus_hyb", tab: "6 \u00b7 Hybrid", title: "Experiment 6 \u00b7 Hybrid quantum-classical blend", short: "Hybrid blend", q: "Do the two model types complement each other?", grp: "Cost and combination", render: function () {
        var xs = D.hybrid.map(function (h) { return h.n; });
        function mk(key, label, color, dash, marker) {
          return { label: label, color: color, dash: dash, marker: marker, points: D.hybrid.map(function (h, i) { return { i: i, x: h.n, y: h[key], tip: { title: label, lines: [["Training samples", String(h.n)], ["PR-AUC", f4(h[key])]] } }; }) };
        }
        linePlot($("ab-chart"), [mk("mlp", "Tiny-MLP", BLUE, "", "circle"), mk("vqc", "Q1-VQC", PURPLE, "", "circle"), mk("blend", "50/50 blend", INK, "7 5", "diamond")],
          { xs: xs, yMin: 0.88, yMax: 0.94, yTicks: [0.88, 0.90, 0.92, 0.94], xLabel: "Training samples (N)", yLabel: "Average PR-AUC", aria: "Hybrid blend" });
        var h0 = D.hybrid[0], h2 = D.hybrid[2];
        return { legend: legendHTML([{ c: BLUE, t: "Classical MLP" }, { c: PURPLE, t: "Quantum VQC" }, { c: INK, t: "Hybrid blend" }]), note: "Mean PR-AUC over 10 seeds.",
          what: "At <b>N = 100</b>, averaging the two models\u2019 probabilities scored <b>" + f4(h0.blend) + "</b>, beating the MLP alone (<b>" + f4(h0.mlp) + "</b>) by <b>" + sg(h0.blend - h0.mlp, 4) + "</b> and the VQC alone (<b>" + f4(h0.vqc) + "</b>) by <b>" + sg(h0.blend - h0.vqc, 4) + "</b>.",
          so: "The two models make different mistakes, so blending helps most when data is scarce. By N = 500 the MLP alone (<b>" + f4(h2.mlp) + "</b>) matches or beats the blend (<b>" + f4(h2.blend) + "</b>): a low-data technique rather than a general upgrade." };
      } }
  ];

  var tabs = $("ab-tabs");
  tabs.innerHTML = STUDIES.map(function (s) {
    return "<button type='button' class='tab' role='tab' id='ab-tab-" + s.id + "' aria-controls='ab-panel' data-id='" + s.id + "' aria-selected='false'>" + esc(s.tab) + "</button>";
  }).join("");
  $("ex-flow").innerHTML = STUDIES.map(function (s, i) {
    return "<li><button type='button' data-id='" + s.id + "' aria-current='false'><span class='n'>EXPERIMENT " + (i + 1) + "</span><b>" + esc(s.short) + "</b><span class='q'>" + esc(s.q) + "</span><span class='grp'>" + esc(s.grp) + "</span></button></li>";
  }).join("");

  function renderStudy() {
    var s = STUDIES.filter(function (x) { return x.id === state.study; })[0], out = s.render();
    $("ab-cur").textContent = s.title;
    $("ab-panel").setAttribute("aria-labelledby", "ab-tab-" + s.id);
    setHTML("ab-legend", out.legend); $("ab-note").textContent = out.note; setHTML("ab-what", out.what); setHTML("ab-so", out.so);
    Array.prototype.forEach.call($("ex-flow").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-current", b.getAttribute("data-id") === s.id ? "true" : "false");
    });
    syncRovingTabs(tabs, "data-id", s.id);
  }
  $("ex-flow").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-id]");
    if (b) { state.study = b.getAttribute("data-id"); renderStudy(); }
  });
  tabs.addEventListener("click", function (e) {
    var b = e.target.closest(".tab");
    if (b) { state.study = b.getAttribute("data-id"); renderStudy(); }
  });
  wireTabKeys(tabs, "data-id", function (id) { state.study = id; renderStudy(); });

  /* ---------- SECTION 06: audit tables ---------- */
  var best = D.htru2.filter(function (r) { return r.family !== "baseline"; }).sort(function (a, b) { return b.mean - a.mean; })[0].name;
  var t1 = "<thead><tr><th>Model</th><th>Type</th><th class='num'>PR-AUC (mean \u00b1 std)</th><th class='num'>MCC</th><th class='num'>F1</th><th class='num'>Weights</th><th class='num'>Collapsed</th></tr></thead><tbody>";
  D.htru2.slice().sort(function (a, b) { return b.mean - a.mean; }).forEach(function (r) {
    var wCell = hasWeights(r.params) ? String(Math.round(r.params)) : "<span class='muted-dash' title='Non-parametric (no trainable weights)'>&mdash;</span>";
    var mccCell = r.mcc != null ? f3(r.mcc) : "&mdash;";
    var f1Cell = r.f1 != null ? f3(r.f1) : "&mdash;";
    t1 += "<tr class='" + r.family + (r.name === best ? " best" : "") + "'><td>" + esc(r.name) + (r.name === best ? " <span class='tag best'>TOP</span>" : "") + "</td><td>" + r.family + "</td><td class='num'>" + f4(r.mean) + " \u00b1 " + f4(r.std) + "</td><td class='num'>" + mccCell + "</td><td class='num'>" + f1Cell + "</td><td class='num'>" + wCell + "</td><td class='num'>" + r.collapse + "/" + r.n + "</td></tr>";
  });
  $("t-main").innerHTML = t1 + "</tbody>";

  var t2 = "<thead><tr><th>ID</th><th>Comparison (A vs. B)</th><th class='num'>\u0394 PR-AUC</th><th class='num'>95% CI</th><th class='num'>Wilcoxon p</th><th>Verdict</th></tr></thead><tbody>";
  D.verdicts.forEach(function (v) {
    var cls = "v-" + v.verdict.split(" ")[0];
    t2 += "<tr><td><b>" + v.id + "</b></td><td>" + esc(v.comparison) + "</td><td class='num'>" + sg(v.delta, 4) + "</td><td class='num'>[" + sg(v.lo, 4) + ", " + sg(v.hi, 4) + "]</td><td class='num'>" + v.p.toFixed(4) + "</td><td><span class='chipv " + cls + "'>" + esc(v.verdict) + "</span></td></tr>";
  });
  $("t-verd").innerHTML = t2 + "</tbody>";

  var t3 = "<thead><tr><th>Model</th><th class='num'>Qubits</th><th class='num'>Depth</th><th class='num'>Two-qubit gates</th><th class='num'>Weights</th><th>Circuits to train (N = 500)</th></tr></thead><tbody>";
  D.hardware.forEach(function (h) {
    var fam = h.model.indexOf("Q") === 0 ? "quantum" : "classical";
    t3 += "<tr class='" + fam + "'><td>" + esc(h.model) + "</td><td class='num'>" + h.qubits + "</td><td class='num'>" + h.depth + "</td><td class='num'>" + h.two_qubit_gates + "</td><td class='num'>" + h.trainable_params + "</td><td>" + esc(h.qpu_circuits_train_N500) + "</td></tr>";
  });
  $("t-hw").innerHTML = t3 + "</tbody>";

  /* ---------- go ---------- */
  renderDataset();
  renderStudy();
})();
