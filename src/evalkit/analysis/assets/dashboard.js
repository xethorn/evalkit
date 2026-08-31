
(function () {
  const ALL = JSON.parse(document.getElementById("payload").textContent).suites;

  // Tabs first: each suite is an independent dashboard over its own samples, and their
  // scores must never be pooled — different suites have different difficulty.
  const tabs = Array.from(document.querySelectorAll("[data-suite-tab]"));
  const panels = Array.from(document.querySelectorAll("[data-suite-panel]"));
  const controlSets = Array.from(document.querySelectorAll("[data-suite-controls]"));
  const docks = Array.from(document.querySelectorAll("[data-suite-dock]"));
  // Each suite registers how to repaint its side panel, so switching tabs can restore the
  // new tab's panel. Without this the panel simply vanished on the second tab: the old
  // dock was hidden and nothing ever told the new one to show itself.
  const repaint = new Map();

  // Collapsing the dock is remembered per browser, because whether you want the diff
  // beside the numbers is a working preference, not a property of the data.
  // The app bar's height is data the layout needs: the dock and its rail hang from the
  // bottom of it, and it grows or shrinks with the window.
  const appbar = document.querySelector("header.appbar");
  const publishBarHeight = () => {
    if (appbar) document.documentElement.style.setProperty("--appbar-h", appbar.offsetHeight + "px");
  };
  publishBarHeight();
  window.addEventListener("resize", publishBarHeight);

  const rail = document.querySelector('[data-role="dock-open"]');

  // Whether a diff is on screen at all — distinct from whether the user collapsed it.
  // The content reclaims the dock's width when there is nothing to dock.
  function setDockPresent(present) {
    document.body.classList.toggle("dock-hidden", !present);
    const collapsed = document.body.classList.contains("dock-closed");
    for (const d of docks) {
      d.hidden = d.hasAttribute("data-inactive") || !present || collapsed;
    }
    if (rail) rail.hidden = !present || !collapsed;
  }
  function setDock(open) {
    document.body.classList.toggle("dock-closed", !open);
    if (rail) rail.hidden = open || document.body.classList.contains("dock-hidden");
    for (const d of docks) {
      d.querySelector('[data-role="dock-toggle"]').setAttribute("aria-expanded", String(open));
    }
    try { localStorage.setItem("evalkit.dock", open ? "open" : "closed"); } catch (e) { void e; }
  }
  for (const d of docks) {
    d.querySelector('[data-role="dock-toggle"]').addEventListener("click", () => setDock(false));
  }
  if (rail) rail.addEventListener("click", () => setDock(true));
  // Opening or closing the dock changes how much width the grid has.
  document.addEventListener("click", (event) => {
    if (!event.target.closest('[data-role="dock-toggle"], [data-role="dock-open"]')) return;
    requestAnimationFrame(() => {
      for (const t of document.querySelectorAll("table.bigtable")) {
        t.dispatchEvent(new CustomEvent("resize-columns", { bubbles: false }));
      }
    });
  });
  try { setDock(localStorage.getItem("evalkit.dock") !== "closed"); } catch (e) { void e; setDock(true); }
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      const wanted = tab.getAttribute("data-suite-tab");
      for (const t of tabs) t.setAttribute("aria-selected", String(t === tab));
      for (const pl of panels) pl.hidden = pl.getAttribute("data-suite-panel") !== wanted;
      // The pickers and the diff dock live outside the panel, so they switch with it.
      for (const c of controlSets) c.hidden = c.getAttribute("data-suite-controls") !== wanted;
      for (const d of docks) {
        const active = d.getAttribute("data-suite-dock") === wanted;
        d.toggleAttribute("data-inactive", !active);
        if (!active) d.hidden = true;
      }
      const paint = repaint.get(wanted);
      if (paint) paint();
    });
  }

  for (const panel of panels) {
    const suite = panel.getAttribute("data-suite-panel");
    wire(
      panel,
      document.querySelector('[data-suite-controls="' + suite + '"]'),
      document.querySelector('[data-suite-dock="' + suite + '"]'),
      ALL[suite]
    );
  }

  function wire(panel, controls, dock, DATA) {
    if (!DATA || !controls) return;
    const lower = new Set(DATA.lowerIsBetter);
    const continuous = new Set(DATA.continuous);
    const byId = new Map(DATA.evaluations.map((e) => [e.id, e]));
    const $ = (sel) => panel.querySelector(sel);
    const pickA = controls.querySelector('[data-role="pick-a"]');
    const pickB = controls.querySelector('[data-role="pick-b"]');
    if (!pickA || !pickB) return;

    const fmt = (scorer, v) => {
      if (v === null || v === undefined) return "–";
      if (scorer === "latency_ms") return (v / 1000).toFixed(1) + "s";
      if (continuous.has(scorer)) return v.toFixed(1);
      return v.toFixed(2);
    };
    const fmtDelta = (scorer, d) => {
      if (d === null) return "–";
      const sign = d > 0 ? "+" : "";
      if (scorer === "latency_ms") return sign + (d / 1000).toFixed(1) + "s";
      if (continuous.has(scorer)) return sign + d.toFixed(1);
      return sign + d.toFixed(2);
    };
    const mean = (xs) => (xs.length ? xs.reduce((x, y) => x + y, 0) / xs.length : null);

    // Flips are a 0-1 pass/fail notion. Applying it to a latency or an error rate produces
    // confident nonsense (a rate falling to zero read as "1 now fails").
    const flipsMeaningful = (scorer) => !continuous.has(scorer);

    // Paired per sample: only samples both evaluations graded can contribute a delta.
    function pairedDeltas(a, b, scorer) {
      const out = [];
      for (const sample of DATA.samples) {
        const av = a.cells[sample] && a.cells[sample][scorer];
        const bv = b.cells[sample] && b.cells[sample][scorer];
        if (!av || !bv) continue;
        out.push({ sample, a: mean(av), b: mean(bv) });
      }
      return out;
    }

    // Order matters. A pass -> fail flip is a fact about one sample; "the mean moved less
    // than the noise floor" is a statement about the average. Letting the second swallow
    // the first is how a real regression gets reported as "within noise".
    function verdict(scorer, delta, noise, fixed, broke) {
      if (delta === null) return { text: "no paired samples", tone: "neutral" };
      const signed = lower.has(scorer) ? -delta : delta;
      const beyond = noise === null || noise === undefined ? true : Math.abs(delta) > noise;
      if (broke && fixed) return { text: "mixed flips", tone: "warning" };
      if (broke) {
        return beyond && signed < 0
          ? { text: broke + " now fail", tone: "critical" }
          : { text: broke + " now fail (mean inside noise)", tone: "warning" };
      }
      if (fixed) {
        return beyond && signed > 0
          ? { text: fixed + " now pass", tone: "good" }
          : { text: fixed + " now pass (mean inside noise)", tone: "warning" };
      }
      if (Math.abs(delta) < 1e-9) return { text: "unchanged", tone: "neutral" };
      if (!beyond) return { text: "within noise", tone: "neutral" };
      return signed > 0 ? { text: "better", tone: "good" } : { text: "worse", tone: "critical" };
    }

    // "See all" surveys every candidate against the baseline; picking one narrows the grid
    // to that comparison. In survey mode a column click focuses one candidate without
    // leaving the survey, so you can read a verdict and a diff and then keep scanning.
    const ALL = "__all__";
    let focused = null;

    // An empty section is still a heading, a paragraph and a table of column labels. When
    // there is no pair to compare, the whole thing goes.
    const compareSection = panel.querySelector('section[id^="compare-"]');
    function setCompareVisible(visible) {
      if (compareSection) compareSection.hidden = !visible;
    }

    function render() {
      const a = byId.get(pickA.value);
      const survey = pickB.value === ALL;
      // Selecting the baseline as its own candidate is not a comparison. Treated as
      // "nothing focused" rather than rendering a column against itself, which produced a
      // single column marked as the candidate but painted as the baseline.
      const chosen = survey ? focused : pickB.value;
      const b = chosen && chosen !== a.id ? byId.get(chosen) : null;
      if (!a) return;

      document.body.classList.toggle("survey", survey);
      // Column visibility follows the mode. The baseline is always in the set: it is the
      // reference every other number is expressed against, so a grid without it is a grid
      // of deltas against nothing.
      const visible = survey ? null : new Set([a.id, pickB.value]);
      for (const el of panel.querySelectorAll("[data-eval]")) {
        el.classList.toggle("col-hidden", visible !== null && !visible.has(el.getAttribute("data-eval")));
      }

      if (!b) {
        // Surveying, nothing focused: the grid still shows every delta against the
        // baseline, but there is no single pair to write a verdict or a diff about. An
        // empty verdict chip over an empty table is worse than showing nothing.
        for (const role of ["verdict", "compare-meta", "compare-body"]) {
          const el = $('[data-role="' + role + '"]');
          if (el) el.innerHTML = "";
        }
        setCompareVisible(false);
        renderExec(a, null);
        diffHtml = "";
        diffTitle = "";
        paintDock();
        for (const el of panel.querySelectorAll("[data-eval]")) {
          el.classList.toggle("sel-a", el.getAttribute("data-eval") === a.id);
          el.classList.remove("sel-b");
        }
        renderGrid(a, a);
        renderChart(a, a);
        return;
      }
      setCompareVisible(true);

      let better = 0, worse = 0, quality = 0;
      const nowFail = new Set(), nowPass = new Set();
      const rows = [];
      for (const scorer of DATA.quality.concat(DATA.diagnostic)) {
        const pairs = pairedDeltas(a, b, scorer);
        const av = mean(pairs.map((x) => x.a));
        const bv = mean(pairs.map((x) => x.b));
        const delta = av === null || bv === null ? null : bv - av;
        const canFlip = flipsMeaningful(scorer);
        const fixed = canFlip ? pairs.filter((x) => x.a < DATA.passThreshold && x.b >= DATA.passThreshold) : [];
        const broke = canFlip ? pairs.filter((x) => x.a >= DATA.passThreshold && x.b < DATA.passThreshold) : [];
        const v = verdict(scorer, delta, DATA.noise[scorer], fixed.length, broke.length);
        const isQuality = DATA.quality.includes(scorer);

        if (isQuality) {
          quality++;
          const noise = DATA.noise[scorer];
          const beyond = delta !== null && (noise === null || noise === undefined || Math.abs(delta) > noise);
          const signed = delta === null ? 0 : lower.has(scorer) ? -delta : delta;
          if (beyond && signed > 0) better++;
          if (beyond && signed < 0) worse++;
          for (const x of broke) nowFail.add(x.sample);
          for (const x of fixed) nowPass.add(x.sample);
        }

        rows.push(
          '<tr class="' + (isQuality ? "" : "diag") + '">' +
            "<th scope=\"row\">" + scorer + (isQuality ? "" : ' <span class="tag">diagnostic</span>') + "</th>" +
            '<td class="num">' + fmt(scorer, av) + "</td>" +
            '<td class="num">' + fmt(scorer, bv) + "</td>" +
            '<td class="num strong">' + fmtDelta(scorer, delta) + "</td>" +
            '<td><span class="pill ' + v.tone + '">' + v.text + "</span></td>" +
            '<td class="num">' + pairs.length + "</td>" +
            '<td class="movers">' +
            (fixed.length ? '<span class="good">+' + fixed.length + " now pass</span> " : "") +
            (broke.length ? '<span class="critical">\u2212' + broke.length + " now fail</span>" : "") +
            (!fixed.length && !broke.length ? '<span class="muted">no flips</span>' : "") +
            "</td></tr>"
        );
      }

      const sameVariation = a.variationId === b.variationId;
      const names = (set) => Array.from(set).join(", ");
      let headline, tone;
      if (sameVariation) {
        headline =
          "Same variation on both sides \u2014 this measures the suite's noise, not a change." +
          (nowFail.size || nowPass.size
            ? " Even so, " + (nowFail.size + nowPass.size) + " sample(s) changed verdict, which is the noise."
            : "");
        tone = "warning";
      } else {
        const parts = [];
        parts.push(
          better || worse
            ? "ahead on " + better + " of " + quality + " quality scorers, behind on " + worse
            : "no quality scorer moved beyond the noise floor"
        );
        if (nowFail.size) parts.push(nowFail.size + " sample(s) now fail: " + names(nowFail));
        if (nowPass.size) parts.push(nowPass.size + " now pass: " + names(nowPass));
        headline = b.label + ": " + parts.join(" \u00b7 ");
        tone = nowFail.size || worse ? (better || nowPass.size ? "warning" : "critical")
             : (better || nowPass.size ? "good" : "neutral");
      }

      renderExec(a, b);

      const verdictEl = $('[data-role="verdict"]');
      verdictEl.className = "verdict-line pill " + tone;
      verdictEl.textContent = headline;
      $('[data-role="compare-body"]').innerHTML = rows.join("");
      $('[data-role="compare-meta"]').innerHTML =
        "<span><b>" + a.label + "</b> " + a.model + " \u00b7 " + a.runs.length + " runs \u00b7 " + a.when + "</span>" +
        '<span aria-hidden="true">\u2192</span>' +
        "<span><b>" + b.label + "</b> " + b.model + " \u00b7 " + b.runs.length + " runs \u00b7 " + b.when + "</span>";

      const key = [a.variationId, b.variationId].sort().join("||");
      const diff = DATA.diffs[key];
      diffTitle = a.label + " \u2192 " + b.label;
      diffHtml = sameVariation
        ? '<p class="note">Both sides are the same variation, so there is nothing to diff.</p>'
        : diff
          ? (diff.from !== a.variationId
              ? '<p class="note">Shown in stored order: ' + diff.from + " \u2192 " + diff.to + ".</p>"
              : "") + diff.html
          : '<p class="note">No diff recorded for this pair.</p>';
      paintDock();

      for (const el of panel.querySelectorAll("[data-eval]")) {
        const id = el.getAttribute("data-eval");
        el.classList.toggle("sel-a", id === a.id);
        el.classList.toggle("sel-b", id === b.id);
      }
      renderGrid(a, b);
      renderChart(a, b);
    }

    // Which evaluation is actually winning, in a sentence. Surveying, that means ranking
    // every candidate against the baseline on the primary scorer; focused, it is the
    // verdict for the chosen pair. Either way it names the evaluation, not just a number.
    function renderExec(a, b) {
      const el = $('[data-role="exec"]');
      if (!el) return;
      const scorer = DATA.quality[0];
      const noise = DATA.noise[scorer];
      const scoreOf = (ev) => {
        const pairs = pairedDeltas(a, ev, scorer);
        const av = mean(pairs.map((x) => x.a));
        const bv = mean(pairs.map((x) => x.b));
        return av === null || bv === null ? null : { delta: bv - av, value: bv };
      };

      if (b) {
        const r = scoreOf(b);
        if (!r) { el.innerHTML = ""; return; }
        const signed = lower.has(scorer) ? -r.delta : r.delta;
        const beyond = noise === null || noise === undefined || Math.abs(r.delta) > noise;
        const word = !beyond ? "is indistinguishable from" : signed > 0 ? "beats" : "loses to";
        el.innerHTML =
          '<p class="exec-line"><b>' + b.label + "</b> " + word + " <b>" + a.label + "</b> on " +
          scorer + ": " + fmt(scorer, r.value) + " vs " + fmt(scorer, r.value - r.delta) +
          " (" + fmtDelta(scorer, r.delta) + ")." + "</p>" +
          '<p class="exec-note">' +
          (beyond ? "Beyond the " : "Inside the ") +
          (noise === null || noise === undefined ? "unmeasured noise floor" : "±" + noise.toFixed(3) + " noise floor") +
          ".</p>";
        return;
      }

      const ranked = DATA.evaluations
        .filter((ev) => ev.id !== a.id && ev.comparable)
        .map((ev) => ({ ev, r: scoreOf(ev) }))
        .filter((x) => x.r !== null)
        .sort((x, y) => (lower.has(scorer) ? x.r.delta - y.r.delta : y.r.delta - x.r.delta));
      if (!ranked.length) { el.innerHTML = ""; return; }

      const best = ranked[0];
      const worst = ranked[ranked.length - 1];
      const beyond = noise === null || noise === undefined || Math.abs(best.r.delta) > noise;
      const lines = [
        '<p class="exec-line">Best so far: <b>' + best.ev.label + "</b> on " + scorer + " — " +
        fmt(scorer, best.r.value) + " (" + fmtDelta(scorer, best.r.delta) + " vs " + a.label + ")" +
        (beyond ? "." : ", which is inside the noise floor.") + "</p>",
      ];
      const worstSigned = lower.has(scorer) ? -worst.r.delta : worst.r.delta;
      if (worstSigned < 0 && (noise === null || Math.abs(worst.r.delta) > noise)) {
        lines.push(
          '<p class="exec-note">Worst: <b>' + worst.ev.label + "</b> " +
          fmtDelta(scorer, worst.r.delta) + ".</p>"
        );
      }
      lines.push(
        '<p class="exec-note">' + ranked.length + " candidates measured against " + a.label + ".</p>"
      );
      el.innerHTML = lines.join("");
    }

    // The hill is drawn once; the page moves the noise band onto the chosen baseline and
    // marks both selected points, so the graph answers the same question as everything
    // else on the page rather than a fixed one of its own.
    function renderChart(a, b) {
      const svg = panel.querySelector("svg.chart");
      if (!svg) return;
      const padT = parseFloat(svg.dataset.padT);
      const plotH = parseFloat(svg.dataset.plotH);
      const yOf = (v) => padT + (1 - Math.max(0, Math.min(1, v))) * plotH;
      const at = (id) => svg.querySelector('circle.pt[data-eval="' + cssEscape(id) + '"]');

      const band = svg.querySelector('[data-role="noise-band"]');
      const baseNode = at(a.id);
      if (band && baseNode) {
        const noise = parseFloat(band.dataset.noise);
        const value = parseFloat(baseNode.dataset.value);
        const top = yOf(value + noise);
        const bottom = yOf(value - noise);
        band.setAttribute("y", String(Math.min(top, bottom)));
        band.setAttribute("height", String(Math.abs(bottom - top)));
      }
      const sameNode = a.id === b.id;
      for (const [role, node] of [["a", baseNode], ["b", sameNode ? null : at(b.id)]]) {
        const guide = svg.querySelector('[data-role="guide-' + role + '"]');
        const label = svg.querySelector('[data-role="guide-' + role + '-label"]');
        for (const el of [guide, label]) {
          if (!el) continue;
          if (!node) { el.setAttribute("hidden", ""); continue; }
          const cx = node.getAttribute("cx");
          if (el === guide) { el.setAttribute("x1", cx); el.setAttribute("x2", cx); }
          else el.setAttribute("x", cx);
          el.removeAttribute("hidden");
        }
      }

      // The summary line names the baseline, so it has to follow the picker rather than
      // keep quoting whichever evaluation was recorded as the baseline on disk.
      const summary = panel.querySelector('[data-role="summary"]');
      if (summary) {
        const scorer = svg.dataset.scorer;
        const bNode = at(b.id);
        const parts = [];
        if (bNode) {
          parts.push("<b>" + scorer + "</b> " + fmt(scorer, parseFloat(bNode.dataset.value)) + " at " + b.label);
        }
        if (baseNode) {
          parts.push("baseline " + fmt(scorer, parseFloat(baseNode.dataset.value)) + " (" + a.label + ")");
        }
        const noise = DATA.noise[scorer];
        parts.push(noise === null || noise === undefined
          ? "noise floor not yet measured"
          : "noise \u00b1" + noise.toFixed(3));
        const totalRuns = DATA.evaluations.reduce((n, e) => n + e.runs.length, 0);
        parts.push(Object.keys(DATA.variations).length + " variations \u00b7 " +
                   DATA.evaluations.length + " evaluations \u00b7 " + totalRuns + " runs");
        summary.innerHTML = parts.join(" \u00b7 ");
      }
    }

    // Cell display mode. Once you have chosen a baseline, "how much better or worse than
    // it" is the question the grid exists to answer, so that is the default; absolute
    // scores stay one click away because you still need them to know where you are.
    let mode = "delta";
    const SEQ = 7;

    function seqClass(scorer, value) {
      const v = lower.has(scorer) ? 1 - value : value;
      return "s" + Math.min(SEQ - 1, Math.max(0, Math.round(Math.max(0, Math.min(1, v)) * (SEQ - 1))));
    }

    // Buckets are relative for measurements (a 2 s move in a 60 s task is not the same
    // event as a 2 s move in a 4 s one) and absolute for 0-1 scores.
    function divClass(scorer, delta, base) {
      const signed = lower.has(scorer) ? -delta : delta;
      const scale = continuous.has(scorer) ? Math.max(Math.abs(base), 1e-9) : 1;
      const size = Math.abs(delta) / scale;
      if (size <= (continuous.has(scorer) ? 0.02 : 0.01)) return "d0";
      const step = size <= 0.1 ? 1 : size <= 0.33 ? 2 : 3;
      return (signed > 0 ? "dp" : "dn") + step;
    }

    const CELL_CLASSES = ["d0", "dp1", "dp2", "dp3", "dn1", "dn2", "dn3", "is-baseline"]
      .concat(Array.from({ length: SEQ }, (_, i) => "s" + i))
      .concat(Array.from({ length: SEQ }, (_, i) => "g" + i));

    function paintCells(a, b) {
      const table = panel.querySelector("table.bigtable");
      if (!table) return;
      for (const row of table.querySelectorAll("tbody tr")) {
        const baseCell = row.querySelector('td.cell[data-eval="' + cssEscape(a.id) + '"]');
        const baseValue = baseCell && baseCell.dataset.value !== undefined
          ? parseFloat(baseCell.dataset.value)
          : null;
        for (const cell of row.querySelectorAll("td.cell[data-scorer]")) {
          const scorer = cell.dataset.scorer;
          // The per-sample cells wrap their number in a <span> (so a flaky dot and a
          // <title> can sit alongside); the mean rows write text directly. Writing blindly
          // to the span threw on every summary row.
          const target = cell.querySelector("span") || cell;
          cell.classList.remove(...CELL_CLASSES);
          if (cell.dataset.value === undefined) continue;   // not graded
          const value = parseFloat(cell.dataset.value);
          const isBase = cell.dataset.eval === a.id;

          if (mode === "absolute" || baseValue === null || isBase) {
            // The gold column is the reference; the blue ramp belongs to the candidates.
            cell.classList.add((isBase ? "g" : "s") + seqClass(scorer, value).slice(1));
            setText(target, fmt(scorer, value));
            continue;
          }
          const delta = value - baseValue;
          cell.classList.add(divClass(scorer, delta, baseValue));
          setText(target, Math.abs(delta) < 1e-9 ? "0" : fmtDelta(scorer, delta));
        }
      }
      panel.querySelectorAll("[data-legend]").forEach((el) => {
        el.hidden = el.getAttribute("data-legend") !== mode;
      });
      void b;
    }

    // Replaces only the text, leaving any <title> tooltip and flaky dot in place.
    function setText(el, text) {
      const node = Array.from(el.childNodes).find((n) => n.nodeType === Node.TEXT_NODE);
      if (node) node.nodeValue = text;
      else el.insertBefore(document.createTextNode(text), el.firstChild);
    }

    function cssEscape(value) {
      return window.CSS && CSS.escape ? CSS.escape(value) : value.replace(/"/g, '\\"');
    }

    // The change/reading columns answer "how did this sample move between the two
    // evaluations you are comparing", so they follow the picker.
    // A `display:none` column still claims its share under fixed table layout, so hiding
    // seven of nine evaluations left the two on screen as narrow as if all nine were
    // there. The visible columns are sized explicitly instead: they take the width the
    // hidden ones are not using, down to a floor that makes a long hill scroll.
    function sizeColumns(table) {
      const heads = Array.from(table.querySelectorAll("thead th[data-eval]"));
      const shown = heads.filter((h) => !h.classList.contains("col-hidden"));
      if (!shown.length) return;
      const fixed = Number(table.dataset.fixed);
      const floor = Number(table.dataset.evalmin);
      const box = table.closest(".bigscroll");
      const available = (box ? box.clientWidth : 0) - fixed - 4;
      const each = Math.max(floor, Math.floor(available / shown.length));
      for (const h of heads) h.style.width = h.classList.contains("col-hidden") ? "" : each + "px";
      table.style.minWidth = fixed + each * shown.length + "px";
    }

    function renderGrid(a, b) {
      const table = panel.querySelector("table.bigtable");
      if (!table) return;
      sizeColumns(table);
      // The dock and the sticky bar settle after this frame; re-measure once they have.
      requestAnimationFrame(() => sizeColumns(table));
      for (const row of table.querySelectorAll("tbody tr[data-scorer]")) {
        const scorer = row.getAttribute("data-scorer");
        const sample = row.getAttribute("data-sample");
        const changeCell = row.querySelector('[data-role="change"]');
        const readingCell = row.querySelector('[data-role="reading"]');
        const av = a.cells[sample] && a.cells[sample][scorer];
        const bv = b.cells[sample] && b.cells[sample][scorer];

        if (!av || !bv) {
          changeCell.textContent = "–";
          readingCell.innerHTML = '<span class="pill neutral">not in both</span>';
          continue;
        }
        const am = mean(av), bm = mean(bv), delta = bm - am;
        const flaky = [av, bv].some(
          (vs) => vs.some((v) => v >= DATA.passThreshold) && vs.some((v) => v < DATA.passThreshold)
        );
        changeCell.textContent = a.id === b.id ? "\u2014" : fmtDelta(scorer, delta);

        // Flakiness is a caveat on a reading, not a substitute for one: labelling a -0.67
        // move "flaky" hid the largest regression on the page behind a shrug.
        let state;
        if (a.id === b.id) {
          state = bm >= DATA.passThreshold ? ["passing", "good"]
                : bm <= 0.001 ? ["failing", "critical"] : ["partial credit", "serious"];
        } else if (flipsMeaningful(scorer) && am < DATA.passThreshold && bm >= DATA.passThreshold) {
          state = ["now passes", "good"];
        } else if (flipsMeaningful(scorer) && am >= DATA.passThreshold && bm < DATA.passThreshold) {
          state = ["now fails", "critical"];
        } else if (Math.abs(delta) <= 0.01) {
          state = bm >= DATA.passThreshold ? ["passing", "good"] : ["unchanged", "neutral"];
        } else {
          const signed = lower.has(scorer) ? -delta : delta;
          state = signed > 0 ? ["improved", "good"] : ["regressed", "critical"];
        }
        readingCell.innerHTML =
          '<span class="pill ' + state[1] + '">' + state[0] + "</span>" +
          (flaky ? ' <span class="pill warning">flaky</span>' : "");
      }
      for (const row of table.querySelectorAll("tbody tr.summary")) {
        row.querySelector('[data-role="change"]').textContent = "";
        row.querySelector('[data-role="reading"]').innerHTML = "";
      }
      paintCells(a, b);
    }

    for (const btn of panel.querySelectorAll("button.mode")) {
      btn.addEventListener("click", () => {
        mode = btn.getAttribute("data-mode");
        for (const other of panel.querySelectorAll("button.mode")) {
          other.setAttribute("aria-pressed", String(other === btn));
        }
        render();
      });
    }

    const bigtable = panel.querySelector("table.bigtable");
    if (bigtable) {
      window.addEventListener("resize", () => sizeColumns(bigtable));
      bigtable.addEventListener("resize-columns", () => sizeColumns(bigtable));
    }

    // The side panel is the "context for what you clicked" surface. By default that is the
    // diff between the two evaluations being compared; clicking a run cell swaps it for
    // that run's result, chat and trace. One panel, because they answer the same question
    // at different depths and would compete for the same space.
    let dockMode = "diff";
    let dockDetail = null;
    let diffHtml = "";
    let diffTitle = "";

    const dockModeButtons = controls.querySelectorAll("[data-dock-mode]");

    function diffAvailable() {
      return Boolean(diffHtml);
    }

    function setDockMode(mode) {
      dockMode = mode === "detail" ? "detail" : "diff";
      paintDock();
    }

    function paintDock() {
      if (!dock) return;
      // Diff needs two evaluations; traces need a clicked run. Whichever is unavailable is
      // disabled rather than hidden, so the panel never silently ignores a click.
      if (dockMode === "diff" && !diffAvailable()) dockMode = "detail";
      for (const btn of dockModeButtons) {
        const mode = btn.getAttribute("data-dock-mode");
        btn.setAttribute("aria-pressed", String(mode === dockMode));
        btn.disabled = mode === "diff" && !diffAvailable();
        btn.title =
          mode === "diff" && !diffAvailable()
            ? "Pick a specific candidate to see a diff"
            : "";
      }

      const heading = dock.querySelector('[data-role="dock-heading"]');
      const sub = dock.querySelector('[data-role="diff-title"]');
      const body = dock.querySelector('[data-role="diff-body"]');

      if (dockMode === "detail") {
        heading.textContent = "Trace";
        sub.textContent = dockDetail ? dockDetail.title : "";
        body.innerHTML = dockDetail
          ? dockDetail.html
          : '<p class="note">Click any cell in <b>Runs</b> to load that run’s result, chat and trace.</p>';
      } else {
        heading.textContent = "Diff";
        sub.textContent = diffTitle;
        body.innerHTML = diffHtml;
      }

      for (const btn of panel.querySelectorAll("[data-open-detail]")) {
        btn.classList.toggle(
          "active",
          dockMode === "detail" &&
            dockDetail !== null &&
            btn.dataset.eval === dockDetail.evId &&
            btn.dataset.sample === dockDetail.sample &&
            btn.dataset.run === String(dockDetail.run)
        );
      }
      setDockPresent(true);
    }

    // Clicking a cell of the sample x run table opens that run in the side panel.
    // The data is already in the page; this only decides what to show.
    const escHtml = (v) =>
      String(v == null ? "" : v).replace(/[&<>"]/g, (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]
      );

    function renderDetail(evId, sample, runIndex) {
      const ev = byId.get(evId);
      if (!ev) return;
      const run = ev.runs.find((r) => String(r.index) === String(runIndex));
      if (!run) return;

      const readings = (run.readings || {})[sample] || {};
      const detail = (run.detail || {})[sample] || {};

      const results = Object.keys(readings)
        .sort()
        .map((scorer) => {
          const r = readings[scorer];
          const bits = [];
          if (r.answer) bits.push('<span class="answer-tag">' + escHtml(r.answer) + "</span> ");
          if (r.explanation) bits.push('<span class="explanation">' + escHtml(r.explanation) + "</span>");
          for (const c of r.criteria || []) {
            bits.push(
              '<div class="criterion ' + (c.met ? "good" : "critical") + '">' + (c.met ? "✓" : "✗") +
              " <b>" + escHtml(c.id) + "</b> <span class=\"muted\">" +
              escHtml(String(c.evidence || "").slice(0, 220)) + "</span></div>"
            );
          }
          for (const c of r.checks || []) {
            bits.push(
              '<div class="criterion ' + (c.ok ? "good" : "critical") + '">' + (c.ok ? "✓" : "✗") +
              " " + escHtml(c.check) + ' <span class="muted">' + escHtml(c.detail) + "</span></div>"
            );
          }
          if ((r.missing || []).length) {
            bits.push('<div class="criterion critical">✗ missing: ' + escHtml(r.missing.join(", ")) + "</div>");
          }
          if (!bits.length) return "";
          return (
            '<div class="reading-row"><div class="reading-head"><code>' + escHtml(scorer) +
            '</code><span class="reading-value">' +
            (r.excluded ? "excluded" : fmt(scorer, r.value)) + "</span></div>" +
            '<div class="reading-body">' + bits.join("") + "</div></div>"
          );
        })
        .join("");

      const chat = (detail.chat || [])
        .map(
          (t) =>
            '<div class="turn ' + escHtml(t.role) + '"><div class="turn-head">' +
            escHtml(t.role.toUpperCase()) +
            (t.origin ? ' <span class="muted">' + escHtml(t.origin) + "</span>" : "") +
            (t.latency ? ' <span class="muted">' + (t.latency / 1000).toFixed(1) + "s</span>" : "") +
            "</div><pre>" + escHtml(t.text) + "</pre></div>"
        )
        .join("");

      const steps = (detail.steps || [])
        .map(
          (s) =>
            "<tr><td><code>" + escHtml(s.name) + "</code></td>" +
            '<td class="muted">' + escHtml(s.subagent) + "</td>" +
            '<td class="muted">' + escHtml(s.status) + "</td>" +
            '<td class="muted">' + escHtml(s.source) + "</td>" +
            '<td class="muted">' + escHtml(s.detail) + "</td></tr>"
        )
        .join("");
      const traceBits = [];
      if ((detail.subagents || []).length) {
        traceBits.push('<p class="note">sub-agents: ' + escHtml(detail.subagents.join(", ")) + "</p>");
      }
      if (steps) {
        traceBits.push(
          '<div class="scroll"><table class="steps"><thead><tr><th>step</th><th>sub-agent</th>' +
          "<th>status</th><th>source</th><th>detail</th></tr></thead><tbody>" + steps + "</tbody></table></div>"
        );
      }
      for (const i of detail.interrupts || []) {
        traceBits.push(
          '<p class="note">approval requested for <code>' + escHtml(i.tool) + "</code> — " +
          escHtml(i.title) + " (" + escHtml(i.decision || "no decision") + ")</p>"
        );
      }
      if (detail.totalMs) traceBits.push('<p class="note">total ' + (detail.totalMs / 1000).toFixed(1) + "s</p>");
      for (const k of detail.infra || []) traceBits.push('<p class="note warn">infra error: ' + escHtml(k) + "</p>");
      for (const n of detail.notes || []) traceBits.push('<p class="note warn">' + escHtml(n) + "</p>");

      const sections = [];
      if (results) {
        sections.push('<details class="sub" open><summary>Result</summary>' + results + "</details>");
      }
      if (chat) {
        sections.push(
          '<details class="sub" open><summary>Chat</summary><div class="chat">' + chat + "</div></details>"
        );
      }
      if (traceBits.length) {
        sections.push('<details class="sub"><summary>Trace</summary>' + traceBits.join("") + "</details>");
      }
      if (!sections.length) {
        sections.push('<p class="note">Nothing recorded for this run.</p>');
      }

      dockDetail = {
        evId: evId,
        sample: sample,
        run: runIndex,
        title: sample + " · run " + runIndex + " of " + ev.label,
        html: sections.join(""),
      };
      dockMode = "detail";
      if (document.body.classList.contains("dock-closed")) setDock(true);
      paintDock();
    }

    panel.addEventListener("click", (event) => {
      const open = event.target.closest("[data-open-detail]");
      if (open) renderDetail(open.dataset.eval, open.dataset.sample, open.dataset.run);
    });
    for (const btn of dockModeButtons) {
      btn.addEventListener("click", () => setDockMode(btn.getAttribute("data-dock-mode")));
    }
    repaint.set(panel.getAttribute("data-suite-panel"), paintDock);

    // Changing what is being compared returns the panel to the diff: the run you had open
    // belongs to the previous question.
    const backToDiff = () => { dockMode = "diff"; };
    pickA.addEventListener("change", () => { backToDiff(); render(); });
    pickB.addEventListener("change", () => { focused = null; backToDiff(); render(); });

    // Column click focuses a candidate while surveying.
    for (const th of panel.querySelectorAll("thead th[data-eval]")) {
      th.tabIndex = 0;
      const choose = () => {
        if (pickB.value !== ALL) return;
        const id = th.getAttribute("data-eval");
        focused = id === pickA.value ? null : id;
        dockMode = "diff";
        render();
      };
      th.addEventListener("click", choose);
      th.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); }
      });
    }
    for (const btn of panel.querySelectorAll("[data-set-a]")) {
      btn.addEventListener("click", () => { pickA.value = btn.getAttribute("data-set-a"); render(); });
    }
    render();
  }
})();
