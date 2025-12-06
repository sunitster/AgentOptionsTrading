// src/dashboard/static/app.js
// WebSocket-first dashboard client (WS forced to 127.0.0.1:8000/ws)
// Expects server broadcast payload shaped like:
// {
//   server_time: "...",
//   snapshot: [...],
//   broker_state: {...},
//   pnl_history: [...],
//   risk_state: {...},
//   current_position: {...},
//   ltp_times: [...],
//   ltp_prices: [...]
// }

(function () {
  "use strict";

  const WS_HOST = "127.0.0.1";
  const WS_PORT = 8000;
  const WS_PATH = "/ws";
  const WS_URL = `ws://${WS_HOST}:${WS_PORT}${WS_PATH}`;

  const FORCE_EXIT_URL = "/api/force-exit";

  // reconnect/backoff config
  const RECONNECT_BASE_MS = 500; // initial
  const RECONNECT_MAX_MS = 30_000; // cap
  const RECONNECT_JITTER = 0.3; // jitter fraction

  // liveness
  let ws = null;
  let wsConnected = false;
  let reconnectAttempts = 0;
  let reconnectTimer = null;

  // rendering helpers and state
  let ltp_history = []; // {t,p}
  const LTP_MAX_POINTS = 500;

  function formatCurrency(num) {
    if (num === null || num === undefined || Number.isNaN(Number(num))) return "₹0.00";
    return "₹" + Number(num).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function showToast(msg) {
    const t = document.getElementById("toast");
    if (!t) return;
    t.textContent = msg;
    t.classList.add("show");
    setTimeout(() => t.classList.remove("show"), 2000);
  }

  // -------------------- rendering functions (preserve original behavior) --------------------
  // All functions are defensive and will never throw.

  function renderState(data) {
    try {
      const serverTimeEl = document.getElementById("server-time");
      if (serverTimeEl) serverTimeEl.textContent = data.server_time || "—";

      const metrics = data.metrics || {};
      const pnl = Number(metrics.net_pnl || 0);

      const startingEl = document.getElementById("metric-starting");
      const capitalEl = document.getElementById("metric-capital");
      const pnlEl = document.getElementById("metric-pnl");
      const tradesEl = document.getElementById("metric-trades");

      if (startingEl) startingEl.textContent = formatCurrency(metrics.starting_balance || 0);
      if (capitalEl) capitalEl.textContent = formatCurrency(metrics.capital || 0);
      if (pnlEl) {
        pnlEl.textContent = formatCurrency(pnl);
        pnlEl.classList.toggle("card-positive", pnl >= 0);
        pnlEl.classList.toggle("card-negative", pnl < 0);
      }
      if (tradesEl) tradesEl.textContent = metrics.trades_count || 0;

      renderPnLChart(data.pnl_history || []);
      renderRiskState(data.risk_state || {});
      renderTrades(data.trades || []);
      renderSnapshot(data.snapshot || []);
    } catch (e) {
      console.error("renderState error", e);
    }
  }

  function renderATM(atm) {
    try {
      const el = document.getElementById("atm-card");
      if (!el) return;
      el.textContent = atm && atm.atm ? `ATM: ${atm.atm} (strikes: ${atm.count_strikes || 0})` : "No ATM data";
    } catch (e) {
      console.error("renderATM", e);
    }
  }

  function renderIC(ic) {
    try {
      const el = document.getElementById("ic-card");
      if (!el) return;
      el.textContent = ic && ic.position ? JSON.stringify(ic.position, null, 2) : "No open IC.";
    } catch (e) {
      console.error("renderIC", e);
    }
  }

  function renderGreeks(g) {
    try {
      const el = document.getElementById("greeks-card");
      if (!el) return;
      el.textContent = g && g.greeks ? JSON.stringify(g.greeks, null, 2) : "—";
      if (g && g.greeks_timeseries && g.greeks_timeseries.length) {
        drawGreeksChart(g.greeks_timeseries);
      } else {
        clearCanvas("greeks-canvas");
      }
    } catch (e) {
      console.error("renderGreeks", e);
    }
  }

  function renderMargin(m) {
    try {
      const txt = document.getElementById("margin-text");
      const usedEl = document.getElementById("margin-used");
      if (!txt || !usedEl) return;
      if (m && m.margin) {
        txt.textContent = JSON.stringify(m.margin);
        const used = parseFloat(m.margin.used_estimate || m.margin.used || 0) || 0;
        const avail = parseFloat(m.margin.available || m.margin.free || m.margin.total || 0) || 0;
        let pct = 0;
        if (avail > 0 || used > 0) pct = Math.min(100, Math.round((used / Math.max(1, used + avail)) * 100));
        usedEl.style.width = pct + "%";
      } else {
        txt.textContent = "—";
        usedEl.style.width = "0%";
      }
    } catch (e) {
      console.error("renderMargin", e);
    }
  }

  function renderRiskState(risk) {
    try {
      const el = document.getElementById("risk-state");
      if (!el) return;
      el.textContent = risk && Object.keys(risk).length ? JSON.stringify(risk, null, 2) : "No risk state yet.";
    } catch (e) {
      console.error("renderRiskState", e);
    }
  }

  function renderTrades(trades) {
    try {
      const tbody = document.querySelector("#trades-table tbody");
      if (!tbody) return;
      tbody.innerHTML = "";
      (trades || []).forEach((t) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${t.time || ""}</td><td>${t.mode || ""}</td><td>${formatCurrency(t.pnl || 0)}</td><td>${formatCurrency(t.balance || 0)}</td>`;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.error("renderTrades", e);
    }
  }

  function renderSnapshot(snapshot) {
    try {
      const tbody = document.querySelector("#snapshot-table tbody");
      if (!tbody) return;
      tbody.innerHTML = "";
      (snapshot || []).slice(0, 80).forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${r.tradingsymbol || ""}</td><td>${r.expiry || ""}</td><td>${r.strike || ""}</td><td>${r.instrument_type || ""}</td><td>${r.ltp || ""}</td>`;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.error("renderSnapshot", e);
    }
  }

  // PnL chart
  function renderPnLChart(pnlHistory) {
    try {
      const canvas = document.getElementById("pnl-chart");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!pnlHistory || !pnlHistory.length) {
        ctx.fillStyle = "#9ca3af";
        ctx.font = "12px system-ui";
        ctx.fillText("No PnL history yet.", 10, 20);
        return;
      }
      const values = pnlHistory.map((p) => Number(p.pnl || 0));
      const n = values.length;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const padding = 12;
      const w = canvas.width;
      const h = canvas.height;
      const xStep = n > 1 ? (w - 2 * padding) / (n - 1) : 0;
      const range = max - min || 1;
      ctx.strokeStyle = "#374151";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, h / 2);
      ctx.lineTo(w - padding, h / 2);
      ctx.stroke();
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = padding + i * xStep;
        const norm = (v - min) / range;
        const y = h - padding - norm * (h - 2 * padding);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    } catch (e) {
      console.error("renderPnLChart", e);
    }
  }

  // LTP chart using snapshot
  function renderLTPChart(ltp) {
    try {
      const canvas = document.getElementById("ltp-chart");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      let snapshot = null;
      if (ltp && Array.isArray(ltp)) snapshot = ltp;
      else if (ltp && ltp.snapshot && Array.isArray(ltp.snapshot)) snapshot = ltp.snapshot;
      else if (ltp && ltp.rows && Array.isArray(ltp.rows)) snapshot = ltp.rows;

      if (!snapshot || !snapshot.length) {
        ctx.fillStyle = "#9ca3af";
        ctx.font = "12px system-ui";
        ctx.fillText("No LTP data yet.", 10, 20);
        return;
      }

      const snap = snapshot.filter((r) => r && r.strike !== undefined).slice(0, 40);
      const values = snap.map((s) => Number(s.ltp || 0));
      if (!values.length) {
        ctx.fillStyle = "#9ca3af";
        ctx.font = "12px system-ui";
        ctx.fillText("No LTP numeric data.", 10, 20);
        return;
      }

      const n = values.length;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const padding = 10;
      const w = canvas.width;
      const h = canvas.height;
      const xStep = n > 1 ? (w - 2 * padding) / (n - 1) : 0;
      const range = max - min || 1;

      ctx.strokeStyle = "#374151";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, h / 2);
      ctx.lineTo(w - padding, h / 2);
      ctx.stroke();

      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = padding + i * xStep;
        const norm = (v - min) / range;
        const y = h - padding - norm * (h - 2 * padding);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    } catch (e) {
      console.error("renderLTPChart", e);
    }
  }

  // Heatmap / ladder
  function renderHeatmap(heat) {
    try {
      const container = document.getElementById("heatmap-ladder");
      if (!container) return;
      container.innerHTML = "";
      if (!heat || !heat.rows || !heat.rows.length) {
        container.innerHTML = '<div class="small">No chain available.</div>';
        return;
      }
      heat.rows.forEach((row) => {
        const div = document.createElement("div");
        div.className = "ladder-row";
        const ce = row.CE ? row.CE.ltp || 0 : "";
        const pe = row.PE ? row.PE.ltp || 0 : "";
        const ceColor = row.CE && row.CE.oi ? heatColor(row.CE.oi) : "";
        const peColor = row.PE && row.PE.oi ? heatColor(row.PE.oi) : "";
        div.innerHTML = `<div class="ce" style="background:${ceColor ? ceColor : "transparent"}">${row.CE ? (row.CE.tradingsymbol || "") + " " + ce : ""}</div>
                         <div class="strike">${row.strike}</div>
                         <div class="pe" style="background:${peColor ? peColor : "transparent"}">${row.PE ? (row.PE.tradingsymbol || "") + " " + pe : ""}</div>`;
        container.appendChild(div);
      });
    } catch (e) {
      console.error("renderHeatmap", e);
    }
  }
  function heatColor(n) {
    const v = Math.min(1, Math.log10(1 + Math.max(0, n)) / 4);
    if (v < 0.33) return "rgba(255,235,59,0.06)";
    if (v < 0.66) return "rgba(255,159,67,0.08)";
    return "rgba(239,68,68,0.12)";
  }

  function drawGreeksChart(points) {
    try {
      const out = document.getElementById("greeks-card");
      if (!out) return;
      const last = points && points.length ? points[points.length - 1] : null;
      out.textContent = last ? JSON.stringify(last, null, 2) : "—";
    } catch (e) {
      console.error("drawGreeksChart", e);
    }
  }

  function clearCanvas(id) {
    try {
      const c = document.getElementById(id);
      if (!c) return;
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
    } catch (e) {
      console.error("clearCanvas", e);
    }
  }

  // -------------------- LTP history helper --------------------
  function appendLtpPoint(spot, ts) {
    try {
      if (spot === null || spot === undefined) return;
      const t = ts || new Date().toISOString();
      ltp_history.push({ t, p: Number(spot) });
      if (ltp_history.length > LTP_MAX_POINTS) {
        ltp_history.splice(0, ltp_history.length - LTP_MAX_POINTS);
      }
    } catch (e) {
      console.error("appendLtpPoint", e);
    }
  }

  // -------------------- control (force exit) --------------------
  async function sendForceExit() {
    try {
      if (wsConnected && ws) {
        ws.send(JSON.stringify({ cmd: "force-exit" }));
        showToast("Exit signal sent (WS)");
        return;
      }
      // fallback
      const res = await fetch(FORCE_EXIT_URL, { method: "POST", headers: { "Content-Type": "application/json" } });
      if (!res.ok) throw new Error("HTTP " + res.status);
      showToast("Exit signal sent (HTTP)");
    } catch (e) {
      console.error("sendForceExit", e);
      showToast("Failed to send exit");
    }
  }

  // -------------------- WebSocket behavior --------------------
  function markWsStatus(connected) {
    wsConnected = !!connected;
    // update UI badge (index.html listens for this event)
    try {
      window.dispatchEvent(new CustomEvent("ws-status", { detail: { connected: !!connected } }));
    } catch (e) {
      // ignore
    }
  }

  function startWebSocket() {
    // if already have ws, close it first
    try {
      if (ws) {
        try {
          ws.close();
        } catch (e) {}
        ws = null;
      }
    } catch (e) {}

    try {
      ws = new WebSocket(WS_URL);
    } catch (e) {
      console.warn("WebSocket construction failed:", e);
      scheduleReconnect();
      return;
    }

    ws.onopen = function () {
      reconnectAttempts = 0;
      markWsStatus(true);
      console.info("WS connected ->", WS_URL);
      showToast("WS connected");
    };

    ws.onmessage = function (ev) {
      // parse and map the payload safely
      try {
        const payload = JSON.parse(ev.data);

        // payload fields: server_time, snapshot, broker_state, pnl_history, risk_state, current_position, ltp_times, ltp_prices
        const server_time = payload.server_time || new Date().toISOString();
        const snapshot = Array.isArray(payload.snapshot) ? payload.snapshot : [];
        const broker_state = payload.broker_state && typeof payload.broker_state === "object" ? payload.broker_state : {};
        const pnl_history = Array.isArray(payload.pnl_history) ? payload.pnl_history : [];
        const risk_state = payload.risk_state && typeof payload.risk_state === "object" ? payload.risk_state : {};
        const current_position = payload.current_position && typeof payload.current_position === "object" ? payload.current_position : {};
        const ltp_times = Array.isArray(payload.ltp_times) ? payload.ltp_times : [];
        const ltp_prices = Array.isArray(payload.ltp_prices) ? payload.ltp_prices : [];

        // Update LTP timeseries if included
        if (ltp_times.length && ltp_prices.length && ltp_times.length === ltp_prices.length) {
          ltp_history = [];
          for (let i = 0; i < ltp_times.length; i++) {
            ltp_history.push({ t: ltp_times[i], p: Number(ltp_prices[i] || 0) });
          }
        } else {
          // fallback: try to pick spot from first snapshot row
          if (snapshot.length > 0 && snapshot[0]) {
            const first = snapshot[0];
            const spot = first.spot || first.ltp || null;
            const ts = first.timestamp || first.time || server_time;
            if (spot !== null && spot !== undefined) appendLtpPoint(spot, ts);
          }
        }

        // Build a state-like object compatible with renderState()
        const metrics = {
          starting_balance: Number(broker_state.starting_balance || 0),
          capital: Number(broker_state.capital || broker_state.starting_balance || 0),
          net_pnl: Number(broker_state.pnl || 0),
          trades_count: Array.isArray(broker_state.trades) ? broker_state.trades.length : 0,
        };

        // Build flat trades for table: try to map common shapes safely
        const flat_trades = Array.isArray(broker_state.trades)
          ? broker_state.trades.map((t) => ({
              time: t.time || t.timestamp || "",
              mode: t.mode || "",
              pnl: (t.ic && t.ic.pnl) || t.pnl || 0,
              balance: t.balance || 0,
            }))
          : [];

        const statePayload = {
          server_time: server_time,
          metrics: metrics,
          pnl_history: pnl_history,
          trades: flat_trades,
          snapshot: snapshot,
          risk_state: risk_state,
          current_position: current_position,
        };

        // call renderers
        renderState(statePayload);

        // Extra renders (ATM, IC, Greeks, Margin, Heatmap, LTP) using safe adapters
        try {
          // ATM: we don't compute here; let the UI keep showing placeholder or we can compute via snapshot if desired
          renderATM({ atm: null, count_strikes: 0, underlying: null });
        } catch (e) {}
        try {
          renderIC({ position: current_position });
        } catch (e) {}
        try {
          renderGreeks({
            greeks: current_position ? current_position.greeks || {} : {},
            greeks_timeseries: current_position ? current_position.greeks_timeseries || [] : [],
          });
        } catch (e) {}
        try {
          renderMargin({ margin: broker_state.margin || current_position.margin || {} });
        } catch (e) {}
        try {
          renderHeatmap({ rows: snapshot });
        } catch (e) {}
        try {
          // render LTP chart with snapshot (the function is defensive)
          renderLTPChart({ snapshot: snapshot });
        } catch (e) {}
        // update last-update UI (index.html shows server time via renderState)
        const lastUpdateEl = document.getElementById("last-update");
        if (lastUpdateEl) lastUpdateEl.textContent = new Date().toLocaleString();
      } catch (err) {
        console.error("WS msg parse error", err, ev.data);
      }
    };

    ws.onerror = function (err) {
      console.error("WS error", err);
      // onerror may be followed by onclose - let onclose handle reconnect
    };

    ws.onclose = function (ev) {
      console.warn("WS closed", ev);
      markWsStatus(false);
      ws = null;
      showToast("WS disconnected — attempting reconnect");
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    // clear existing
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    reconnectAttempts = Math.min(1000, reconnectAttempts + 1);
    let backoff = Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * Math.pow(1.8, reconnectAttempts));
    // jitter
    const jitter = backoff * RECONNECT_JITTER * (Math.random() * 2 - 1);
    backoff = Math.max(200, Math.floor(backoff + jitter));
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      startWebSocket();
    }, backoff);
    console.info("Scheduled WS reconnect in", backoff, "ms (attempt)", reconnectAttempts);
  }

  // -------------------- Init wiring --------------------
  function setupUI() {
    try {
      const forceBtn = document.getElementById("force-exit-btn");
      if (forceBtn) forceBtn.addEventListener("click", sendForceExit);

      // allow manual reconnect button if present
      const refreshBtn = document.getElementById("refresh-btn");
      if (refreshBtn) refreshBtn.addEventListener("click", () => {
        if (wsConnected && ws) {
          // ask server to send immediate data by closing/reconnecting the ws quickly (simple)
          try { ws.send(JSON.stringify({ cmd: "ping" })); } catch (e) {}
          showToast("Requested update (WS)");
        } else {
          showToast("WS not connected; waiting for reconnect...");
        }
      });
    } catch (e) {
      console.error("setupUI", e);
    }
  }

  // -------------------- start --------------------
  function start() {
    setupUI();
    markWsStatus(false);
    startWebSocket();
    // nothing else — no HTTP polling to avoid /api/* spam
  }

  // Expose a tiny debug hook if needed
  window.__dashboard_client = {
    getWsUrl: () => WS_URL,
    isWsConnected: () => wsConnected,
    reconnectNow: () => {
      reconnectAttempts = 0;
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      startWebSocket();
    },
    sendDebug: (obj) => {
      if (wsConnected && ws) ws.send(JSON.stringify(obj));
    }
  };

  // Start on load
  window.addEventListener("load", start);
})();
