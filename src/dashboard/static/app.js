// src/dashboard/static/app.js
(function(){
    const STATE_URL = "/api/state";
    const ATM_URL = "/api/atm";
    const IC_URL = "/api/ic-position";
    const GREEKS_URL = "/api/greeks";
    const MARGIN_URL = "/api/margin";
    const HEATMAP_URL = "/api/heatmap";
    const LTP_URL = "/api/ltp-chart";
    const FORCE_EXIT_URL = "/api/force-exit";
    
    let autoRefreshInterval = null;
    const REFRESH_SECONDS = 5;
    
    function formatCurrency(num){ if(num===null||num===undefined||isNaN(num)) return "₹0.00"; return "₹"+Number(num).toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2}); }
    function showToast(msg){ const t=document.getElementById('toast'); t.textContent=msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),2000); }
    
    async function fetchState(){
      try{
        const res = await fetch(STATE_URL);
        if(!res.ok) throw new Error('HTTP '+res.status);
        const data = await res.json();
        renderState(data);
      }catch(e){ console.error(e); showToast('Failed to load state'); }
    }
    
    async function fetchExtras(){
      try{
        const [atmRes, icRes, greeksRes, marginRes, heatmapRes, ltpRes] = await Promise.all([
          fetch(ATM_URL), fetch(IC_URL), fetch(GREEKS_URL), fetch(MARGIN_URL), fetch(HEATMAP_URL), fetch(LTP_URL)
        ]);
        const atm = atmRes.ok?await atmRes.json():{};
        const ic = icRes.ok?await icRes.json():{};
        const greeks = greeksRes.ok?await greeksRes.json():{};
        const margin = marginRes.ok?await marginRes.json():{};
        const heatmap = heatmapRes.ok?await heatmapRes.json():{};
        const ltp = ltpRes.ok?await ltpRes.json():{};
        renderATM(atm); renderIC(ic); renderGreeks(greeks); renderMargin(margin); renderHeatmap(heatmap); renderLTPChart(ltp);
      }catch(e){ console.error('extras error',e); }
    }
    
    function renderState(data){
      document.getElementById('server-time').textContent = data.server_time||'—';
      const metrics = data.metrics||{};
      const pnl = metrics.net_pnl||0;
      document.getElementById('metric-starting').textContent = formatCurrency(metrics.starting_balance||0);
      document.getElementById('metric-capital').textContent = formatCurrency(metrics.capital||0);
      const pnlEl=document.getElementById('metric-pnl');
      pnlEl.textContent = formatCurrency(pnl);
      pnlEl.classList.toggle('card-positive', pnl>=0);
      pnlEl.classList.toggle('card-negative', pnl<0);
      document.getElementById('metric-trades').textContent = metrics.trades_count||0;
      renderPnLChart(data.pnl_history||[]);
      renderRiskState(data.risk_state||{});
      renderTrades(data.trades||[]);
      renderSnapshot(data.snapshot||[]);
    }
    
    function renderATM(atm){
      const el = document.getElementById('atm-card');
      el.textContent = atm.atm ? ('ATM: ' + atm.atm + ' (strikes: ' + (atm.count_strikes||0) + ')') : 'No ATM data';
    }
    
    function renderIC(ic){
      const el = document.getElementById('ic-card');
      el.textContent = ic.position ? JSON.stringify(ic.position, null, 2) : 'No open IC.';
    }
    
    function renderGreeks(g){
      const el = document.getElementById('greeks-card');
      el.textContent = g.greeks ? JSON.stringify(g.greeks, null, 2) : '—';
      // draw greeks timeseries if present
      if(g.greeks_timeseries && g.greeks_timeseries.length){
        drawGreeksChart(g.greeks_timeseries);
      } else {
        clearCanvas('greeks-canvas'); // optional if future canvas added
      }
    }
    
    function renderMargin(m){
      const el = document.getElementById('margin-card');
      const txt = document.getElementById('margin-text');
      const usedEl = document.getElementById('margin-used');
      if(m.margin){
        txt.textContent = JSON.stringify(m.margin);
        // compute percent if we have used/available
        const used = parseFloat(m.margin.used_estimate || m.margin.used || 0);
        const avail = parseFloat(m.margin.available || m.margin.free || m.margin.total || 0);
        let pct = 0;
        if(avail > 0) pct = Math.min(100, Math.round((used / (used + avail)) * 100));
        usedEl.style.width = pct + '%';
      } else {
        txt.textContent = '—';
        usedEl.style.width = '0%';
      }
    }
    
    function renderRiskState(risk){
      const el=document.getElementById('risk-state');
      el.textContent = Object.keys(risk).length ? JSON.stringify(risk, null, 2) : 'No risk state yet.';
    }
    
    function renderTrades(trades){
      const tbody=document.querySelector('#trades-table tbody');
      tbody.innerHTML = '';
      trades.forEach(t=>{
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${t.time||''}</td><td>${t.mode||''}</td><td>${formatCurrency(t.pnl||0)}</td><td>${formatCurrency(t.balance||0)}</td>`;
        tbody.appendChild(tr);
      });
    }
    
    function renderSnapshot(snapshot){
      const tbody=document.querySelector('#snapshot-table tbody');
      tbody.innerHTML = '';
      snapshot.slice(0,80).forEach(r=>{
        const tr=document.createElement('tr');
        tr.innerHTML = `<td>${r.tradingsymbol||''}</td><td>${r.expiry||''}</td><td>${r.strike||''}</td><td>${r.instrument_type||''}</td><td>${r.ltp||''}</td>`;
        tbody.appendChild(tr);
      });
    }
    
    /* PnL chart (same as before) */
    function renderPnLChart(pnlHistory){
      const canvas=document.getElementById('pnl-chart'); const ctx=canvas.getContext('2d');
      ctx.clearRect(0,0,canvas.width,canvas.height);
      if(!pnlHistory||!pnlHistory.length){ ctx.fillStyle='#9ca3af'; ctx.font='12px system-ui'; ctx.fillText('No PnL history yet.',10,20); return; }
      const values=pnlHistory.map(p=>p.pnl||0); const n=values.length;
      const min=Math.min(...values); const max=Math.max(...values);
      const padding=12; const w=canvas.width; const h=canvas.height;
      const xStep=n>1?(w-2*padding)/(n-1):0; const range=max-min||1;
      ctx.strokeStyle='#374151'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(padding,h/2); ctx.lineTo(w-padding,h/2); ctx.stroke();
      ctx.strokeStyle='#3b82f6'; ctx.lineWidth=2; ctx.beginPath();
      values.forEach((v,i)=>{ const x=padding+i*xStep; const norm=(v-min)/range; const y=h-padding-norm*(h-2*padding); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
      ctx.stroke();
    }
    
    /* LTP chart uses snapshot; picks first 40 strikes and plots ltp */
    function renderLTPChart(ltp){
      const canvas=document.getElementById('ltp-chart'); const ctx=canvas.getContext('2d');
      ctx.clearRect(0,0,canvas.width,canvas.height);
      if(!ltp || !ltp.snapshot || !ltp.snapshot.length){ ctx.fillStyle='#9ca3af'; ctx.font='12px system-ui'; ctx.fillText('No LTP data yet.',10,20); return; }
      const snap = ltp.snapshot.filter(r=>r.strike!==undefined).slice(0,40);
      const values = snap.map(s=>s.ltp||0);
      if(!values.length){ ctx.fillStyle='#9ca3af'; ctx.font='12px system-ui'; ctx.fillText('No LTP numeric data.',10,20); return; }
      const n = values.length; const min=Math.min(...values); const max=Math.max(...values); const padding=10; const w=canvas.width; const h=canvas.height;
      const xStep=n>1?(w-2*padding)/(n-1):0; const range=max-min||1;
      ctx.strokeStyle='#374151'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(padding,h/2); ctx.lineTo(w-padding,h/2); ctx.stroke();
      ctx.strokeStyle='#10b981'; ctx.lineWidth=2; ctx.beginPath();
      values.forEach((v,i)=>{ const x=padding+i*xStep; const norm=(v-min)/range; const y=h-padding-norm*(h-2*padding); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
      ctx.stroke();
    }
    
    /* Heatmap ladder rendering */
    function renderHeatmap(heat){
      const container = document.getElementById('heatmap-ladder');
      container.innerHTML = '';
      if(!heat.rows || !heat.rows.length){ container.innerHTML = '<div class="small">No chain available.</div>'; return; }
      // Each row contains {strike, CE: {ltp,...} , PE: {...}}
      heat.rows.forEach(row => {
        const div = document.createElement('div');
        div.className = 'ladder-row';
        const ce = row.CE ? (row.CE.ltp || 0) : '';
        const pe = row.PE ? (row.PE.ltp || 0) : '';
        // color intensity based on size (oi) if available
        const ceColor = row.CE && row.CE.oi ? heatColor(row.CE.oi) : '';
        const peColor = row.PE && row.PE.oi ? heatColor(row.PE.oi) : '';
        div.innerHTML = `<div class="ce" style="background:${ceColor?ceColor:'transparent'}">${row.CE ? (row.CE.tradingsymbol||'') + ' ' + ce : ''}</div>
                         <div class="strike">${row.strike}</div>
                         <div class="pe" style="background:${peColor?peColor:'transparent'}">${row.PE ? (row.PE.tradingsymbol||'') + ' ' + pe : ''}</div>`;
        container.appendChild(div);
      });
    }
    
    // Map numeric to translucent color
    function heatColor(n){
      // simple ramp: low->transparent, medium->yellow, high->red
      const v = Math.min(1, Math.log10(1 + Math.max(0, n))/4); // scale log
      if(v < 0.33) return 'rgba(255,235,59,0.06)';
      if(v < 0.66) return 'rgba(255,159,67,0.08)';
      return 'rgba(239,68,68,0.12)';
    }
    
    /* Draw greeks timeseries as small chart (delta/theta/vega) */
    function drawGreeksChart(points){
      // points: [{time, delta, theta, vega, gamma}, ...]
      // we draw only delta and theta for clarity on pnl-card canvas if present
      // for now we will overlay into pnl-chart area small inset (simple)
      // Provide a basic separate small canvas? We reuse ltp-chart top-left small area by drawing overlay.
      // Simpler: create an offscreen small canvas and draw into greeks-card using a pre element fallback.
      const out = document.getElementById('greeks-card');
      try{
        const last = points[points.length-1];
        out.textContent = JSON.stringify(last, null, 2);
      }catch(e){
        out.textContent = '—';
      }
    }
    
    /* helpers */
    function clearCanvas(id){ const c=document.getElementById(id); if(!c) return; const ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height); }
    
    /* Force exit */
    async function sendForceExit(){
      try{
        const res = await fetch(FORCE_EXIT_URL, {method:'POST', headers:{'Content-Type':'application/json'}});
        if(!res.ok) throw new Error('HTTP '+res.status);
        showToast('Exit signal sent');
      }catch(e){ console.error('force',e); showToast('Failed to send exit'); }
    }
    
    function setup(){
      document.getElementById('refresh-seconds').textContent = REFRESH_SECONDS;
      document.getElementById('refresh-btn').addEventListener('click', ()=>{ fetchState(); fetchExtras(); });
      document.getElementById('force-exit-btn').addEventListener('click', ()=>{ sendForceExit(); });
      fetchState(); fetchExtras();
      if(autoRefreshInterval) clearInterval(autoRefreshInterval);
      autoRefreshInterval = setInterval(()=>{ fetchState(); fetchExtras(); }, REFRESH_SECONDS*1000);
    }
    
    window.addEventListener('load', setup);
    
    })();
    