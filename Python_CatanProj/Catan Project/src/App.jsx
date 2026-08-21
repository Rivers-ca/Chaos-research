import React, { useState, useEffect, useMemo, useCallback } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

const PC = ["#3266ad","#c44e3c","#3a8f5f","#9c6dbf"];
const PN = ["P0","P1","P2","P3"];
const TC = {FOREST:"#4a7c3f",HILLS:"#c44e3c",FIELDS:"#d4a820",PASTURE:"#7ab648",MOUNTAINS:"#6b7b8d",DESERT:"#c8b88a"};
const TABS = ["Game replay","Board heatmap","Benchmarks","Build sequences"];
const EMPTY = {vp:{},dice:{},meta:{winner:-1,total_turns:0,final_vp:[],final_settlements:[],final_cities:[],final_roads:[]},nodes:[],tiles:[],bench:{},mb:{top_sequences:[],win_correlation:{}},gl:[],win:[]};
const SKEY = "catan-research-data";

function Metric({label, value, sub}) {
  return <div style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:"12px 16px",minWidth:0}}>
    <div style={{fontSize:12,color:"var(--color-text-secondary)",marginBottom:4}}>{label}</div>
    <div style={{fontSize:22,fontWeight:500,color:"var(--color-text-primary)"}}>{value}</div>
    {sub && <div style={{fontSize:11,color:"var(--color-text-tertiary)",marginTop:2}}>{sub}</div>}
  </div>;
}

function Empty({msg, sub}) {
  return <div style={{padding:"3rem 1rem",textAlign:"center"}}>
    <div style={{fontSize:14,color:"var(--color-text-secondary)",marginBottom:8}}>{msg}</div>
    {sub && <div style={{fontSize:12,color:"var(--color-text-tertiary)"}}>{sub}</div>}
  </div>;
}

function GameReplay({D}) {
  const pids = useMemo(() => Object.keys(D.vp || {}), [D.vp]);
  
  // Moved Hooks UP above the early return
  const vpData = useMemo(() => {
    if (!pids.length) return [];
    const ref = D.vp[pids[0]] || [];
    return ref.map((_, i) => {
      const pt = { t: i * 2 };
      pids.forEach(k => { pt[`p${k}`] = (D.vp[k] || [])[i] || 0; });
      return pt;
    });
  }, [D.vp, pids]);

  const diceData = useMemo(() =>
    Object.entries(D.dice || {}).map(([k, v]) => ({
      roll: k, count: v,
      expected: Math.round((D.meta?.total_turns || 0) * (6 - Math.abs(7 - +k)) / 36)
    })).sort((a, b) => +a.roll - +b.roll),
  [D.dice, D.meta]);

  // Early return moved DOWN
  const np = pids.length;
  if (!np) return <Empty msg="No game replay data loaded" sub="Load data to see VP curves and dice distribution"/>;
  
  const m = D.meta || {};
  const fvp = m.final_vp || [];
  const fc = m.final_cities || [];
  
  return <div>
    <div style={{display:"grid",gridTemplateColumns:`repeat(${Math.min(np, 4)},1fr)`,gap:10,marginBottom:20}}>
      <Metric label="Winner" value={m.winner >= 0 ? `Player ${m.winner}` : "—"} sub={m.winner >= 0 && fvp[m.winner] != null ? `${fvp[m.winner]} VP` : ""}/>
      <Metric label="Game length" value={m.total_turns ? `${m.total_turns} turns` : "—"}/>
      <Metric label="Final VP" value={fvp.length ? fvp.join(" / ") : "—"}/>
      {fc.length > 0 && <Metric label="Cities built" value={fc.reduce((a, b) => a + b, 0)} sub={fc.join(" / ")}/>}
    </div>
    <div style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:8}}>Victory point progression</div>
    <div style={{height:220}}>
      <ResponsiveContainer><LineChart data={vpData} margin={{top:5,right:10,bottom:5,left:0}}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)"/>
        <XAxis dataKey="t" tick={{fontSize:11}} label={{value:"Turn",position:"insideBottom",offset:-2,fontSize:11}}/>
        <YAxis domain={[0,12]} tick={{fontSize:11}}/>
        <Tooltip contentStyle={{fontSize:12,background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:8}}/>
        {pids.map((k, i) => <Line key={k} type="stepAfter" dataKey={`p${k}`} stroke={PC[i % PC.length]} strokeWidth={2} dot={false} name={`P${k}`}/>)}
      </LineChart></ResponsiveContainer>
    </div>
    {diceData.length > 0 && <>
      <div style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",margin:"16px 0 8px"}}>Dice distribution vs expected</div>
      <div style={{height:180}}>
        <ResponsiveContainer><BarChart data={diceData} margin={{top:5,right:10,bottom:5,left:0}}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)"/>
          <XAxis dataKey="roll" tick={{fontSize:11}}/>
          <YAxis tick={{fontSize:11}}/>
          <Tooltip contentStyle={{fontSize:12,background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:8}}/>
          <Bar dataKey="count" fill="#3266ad" radius={[3,3,0,0]} name="Actual"/>
          <Bar dataKey="expected" fill="#b5b3ab" radius={[3,3,0,0]} name="Expected"/>
        </BarChart></ResponsiveContainer>
      </div>
    </>}
  </div>;
}

function BoardHeatmap({D}) {
  const [mode, setMode] = useState("settlement");
  const [hover, setHover] = useState(null);
  
  // Moved Hook UP
  const maxWc = useMemo(() => Math.max(1, ...(D.nodes || []).map(n => n.wc || 0)), [D.nodes]);
  
  // Early return moved DOWN
  if (!D.nodes?.length) return <Empty msg="No placement data loaded" sub="Run simulations to generate heatmap data"/>;
  
  const S = 42, OX = 80, OY = 60;
  const tx = x => OX + x * S;
  const ty = y => OY + y * S;
  const hexPath = (cx, cy) => {
    const pts = [];
    for (let i = 0; i < 6; i++) { const a = Math.PI / 180 * (60 * i - 30); pts.push(`${cx + S * 0.58 * Math.cos(a)},${cy + S * 0.58 * Math.sin(a)}`); }
    return `M${pts.join("L")}Z`;
  };
  
  const intensity = n => mode === "settlement" ? (n.si || 0) : mode === "city" ? (n.ci || 0) : (n.wc || 0) / maxWc;
  const hN = hover !== null ? D.nodes.find(n => n.id === hover) : null;
  const sk = mode === "settlement" ? "sc" : mode === "city" ? "cc" : "wc";
  return <div>
    <div style={{display:"flex",gap:8,marginBottom:12}}>
      {["settlement","city","winner"].map(m => <button key={m} onClick={() => setMode(m)} style={{fontSize:12,padding:"4px 14px",borderRadius:20,border:mode===m?"2px solid var(--color-border-info)":"0.5px solid var(--color-border-tertiary)",background:mode===m?"var(--color-background-info)":"transparent",color:"var(--color-text-primary)",cursor:"pointer",textTransform:"capitalize"}}>{m} frequency</button>)}
    </div>
    <div style={{display:"flex",gap:16}}>
      <svg viewBox="0 -30 420 380" style={{width:"100%",maxWidth:480}}>
        {(D.tiles || []).map(t => <path key={t.id} d={hexPath(tx(t.cx), ty(t.cy))} fill={t.resource ? (TC[t.type] || "#999") + "30" : "#c8b88a30"} stroke="var(--color-border-tertiary)" strokeWidth={0.5}/>)}
        {(D.tiles || []).filter(t => t.number > 0).map(t => <text key={`tn${t.id}`} x={tx(t.cx)} y={ty(t.cy) + 1} textAnchor="middle" dominantBaseline="middle" fontSize={11} fontWeight={500} fill="var(--color-text-secondary)">{t.number}</text>)}
        {D.nodes.map(n => {
          const v = Math.min(1, Math.max(0, intensity(n)));
          const r = 3 + v * 9;
          const c = mode === "winner" ? "rgba(212,168,32," : "rgba(50,102,173,";
          return <circle key={n.id} cx={tx(n.x)} cy={ty(n.y)} r={r} fill={`${c}${0.15 + v * 0.85})`} stroke={hover === n.id ? "var(--color-text-primary)" : "none"} strokeWidth={1.5} style={{cursor:"pointer"}} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)}/>;
        })}
      </svg>
      <div style={{minWidth:160,fontSize:12}}>
        {hN ? <div style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:12}}>
          <div style={{fontWeight:500,marginBottom:6}}>Node {hN.id}</div>
          <div style={{color:"var(--color-text-secondary)"}}>Pips: {hN.p}</div>
          <div style={{color:"var(--color-text-secondary)"}}>Resources: {(hN.r || []).join(", ") || "none"}</div>
          <div style={{marginTop:8,borderTop:"0.5px solid var(--color-border-tertiary)",paddingTop:8}}>
            <div>Settlements: {hN.sc}</div>
            <div>Cities: {hN.cc}</div>
            <div>Winner picks: {hN.wc}</div>
          </div>
        </div>
        : <div style={{color:"var(--color-text-tertiary)",padding:12}}>Hover a node for details</div>}
        <div style={{marginTop:8,fontSize:11}}>
          <div style={{fontWeight:500,marginBottom:4}}>Top 5 nodes</div>
          {D.nodes.slice().sort((a, b) => (b[sk] || 0) - (a[sk] || 0)).slice(0, 5).map(n => <div key={n.id} style={{display:"flex",justifyContent:"space-between",padding:"2px 0",color:"var(--color-text-secondary)"}}>
            <span>#{n.id} ({(n.r || []).slice(0, 2).map(r => r.slice(0, 3)).join("+") || "-"})</span>
            <span style={{fontWeight:500,color:"var(--color-text-primary)"}}>{n[sk] || 0}</span>
          </div>)}
        </div>
      </div>
    </div>
  </div>;
}

function Benchmarks({D}) {
  const matchups = useMemo(() => Object.values(D.bench || {}), [D.bench]);
  const gl = D.gl || [];
  const wins = D.win || [];
  
  // Moved Hooks UP
  const glData = useMemo(() => {
    if (!gl.length) return [];
    const bins = {};
    gl.forEach(t => { const b = Math.floor(t / 25) * 25; bins[b] = (bins[b] || 0) + 1; });
    return Object.entries(bins).map(([k, v]) => ({bin: `${k}-${+k + 24}`, count: v})).sort((a, b) => +a.bin.split("-")[0] - +b.bin.split("-")[0]);
  }, [gl]);
  
  const np = useMemo(() => wins.length ? Math.max(4, ...wins) + 1 : 4, [wins]);
  
  const wc = useMemo(() => {
    const c = new Array(np).fill(0);
    wins.forEach(w => { if (w >= 0 && w < np) c[w]++; });
    return c;
  }, [wins, np]);

  // Early return moved DOWN
  if (!gl.length && !matchups.length) return <Empty msg="No benchmark data loaded" sub="Run simulations to generate benchmark comparisons"/>;
  
  return <div>
    {gl.length > 0 && <>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:20}}>
        <Metric label="Games simulated" value={gl.length}/>
        <Metric label="Mean game length" value={`${Math.round(gl.reduce((a, b) => a + b, 0) / gl.length)} turns`}/>
        <Metric label="Shortest" value={`${Math.min(...gl)} turns`}/>
        <Metric label="Longest" value={`${Math.max(...gl)} turns`}/>
      </div>
      <div style={{fontSize:14,fontWeight:500,marginBottom:8}}>Win distribution ({gl.length} games)</div>
      <div style={{display:"flex",gap:6,marginBottom:20}}>
        {wc.slice(0, np).map((w, i) => <div key={i} style={{flex: w || 0.5, background: PC[i % PC.length], borderRadius:6, padding:"8px 12px", color:"#fff", fontSize:12, textAlign:"center", minWidth:40}}>
          <div style={{fontWeight:500}}>{PN[i] || `P${i}`}</div><div>{w} wins</div>
        </div>)}
      </div>
    </>}
    {glData.length > 0 && <>
      <div style={{fontSize:14,fontWeight:500,marginBottom:8}}>Game length distribution</div>
      <div style={{height:160,marginBottom:24}}>
        <ResponsiveContainer><BarChart data={glData} margin={{top:5,right:10,bottom:5,left:0}}>
          <XAxis dataKey="bin" tick={{fontSize:10}} angle={-30} textAnchor="end" height={40}/>
          <YAxis tick={{fontSize:11}}/>
          <Bar dataKey="count" fill="#3a8f5f" radius={[3,3,0,0]}/>
        </BarChart></ResponsiveContainer>
      </div>
    </>}
    {matchups.length > 0 && <>
      <div style={{fontSize:14,fontWeight:500,marginBottom:12}}>Head-to-head matchup results</div>
      {matchups.map((m, i) => <div key={i} style={{background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)",padding:"14px 18px",marginBottom:10}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
          <span style={{fontWeight:500,fontSize:14}}>{m.label_a} vs {m.label_b}</span>
          {m.games_per_sec != null && <span style={{fontSize:12,color:"var(--color-text-secondary)"}}>{Math.round(m.games_per_sec)} g/s</span>}
        </div>
        <div style={{display:"flex",gap:4,marginBottom:8}}>
          <div style={{flex: m.win_rate_a || 1, background:"#3266ad", borderRadius:4, height:24, display:"flex", alignItems:"center", justifyContent:"center", color:"#fff", fontSize:11, fontWeight:500, minWidth:30}}>{m.win_rate_a}%</div>
          <div style={{flex: m.win_rate_b || 1, background:"#c44e3c", borderRadius:4, height:24, display:"flex", alignItems:"center", justifyContent:"center", color:"#fff", fontSize:11, fontWeight:500, minWidth:30}}>{m.win_rate_b}%</div>
        </div>
        <div style={{display:"flex",gap:20,fontSize:11,color:"var(--color-text-secondary)",flexWrap:"wrap"}}>
          {m.turns_mean != null && <span>Mean {Math.round(m.turns_mean)} turns</span>}
          {m.turns_median != null && <span>Median {Math.round(m.turns_median)}</span>}
          {m.vp_a_mean != null && <span>VP: {m.vp_a_mean} vs {m.vp_b_best_mean}</span>}
        </div>
      </div>)}
    </>}
  </div>;
}

function BuildSeqs({D}) {
  const mb = D.mb || {};
  const seqs = (mb.top_sequences || []).slice(0, 12);
  const wc = mb.win_correlation || {};
  
  // Moved Hooks UP
  const barData = useMemo(() => seqs.map(([name, count]) => {
    const w = wc[name];
    return {name: name.length > 22 ? name.slice(0, 20) + "…" : name, full: name, total: count, winners: w ? w.by_winners : 0, winShare: w ? w.win_share : 0};
  }), [seqs, wc]);
  
  const sorted = useMemo(() => [...barData].sort((a, b) => b.winShare - a.winShare), [barData]);

  // Early return moved DOWN
  if (!seqs.length) return <Empty msg="No build sequence data loaded" sub="Run simulations to analyze build patterns"/>;
  
  return <div>
    <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10,marginBottom:20}}>
      <Metric label="Total build actions" value={seqs.reduce((a, b) => a + b[1], 0).toLocaleString()}/>
      <Metric label="Unique sequences" value={Object.keys(wc).length || seqs.length}/>
      <Metric label="Highest win share" value={sorted[0] ? `${sorted[0].winShare}%` : "—"} sub={sorted[0]?.full}/>
    </div>
    <div style={{fontSize:14,fontWeight:500,marginBottom:8}}>Build sequences by frequency</div>
    <div style={{height: Math.max(200, barData.length * 26 + 40)}}>
      <ResponsiveContainer><BarChart data={barData} layout="vertical" margin={{top:5,right:10,bottom:5,left:100}}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)"/>
        <XAxis type="number" tick={{fontSize:11}}/>
        <YAxis type="category" dataKey="name" tick={{fontSize:10}} width={95}/>
        <Tooltip contentStyle={{fontSize:12,background:"var(--color-background-primary)",border:"0.5px solid var(--color-border-tertiary)",borderRadius:8}} formatter={(v,n)=>[v,n]}/>
        <Bar dataKey="total" fill="#b5b3ab" radius={[0,3,3,0]} name="Total"/>
        <Bar dataKey="winners" fill="#3266ad" radius={[0,3,3,0]} name="By winners"/>
      </BarChart></ResponsiveContainer>
    </div>
    <div style={{fontSize:14,fontWeight:500,margin:"20px 0 10px"}}>Win correlation — which sequences predict winning?</div>
    <div style={{display:"grid",gap:6}}>
      {sorted.slice(0, 10).map((s, i) => <div key={i} style={{display:"flex",alignItems:"center",gap:10,fontSize:12}}>
        <div style={{minWidth:160,color:"var(--color-text-secondary)",fontFamily:"var(--font-mono)",fontSize:11}}>{s.name}</div>
        <div style={{flex:1,background:"var(--color-background-secondary)",borderRadius:4,height:20,overflow:"hidden"}}>
          <div style={{height:"100%",width:`${Math.min(100, s.winShare * 2)}%`,background:s.winShare > 35 ? "#3a8f5f" : "#3266ad",borderRadius:4,transition:"width 0.3s"}}/>
        </div>
        <div style={{minWidth:40,textAlign:"right",fontWeight:500,fontSize:12,color:s.winShare > 35 ? "#3a8f5f" : "var(--color-text-primary)"}}>{s.winShare}%</div>
      </div>)}
    </div>
    <div style={{fontSize:11,color:"var(--color-text-tertiary)",marginTop:12}}>Win share = % of times this sequence was executed by the eventual winner. Baseline (random) = 25%.</div>
  </div>;
}

export default function Dashboard() {
  const [tab, setTab] = useState(0);
  const [data, setData] = useState(EMPTY);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("");
  const [jsonInput, setJsonInput] = useState("");
  const [showInput, setShowInput] = useState(false);

  // 1. Fixed the auto-load logic
  useEffect(() => {
    try {
      const saved = localStorage.getItem(SKEY);
      if (saved) {
        setData(JSON.parse(saved));
        setStatus("Loaded from storage");
      } else {
        setStatus("No stored data — load JSON below");
      }
    } catch (e) {
      setStatus("Error loading from storage");
    }
    setLoading(false);
  }, []);

  // 2. Fixed the data import logic
  const handleLoad = useCallback((text) => {
    try {
      const parsed = JSON.parse(text);
      if (parsed.vp || parsed.nodes || parsed.gl || parsed.bench) {
        setData(parsed);
        // Standard browser save
        localStorage.setItem(SKEY, JSON.stringify(parsed));
        setStatus(`Loaded ${Object.keys(parsed).length} sections at ${new Date().toLocaleTimeString()}`);
        setShowInput(false);
        setJsonInput("");
      } else {
        setStatus("Invalid format: needs vp, nodes, gl, or bench keys");
      }
    } catch (e) {
      setStatus("JSON parse error: " + e.message);
    }
  }, []);

  // 3. Fixed the clear logic
  const handleClear = useCallback(() => {
    setData(EMPTY);
    localStorage.removeItem(SKEY);
    setStatus("Data cleared");
  }, []);

  const hasData = (data.gl?.length > 0) || (data.nodes?.length > 0) || Object.keys(data.vp || {}).length > 0;
  const views = [<GameReplay D={data}/>, <BoardHeatmap D={data}/>, <Benchmarks D={data}/>, <BuildSeqs D={data}/>];

  return <div style={{padding:"1rem 0",maxWidth:700}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16,flexWrap:"wrap",gap:8}}>
      <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
        {TABS.map((t, i) => <button key={i} onClick={() => setTab(i)} style={{fontSize:13,padding:"6px 16px",borderRadius:20,border:tab===i?"2px solid var(--color-border-info)":"0.5px solid var(--color-border-tertiary)",background:tab===i?"var(--color-background-info)":"transparent",color:"var(--color-text-primary)",cursor:"pointer",fontWeight:tab===i?500:400}}>{t}</button>)}
      </div>
      <div style={{display:"flex",gap:6}}>
        <button onClick={() => setShowInput(!showInput)} style={{fontSize:11,padding:"4px 12px",borderRadius:16,border:"0.5px solid var(--color-border-secondary)",background:"transparent",color:"var(--color-text-secondary)",cursor:"pointer"}}>{showInput ? "Cancel" : "Load JSON"}</button>
        {hasData && <button onClick={handleClear} style={{fontSize:11,padding:"4px 12px",borderRadius:16,border:"0.5px solid var(--color-border-tertiary)",background:"transparent",color:"var(--color-text-tertiary)",cursor:"pointer"}}>Clear</button>}
      </div>
    </div>

    {status && <div style={{fontSize:11,color:"var(--color-text-tertiary)",marginBottom:12,padding:"6px 12px",background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-md)"}}>{loading ? "Loading..." : status}</div>}

    {showInput && <div style={{marginBottom:16}}>
      <textarea value={jsonInput} onChange={e => setJsonInput(e.target.value)} placeholder='Paste the JSON output from catan_analytics.py here...' style={{width:"100%",minHeight:120,padding:12,fontSize:12,fontFamily:"var(--font-mono)",borderRadius:"var(--border-radius-md)",border:"0.5px solid var(--color-border-secondary)",background:"var(--color-background-primary)",color:"var(--color-text-primary)",resize:"vertical"}}/>
      <button onClick={() => handleLoad(jsonInput)} disabled={!jsonInput.trim()} style={{marginTop:8,fontSize:12,padding:"6px 20px",borderRadius:16,border:"0.5px solid var(--color-border-info)",background:jsonInput.trim()?"var(--color-background-info)":"transparent",color:"var(--color-text-primary)",cursor:jsonInput.trim()?"pointer":"default"}}>Load data</button>
    </div>}

    {views[tab]}

    {!hasData && !loading && !showInput && <div style={{marginTop:20,padding:16,background:"var(--color-background-secondary)",borderRadius:"var(--border-radius-lg)",fontSize:12,color:"var(--color-text-secondary)"}}>
      <div style={{fontWeight:500,marginBottom:8,color:"var(--color-text-primary)"}}>How to load data</div>
      <div style={{lineHeight:1.8}}>
        1. Run <code style={{fontSize:11,padding:"2px 6px",background:"var(--color-background-tertiary)",borderRadius:4}}>python3 catan_analytics.py</code> to generate research data<br/>
        2. The script produces <code style={{fontSize:11,padding:"2px 6px",background:"var(--color-background-tertiary)",borderRadius:4}}>catan_dashboard_data.json</code><br/>
        3. Click <span style={{fontWeight:500}}>Load JSON</span> above and paste the contents<br/>
        Data persists across sessions automatically via storage.
      </div>
    </div>}
  </div>;
}