import { useState, useEffect, useRef, useCallback } from "react";
import SystemPanel from "./SystemPanel.jsx";

const API = "http://localhost:8000";
const RAM_BUDGET_GB = 8;
const ATTENTION_GB = 3.2;

const DOMAIN_CONFIG = {
  code:      { color: "#00ffe5", glow: "#00ffe540", icon: "{ }" },
  math:      { color: "#ff6b35", glow: "#ff6b3540", icon: "∑" },
  language:  { color: "#c084fc", glow: "#c084fc40", icon: "Aa" },
  reasoning: { color: "#38bdf8", glow: "#38bdf840", icon: "⟳" },
  creative:  { color: "#f472b6", glow: "#f472b640", icon: "✦" },
  factual:   { color: "#fbbf24", glow: "#fbbf2440", icon: "◎" },
};

const EXAMPLES = [
  { label:"CODE",    domain:"code",      q:"Write a Python function to find the shortest path in a weighted graph" },
  { label:"MATH",    domain:"math",      q:"Solve the integral of x squared times sin(x) dx step by step" },
  { label:"LOGIC",   domain:"reasoning", q:"Why does correlation not imply causation? Give a concrete example" },
  { label:"STORY",   domain:"creative",  q:"Write a short story about a lost astronaut who finds an alien library" },
  { label:"FACTS",   domain:"factual",   q:"What is the difference between RAM and VRAM in AI inference?" },
];

function fmt(n, dec=2) { return typeof n === "number" ? n.toFixed(dec) : (n ?? "—"); }

// Hexagon SVG path
function HexNode({ active, cached, predicted, id, size=36 }) {
  const s = size / 2;
  const h = s * Math.sqrt(3) / 2;
  const pts = [
    [s, 0], [s*2, h], [s*2, h*2], [s, h*3], [0, h*2], [0, h]
  ].map(([x,y]) => `${x},${y}`).join(" ");

  const color = active ? "#00ffe5" : predicted ? "#fbbf24" : cached ? "#38bdf8" : "#1e2d3d";
  const glow = active ? "0 0 12px #00ffe5, 0 0 24px #00ffe540"
             : predicted ? "0 0 8px #fbbf24"
             : cached ? "0 0 6px #38bdf840" : "none";

  return (
    <svg width={size*2} height={size*3} style={{ filter: active||cached ? `drop-shadow(0 0 6px ${color})` : "none", transition:"all 0.3s" }}>
      <polygon points={pts} fill={active?"#001a1a":predicted?"#1a1200":cached?"#001220":"#060d14"}
        stroke={color} strokeWidth={active?1.5:cached?1:0.5} style={{ transition:"all 0.3s" }}/>
      <text x={s} y={h*1.5+4} textAnchor="middle" fontSize={9}
        fill={active?"#00ffe5":cached?"#38bdf8":"#1e2d3d"} fontFamily="'Share Tech Mono',monospace">
        E{id}
      </text>
      {active && <polygon points={pts} fill="none" stroke="#00ffe5" strokeWidth={3} opacity={0.3}>
        <animate attributeName="opacity" values="0.3;0.8;0.3" dur="1.2s" repeatCount="indefinite"/>
      </polygon>}
    </svg>
  );
}

// Corner bracket decoration
function Bracket({ color="#00ffe520", size=12 }) {
  return (
    <svg width={size} height={size} style={{ position:"absolute", pointerEvents:"none" }}>
      <path d={`M${size},0 L0,0 L0,${size}`} fill="none" stroke={color} strokeWidth={1}/>
    </svg>
  );
}

export default function LazyMoECool() {
  const [query, setQuery]               = useState("");
  const [isRunning, setIsRunning]       = useState(false);
  const [phase, setPhase]               = useState("idle");
  const [tokenStream, setTokenStream]   = useState("");
  const [log, setLog]                   = useState([]);
  const [analysis, setAnalysis]         = useState(null);
  const [cacheSnap, setCacheSnap]       = useState([]);
  const [cacheStats, setCacheStats]     = useState({ hits:0, misses:0, hit_rate:0 });
  const [kvStats, setKvStats]           = useState({ tokens:0, ram_used_gb:0, raw_fp16_gb:0, compression_ratio:1 });
  const [kvHistory, setKvHistory]       = useState([]);
  const [expertActivity, setExpertActivity] = useState(Array(8).fill(0));
  const [finalStats, setFinalStats]     = useState(null);
  const [health, setHealth]             = useState(null);
  const [modelInfo, setModelInfo]       = useState(null);
  const [mockMode, setMockMode]         = useState(false);
  const [numExperts, setNumExperts]     = useState(8);
  const [tick, setTick]                 = useState(0);
  const [showSystem, setShowSystem]     = useState(false);

  const logRef    = useRef([]);
  const logEnd    = useRef(null);
  const streamRef = useRef("");

  // Heartbeat animation
  useEffect(() => {
    const id = setInterval(() => setTick(t => t+1), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => {
        setHealth(d);
        setMockMode(d.mock_mode);
        if (d.model) {
          setModelInfo(d.model);
          setNumExperts(d.model.is_moe ? (d.model.num_experts || 8) : 1);
        }
      })
      .catch(() => setHealth({ status:"offline", mock_mode:true }));
  }, []);

  useEffect(() => { logEnd.current?.scrollIntoView({ behavior:"smooth" }); }, [log]);

  const addLog = useCallback((msg, type="info") => {
    const e = { msg, type, ts: Date.now() };
    logRef.current = [...logRef.current.slice(-80), e];
    setLog([...logRef.current]);
  }, []);

  const resetAll = useCallback(async () => {
    if (isRunning) setIsRunning(false);
    streamRef.current = "";
    setTokenStream(""); setPhase("idle"); setAnalysis(null);
    setCacheSnap([]); setKvStats({ tokens:0, ram_used_gb:0, raw_fp16_gb:0, compression_ratio:1 });
    setKvHistory([]); setExpertActivity(Array(numExperts).fill(0));
    setFinalStats(null); logRef.current = []; setLog([]);
    await fetch(`${API}/reset`, { method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ clear_kv:true, clear_experts:true }) }).catch(()=>{});
  }, [isRunning, numExperts]);

  const runInference = useCallback(async () => {
    if (!query.trim() || isRunning) return;
    await resetAll();
    setIsRunning(true);
    streamRef.current = "";
    addLog("▶ INITIALIZING INFERENCE PIPELINE", "system");

    try {
      const response = await fetch(`${API}/infer`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ query, max_tokens:400, temperature:0.7 }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream:true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try { handleEvent(JSON.parse(line.slice(6))); } catch(_) {}
        }
      }
    } catch(err) {
      addLog(`✗ ${err.message}`, "miss");
    } finally {
      setIsRunning(false);
      setExpertActivity(Array(numExperts).fill(0));
    }
  }, [query, isRunning, resetAll, addLog, numExperts]);

  const handleEvent = useCallback((evt) => {
    switch(evt.type) {
      case "phase":
        setPhase(evt.phase);
        const phaseMsg = {
          analyzing:   "◈ QUERY ANALYZER — embedding + domain classification",
          prefetching: "⚡ PREFETCHER — loading expert weights SSD → RAM",
          inferring:   "▶ INFERENCE ENGINE — token generation active",
        }[evt.phase];
        if (phaseMsg) addLog(phaseMsg, "system");
        break;
      case "analysis":
        setAnalysis(evt);
        addLog(`  DOMAIN   ${evt.domain.toUpperCase()} (${(evt.confidence*100).toFixed(0)}% confidence)`, "success");
        addLog(`  EXPERTS  routing to [${evt.active_experts.join(", ")}]`, "success");
        break;
      case "expert": {
        const { expert_id, hit, load_ms } = evt;
        addLog(
          hit ? `  E${expert_id}  ██ HIT   ${load_ms.toFixed(0)}ms` : `  E${expert_id}  ░░ MISS  ${load_ms.toFixed(0)}ms (SSD load)`,
          hit ? "hit" : "miss"
        );
        if (evt.cache_snapshot) setCacheSnap(evt.cache_snapshot);
        if (evt.cache_stats) setCacheStats(evt.cache_stats);
        break;
      }
      case "token": {
        streamRef.current += evt.token;
        setTokenStream(streamRef.current);
        if (evt.active_experts?.length > 0) {
          const act = Array(numExperts).fill(0);
          evt.active_experts.forEach(e => { if(e < numExperts) act[e] = 1; });
          setExpertActivity(act);
        }
        break;
      }
      case "kv_update":
        setKvStats(evt);
        setKvHistory(h => [...h, { tok:evt.tokens, raw:evt.raw_fp16_gb, tq:evt.ram_used_gb }].slice(-60));
        break;
      case "done":
        setPhase("done");
        setFinalStats(evt);
        setMockMode(evt.mock_mode);
        if (evt.cache_stats) setCacheStats(evt.cache_stats);
        if (evt.kv_stats) setKvStats(evt.kv_stats);
        addLog(`◼ COMPLETE  ${evt.tokens} tokens · ${evt.elapsed_sec}s · ${evt.tokens_per_sec} tok/s`, "success");
        addLog(`  CACHE HIT RATE  ${((evt.cache_stats?.hit_rate||0)*100).toFixed(0)}%`, "success");
        break;
      case "error":
        setPhase("error");
        addLog(`✗ ERROR: ${evt.message}`, "miss");
        break;
    }
  }, [addLog, numExperts]);

  // RAM
  const expertGB = cacheSnap.reduce((a,s) => a+(s.size_gb||0), 0);
  const kvGB = kvStats.ram_used_gb || 0;
  const totalGB = ATTENTION_GB + expertGB + kvGB;
  const oom = totalGB > RAM_BUDGET_GB;

  // KV chart
  const CW=220, CH=64;
  const maxTok = Math.max(...kvHistory.map(h=>h.tok), 50);
  const maxRaw = Math.max(...kvHistory.map(h=>h.raw), 0.01);
  const xS = t => (t/maxTok)*CW;
  const yS = v => CH-(v/maxRaw)*CH;
  const pts = key => kvHistory.map(h=>`${xS(h.tok).toFixed(1)},${yS(h[key]).toFixed(1)}`).join(" ");

  const domainColor = DOMAIN_CONFIG[analysis?.domain]?.color || "#00ffe5";
  const statusColor = { idle:"#1e3d2f", analyzing:"#1a1500", prefetching:"#0a1520", inferring:"#001a12", done:"#001a12", error:"#1a0000" }[phase];
  const statusBorder = { idle:"#1e2d3d", analyzing:"#fbbf2460", prefetching:"#38bdf860", inferring:"#00ffe560", done:"#00ffe560", error:"#ef444460" }[phase];

  const expertCols = numExperts <= 8 ? 4 : numExperts <= 16 ? 4 : 8;
  const showExperts = Math.min(numExperts, 32);

  return (
    <div style={{ minHeight:"100vh", background:"#020609", color:"#8baabf", overflowX:"hidden", position:"relative" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        body{background:#020609}
        ::-webkit-scrollbar{width:2px}
        ::-webkit-scrollbar-track{background:#020609}
        ::-webkit-scrollbar-thumb{background:#00ffe520;border-radius:1px}

        .panel {
          border: 1px solid #0d2030;
          background: #030b12;
          position: relative;
        }
        .panel::before {
          content:'';position:absolute;top:0;left:0;right:0;
          height:1px;background:linear-gradient(90deg,transparent,#00ffe520,transparent);
        }

        .glow-cyan { box-shadow: 0 0 20px #00ffe508, inset 0 0 20px #00ffe504; }
        .glow-active { box-shadow: 0 0 30px #00ffe515, inset 0 0 30px #00ffe508; }

        .scan {
          position:fixed;top:0;left:0;right:0;bottom:0;pointer-events:none;z-index:1000;
          background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,229,0.012) 2px,rgba(0,255,229,0.012) 4px);
        }

        .log-e { animation: slideIn 0.15s ease; }
        @keyframes slideIn { from{opacity:0;transform:translateX(-4px)} to{opacity:1;transform:none} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        @keyframes glow-pulse { 0%,100%{box-shadow:0 0 8px #00ffe530} 50%{box-shadow:0 0 20px #00ffe560,0 0 40px #00ffe520} }
        @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        @keyframes flicker { 0%,100%{opacity:1} 95%{opacity:0.85} 96%{opacity:1} 98%{opacity:0.9} }

        .status-dot {
          width:7px;height:7px;border-radius:50%;
          animation: glow-pulse 2s infinite;
        }

        textarea { caret-color: #00ffe5; }
        textarea:focus { outline:none; }
        button:disabled { opacity:0.3; cursor:not-allowed !important; }

        .hex-grid { display:grid; gap:3px; }
        .token-output { animation:flicker 8s infinite; }

        .progress-bar {
          height:100%;border-radius:1px;
          background:linear-gradient(90deg,#00ffe5,#38bdf8);
          transition:width 0.6s ease;
          box-shadow:0 0 8px #00ffe540;
        }
        .progress-bar-expert {
          background:linear-gradient(90deg,#38bdf8,#c084fc);
          box-shadow:0 0 8px #38bdf840;
        }
        .progress-bar-kv {
          background:linear-gradient(90deg,#f472b6,#c084fc);
          box-shadow:0 0 8px #f472b640;
        }
        .progress-bar-warn {
          background:linear-gradient(90deg,#ef4444,#f97316);
          box-shadow:0 0 8px #ef444440;
        }

        .metric-card {
          background:#030b12;
          border:1px solid #0d2030;
          border-radius:2px;
          padding:8px 10px;
          position:relative;
          overflow:hidden;
        }
        .metric-card::after {
          content:'';position:absolute;bottom:0;left:0;right:0;
          height:1px;background:linear-gradient(90deg,transparent,#00ffe520,transparent);
        }

        .btn-run {
          background:linear-gradient(135deg,#003d30,#001a20);
          border:1px solid #00ffe540;
          color:#00ffe5;
          font-family:'Share Tech Mono',monospace;
          font-size:11px;letter-spacing:0.15em;
          cursor:pointer;
          padding:0 20px;
          border-radius:2px;
          transition:all 0.2s;
          white-space:nowrap;
        }
        .btn-run:hover:not(:disabled) {
          background:linear-gradient(135deg,#004d3c,#002028);
          box-shadow:0 0 15px #00ffe530;
        }
        .btn-run.running {
          background:#030b12;
          border-color:#1e3040;
          color:#1e3040;
          animation:pulse 1.5s infinite;
        }

        .example-btn {
          font-family:'Share Tech Mono',monospace;
          font-size:8px;letter-spacing:0.1em;
          padding:3px 8px;
          background:transparent;
          border:1px solid #0d2030;
          color:#1e3040;cursor:pointer;
          border-radius:1px;
          transition:all 0.2s;
        }
        .example-btn:hover { border-color:#00ffe530; color:#00ffe570; }
      `}</style>

      {/* Scanline overlay */}
      <div className="scan"/>

      {/* System panel */}
      {showSystem && <SystemPanel onClose={()=>setShowSystem(false)}/>}

      {/* Header */}
      <div style={{ borderBottom:"1px solid #0d2030", padding:"10px 20px", display:"flex", alignItems:"center", gap:12, background:"linear-gradient(180deg,#030e18,#020609)", flexWrap:"wrap" }}>

        {/* Logo */}
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ width:28, height:28, border:"1px solid #00ffe540", display:"flex", alignItems:"center", justifyContent:"center", position:"relative" }}>
            <span style={{ fontFamily:"'Share Tech Mono'", fontSize:10, color:"#00ffe5", letterSpacing:0 }}>LM</span>
            <div style={{ position:"absolute", top:-1, left:-1, width:4, height:4, borderTop:"1px solid #00ffe5", borderLeft:"1px solid #00ffe5" }}/>
            <div style={{ position:"absolute", bottom:-1, right:-1, width:4, height:4, borderBottom:"1px solid #00ffe5", borderRight:"1px solid #00ffe5" }}/>
          </div>
          <div>
            <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:15, color:"#00ffe5", letterSpacing:"0.2em" }}>LAZY-MOE</div>
            <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em" }}>v0.3 · UNIVERSAL INFERENCE</div>
          </div>
        </div>

        {/* Status */}
        <div style={{ display:"flex", alignItems:"center", gap:6, marginLeft:8 }}>
          <div className="status-dot" style={{ background: phase==="inferring"?"#00ffe5":phase==="done"?"#00ffe5":phase==="error"?"#ef4444":"#1e3040" }}/>
          <span style={{ fontFamily:"'Share Tech Mono'", fontSize:9, color: phase==="inferring"?"#00ffe5":phase==="done"?"#00ffe5":phase==="error"?"#ef4444":"#1e3040", letterSpacing:"0.1em" }}>
            {({idle:"STANDBY",analyzing:"ANALYZING",prefetching:"PREFETCHING",inferring:"INFERRING",done:"COMPLETE",error:"ERROR"})[phase]}
          </span>
        </div>

        {/* Mock badge */}
        {mockMode && (
          <div style={{ fontFamily:"'Share Tech Mono'", fontSize:8, padding:"2px 8px", border:"1px solid #fbbf2440", color:"#fbbf2480", letterSpacing:"0.1em" }}>
            MOCK MODE
          </div>
        )}

        {/* Model info */}
        {modelInfo && (
          <div style={{ display:"flex", gap:8, alignItems:"center", marginLeft:4 }}>
            <span style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:12, color:"#38bdf8", letterSpacing:"0.1em" }}>{modelInfo.name.toUpperCase()}</span>
            <span style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#1e3040" }}>{modelInfo.params_b}B</span>
            {modelInfo.is_moe && <span style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#fbbf24", padding:"1px 5px", border:"1px solid #fbbf2430" }}>MOE×{modelInfo.num_experts}</span>}
            <span style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#1e3040" }}>{modelInfo.quant}</span>
            <span style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#1e3040" }}>{modelInfo.size_gb}GB</span>
          </div>
        )}

        {/* Ticker */}
        <div style={{ marginLeft:"auto", fontFamily:"'Share Tech Mono'", fontSize:8, color:"#1e3040", display:"flex", gap:12 }}>
          <span>ATTN {ATTENTION_GB.toFixed(1)}G</span>
          <span>EXP {expertGB.toFixed(2)}G</span>
          <span style={{ color: oom?"#ef4444":"#1e3040" }}>KV {kvGB.toFixed(3)}G</span>
          <span style={{ color: oom?"#ef4444":"#00ffe560" }}>TOT {totalGB.toFixed(2)}G/{RAM_BUDGET_GB}G</span>
          <button onClick={resetAll} style={{ fontFamily:"'Share Tech Mono'", fontSize:8, padding:"2px 8px", background:"transparent", border:"1px solid #1e3040", color:"#1e3040", cursor:"pointer", letterSpacing:"0.1em" }}>
            RESET
          </button>
          <button onClick={()=>setShowSystem(true)} style={{ fontFamily:"'Share Tech Mono'", fontSize:8, padding:"2px 8px", background:"transparent", border:"1px solid #00ffe530", color:"#00ffe570", cursor:"pointer", letterSpacing:"0.1em" }}>
            ⬡ SYSTEM
          </button>
        </div>
      </div>

      {/* Main layout */}
      <div style={{ display:"grid", gridTemplateColumns:"260px 1fr 250px 240px", height:"calc(100vh - 50px)" }}>

        {/* ── COL 1: Expert Cache ── */}
        <div className="panel" style={{ borderRight:"1px solid #0d2030", display:"flex", flexDirection:"column", overflow:"hidden" }}>

          {/* RAM bar */}
          <div style={{ padding:"10px 14px", borderBottom:"1px solid #0d2030" }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6, fontFamily:"'Share Tech Mono'", fontSize:8, letterSpacing:"0.1em" }}>
              <span style={{ color:"#1e3040" }}>MEMORY ALLOCATION</span>
              <span style={{ color:oom?"#ef4444":totalGB>6?"#fbbf24":"#00ffe5" }}>
                {totalGB.toFixed(2)} / {RAM_BUDGET_GB}.00 GB
              </span>
            </div>

            {/* Segmented bar */}
            <div style={{ height:6, background:"#020609", borderRadius:1, overflow:"hidden", display:"flex", border:"1px solid #0d2030" }}>
              <div className="progress-bar" style={{ width:`${(ATTENTION_GB/RAM_BUDGET_GB*100).toFixed(1)}%` }}/>
              <div className="progress-bar progress-bar-expert" style={{ width:`${(expertGB/RAM_BUDGET_GB*100).toFixed(1)}%` }}/>
              <div className="progress-bar progress-bar-kv" style={{ width:`${(kvGB/RAM_BUDGET_GB*100).toFixed(1)}%` }}/>
            </div>

            <div style={{ display:"flex", gap:10, marginTop:5, flexWrap:"wrap" }}>
              {[
                ["ATTN", ATTENTION_GB, "#00ffe5"],
                ["EXPERT", expertGB, "#38bdf8"],
                ["KV·TQ", kvGB, "#f472b6"],
              ].map(([l,v,c])=>(
                <span key={l} style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:c, display:"flex", alignItems:"center", gap:3 }}>
                  <span style={{ width:5,height:5,background:c,display:"inline-block",boxShadow:`0 0 4px ${c}` }}/>
                  {l} {v.toFixed(2)}G
                </span>
              ))}
            </div>

            {oom && (
              <div style={{ marginTop:5, fontFamily:"'Share Tech Mono'", fontSize:8, color:"#ef4444", letterSpacing:"0.1em" }}>
                ⚠ OUT OF MEMORY RISK
              </div>
            )}
          </div>

          {/* Model config */}
          {modelInfo && (
            <div style={{ padding:"10px 14px", borderBottom:"1px solid #0d2030" }}>
              <div style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:6 }}>MODEL ARCHITECTURE</div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"2px 8px" }}>
                {[
                  ["LAYERS",   modelInfo.layers],
                  ["KV HEADS", modelInfo.num_kv_heads||"—"],
                  ["CTX",      `${modelInfo.ctx||2048}`],
                  ["KV BITS",  `${modelInfo.kv_bits||3}b`],
                  ["EXPERTS",  modelInfo.is_moe?modelInfo.num_experts:"dense"],
                  ["QUANT",    modelInfo.quant||"Q4"],
                ].map(([l,v])=>(
                  <div key={l} style={{ display:"flex", justifyContent:"space-between", padding:"2px 0", borderBottom:"1px solid #0a1520" }}>
                    <span style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040" }}>{l}</span>
                    <span style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#38bdf8" }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Expert nodes */}
          <div style={{ padding:"10px 14px", borderBottom:"1px solid #0d2030" }}>
            <div style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:8 }}>
              EXPERT NODES [{numExperts}]
            </div>
            <div style={{ display:"grid", gridTemplateColumns:`repeat(${expertCols}, 1fr)`, gap:2 }}>
              {Array.from({length:showExperts},(_,i)=>{
                const inCache = cacheSnap.some(s=>s.expert_id===i);
                const active = expertActivity[i]===1;
                const predicted = analysis?.active_experts?.includes(i) && phase!=="idle";
                const color = active?"#00ffe5":predicted?"#fbbf24":inCache?"#38bdf8":"#0d2030";
                return (
                  <div key={i} style={{ aspectRatio:"1", border:`1px solid ${color}`, background:active?"#001a14":predicted?"#1a1200":inCache?"#00121a":"#020609", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:1, transition:"all 0.2s",
                    boxShadow:active?`0 0 8px ${color}, inset 0 0 8px ${color}20`:predicted?`0 0 4px ${color}`:"none" }}>
                    <span style={{ fontSize:10, color, opacity:inCache||active?1:0.3, transition:"all 0.2s" }}>
                      {active?"◉":inCache?"◎":"○"}
                    </span>
                    <span style={{ fontFamily:"'Share Tech Mono'", fontSize:6, color: active?"#00ffe5":inCache?"#38bdf8":"#1e3040" }}>
                      {i}
                    </span>
                    {active && <div style={{ position:"absolute", width:"100%", height:"100%", border:`1px solid #00ffe5`, opacity:0.3, animation:"pulse 1s infinite" }}/>}
                  </div>
                );
              })}
            </div>
            {numExperts > 32 && (
              <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", marginTop:5 }}>
                +{numExperts-32} experts on SSD
              </div>
            )}
            <div style={{ display:"flex", gap:10, marginTop:6, fontFamily:"'Share Tech Mono'", fontSize:7 }}>
              <span style={{ color:"#00ffe5" }}>◉ active</span>
              <span style={{ color:"#38bdf8" }}>◎ cached</span>
              <span style={{ color:"#1e3040" }}>○ on SSD</span>
            </div>
          </div>

          {/* LRU slots */}
          <div style={{ padding:"10px 14px", flex:1, overflowY:"auto" }}>
            <div style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:8 }}>
              LRU CACHE ({cacheSnap.length}/{modelInfo?.expert_cache_slots||3})
            </div>
            {cacheSnap.length===0
              ? <div style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#0d2030", fontStyle:"italic" }}>NO EXPERTS LOADED</div>
              : cacheSnap.map((s,rank)=>(
                  <div key={s.expert_id} style={{ padding:"7px 10px", marginBottom:3, background:"#020e18", border:"1px solid #0d2030", borderLeft:`2px solid ${rank===0&&cacheSnap.length>=(modelInfo?.expert_cache_slots||3)?"#ef444460":"#38bdf840"}` }}>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:2 }}>
                      <span style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:12, color:"#38bdf8" }}>EXPERT {s.expert_id}</span>
                      <span style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#1e3040" }}>{fmt(s.size_gb,2)}GB</span>
                    </div>
                    <div style={{ display:"flex", justifyContent:"space-between", fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040" }}>
                      <span>HITS: {s.hits}</span>
                      <span style={{ color:rank===0&&cacheSnap.length>=(modelInfo?.expert_cache_slots||3)?"#ef444470":"#1e3040" }}>
                        {rank===0&&cacheSnap.length>=(modelInfo?.expert_cache_slots||3)?"↑ EVICT NEXT":`RANK ${rank+1}`}
                      </span>
                    </div>
                  </div>
                ))
            }
          </div>

          {/* Stats */}
          <div style={{ padding:"10px 14px", borderTop:"1px solid #0d2030", display:"grid", gridTemplateColumns:"1fr 1fr", gap:4 }}>
            {[
              ["CACHE HITS",  cacheStats.hits,   "#00ffe5"],
              ["CACHE MISS",  cacheStats.misses, "#ef4444"],
              ["TOKENS",      finalStats?.tokens||0, "#38bdf8"],
              ["HIT RATE",    cacheStats.hit_rate!=null?`${(cacheStats.hit_rate*100).toFixed(0)}%`:"—", (cacheStats.hit_rate||0)>0.6?"#00ffe5":"#fbbf24"],
            ].map(([l,v,c])=>(
              <div className="metric-card" key={l}>
                <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em", marginBottom:3 }}>{l}</div>
                <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:22, color:c, lineHeight:1 }}>{v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── COL 2: Query + Output ── */}
        <div style={{ display:"flex", flexDirection:"column", overflow:"hidden", background:"#020609" }}>

          {/* Input */}
          <div style={{ padding:"14px 18px", borderBottom:"1px solid #0d2030", background:"#030b12" }}>
            <div style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#1e3040", letterSpacing:"0.15em", marginBottom:8 }}>
              ▸ QUERY INPUT
            </div>
            <div style={{ display:"flex", gap:8 }}>
              <div style={{ flex:1, position:"relative", border:"1px solid #0d2030", background:"#020609" }}>
                {/* Corner brackets */}
                <div style={{ position:"absolute", top:-1, left:-1, width:8, height:8, borderTop:"1px solid #00ffe540", borderLeft:"1px solid #00ffe540" }}/>
                <div style={{ position:"absolute", bottom:-1, right:-1, width:8, height:8, borderBottom:"1px solid #00ffe540", borderRight:"1px solid #00ffe540" }}/>
                <textarea
                  value={query}
                  onChange={e=>setQuery(e.target.value)}
                  onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();runInference();}}}
                  placeholder="Enter query... (ENTER to execute)"
                  disabled={isRunning}
                  style={{ width:"100%", background:"transparent", border:"none", padding:"10px 12px", color:"#8baabf", fontFamily:"'JetBrains Mono',monospace", fontSize:12, resize:"none", height:60, lineHeight:1.5 }}
                />
              </div>
              <button
                onClick={runInference}
                disabled={isRunning||!query.trim()}
                className={`btn-run ${isRunning?"running":""}`}
              >
                {isRunning ? "EXEC..." : "EXECUTE ↵"}
              </button>
            </div>

            {/* Example queries */}
            <div style={{ display:"flex", gap:5, marginTop:8, flexWrap:"wrap" }}>
              {EXAMPLES.map(ex=>(
                <button key={ex.domain} onClick={()=>setQuery(ex.q)} disabled={isRunning} className="example-btn"
                  style={{ borderColor: DOMAIN_CONFIG[ex.domain]?.color+"30", color: DOMAIN_CONFIG[ex.domain]?.color+"70" }}>
                  {DOMAIN_CONFIG[ex.domain]?.icon} {ex.label}
                </button>
              ))}
            </div>
          </div>

          {/* Analysis bar */}
          {analysis && (
            <div style={{ padding:"8px 18px", borderBottom:"1px solid #0d2030", background:statusColor||"#030b12", borderLeft:`3px solid ${domainColor}40`, display:"flex", gap:20, alignItems:"center", flexWrap:"wrap" }}>
              <div>
                <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em" }}>DOMAIN</div>
                <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:14, color:domainColor, letterSpacing:"0.1em" }}>
                  {DOMAIN_CONFIG[analysis.domain]?.icon} {analysis.domain.toUpperCase()}
                </div>
              </div>
              <div>
                <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em" }}>ROUTING</div>
                <div style={{ fontFamily:"'Share Tech Mono'", fontSize:11, color:"#8baabf" }}>E[{analysis.active_experts?.join(",")||""}]</div>
              </div>
              <div>
                <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em" }}>CONFIDENCE</div>
                <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:14, color:(analysis.confidence||0)>0.7?"#00ffe5":"#fbbf24" }}>
                  {((analysis.confidence||0)*100).toFixed(0)}%
                </div>
              </div>
              {finalStats && <>
                <div>
                  <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em" }}>SPEED</div>
                  <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:14, color:"#38bdf8" }}>
                    {finalStats.tokens_per_sec} tok/s
                  </div>
                </div>
                <div>
                  <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em" }}>TIME</div>
                  <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:14, color:"#8baabf" }}>
                    {finalStats.elapsed_sec}s
                  </div>
                </div>
              </>}
            </div>
          )}

          {/* Output */}
          <div style={{ flex:1, padding:"18px 20px", overflowY:"auto", position:"relative" }}>
            {phase==="idle" ? (
              <div style={{ textAlign:"center", marginTop:60, color:"#0d2030" }}>
                <div style={{ fontFamily:"'Share Tech Mono'", fontSize:32, marginBottom:12, color:"#0d2030", letterSpacing:8 }}>◈</div>
                <div style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:16, letterSpacing:"0.3em", color:"#1e3040" }}>LAZY-MOE INFERENCE SYSTEM</div>
                <div style={{ fontFamily:"'Share Tech Mono'", fontSize:8, marginTop:8, color:"#0d2030", letterSpacing:"0.1em" }}>
                  {health?.status==="offline" ? "SERVER OFFLINE — RUN: python server.py"
                    : modelInfo ? `${modelInfo.name} · ${modelInfo.layers} LAYERS · ${modelInfo.is_moe?`MOE ${modelInfo.num_experts} EXPERTS`:"DENSE"} · ${modelInfo.quant}`
                    : "AWAITING QUERY"}
                </div>
                <div style={{ marginTop:24, display:"flex", justifyContent:"center", gap:3 }}>
                  {[..."LazyMoE"].map((c,i)=>(
                    <span key={i} style={{ fontFamily:"'Share Tech Mono'", fontSize:10, color:`hsl(${170+i*5},80%,${25+i*2}%)`, animation:`pulse ${1.5+i*0.2}s infinite`, animationDelay:`${i*0.1}s` }}>
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <div>
                <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:12 }}>
                  <span style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#1e3040", letterSpacing:"0.15em" }}>OUTPUT STREAM</span>
                  {phase==="inferring" && (
                    <span style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#00ffe5", animation:"pulse 0.8s infinite" }}>▶ LIVE</span>
                  )}
                  {mockMode && <span style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#fbbf24", padding:"1px 6px", border:"1px solid #fbbf2430" }}>MOCK</span>}
                </div>

                {!tokenStream && (
                  <div style={{ fontFamily:"'Share Tech Mono'", fontSize:9, color:"#1e3040", animation:"pulse 1s infinite" }}>
                    {phase==="analyzing"?"ANALYZING QUERY DOMAIN...":phase==="prefetching"?"LOADING EXPERT WEIGHTS INTO RAM...":"WAITING FOR MODEL OUTPUT..."}
                  </div>
                )}

                <div className="token-output" style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:13, lineHeight:1.9, color:"#a8c4d4" }}>
                  {tokenStream}
                  {phase==="inferring" && (
                    <span style={{ display:"inline-block", width:8, height:14, background:"#00ffe5", marginLeft:2, animation:"pulse 0.7s infinite", boxShadow:"0 0 8px #00ffe5", verticalAlign:"middle" }}/>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div style={{ padding:"6px 18px", borderTop:"1px solid #0d2030", display:"flex", gap:12, alignItems:"center", background:"#030b12" }}>
            {[
              [API, "#1e3040"],
              ["SSE STREAM", "#1e3040"],
              [modelInfo?.name||"NO MODEL", "#1e3040"],
              [`KV ${modelInfo?.kv_bits||3}b TURBOQUANT`, "#f472b640"],
            ].map(([t,c],i)=>(
              <span key={i} style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:c, letterSpacing:"0.05em" }}>{t}</span>
            ))}
            <span style={{ marginLeft:"auto", fontFamily:"'Share Tech Mono'", fontSize:7, color:"#0d2030" }}>
              {new Date().toLocaleTimeString()}
            </span>
          </div>
        </div>

        {/* ── COL 3: TurboQuant ── */}
        <div className="panel" style={{ borderLeft:"1px solid #0d2030", display:"flex", flexDirection:"column", overflow:"hidden" }}>
          <div style={{ padding:"10px 14px", borderBottom:"1px solid #0d2030" }}>
            <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:11, color:"#f472b6", letterSpacing:"0.2em" }}>TURBOQUANT</div>
            <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em" }}>KV CACHE COMPRESSION</div>
          </div>

          {/* KV metrics */}
          <div style={{ padding:"10px 14px", borderBottom:"1px solid #0d2030", display:"grid", gridTemplateColumns:"1fr 1fr", gap:4 }}>
            {[
              ["TOKENS",    kvStats.tokens,                             "#8baabf"],
              ["RATIO",     `${fmt(kvStats.compression_ratio,1)}×`,    "#f472b6"],
              ["RAW fp16",  `${fmt(kvStats.raw_fp16_gb,3)}G`,          "#ff6b35"],
              ["TQ 3-bit",  `${fmt(kvStats.ram_used_gb,3)}G`,          "#f472b6"],
            ].map(([l,v,c])=>(
              <div className="metric-card" key={l}>
                <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.08em", marginBottom:3 }}>{l}</div>
                <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:18, color:c, lineHeight:1 }}>{v}</div>
              </div>
            ))}
          </div>

          {/* KV chart */}
          <div style={{ padding:"10px 14px", borderBottom:"1px solid #0d2030" }}>
            <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em", marginBottom:8 }}>KV RAM GROWTH</div>
            <div style={{ background:"#020609", border:"1px solid #0d2030", padding:4, position:"relative" }}>
              <svg width={CW} height={CH}>
                {/* Grid */}
                {[0,0.25,0.5,0.75,1].map(f=>(
                  <line key={f} x1={0} y1={CH*f} x2={CW} y2={CH*f} stroke="#0d2030" strokeWidth={0.5} strokeDasharray={f===0||f===1?"none":"2,4"}/>
                ))}
                {kvHistory.length>1&&<>
                  {/* Area fill for TQ */}
                  <polygon
                    points={`0,${CH} ${pts("tq")} ${CW},${CH}`}
                    fill="#f472b608"
                  />
                  {/* Raw fp16 dashed */}
                  <polyline points={pts("raw")} fill="none" stroke="#ff6b3540" strokeWidth={1} strokeDasharray="4,3"/>
                  {/* TQ solid */}
                  <polyline points={pts("tq")} fill="none" stroke="#f472b6" strokeWidth={1.5}
                    filter="url(#glow)"/>
                  <defs>
                    <filter id="glow">
                      <feGaussianBlur stdDeviation="2" result="blur"/>
                      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                  </defs>
                </>}
                {kvHistory.length===0 && (
                  <text x={CW/2} y={CH/2} textAnchor="middle" fontSize={8} fill="#0d2030" fontFamily="Share Tech Mono">NO DATA</text>
                )}
              </svg>
            </div>
            <div style={{ display:"flex", gap:10, marginTop:4, fontFamily:"'Share Tech Mono'", fontSize:7 }}>
              <span style={{ color:"#ff6b3540" }}>── RAW fp16</span>
              <span style={{ color:"#f472b6" }}>── TURBOQUANT</span>
            </div>
          </div>

          {/* Pipeline */}
          <div style={{ padding:"10px 14px", borderBottom:"1px solid #0d2030" }}>
            <div style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:8 }}>PIPELINE</div>
            {[
              {n:"01",label:"RANDOM ROTATE",  desc:"spread outlier activations",  c:"#c084fc"},
              {n:"02",label:"POLARQUANT",      desc:"3-bit angle quantization",    c:"#f472b6"},
              {n:"03",label:"QJL RESIDUAL",    desc:"1-bit bias correction",       c:"#38bdf8"},
              {n:"04",label:"ATTENTION",       desc:"unbiased inner products",     c:"#00ffe5"},
            ].map(s=>(
              <div key={s.n} style={{ display:"flex", gap:10, marginBottom:8, alignItems:"flex-start" }}>
                <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:14, color:s.c, minWidth:24, lineHeight:1 }}>{s.n}</div>
                <div>
                  <div style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:11, color:s.c, letterSpacing:"0.05em" }}>{s.label}</div>
                  <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040" }}>{s.desc}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Comparison */}
          <div style={{ padding:"10px 14px", flex:1, overflowY:"auto" }}>
            <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040", letterSpacing:"0.1em", marginBottom:8 }}>
              KV @ 8K CTX · {modelInfo?`${modelInfo.params_b}B`:"70B"}
            </div>
            {[
              {m:"fp16 RAW", gb:"21.0", comp:"1×",   bar:1.0,   c:"#1e3040"},
              {m:"KIVI 4b",  gb:"5.3",  comp:"4×",   bar:0.25,  c:"#fbbf24"},
              {m:"TURBOQUANT",gb:"3.5", comp:"6×",   bar:0.167, c:"#f472b6"},
            ].map(r=>(
              <div key={r.m} style={{ marginBottom:8 }}>
                <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3 }}>
                  <span style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:r.c }}>{r.m}</span>
                  <div style={{ display:"flex", gap:8 }}>
                    <span style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#1e3040" }}>{r.gb}GB</span>
                    <span style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:10, color:r.c }}>{r.comp}</span>
                  </div>
                </div>
                <div style={{ height:2, background:"#0d2030" }}>
                  <div style={{ height:"100%", width:`${r.bar*100}%`, background:r.c, boxShadow:`0 0 4px ${r.c}` }}/>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── COL 4: System Log ── */}
        <div className="panel" style={{ borderLeft:"1px solid #0d2030", display:"flex", flexDirection:"column", overflow:"hidden" }}>
          <div style={{ padding:"10px 14px", borderBottom:"1px solid #0d2030", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <div>
              <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:11, color:"#1e3040", letterSpacing:"0.2em" }}>SYSTEM LOG</div>
              <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#0d2030" }}>REAL-TIME PIPELINE EVENTS</div>
            </div>
            <button onClick={()=>{logRef.current=[];setLog([]);}}
              style={{ fontFamily:"'Share Tech Mono'", fontSize:7, padding:"2px 6px", background:"transparent", border:"1px solid #0d2030", color:"#1e3040", cursor:"pointer", letterSpacing:"0.1em" }}>
              CLR
            </button>
          </div>

          <div style={{ flex:1, overflowY:"auto", padding:"8px 12px" }}>
            {log.length===0
              ? <div style={{ fontFamily:"'Share Tech Mono'", fontSize:8, color:"#0d2030", fontStyle:"italic", marginTop:8 }}>AWAITING INFERENCE...</div>
              : log.map((e,i)=>(
                  <div key={i} className="log-e" style={{ fontFamily:"'Share Tech Mono'", fontSize:8, lineHeight:1.8, marginBottom:1, letterSpacing:"0.02em",
                    color: e.type==="success"?"#00ffe5":e.type==="hit"?"#38bdf8":e.type==="miss"?"#ef4444":e.type==="evict"?"#fbbf24":e.type==="system"?"#c084fc":"#1e3040" }}>
                    {e.msg}
                  </div>
                ))
            }
            <div ref={logEnd}/>
          </div>

          {/* Pipeline status */}
          <div style={{ padding:"10px 14px", borderTop:"1px solid #0d2030" }}>
            <div style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:8 }}>PIPELINE STATUS</div>
            {[
              {id:"analyzing",   n:"01", label:"QUERY ANALYZE",  desc:"domain · expert predict"},
              {id:"prefetching", n:"02", label:"EXPERT PREFETCH", desc:"SSD → LRU cache"},
              {id:"inferring",   n:"03", label:"INFERENCE + KV",  desc:"tokens · TurboQuant"},
            ].map(({id,n,label,desc})=>{
              const done = phase==="done"||(id==="analyzing"&&["prefetching","inferring","done"].includes(phase))||(id==="prefetching"&&["inferring","done"].includes(phase));
              const active = phase===id;
              const color = done?"#00ffe5":active?"#fbbf24":"#0d2030";
              return (
                <div key={id} style={{ display:"flex", gap:10, marginBottom:8, padding:"6px 8px", background:active?"#020e18":"transparent", border:`1px solid ${active?"#fbbf2420":"transparent"}` }}>
                  <div style={{ fontFamily:"'Rajdhani'", fontWeight:700, fontSize:14, color, lineHeight:1, minWidth:20 }}>{n}</div>
                  <div>
                    <div style={{ fontFamily:"'Rajdhani'", fontWeight:600, fontSize:11, color, letterSpacing:"0.05em" }}>{label}</div>
                    <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#0d2030" }}>{desc}</div>
                  </div>
                  {done && <span style={{ marginLeft:"auto", color:"#00ffe5", fontSize:10 }}>✓</span>}
                  {active && <span style={{ marginLeft:"auto", color:"#fbbf24", fontSize:8, animation:"pulse 1s infinite", fontFamily:"'Share Tech Mono'" }}>ACTIVE</span>}
                </div>
              );
            })}
          </div>

          {/* Supported models */}
          <div style={{ padding:"10px 14px", borderTop:"1px solid #0d2030" }}>
            <div style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#0d2030", letterSpacing:"0.1em", marginBottom:6 }}>SUPPORTED MODELS</div>
            {["Mistral · Mixtral", "Llama 3 8B–405B", "Qwen2/2.5", "DeepSeek V2/V3/R1", "Phi · Gemma · Falcon"].map((m,i)=>(
              <div key={i} style={{ fontFamily:"'Share Tech Mono'", fontSize:7, color:"#0d2030", marginBottom:2 }}>· {m}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
