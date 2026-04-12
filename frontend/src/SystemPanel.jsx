import { useState, useEffect } from "react";

const API = "http://localhost:8000";

const STATUS_CONFIG = {
  great: { color:"#00ffe5", bg:"#001a14", label:"RUNS GREAT",  icon:"◉" },
  ok:    { color:"#38bdf8", bg:"#001220", label:"RUNS OK",     icon:"◎" },
  mmap:  { color:"#fbbf24", bg:"#1a1200", label:"SSD STREAM",  icon:"◈" },
  no:    { color:"#ef4444", bg:"#1a0000", label:"NOT ENOUGH RAM", icon:"✗" },
};

const VENDOR_COLORS = {
  intel: "#38bdf8", amd: "#ef4444", nvidia: "#22c55e",
  apple: "#c084fc", unknown: "#1e3040",
};

function StatBar({ value, max, color }) {
  const pct = Math.min((value/max)*100, 100);
  return (
    <div style={{ height:3, background:"#0d2030", borderRadius:1, overflow:"hidden" }}>
      <div style={{ height:"100%", width:`${pct}%`, background:color, boxShadow:`0 0 4px ${color}`, transition:"width 0.8s ease" }}/>
    </div>
  );
}

export default function SystemPanel({ onClose }) {
  const [sysInfo, setSysInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all"); // all | great | ok | mmap | no

  useEffect(() => {
    fetch(`${API}/system`)
      .then(r => r.json())
      .then(d => { setSysInfo(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const filtered = sysInfo?.model_compatibility?.filter(m =>
    filter === "all" || m.status === filter
  ) || [];

  const statusCounts = sysInfo?.model_compatibility?.reduce((acc, m) => {
    acc[m.status] = (acc[m.status] || 0) + 1;
    return acc;
  }, {}) || {};

  return (
    <div style={{
      position:"fixed", top:0, left:0, right:0, bottom:0,
      background:"rgba(2,6,9,0.95)", zIndex:9999,
      display:"flex", alignItems:"center", justifyContent:"center",
      backdropFilter:"blur(4px)",
    }}>
      <div style={{
        width:"min(900px,95vw)", maxHeight:"90vh",
        background:"#030b12", border:"1px solid #0d2030",
        display:"flex", flexDirection:"column", position:"relative",
        overflow:"hidden",
      }}>
        {/* Corner decorations */}
        {[["top:0,left:0","borderTop,borderLeft"],["top:0,right:0","borderTop,borderRight"],
          ["bottom:0,left:0","borderBottom,borderLeft"],["bottom:0,right:0","borderBottom,borderRight"]
        ].map(([pos],i) => (
          <div key={i} style={{
            position:"absolute", width:12, height:12,
            ...(Object.fromEntries(pos.split(",").map(p => p.split(":")))),
            border:"1px solid #00ffe540",
            ...(i===0?{borderRight:"none",borderBottom:"none"}:
                i===1?{borderLeft:"none",borderBottom:"none"}:
                i===2?{borderRight:"none",borderTop:"none"}:
                      {borderLeft:"none",borderTop:"none"}),
          }}/>
        ))}

        {/* Header */}
        <div style={{ padding:"16px 20px", borderBottom:"1px solid #0d2030", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <div>
            <div style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:700, fontSize:16, color:"#00ffe5", letterSpacing:"0.2em" }}>
              SYSTEM DIAGNOSTICS
            </div>
            <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:8, color:"#1e3040", letterSpacing:"0.1em", marginTop:2 }}>
              HARDWARE DETECTION · MODEL COMPATIBILITY MATRIX
            </div>
          </div>
          <button onClick={onClose} style={{
            fontFamily:"'Share Tech Mono',monospace", fontSize:10, padding:"4px 12px",
            background:"transparent", border:"1px solid #1e3040", color:"#1e3040",
            cursor:"pointer", letterSpacing:"0.1em",
          }}>✕ CLOSE</button>
        </div>

        {loading ? (
          <div style={{ padding:40, textAlign:"center", fontFamily:"'Share Tech Mono',monospace", fontSize:10, color:"#1e3040" }}>
            SCANNING HARDWARE...
            <div style={{ marginTop:12, display:"flex", justifyContent:"center", gap:4 }}>
              {Array(6).fill(0).map((_,i)=>(
                <div key={i} style={{ width:4, height:4, background:"#00ffe5", borderRadius:1, animation:"pulse 1s infinite", animationDelay:`${i*0.1}s` }}/>
              ))}
            </div>
          </div>
        ) : !sysInfo ? (
          <div style={{ padding:40, textAlign:"center", fontFamily:"'Share Tech Mono',monospace", fontSize:10, color:"#ef4444" }}>
            COULD NOT CONNECT TO SERVER
          </div>
        ) : (
          <div style={{ display:"flex", flex:1, overflow:"hidden" }}>

            {/* LEFT: Hardware info */}
            <div style={{ width:280, borderRight:"1px solid #0d2030", overflowY:"auto", flexShrink:0 }}>

              {/* OS */}
              <div style={{ padding:"12px 16px", borderBottom:"1px solid #0d2030" }}>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:6 }}>OPERATING SYSTEM</div>
                <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:9, color:"#38bdf8" }}>{sysInfo.os.name} {sysInfo.os.arch}</div>
                <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:"#1e3040", marginTop:2 }}>{sysInfo.os.version}</div>
              </div>

              {/* CPU */}
              <div style={{ padding:"12px 16px", borderBottom:"1px solid #0d2030" }}>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:6 }}>CPU</div>
                <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:8, color: VENDOR_COLORS[sysInfo.cpu.vendor]||"#8baabf", marginBottom:4, lineHeight:1.4 }}>{sysInfo.cpu.name}</div>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"3px 8px" }}>
                  {[
                    ["PHYSICAL", `${sysInfo.cpu.cores_physical} cores`],
                    ["LOGICAL",  `${sysInfo.cpu.cores_logical} threads`],
                    ["SPEED",    sysInfo.cpu.freq_mhz ? `${(sysInfo.cpu.freq_mhz/1000).toFixed(1)} GHz` : "—"],
                    ["VENDOR",   sysInfo.cpu.vendor.toUpperCase()],
                  ].map(([l,v])=>(
                    <div key={l}>
                      <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:"#1e3040" }}>{l}</div>
                      <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:8, color:"#8baabf" }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* RAM */}
              <div style={{ padding:"12px 16px", borderBottom:"1px solid #0d2030" }}>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:6 }}>
                  {sysInfo.apple_silicon ? "UNIFIED MEMORY" : "RAM"}
                </div>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:700, fontSize:24, color:"#00ffe5", lineHeight:1, marginBottom:6 }}>
                  {sysInfo.apple_silicon ? sysInfo.unified_memory_gb : sysInfo.ram.total_gb}
                  <span style={{ fontSize:12, color:"#1e3040", marginLeft:4 }}>GB</span>
                </div>
                <StatBar value={sysInfo.ram.used_pct} max={100} color="#00ffe5"/>
                <div style={{ display:"flex", justifyContent:"space-between", marginTop:4, fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:"#1e3040" }}>
                  <span>USED {sysInfo.ram.used_pct}%</span>
                  <span>FREE {sysInfo.ram.available_gb.toFixed(1)}GB</span>
                </div>
                <div style={{ marginTop:6, padding:"4px 8px", background:"#001a14", border:"1px solid #00ffe520", fontFamily:"'Share Tech Mono',monospace", fontSize:7 }}>
                  <span style={{ color:"#1e3040" }}>EFFECTIVE FOR MODELS: </span>
                  <span style={{ color:"#00ffe5" }}>{sysInfo.ram.effective_gb}GB</span>
                </div>
              </div>

              {/* GPU */}
              <div style={{ padding:"12px 16px", borderBottom:"1px solid #0d2030" }}>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:6 }}>GPU</div>
                {sysInfo.gpu.devices.length === 0 ? (
                  <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:8, color:"#1e3040" }}>NO GPU DETECTED</div>
                ) : sysInfo.gpu.devices.map((g,i)=>(
                  <div key={i} style={{ marginBottom:8 }}>
                    <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:8, color: VENDOR_COLORS[g.vendor]||"#8baabf", marginBottom:2 }}>{g.name}</div>
                    <div style={{ display:"flex", gap:12, fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:"#1e3040" }}>
                      <span>{g.integrated?"INTEGRATED":"DISCRETE"}</span>
                      {g.vram_gb > 0 && <span>{g.vram_gb}GB VRAM</span>}
                    </div>
                  </div>
                ))}
              </div>

              {/* Disk */}
              <div style={{ padding:"12px 16px" }}>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:600, fontSize:10, color:"#1e3040", letterSpacing:"0.15em", marginBottom:6 }}>STORAGE</div>
                <div style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:700, fontSize:20, color:"#c084fc", lineHeight:1, marginBottom:6 }}>
                  {sysInfo.disk.free_gb.toFixed(0)}
                  <span style={{ fontSize:11, color:"#1e3040", marginLeft:4 }}>GB FREE</span>
                </div>
                <StatBar value={sysInfo.disk.total_gb - sysInfo.disk.free_gb} max={sysInfo.disk.total_gb} color="#c084fc"/>
                <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:"#1e3040", marginTop:4 }}>
                  {sysInfo.disk.total_gb.toFixed(0)}GB TOTAL
                </div>
              </div>
            </div>

            {/* RIGHT: Model compatibility */}
            <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden" }}>

              {/* Filter tabs */}
              <div style={{ padding:"10px 16px", borderBottom:"1px solid #0d2030", display:"flex", gap:6, alignItems:"center", flexWrap:"wrap" }}>
                <span style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:8, color:"#1e3040", letterSpacing:"0.1em", marginRight:4 }}>FILTER:</span>
                {[
                  ["all",   "ALL",        "#8baabf", Object.values(statusCounts).reduce((a,b)=>a+b,0)],
                  ["great", "RUNS GREAT", "#00ffe5", statusCounts.great||0],
                  ["ok",    "RUNS OK",    "#38bdf8", statusCounts.ok||0],
                  ["mmap",  "SSD STREAM", "#fbbf24", statusCounts.mmap||0],
                  ["no",    "TOO HEAVY",  "#ef4444", statusCounts.no||0],
                ].map(([val, label, color, count])=>(
                  <button key={val} onClick={()=>setFilter(val)} style={{
                    fontFamily:"'Share Tech Mono',monospace", fontSize:7, padding:"3px 8px",
                    background:filter===val?`${color}20`:"transparent",
                    border:`1px solid ${filter===val?color:"#0d2030"}`,
                    color:filter===val?color:"#1e3040",
                    cursor:"pointer", letterSpacing:"0.08em",
                  }}>
                    {label} ({count})
                  </button>
                ))}
              </div>

              {/* Model list */}
              <div style={{ flex:1, overflowY:"auto", padding:"8px 12px" }}>
                {filtered.map((model, i) => {
                  const sc = STATUS_CONFIG[model.status];
                  return (
                    <div key={i} style={{
                      marginBottom:4, padding:"8px 12px",
                      background:sc.bg, border:`1px solid ${sc.color}20`,
                      borderLeft:`3px solid ${sc.color}`,
                      display:"flex", alignItems:"center", gap:12,
                    }}>
                      {/* Status */}
                      <div style={{ minWidth:90 }}>
                        <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:8, color:sc.color, letterSpacing:"0.05em" }}>
                          {sc.icon} {sc.label}
                        </div>
                      </div>

                      {/* Name + params */}
                      <div style={{ flex:1 }}>
                        <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:2 }}>
                          <span style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:600, fontSize:13, color:"#8baabf" }}>
                            {model.name}
                          </span>
                          {model.is_moe && (
                            <span style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, padding:"1px 5px", border:"1px solid #fbbf2430", color:"#fbbf2470" }}>MoE</span>
                          )}
                        </div>
                        <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:"#1e3040" }}>
                          {model.note}
                        </div>
                      </div>

                      {/* Stats */}
                      <div style={{ display:"flex", gap:12, flexShrink:0 }}>
                        <div style={{ textAlign:"right" }}>
                          <div style={{ fontFamily:"'Rajdhani',sans-serif", fontWeight:700, fontSize:14, color:sc.color, lineHeight:1 }}>
                            {model.params}B
                          </div>
                          <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:"#1e3040" }}>PARAMS</div>
                        </div>
                        <div style={{ textAlign:"right" }}>
                          <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:8, color:"#8baabf" }}>
                            {model.min_ram}GB+
                          </div>
                          <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:"#1e3040" }}>MIN RAM</div>
                        </div>
                        <div style={{ textAlign:"right", minWidth:70 }}>
                          <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, color: model.status==="great"?"#00ffe5":model.status==="no"?"#1e3040":"#8baabf" }}>
                            {model.speed_label}
                          </div>
                          <div style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:"#1e3040" }}>SPEED</div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Bottom hint */}
              <div style={{ padding:"8px 16px", borderTop:"1px solid #0d2030", display:"flex", gap:16 }}>
                {Object.entries(STATUS_CONFIG).map(([k,v])=>(
                  <span key={k} style={{ fontFamily:"'Share Tech Mono',monospace", fontSize:7, color:v.color, display:"flex", alignItems:"center", gap:4 }}>
                    <span>{v.icon}</span> {v.label}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
      <style>{`
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
      `}</style>
    </div>
  );
}
