"""Render a recorded duel trajectory to a self-contained interactive HTML file.

The page shows a top-down SVG of the arena (x/z) with player markers, HP bars,
a charge/cooldown bar, and ground/air status. Use the slider or space bar to
scrub, or press play to animate.
"""
import os
import json

from mcbot.sim import consts as C

_W = 560
_PX = _W / 26.0          # pixels per block (arena half-size 12 -> 24 wide + margin)
_CX, _CY = _W / 2, _W / 2

_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body{{background:#0f1115;color:#e8e8e8;font-family:system-ui,sans-serif;margin:0;padding:18px}}
 h1{{font-size:16px;margin:0 0 4px}}
 .sub{{color:#888;font-size:12px;margin-bottom:12px}}
 .grid{{display:flex;gap:18px;flex-wrap:wrap}}
 svg{{background:#1a1d24;border:1px solid #2a2f3a;border-radius:8px}}
 .panel{{min-width:220px}}
 .hprow{{margin:6px 0}}
 .hpbar{{background:#2a2f3a;border-radius:4px;height:12px;width:100%;overflow:hidden}}
 .hpfill{{height:100%;width:0%;transition:width .05s}}
 .controls{{margin:10px 0}}
 button{{background:#2a2f3a;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 12px;cursor:pointer;margin-right:6px}}
 button:hover{{background:#3a4150}}
 input[type=range]{{width:100%}}
 .tag{{font-size:12px;color:#aaa}}
</style></head><body>
<h1>{title}</h1>
<div class="sub">{a}  vs  {b} &middot; {frames} ticks (1 tick = 50 ms) &middot;
A=blue B=red &middot; small dot = airborne</div>
<div class="grid">
 <svg id="svg" width="{W}" height="{W}"></svg>
 <div class="panel">
   <div class="controls">
     <button onclick="play()">&#9654; play</button>
     <button onclick="pause()">&#10074;&#10074;</button>
     <button onclick="step(1)">+1</button>
     <button onclick="step(-1)">-1</button>
   </div>
   <div class="hprow"><span class="tag">A (blue)</span>
     <div class="hpbar"><div class="hpfill" id="hpa" style="background:#4a9eff"></div></div></div>
   <div class="hprow"><span class="tag">B (red)</span>
     <div class="hpbar"><div class="hpfill" id="hpb" style="background:#ff5d5d"></div></div></div>
   <div class="hprow"><span class="tag">charge</span>
     <div class="hpbar"><div class="hpfill" id="chg" style="background:#ffd75d"></div></div></div>
   <div class="controls"><input type="range" id="slider" min="0" max="0" value="0"
     oninput="setFrame(+this.value)"></div>
   <div class="tag" id="tick">tick 0</div>
   <div class="tag" id="status"></div>
 </div>
</div>
<script>
const FRAMES = __FRAMES_JSON__;
const W = {W}, CX = {CX}, CY = {CY}, PX = {PX};
const svg = document.getElementById('svg');
const NS='http://www.w3.org/2000/svg';
const slider = document.getElementById('slider');
let idx=0, timer=null, playing=false;
function el(tag, attrs){{const e=document.createElementNS(NS,tag);
 for(const k in attrs) e.setAttribute(k,attrs[k]); return e;}}
function px(x){{return CX + x*PX;}}
function py(z){{return CY + z*PX;}}
// grid lines
for(let i=-12;i<=12;i+=3){{
  svg.appendChild(el('line',{{x1:px(i),y1:0,x2:px(i),y2:W,stroke:'#262b36','stroke-width':1}}));
  svg.appendChild(el('line',{{x1:0,y1:py(i),x2:W,y2:py(i),stroke:'#262b36','stroke-width':1}}));
}}
let aDot=el('circle',{{r:10,fill:'#4a9eff','stroke':'#fff','stroke-width':1.5}});
let bDot=el('circle',{{r:10,fill:'#ff5d5d','stroke':'#fff','stroke-width':1.5}});
let aAir=el('circle',{{r:4,fill:'#fff'}}), bAir=el('circle',{{r:4,fill:'#fff'}});
let aMark=el('circle',{{r:4,fill:'#4a9eff'}}), bMark=el('circle',{{r:4,fill:'#ff5d5d'}});
svg.appendChild(aDot);svg.appendChild(bDot);svg.appendChild(aMark);svg.appendChild(bMark);
svg.appendChild(aAir);svg.appendChild(bAir);
function setFrame(i){{
  idx=i; const f=FRAMES[i];
  const a=f[0], b=f[1];
  aDot.setAttribute('cx',px(a[0])); aDot.setAttribute('cy',py(a[1]));
  bDot.setAttribute('cx',px(b[0])); bDot.setAttribute('cy',py(b[1]));
  const airA = a[10]<0.5, airB = b[10]<0.5;
  aAir.setAttribute('cx',px(a[0])); aAir.setAttribute('cy',py(a[1])+(airA?-6:0));
  bAir.setAttribute('cx',px(b[0])); bAir.setAttribute('cy',py(b[1])+(airB?-6:0));
  aAir.setAttribute('display',airA?'':'none');
  bAir.setAttribute('display',airB?'':'none');
  document.getElementById('hpa').style.width=(a[6]/20*100)+'%';
  document.getElementById('hpb').style.width=(b[6]/20*100)+'%';
  document.getElementById('chg').style.width=(Math.min(1,Math.max(0,a[7]))*100)+'%';
  document.getElementById('tick').textContent='tick '+i+'/'+(FRAMES.length-1);
  const sa=a[8]>0.5?'sprint ':'', sb=b[8]>0.5?'sprint ':'';
  document.getElementById('status').textContent='A hp '+a[6].toFixed(1)+
    (a[10]<0.5?' (air) ':' ') + sa + '| B hp '+b[6].toFixed(1)+(b[10]<0.5?' (air) ':' ') + sb;
  slider.value=i;
}}
function step(d){{setFrame(Math.max(0,Math.min(FRAMES.length-1,idx+d)));}}
function play(){{if(timer)return;playing=true;timer=setInterval(()=>{{step(1); if(idx>=FRAMES.length-1)pause();}},50);}}
function pause(){{playing=false;clearInterval(timer);timer=null;}}
document.addEventListener('keydown',e=>{{if(e.code==='Space'){{e.preventDefault();playing?pause():play();}}}});
slider.setAttribute('max',(FRAMES.length-1));
setFrame(0);
</script>
</body></html>
"""


def write_replay(trace, out_path):
    frames = trace["frames"]
    data = []
    for f in frames:
        # f shape (1, 2, NOBS); flatten to [[a0..a25],[b0..b25]]
        a = [float(x) for x in f[0, 0]]
        b = [float(x) for x in f[0, 1]]
        data.append([a, b])
    html = _HTML.format(
        title=trace.get("name", "duel"),
        a=trace.get("a", "A"),
        b=trace.get("b", "B"),
        frames=len(data),
        W=_W, CX=_CX, CY=_CY, PX=_PX,
    )
    html = html.replace("__FRAMES_JSON__", json.dumps(data))
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path
