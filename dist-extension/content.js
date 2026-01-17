const s=document.createElement("style");s.textContent=`
  .diver-ai-btn {
    background: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 8px;
    cursor: pointer;
    margin-left: 10px;
    font-family: -apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, Ubuntu, sans-serif;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 9999;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .diver-ai-btn:hover {
    background: #334155;
    transform: scale(1.05) translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6), 0 0 0 1px #10b981;
  }
  .diver-ai-btn:active {
    transform: scale(0.95);
  }
  @keyframes heartbeat {
    0% { transform: scale(1); }
    15% { transform: scale(1.1); }
    30% { transform: scale(1); }
    45% { transform: scale(1.15); }
    70% { transform: scale(1); }
  }
  .diver-ai-btn img {
    animation: heartbeat 3s infinite ease-in-out;
  }
`;document.head.appendChild(s);function a(){if(document.querySelector(".diver-ai-btn"))return;const t=['[class*="layout-header"] [class*="group-left"]',"#header-toolbar-symbol-search","#header-wrapper","#ybar-inner-wrap","header",".header-main"];let o=null;for(const r of t){const i=document.querySelector(r);if(i){o=i;break}}const e=document.createElement("button");e.className="diver-ai-btn";const n=chrome.runtime.getURL("pulse-icon.png");e.innerHTML=`<img src="${n}" alt="DiverAI" style="height: 24px; width: 24px; vertical-align: middle; pointer-events:none; border-radius: 6px;" />`,e.style.padding="8px",e.style.display="flex",e.style.alignItems="center",e.style.justifyContent="center",e.style.backgroundColor="#1e293b",e.style.border="1px solid #334155",e.style.borderRadius="12px",e.style.transition="all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",e.style.backdropFilter="blur(8px)",e.onclick=()=>{try{chrome.runtime.sendMessage({action:"OPEN_SIDEBAR"},r=>{chrome.runtime.lastError&&console.debug("DiverAI: Connection to background script lost/pending. Re-injection may be needed.")})}catch{console.debug("DiverAI: Message sending failed (likely extension context invalidated).")}},o?o.appendChild(e):(e.style.position="fixed",e.style.bottom="20px",e.style.right="20px",e.style.zIndex="2147483647",e.style.boxShadow="0 4px 14px rgba(0,0,0,0.4)",document.body.appendChild(e))}function l(t,o){const e=document.getElementById("diver-ai-rr-overlay");e&&e.remove();const n=document.querySelector(".chart-container-border")||document.body,r=document.createElement("div");r.id="diver-ai-rr-overlay",r.style.position="absolute",r.style.inset="0",r.style.pointerEvents="none",r.style.zIndex="100",document.querySelector('[class*="price-axis"]'),n.getBoundingClientRect(),r.innerHTML=`
    <div style="position: absolute; top: 20px; left: 20px; background: rgba(15, 23, 42, 0.9); border: 1px solid #3b82f6; border-radius: 12px; padding: 15px; font-family: monospace; color: white; backdrop-filter: blur(8px); box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
      <div style="font-size: 10px; color: #64748b; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px;">Protocol Overlay: ${o}</div>
      <div style="display: grid; gap: 8px;">
        <div style="color: #3b82f6;">ENTRY: ${t.entry}</div>
        <div style="color: #10b981;">TP1: ${t.tp1}</div>
        <div style="color: #10b981;">TP2: ${t.tp2}</div>
        <div style="color: #f43f5e;">SL: ${t.sl}</div>
        <div style="margin-top: 5px; font-weight: bold;">R/R: ${t.rr}</div>
      </div>
    </div>
  `,n.appendChild(r),setTimeout(()=>r.remove(),3e4)}chrome.runtime.onMessage.addListener((t,o,e)=>{if(t.action==="DRAW_RR_OVERLAY")return console.log("[Content Script] Received DRAW_RR_OVERLAY request:",t),l(t.targets,t.ticker),e({success:!0}),!0});const d=new MutationObserver(t=>{a()});d.observe(document.body,{childList:!0,subtree:!0});setTimeout(a,2e3);
