const r=document.createElement("style");r.textContent=`
  .diver-ai-btn {
    background: #10b981;
    color: #020617;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 700;
    cursor: pointer;
    margin-left: 10px;
    font-family: -apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, Ubuntu, sans-serif;
    transition: all 0.2s;
    z-index: 9999;
  }
  .diver-ai-btn:hover {
    background: #34d399;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
  }
`;document.head.appendChild(r);function s(){if(document.querySelector(".diver-ai-btn"))return;const n=['[class*="layout-header"] [class*="group-left"]',"#header-toolbar-symbol-search","#header-wrapper","#ybar-inner-wrap","header",".header-main"];let t=null;for(const i of n){const o=document.querySelector(i);if(o){t=o;break}}const e=document.createElement("button");e.className="diver-ai-btn";const a=chrome.runtime.getURL("logo-text.png");e.innerHTML=`<img src="${a}" alt="DiverAI" style="height: 20px; vertical-align: middle;pointer-events:none;" />`,e.style.padding="6px 12px",e.style.display="flex",e.style.alignItems="center",e.style.justifyContent="center",e.onclick=()=>{chrome.runtime.sendMessage({action:"OPEN_SIDEBAR"})},t?t.appendChild(e):(e.style.position="fixed",e.style.bottom="20px",e.style.right="20px",e.style.zIndex="2147483647",e.style.boxShadow="0 4px 14px rgba(0,0,0,0.4)",document.body.appendChild(e))}const l=new MutationObserver(n=>{s()});l.observe(document.body,{childList:!0,subtree:!0});setTimeout(s,2e3);
