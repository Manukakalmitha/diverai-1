const n=l;function s(){const x=["#1e293b",'<img src="',"boxShadow",'[class*="layout-header"] [class*="group-left"]',"createElement","getURL","querySelector","#header-wrapper","zIndex","12RVAFeO","padding","center","136086tYWutq","8px","lastError","DiverAI: Message sending failed (likely extension context invalidated).","runtime","style","120716yxaCbh",".diver-ai-btn","position","appendChild","OPEN_SIDEBAR","12327720xSWrQV","innerHTML","1907568oLaGFD","3502776ngUmSv","header","all 0.3s cubic-bezier(0.4, 0, 0.2, 1)","transition","7YgDepn","button","2147483647","blur(8px)","className","justifyContent","textContent","669215BhXYNM","debug","backgroundColor","883664WKKEbG","20px","DiverAI: Connection to background script lost/pending. Re-injection may be needed.","right","75IPOVgR","pulse-icon.png",'" alt="DiverAI" style="height: 24px; width: 24px; vertical-align: middle; pointer-events:none; border-radius: 6px;" />',"bottom","1px solid #334155","display","alignItems"];return s=function(){return x},s()}(function(x,r){const a=l,e=x();for(;;)try{if(-parseInt(a(150))/1+-parseInt(a(127))/2+parseInt(a(131))/3*(parseInt(a(156))/4)+parseInt(a(124))/5*(-parseInt(a(147))/6)+parseInt(a(168))/7*(-parseInt(a(164))/8)+-parseInt(a(163))/9+parseInt(a(161))/10===r)break;e.push(e.shift())}catch{e.push(e.shift())}})(s,491844);const d=document[n(142)](n(155));d[n(123)]=`
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
`,document.head[n(159)](d);function b(){const x=n;if(document[x(144)](x(157)))return;const r=[x(141),"#header-toolbar-symbol-search",x(145),"#ybar-inner-wrap",x(165),".header-main"];let a=null;for(const t of r){const o=document[x(144)](t);if(o){a=o;break}}const e=document[x(142)](x(169));e[x(172)]="diver-ai-btn";const i=chrome.runtime[x(143)](x(132));e[x(162)]=x(139)+i+x(133),e[x(155)][x(148)]=x(151),e[x(155)][x(136)]="flex",e.style[x(137)]=x(149),e[x(155)][x(122)]=x(149),e[x(155)][x(126)]=x(138),e[x(155)].border=x(135),e[x(155)].borderRadius="12px",e.style[x(167)]=x(166),e[x(155)].backdropFilter=x(171),e.onclick=()=>{const t=x;try{chrome[t(154)].sendMessage({action:t(160)},o=>{const c=t;chrome.runtime[c(152)]&&console[c(125)](c(129))})}catch{console.debug(t(153))}},a?a[x(159)](e):(e[x(155)][x(158)]="fixed",e[x(155)][x(134)]=x(128),e.style[x(130)]=x(128),e[x(155)][x(146)]=x(170),e.style[x(140)]="0 4px 14px rgba(0,0,0,0.4)",document.body[x(159)](e))}const p=new MutationObserver(x=>{b()});function l(x,r){return x=x-122,s()[x]}p.observe(document.body,{childList:!0,subtree:!0}),setTimeout(b,2e3);
