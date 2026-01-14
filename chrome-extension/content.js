// Inject Styling
const style = document.createElement('style');
style.textContent = `
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
`;
document.head.appendChild(style);

// Function to add the button
function addDiverButton() {
  if (document.querySelector('.diver-ai-btn')) return;

  // Platform-Specific Selectors (Tries to find a native toolbar)
  const selectors = [
    // TradingView
    '[class*="layout-header"] [class*="group-left"]',
    '#header-toolbar-symbol-search',
    // Yahoo Finance
    '#header-wrapper',
    '#ybar-inner-wrap',
    // CoinGecko / CMC
    'header',
    '.header-main'
  ];

  let targetArea = null;

  // Try to find a header/toolbar to inject into
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el) {
      targetArea = el;
      break;
    }
  }

  const btn = document.createElement('button');
  btn.className = 'diver-ai-btn';
  // Use Pulse Icon Logo
  const logoUrl = chrome.runtime.getURL('pulse-icon.png');
  btn.innerHTML = `<img src="${logoUrl}" alt="DiverAI" style="height: 24px; width: 24px; vertical-align: middle; pointer-events:none; border-radius: 6px;" />`;
  btn.style.padding = '8px';
  btn.style.display = 'flex';
  btn.style.alignItems = 'center';
  btn.style.justifyContent = 'center';
  btn.style.backgroundColor = '#1e293b'; // Sleek dark slate
  btn.style.border = '1px solid #334155';
  btn.style.borderRadius = '12px';
  btn.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
  btn.style.backdropFilter = 'blur(8px)';

  btn.onclick = () => {
    try {
      chrome.runtime.sendMessage({ action: 'OPEN_SIDEBAR' }, (response) => {
        if (chrome.runtime.lastError) {
          console.debug("DiverAI: Connection to background script lost/pending. Re-injection may be needed.");
        }
      });
    } catch (e) {
      console.debug("DiverAI: Message sending failed (likely extension context invalidated).");
    }
  };

  if (targetArea) {
    // Inject into native header (cleaner integration)
    targetArea.appendChild(btn);
  } else {
    // UNIVERSAL FALLBACK: Fixed Floating Pill (Bottom Right)
    // Remove "⚡ Diver AI Scan" text if fallback looks cleaner with just logo? 
    // Or keep it simple.
    btn.style.position = 'fixed';
    btn.style.bottom = '20px';
    btn.style.right = '20px';
    btn.style.zIndex = '2147483647'; // Max Z-Index
    btn.style.boxShadow = '0 4px 14px rgba(0,0,0,0.4)';
    document.body.appendChild(btn);
  }
}

// Observe DOM changes to re-inject if lost (SPA navigation)
const observer = new MutationObserver((mutations) => {
  addDiverButton();
});

observer.observe(document.body, {
  childList: true,
  subtree: true
});

// Initial try
setTimeout(addDiverButton, 2000);
