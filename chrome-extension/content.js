// Inject Styling
const style = document.createElement('style');
style.textContent = `
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
  // Use Image Logo instead of Text
  const logoUrl = chrome.runtime.getURL('logo-text.png');
  btn.innerHTML = `<img src="${logoUrl}" alt="DiverAI" style="height: 20px; vertical-align: middle;pointer-events:none;" />`;
  btn.style.padding = '6px 12px';
  btn.style.display = 'flex';
  btn.style.alignItems = 'center';
  btn.style.justifyContent = 'center';

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
