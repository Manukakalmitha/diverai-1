import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Unbreakable CSS Guard: Prevents extension-triggered 'cssRules of null' crashes
try {
  const originalCssRules = Object.getOwnPropertyDescriptor(CSSStyleSheet.prototype, 'cssRules');
  if (originalCssRules) {
    Object.defineProperty(CSSStyleSheet.prototype, 'cssRules', {
      get: function () {
        try {
          return originalCssRules.get.call(this) || [];
        } catch (e) {
          // If the stylesheet is detached or ownerNode is null, return empty array instead of crashing
          return [];
        }
      },
      configurable: true,
      enumerable: true
    });
  }
} catch (e) {
  console.debug("CSS Guard initialization failed, skipping.");
}

// Hard block against browser translation extensions
const blockTranslation = () => {
  const html = document.documentElement;
  html.setAttribute('translate', 'no');
  html.classList.add('notranslate');

  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === 'attributes') {
        if (mutation.attributeName === 'translate' && html.getAttribute('translate') !== 'no') {
          html.setAttribute('translate', 'no');
        }
        // Force-strip extension injected classes that trigger translation
        if (mutation.attributeName === 'class' && html.classList.contains('translated-ltr')) {
          html.classList.remove('translated-ltr', 'translated-rtl');
        }
      }
    });
  });

  observer.observe(html, { attributes: true });
};

blockTranslation();

// Global resilience against extension interference
window.addEventListener('unhandledrejection', (event) => {
  // Silent ignore for known browser extension errors that trigger promise rejection
  const ignoredErrors = [
    'translate-page',
    'save-page',
    'establish connection',
    'Receiving end does not exist',
    'cssRules',
    'null (reading \'cssRules\')'
  ];
  if (ignoredErrors.some(msg => event.reason?.message?.includes(msg) || event.reason?.toString()?.includes(msg))) {
    event.preventDefault();
    console.debug("Mitigated extension-related promise rejection.");
    return;
  }
  console.error("Unhandled Rejection:", event.reason);
});

// Suppress known non-fatal errors from polluting the user experience
const originalConsoleError = console.error;
console.error = (...args) => {
  const msg = args[0]?.toString() || "";
  const ignoredKeywords = ["cssRules", "translate-page", "null (reading 'cssRules')", "Cannot read properties of null"];

  if (ignoredKeywords.some(kw => msg.includes(kw))) {
    console.debug("Intercepted and suppressed extension error:", msg);
    return;
  }
  originalConsoleError.apply(console, args);
};

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
