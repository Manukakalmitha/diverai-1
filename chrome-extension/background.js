// Open Side Panel on Icon Click
chrome.action.onClicked.addListener((tab) => {
    chrome.sidePanel.open({ tabId: tab.id });
});

// Handle Messages
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'OPEN_SIDEBAR') {
        chrome.sidePanel.open({ tabId: sender.tab.id });
        sendResponse({ status: 'opening' });
    } else if (request.action === 'CAPTURE_SCREENSHOT') {
        chrome.tabs.captureVisibleTab(
            null,
            { format: 'jpeg', quality: 50 },
            (dataUrl) => {
                if (chrome.runtime.lastError) {
                    console.error(chrome.runtime.lastError);
                    sendResponse({ error: chrome.runtime.lastError.message });
                } else {
                    sendResponse({ dataUrl: dataUrl });
                }
            }
        );
        return true; // Indicates async response
    }
});
