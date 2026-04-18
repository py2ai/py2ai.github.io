// Copy Code to Clipboard - Dynamic Button Injection
// This script automatically adds copy buttons to all code blocks

(function() {
  'use strict';

  // Line count thresholds for showing the copy button
  var MIN_LINES_FOR_COPY = 3;  // Hide copy button if ≤ this many lines
  var MAX_LINES_FOR_COPY = 20; // Hide copy button if ≥ this many lines

  // Count the number of lines in a code block
  function getCodeLineCount(element) {
    // Try to find the <code> element inside the highlight block
    var code = element.querySelector('code') || element;

    // If Rouge line-number spans exist (.rouge-line), count those
    var rougeLines = element.querySelectorAll('.rouge-line');
    if (rougeLines.length > 0) {
      return rougeLines.length;
    }

    // If line number elements exist (.lineno), count those
    var linenoElements = element.querySelectorAll('.lineno');
    if (linenoElements.length > 0) {
      return linenoElements.length;
    }

    // Fallback: count by newline characters in text content
    var text = code.textContent || code.innerText || '';
    // Trim trailing newlines so a trailing newline doesn't create an extra line
    text = text.replace(/\n+$/, '');
    if (text === '') {
      return 0;
    }
    return text.split('\n').length;
  }

  // Determine whether a copy button should be shown for a code block
  // Respects 'always-copy' and 'never-copy' CSS classes as overrides
  function shouldShowCopyButton(codeBlock) {
    // Check for manual override classes on the code block or its parent .highlighter-rouge
    var checkEl = codeBlock.closest('.highlighter-rouge') || codeBlock;
    if (checkEl.classList.contains('always-copy')) {
      return true;
    }
    if (checkEl.classList.contains('never-copy')) {
      return false;
    }

    var lineCount = getCodeLineCount(codeBlock);
    return lineCount > MIN_LINES_FOR_COPY && lineCount < MAX_LINES_FOR_COPY;
  }

  // Check if a code block already has a copy button (from codeHeader include)
  function hasExistingButton(codeBlock) {
    // Check previous siblings for code-header
    let prev = codeBlock.previousElementSibling;
    while (prev) {
      if (prev.classList && prev.classList.contains('code-header')) {
        return true;
      }
      // Skip whitespace/text nodes
      prev = prev.previousElementSibling;
    }
    return false;
  }

  // Copy code to clipboard
  async function copyToClipboard(codeText, button) {
    try {
      await navigator.clipboard.writeText(codeText);
      showFeedback(button, 'Copied!');
    } catch (err) {
      // Fallback for older browsers
      try {
        const textArea = document.createElement('textarea');
        textArea.value = codeText;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showFeedback(button, 'Copied!');
      } catch (fallbackErr) {
        console.error('Failed to copy code:', fallbackErr);
        showFeedback(button, 'Failed to copy');
      }
    }
  }

  // Show visual feedback
  function showFeedback(button, message) {
    const originalText = button.textContent;
    button.textContent = message;
    button.classList.add('copied');
    
    setTimeout(() => {
      button.textContent = originalText;
      button.classList.remove('copied');
    }, 2000);
  }

  // Get code text from a code block
  function getCodeText(codeBlock) {
    const pre = codeBlock.querySelector('pre') || codeBlock;
    const code = pre.querySelector('code') || pre;
    return code.textContent || code.innerText;
  }

  // Wait for DOM to be ready
  function init() {
    // First, attach handlers to existing buttons (from codeHeader include)
    // Also hide/remove buttons for code blocks outside the line-count range
    const existingButtons = document.querySelectorAll('.copy-code-button');
    existingButtons.forEach((button) => {
      // Find the associated code block
      const header = button.closest('.code-header');
      const codeBlock = header ? header.nextElementSibling : null;

      // If we found the code block, check whether the button should be shown
      if (codeBlock && !shouldShowCopyButton(codeBlock)) {
        // Hide the entire code-header (contains only the copy button)
        header.style.display = 'none';
        return; // Skip attaching handler — button is hidden
      }

      // Skip if already has handler
      if (button.hasAttribute('data-handler-attached')) {
        return;
      }
      button.setAttribute('data-handler-attached', 'true');
      
      button.addEventListener('click', async function() {
        const codeBlock = header.nextElementSibling;
        
        if (!codeBlock) return;
        
        const codeText = getCodeText(codeBlock);
        await copyToClipboard(codeText, button);
      });
    });

    // Then, add buttons to code blocks that don't have one
    // Only process .highlighter-rouge elements (not nested div.highlight)
    const codeBlocks = document.querySelectorAll('.highlighter-rouge');
    
    codeBlocks.forEach((codeBlock) => {
      // Skip if already has a code-header sibling (legacy posts with include)
      if (hasExistingButton(codeBlock)) {
        return;
      }

      // Skip if the code block is outside the line-count range
      if (!shouldShowCopyButton(codeBlock)) {
        return;
      }
      
      // Create the copy button container
      const header = document.createElement('div');
      header.className = 'code-header';
      
      // Create the copy button
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-code-button';
      copyButton.textContent = 'Copy code to clipboard';
      copyButton.setAttribute('type', 'button');
      copyButton.setAttribute('aria-label', 'Copy code to clipboard');
      copyButton.setAttribute('data-handler-attached', 'true');
      
      header.appendChild(copyButton);
      
      // Insert header before code block
      codeBlock.parentNode.insertBefore(header, codeBlock);
      
      // Add click handler
      copyButton.addEventListener('click', async function() {
        const codeText = getCodeText(codeBlock);
        await copyToClipboard(codeText, copyButton);
      });
    });

    // Also handle standalone div.highlight elements (not inside .highlighter-rouge)
    const standaloneHighlights = document.querySelectorAll('div.highlight');
    standaloneHighlights.forEach((codeBlock) => {
      // Skip if inside a .highlighter-rouge (already handled above)
      if (codeBlock.closest('.highlighter-rouge')) {
        return;
      }
      
      // Skip if already has a code-header sibling
      if (hasExistingButton(codeBlock)) {
        return;
      }

      // Skip if the code block is outside the line-count range
      if (!shouldShowCopyButton(codeBlock)) {
        return;
      }
      
      // Create the copy button container
      const header = document.createElement('div');
      header.className = 'code-header';
      
      // Create the copy button
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-code-button';
      copyButton.textContent = 'Copy code to clipboard';
      copyButton.setAttribute('type', 'button');
      copyButton.setAttribute('aria-label', 'Copy code to clipboard');
      copyButton.setAttribute('data-handler-attached', 'true');
      
      header.appendChild(copyButton);
      
      // Insert header before code block
      codeBlock.parentNode.insertBefore(header, codeBlock);
      
      // Add click handler
      copyButton.addEventListener('click', async function() {
        const codeText = getCodeText(codeBlock);
        await copyToClipboard(codeText, copyButton);
      });
    });
  }

  // Run on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
