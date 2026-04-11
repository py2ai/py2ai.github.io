// Copy Code to Clipboard - Dynamic Button Injection
// This script automatically adds copy buttons to all code blocks

(function() {
  'use strict';

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
    const existingButtons = document.querySelectorAll('.copy-code-button');
    existingButtons.forEach((button) => {
      // Skip if already has handler
      if (button.hasAttribute('data-handler-attached')) {
        return;
      }
      button.setAttribute('data-handler-attached', 'true');
      
      button.addEventListener('click', async function() {
        const header = button.closest('.code-header');
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
