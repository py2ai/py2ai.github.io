/**
 * Zoomable SVG Viewer - pyshine.com
 * Automatically wraps SVG images in blog posts with zoom/pan/drag controls.
 * Only affects individual SVGs without zooming the entire page.
 */
(function () {
  'use strict';

  var MIN_ZOOM = 0.5;   // 50%
  var MAX_ZOOM = 5.0;    // 500%
  var ZOOM_STEP = 0.25;  // 25% increments
  var DRAG_THRESHOLD = 3; // pixels before drag starts

  /**
   * Creates the viewer DOM structure around an SVG image.
   * @param {HTMLImageElement} img - The SVG image element to wrap
   * @returns {HTMLElement} The .svg-viewer container
   */
  function createViewer(img) {
    var viewer = document.createElement('div');
    viewer.className = 'svg-viewer';
    viewer.setAttribute('role', 'figure');
    viewer.setAttribute('aria-label', img.alt || 'Zoomable diagram');

    // Preserve any classes from the original img
    if (img.className) {
      viewer.classList.add(img.className);
    }

    // Controls bar
    var controls = document.createElement('div');
    controls.className = 'svg-viewer-controls';

    var btnZoomIn = document.createElement('button');
    btnZoomIn.className = 'svg-viewer-btn';
    btnZoomIn.setAttribute('aria-label', 'Zoom in');
    btnZoomIn.setAttribute('title', 'Zoom in');
    btnZoomIn.textContent = '+';

    var btnZoomOut = document.createElement('button');
    btnZoomOut.className = 'svg-viewer-btn';
    btnZoomOut.setAttribute('aria-label', 'Zoom out');
    btnZoomOut.setAttribute('title', 'Zoom out');
    btnZoomOut.textContent = '\u2212'; // minus sign

    var separator = document.createElement('div');
    separator.className = 'svg-viewer-separator';

    var zoomLevel = document.createElement('span');
    zoomLevel.className = 'svg-viewer-zoom-level';
    zoomLevel.textContent = '100%';
    zoomLevel.setAttribute('aria-live', 'polite');

    var separator2 = document.createElement('div');
    separator2.className = 'svg-viewer-separator';

    var btnReset = document.createElement('button');
    btnReset.className = 'svg-viewer-btn svg-viewer-btn--reset';
    btnReset.setAttribute('aria-label', 'Reset zoom');
    btnReset.setAttribute('title', 'Reset zoom');
    btnReset.textContent = '\u21BA'; // anticlockwise arrow

    controls.appendChild(btnZoomIn);
    controls.appendChild(btnZoomOut);
    controls.appendChild(separator);
    controls.appendChild(zoomLevel);
    controls.appendChild(separator2);
    controls.appendChild(btnReset);

    // Content area (scrollable/pannable)
    var content = document.createElement('div');
    content.className = 'svg-viewer-content';
    content.setAttribute('tabindex', '0');
    content.setAttribute('role', 'img');

    // Inner wrapper (gets the width change for zoom)
    var inner = document.createElement('div');
    inner.className = 'svg-viewer-inner';

    // Clone the image to avoid reference issues
    var newImg = img.cloneNode(true);
    newImg.removeAttribute('tabindex');
    inner.appendChild(newImg);

    content.appendChild(inner);
    viewer.appendChild(controls);
    viewer.appendChild(content);

    // State
    var currentZoom = 1.0;
    var isDragging = false;
    var dragStartX = 0;
    var dragStartY = 0;
    var scrollStartX = 0;
    var scrollStartY = 0;
    var hasMoved = false;

    /**
     * Updates the zoom level display and applies the width change.
     */
    function updateZoom() {
      var percent = Math.round(currentZoom * 100);
      zoomLevel.textContent = percent + '%';
      // Use actual width change instead of CSS transform for proper scroll overflow
      inner.style.transform = '';
      inner.style.width = (currentZoom * 100) + '%';

      // Update cursor based on zoom level
      if (currentZoom > 1.0) {
        content.classList.add('draggable');
      } else {
        content.classList.remove('draggable');
      }
    }

    /**
     * Zoom in by one step.
     */
    function zoomIn() {
      var newZoom = Math.min(currentZoom + ZOOM_STEP, MAX_ZOOM);
      if (newZoom !== currentZoom) {
        currentZoom = newZoom;
        updateZoom();
      }
    }

    /**
     * Zoom out by one step.
     */
    function zoomOut() {
      var newZoom = Math.max(currentZoom - ZOOM_STEP, MIN_ZOOM);
      if (newZoom !== currentZoom) {
        currentZoom = newZoom;
        updateZoom();
      }
    }

    /**
     * Reset zoom to 100%.
     */
    function resetZoom() {
      if (currentZoom !== 1.0) {
        currentZoom = 1.0;
        content.scrollLeft = 0;
        content.scrollTop = 0;
        updateZoom();
      }
    }

    // Button click handlers
    btnZoomIn.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      zoomIn();
    });

    btnZoomOut.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      zoomOut();
    });

    btnReset.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      resetZoom();
    });

    // Keyboard shortcuts when content is focused
    content.addEventListener('keydown', function (e) {
      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        zoomIn();
      } else if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        zoomOut();
      } else if (e.key === '0') {
        e.preventDefault();
        resetZoom();
      }
    });

    // Mouse wheel zoom (Ctrl+wheel or just wheel when content is hovered)
    content.addEventListener('wheel', function (e) {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        e.stopPropagation();

        if (e.deltaY < 0) {
          zoomIn();
        } else if (e.deltaY > 0) {
          zoomOut();
        }
      }
    }, { passive: false });

    // Mouse drag-to-pan
    content.addEventListener('mousedown', function (e) {
      if (currentZoom <= 1.0) return;
      if (e.button !== 0) return; // left button only

      isDragging = true;
      hasMoved = false;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      scrollStartX = content.scrollLeft;
      scrollStartY = content.scrollTop;
      content.classList.add('dragging');
      e.preventDefault();
    });

    document.addEventListener('mousemove', function (e) {
      if (!isDragging) return;

      var dx = e.clientX - dragStartX;
      var dy = e.clientY - dragStartY;

      if (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD) {
        hasMoved = true;
      }

      content.scrollLeft = scrollStartX - dx;
      content.scrollTop = scrollStartY - dy;
    });

    document.addEventListener('mouseup', function () {
      if (isDragging) {
        isDragging = false;
        content.classList.remove('dragging');
      }
    });

    // Touch support for mobile
    var touchStartX = 0;
    var touchStartY = 0;
    var touchScrollStartX = 0;
    var touchScrollStartY = 0;
    var initialPinchDistance = 0;
    var initialPinchZoom = 1.0;
    var isTouchDragging = false;

    /**
     * Calculate distance between two touch points.
     */
    function getTouchDistance(touches) {
      var dx = touches[0].clientX - touches[1].clientX;
      var dy = touches[0].clientY - touches[1].clientY;
      return Math.sqrt(dx * dx + dy * dy);
    }

    content.addEventListener('touchstart', function (e) {
      if (e.touches.length === 2) {
        // Pinch-to-zoom start
        e.preventDefault();
        initialPinchDistance = getTouchDistance(e.touches);
        initialPinchZoom = currentZoom;
      } else if (e.touches.length === 1 && currentZoom > 1.0) {
        // Single touch drag start
        isTouchDragging = true;
        touchStartX = e.touches[0].clientX;
        touchStartY = e.touches[0].clientY;
        touchScrollStartX = content.scrollLeft;
        touchScrollStartY = content.scrollTop;
      }
    }, { passive: false });

    content.addEventListener('touchmove', function (e) {
      if (e.touches.length === 2) {
        // Pinch-to-zoom
        e.preventDefault();
        var currentDistance = getTouchDistance(e.touches);
        var scale = currentDistance / initialPinchDistance;
        var newZoom = Math.min(Math.max(initialPinchZoom * scale, MIN_ZOOM), MAX_ZOOM);
        currentZoom = newZoom;
        updateZoom();
      } else if (e.touches.length === 1 && isTouchDragging && currentZoom > 1.0) {
        // Single touch drag
        e.preventDefault();
        var dx = e.touches[0].clientX - touchStartX;
        var dy = e.touches[0].clientY - touchStartY;
        content.scrollLeft = touchScrollStartX - dx;
        content.scrollTop = touchScrollStartY - dy;
      }
    }, { passive: false });

    content.addEventListener('touchend', function () {
      isTouchDragging = false;
    });

    // Double-click/tap to reset
    content.addEventListener('dblclick', function (e) {
      e.preventDefault();
      if (currentZoom !== 1.0) {
        resetZoom();
      } else {
        currentZoom = 2.0; // double-click zooms to 200%
        updateZoom();
      }
    });

    // Prevent image drag (native browser image dragging)
    newImg.addEventListener('dragstart', function (e) {
      e.preventDefault();
    });

    // Initialize
    updateZoom();

    return viewer;
  }

  /**
   * Initialize the SVG viewer on all qualifying images in the post content.
   */
  function init() {
    // Target images inside .post-content that have SVG sources
    var postContent = document.querySelector('.post-content');
    if (!postContent) return;

    var images = postContent.querySelectorAll('img');
    var svgImages = [];

    for (var i = 0; i < images.length; i++) {
      var img = images[i];
      var src = (img.getAttribute('src') || '').toLowerCase();

      // Only target SVG images
      if (src.endsWith('.svg') || src.includes('.svg?') || src.includes('/svg/')) {
        // Skip images that are already inside a viewer or are tiny (icons, etc.)
        if (img.closest('.svg-viewer')) continue;
        if (img.closest('.hero')) continue;
        if (img.closest('.related-post-card')) continue;
        if (img.width > 0 && img.width < 80) continue; // skip small icons

        svgImages.push(img);
      }
    }

    // Wrap each SVG image with the viewer
    for (var j = 0; j < svgImages.length; j++) {
      var imgEl = svgImages[j];
      var viewer = createViewer(imgEl);

      // Replace the original image with the viewer
      imgEl.parentNode.replaceChild(viewer, imgEl);
    }
  }

  // Run on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // DOM already loaded (e.g., script loaded with async/defer)
    init();
  }
})();