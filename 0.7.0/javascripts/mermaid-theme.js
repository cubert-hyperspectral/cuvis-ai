// Initialize mermaid with dynamic theming for Material for MkDocs
(function() {
  // Function to get current Material theme
  function getCurrentTheme() {
    const scheme = document.body.getAttribute('data-md-color-scheme');
    return scheme === 'slate' ? 'dark' : 'default';
  }

  // Function to get mermaid config for theme
  function getMermaidConfig(theme) {
    if (theme === 'dark') {
      return {
        theme: 'dark',
        themeVariables: {
          primaryColor: '#404040',
          primaryTextColor: '#e0e0e0',
          primaryBorderColor: '#666666',
          lineColor: '#888888',
          secondaryColor: '#2d2d2d',
          tertiaryColor: '#353535',
          background: '#1e1e1e',
          mainBkg: '#2d2d2d',
          secondBkg: '#353535',
          mainContrastColor: '#e0e0e0',
          darkMode: true,
          clusterBkg: '#2d2d2d',
          clusterBorder: '#666666',
          edgeLabelBackground: '#2d2d2d',
          tertiaryTextColor: '#e0e0e0',
          fontSize: '14px'
        }
      };
    } else {
      return {
        theme: 'default',
        themeVariables: {
          fontSize: '14px'
        }
      };
    }
  }

  // Initialize mermaid on page load
  function initMermaid() {
    if (typeof mermaid === 'undefined') {
      console.error('Mermaid is not loaded');
      return;
    }

    const theme = getCurrentTheme();
    const config = getMermaidConfig(theme);

    mermaid.initialize({
      startOnLoad: true,
      ...config
    });
  }

  // Re-render all mermaid diagrams
  function reRenderMermaid() {
    if (typeof mermaid === 'undefined') {
      return;
    }

    const theme = getCurrentTheme();
    const config = getMermaidConfig(theme);

    // Re-initialize mermaid with new theme
    mermaid.initialize({
      startOnLoad: false,
      ...config
    });

    // Find all mermaid elements and re-render
    const elements = document.querySelectorAll('.mermaid');
    elements.forEach((element) => {
      // Store original content if not already stored
      if (!element.hasAttribute('data-original-content')) {
        const preElement = element.querySelector('pre');
        if (preElement) {
          element.setAttribute('data-original-content', preElement.textContent);
        } else {
          element.setAttribute('data-original-content', element.textContent);
        }
      }

      // Get original content
      const originalContent = element.getAttribute('data-original-content');

      // Clear and re-render
      element.innerHTML = originalContent;
      element.removeAttribute('data-processed');

      try {
        mermaid.init(undefined, element);
      } catch (e) {
        console.error('Error rendering mermaid diagram:', e);
      }
    });
  }

  // Watch for theme changes
  function watchThemeChanges() {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' &&
            mutation.attributeName === 'data-md-color-scheme') {
          setTimeout(reRenderMermaid, 100);
        }
      });
    });

    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-md-color-scheme']
    });
  }

  // Wait for mermaid to be available
  function waitForMermaid() {
    if (typeof mermaid !== 'undefined') {
      initMermaid();
      watchThemeChanges();
    } else {
      setTimeout(waitForMermaid, 100);
    }
  }

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', waitForMermaid);
  } else {
    waitForMermaid();
  }
})();
