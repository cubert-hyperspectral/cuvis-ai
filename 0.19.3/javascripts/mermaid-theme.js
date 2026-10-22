// Initialize mermaid with dynamic theming for Material for MkDocs
(function() {
  // Function to get current Material theme
  function getCurrentTheme() {
    const scheme = document.body.getAttribute('data-md-color-scheme');
    return scheme === 'slate' ? 'dark' : 'default';
  }

  // Function to get mermaid config for theme
  // Colors track Cubert brand book pages 3-4 (cool dark/light bg + Tron Turquoise edges).
  function getMermaidConfig(theme) {
    if (theme === 'dark') {
      return {
        theme: 'base',
        themeVariables: {
          primaryColor: '#191932',
          primaryTextColor: '#fafafa',
          primaryBorderColor: '#00f0c8',
          lineColor: '#00f0c8',
          secondaryColor: '#0f0f24',
          tertiaryColor: '#1f1f3d',
          background: '#191932',
          mainBkg: '#0f0f24',
          secondBkg: '#1f1f3d',
          mainContrastColor: '#fafafa',
          darkMode: true,
          clusterBkg: '#0f0f24',
          clusterBorder: '#00f0c8',
          edgeLabelBackground: '#0f0f24',
          tertiaryTextColor: '#fafafa',
          fontFamily: 'Roboto, system-ui, sans-serif',
          fontSize: '14px'
        }
      };
    } else {
      return {
        theme: 'base',
        themeVariables: {
          primaryColor: '#ebf0f5',
          primaryTextColor: '#191932',
          primaryBorderColor: '#00b48c',
          lineColor: '#00b48c',
          secondaryColor: '#fafafa',
          tertiaryColor: '#ffffff',
          background: '#ebf0f5',
          mainBkg: '#fafafa',
          mainContrastColor: '#191932',
          clusterBkg: '#fafafa',
          clusterBorder: '#00b48c',
          edgeLabelBackground: '#ebf0f5',
          fontFamily: 'Roboto, system-ui, sans-serif',
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
