(function () {
  const STORAGE_KEY = "webifier-theme";
  const root = document.documentElement;

  function prefersDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function normalizeModes(value) {
    const modes = (value || "system,light,dark")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    return modes.length ? modes : ["system", "light", "dark"];
  }

  function bootstrapThemeFor(selected) {
    if (selected === "system") {
      return prefersDark() ? "dark" : "light";
    }
    return selected === "dark" ? "dark" : "light";
  }

  function resolvedThemeFor(selected) {
    if (selected === "system") {
      return prefersDark() ? "dark" : "light";
    }
    return selected;
  }

  function labelFor(selected, resolved) {
    if (selected === "system") return `System theme (${resolved})`;
    return `${selected.charAt(0).toUpperCase()}${selected.slice(1)} theme`;
  }

  function applyTheme(selected) {
    const resolved = resolvedThemeFor(selected);
    root.dataset.wfTheme = selected;
    root.dataset.wfResolvedTheme = resolved;
    root.dataset.bsTheme = bootstrapThemeFor(selected);
    updateUtterancesTheme(resolved);

    document.querySelectorAll(".webifier-theme-toggle").forEach((button) => {
      const label = button.querySelector(".webifier-theme-label");
      const text = labelFor(selected, resolved);
      button.setAttribute("aria-label", text);
      button.setAttribute("title", text);
      if (label) label.textContent = text;
    });
  }

  function updateUtterancesTheme(resolved) {
    const theme = resolved === "dark" ? "github-dark" : "github-light";
    document.querySelectorAll("iframe.utterances-frame").forEach((iframe) => {
      if (iframe.contentWindow) {
        iframe.contentWindow.postMessage({ type: "set-theme", theme }, "https://utteranc.es");
      }
    });
  }

  function currentTheme(defaultTheme, modes) {
    const stored = localStorage.getItem(STORAGE_KEY);
    const selected = stored || defaultTheme || "system";
    return modes.includes(selected) ? selected : modes[0];
  }

  function nextTheme(current, modes) {
    const index = modes.indexOf(current);
    return modes[(index + 1) % modes.length] || "system";
  }

  function setup() {
    const buttons = document.querySelectorAll(".webifier-theme-toggle");
    const firstButton = buttons[0];
    const modes = normalizeModes(firstButton && firstButton.dataset.themeModes);
    const defaultTheme = firstButton && firstButton.dataset.themeDefault;

    applyTheme(currentTheme(defaultTheme, modes));

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const selected = nextTheme(root.dataset.wfTheme || currentTheme(defaultTheme, modes), modes);
        localStorage.setItem(STORAGE_KEY, selected);
        applyTheme(selected);
      });
    });

    window.addEventListener("message", () => {
      updateUtterancesTheme(root.dataset.wfResolvedTheme || resolvedThemeFor(currentTheme(defaultTheme, modes)));
    });

    window.setTimeout(() => {
      updateUtterancesTheme(root.dataset.wfResolvedTheme || resolvedThemeFor(currentTheme(defaultTheme, modes)));
    }, 1500);

    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
        if ((root.dataset.wfTheme || "system") === "system") {
          applyTheme("system");
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();
