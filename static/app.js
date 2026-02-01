// static/app.js

document.addEventListener("DOMContentLoaded", () => {
  const modeInput = document.getElementById("mode-input");
  const chips = document.querySelectorAll(".dd-mode-chip");
  const body = document.body;
  const themeToggle = document.getElementById("theme-toggle");
  const form = document.getElementById("dream-form");
  const loadingIndicator = document.getElementById("loading-indicator");

  // --- Mode chips ---
  if (chips && modeInput) {
    chips.forEach((chip) => {
      chip.addEventListener("click", () => {
        const mode = chip.getAttribute("data-mode");
        if (!mode) return;

        modeInput.value = mode;

        chips.forEach((c) => c.classList.remove("dd-mode-chip--active"));
        chip.classList.add("dd-mode-chip--active");
      });
    });
  }

  // --- Theme toggle (dark default) ---
  if (themeToggle) {
    const storedTheme = localStorage.getItem("dd_theme");
    if (storedTheme === "light") {
      body.classList.remove("dd-dark");
      themeToggle.textContent = "☀️";
    } else {
      body.classList.add("dd-dark");
      themeToggle.textContent = "🌙";
    }

    themeToggle.addEventListener("click", () => {
      const isDark = body.classList.toggle("dd-dark");
      localStorage.setItem("dd_theme", isDark ? "dark" : "light");
      themeToggle.textContent = isDark ? "🌙" : "☀️";
    });
  }

  // --- Show loading indicator on submit ---
  if (form) {
    form.addEventListener("submit", () => {
      const submitButton = form.querySelector(".dd-primary-button");
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.textContent = "Analyzing your dream…";
      }
      if (loadingIndicator) {
        loadingIndicator.style.display = "flex";
      }
    });
  }
});
