// static/app.js

document.addEventListener("DOMContentLoaded", () => {
  const modeInput = document.getElementById("mode-input");
  const chips = document.querySelectorAll(".dd-mode-chip");
  const body = document.body;
  const themeToggle = document.getElementById("theme-toggle");

  // --- Mode chips ---
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const mode = chip.getAttribute("data-mode");
      if (!mode) return;

      modeInput.value = mode;

      chips.forEach((c) => c.classList.remove("dd-mode-chip--active"));
      chip.classList.add("dd-mode-chip--active");
    });
  });

  // --- Theme toggle (dark default) ---
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
});
