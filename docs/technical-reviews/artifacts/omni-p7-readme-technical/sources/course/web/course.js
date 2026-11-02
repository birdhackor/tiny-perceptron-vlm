"use strict";

function initializeDiagrams() {
  document.querySelectorAll(".diagram-zoom").forEach((button) => {
    if (button.dataset.initialized) return;
    button.dataset.initialized = "true";
    const viewport = document.getElementById(button.getAttribute("aria-controls"));
    const hint = button.closest("figure").querySelector(".diagram-scroll-hint");
    button.hidden = false;
    button.addEventListener("click", () => {
      const enlarged = viewport.classList.toggle("is-enlarged");
      button.setAttribute("aria-expanded", String(enlarged));
      button.textContent = enlarged ? "縮回全圖" : "放大圖解";
      hint.hidden = !enlarged;
    });
  });
}

// Zensical 的即時換頁會替換正文；每次換頁都要啟用新圖解的按鈕。
if (typeof document$ !== "undefined") document$.subscribe(initializeDiagrams);
else initializeDiagrams();
