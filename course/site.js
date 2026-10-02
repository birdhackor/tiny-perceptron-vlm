"use strict";
const toc = document.querySelector(".toc");
if (toc && window.matchMedia("(min-width: 901px)").matches) toc.open = true;
const search = document.getElementById("search");
if (search) {
  search.addEventListener("input", () => {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    document.querySelectorAll(".lesson").forEach((item) => {
      item.hidden = !item.textContent.toLowerCase().includes(query);
      if (!item.hidden) visible += 1;
    });
    document.getElementById("search-count").textContent = `${visible} 個小節`;
  });
}
document.querySelectorAll(".diagram-zoom").forEach((button) => {
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
