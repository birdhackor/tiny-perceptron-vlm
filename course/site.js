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
