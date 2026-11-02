"use strict";

window.MathJax = {
  tex: { inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]], processEscapes: true },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
};

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (!window.MathJax.typesetPromise) return;
    window.MathJax.typesetClear();
    window.MathJax.texReset();
    window.MathJax.typesetPromise().catch((error) => console.error("公式排版失敗", error));
  });
}
