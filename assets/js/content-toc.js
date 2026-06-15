(function () {
  document.addEventListener("DOMContentLoaded", function () {
    Array.prototype.slice.call(document.querySelectorAll(".wf-content-toc")).forEach(function (toc, index) {
      var button = toc.querySelector(".wf-content-toc-toggle");
      var list = toc.querySelector("ol");
      if (!button || !list) {
        return;
      }
      if (!list.id) {
        list.id = "wf-content-toc-list-" + index;
      }
      button.setAttribute("aria-controls", list.id);
      button.setAttribute("aria-expanded", toc.classList.contains("is-collapsed") ? "false" : "true");
      button.addEventListener("click", function () {
        var collapsed = toc.classList.toggle("is-collapsed");
        button.setAttribute("aria-expanded", collapsed ? "false" : "true");
      });
    });
  });
})();
