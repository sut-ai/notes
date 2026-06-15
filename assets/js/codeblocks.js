(function () {
  document.addEventListener("DOMContentLoaded", function () {
    var candidates = Array.prototype.slice.call(document.querySelectorAll(".codehilite, .highlight, pre"));
    candidates.forEach(function (candidate) {
      var block = candidate;
      var pre = candidate.matches("pre") ? candidate : candidate.querySelector("pre");
      if (!pre || pre.dataset.wfCodeReady === "true") {
        return;
      }
      if (candidate.matches("pre")) {
        var highlightedParent = candidate.closest(".codehilite, .highlight");
        if (highlightedParent) {
          return;
        }
        block = document.createElement("div");
        block.className = "wf-code-block";
        candidate.parentNode.insertBefore(block, candidate);
        block.appendChild(candidate);
      }
      block.classList.add("has-copy");
      pre.dataset.wfCodeReady = "true";

      var button = document.createElement("button");
      button.type = "button";
      button.className = "wf-code-copy";
      button.textContent = "Copy";
      button.setAttribute("aria-label", "Copy code to clipboard");
      block.appendChild(button);

      button.addEventListener("click", function () {
        var code = pre.innerText;
        var copied = navigator.clipboard && window.isSecureContext
          ? navigator.clipboard.writeText(code)
          : new Promise(function (resolve, reject) {
              var textarea = document.createElement("textarea");
              textarea.value = code;
              textarea.setAttribute("readonly", "");
              textarea.style.position = "fixed";
              textarea.style.top = "-1000px";
              document.body.appendChild(textarea);
              textarea.select();
              try {
                document.execCommand("copy") ? resolve() : reject();
              } catch (error) {
                reject(error);
              }
              document.body.removeChild(textarea);
            });
        copied.then(function () {
          button.textContent = "Copied";
          button.classList.add("is-copied");
          window.setTimeout(function () {
            button.textContent = "Copy";
            button.classList.remove("is-copied");
          }, 1200);
        }).catch(function () {
          button.textContent = "Failed";
          window.setTimeout(function () {
            button.textContent = "Copy";
          }, 1200);
        });
      });
    });
  });
})();
