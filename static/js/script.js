/* ==========================================================================
   CardioPredict - Client-side interactions
   ========================================================================== */
(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    // --- Bootstrap form validation feedback -------------------------------
    const forms = document.querySelectorAll("form");
    forms.forEach(function (form) {
      form.addEventListener("submit", function (event) {
        if (!form.checkValidity()) {
          event.preventDefault();
          event.stopPropagation();
        }
        form.classList.add("was-validated");

        // Show a loading state on the submit button.
        const btn = form.querySelector('button[type="submit"]');
        if (btn && form.checkValidity()) {
          btn.disabled = true;
          const original = btn.innerHTML;
          btn.dataset.original = original;
          btn.innerHTML =
            '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Processing...';
        }
      });
    });

    // --- Show selected file name on the upload input ----------------------
    const fileInput = document.getElementById("dataset");
    if (fileInput) {
      fileInput.addEventListener("change", function () {
        console.log("[v0] dataset selected:", fileInput.files[0]?.name);
      });
    }

    // --- Auto-dismiss flash alerts after 5 seconds ------------------------
    setTimeout(function () {
      document.querySelectorAll(".alert-dismissible").forEach(function (alert) {
        const instance = bootstrap.Alert.getOrCreateInstance(alert);
        instance.close();
      });
    }, 5000);

    // --- Animate probability bars on the prediction page ------------------
    document.querySelectorAll(".progress-bar").forEach(function (bar) {
      const target = bar.style.width;
      bar.style.width = "0%";
      requestAnimationFrame(function () {
        setTimeout(function () {
          bar.style.transition = "width 0.9s ease";
          bar.style.width = target;
        }, 100);
      });
    });

    // --- Smooth scroll to result after a prediction -----------------------
    const resultBox = document.querySelector(".result-box");
    if (resultBox && window.innerWidth < 992) {
      resultBox.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
})();
