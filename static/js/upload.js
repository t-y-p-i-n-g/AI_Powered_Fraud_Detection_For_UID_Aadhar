document.addEventListener("DOMContentLoaded", function () {
  // --- Get all DOM elements once ---
  const frontZone = document.getElementById("front-drop-zone");
  const backZone = document.getElementById("back-drop-zone");
  const frontUploadInput = document.getElementById("front-upload");
  const backUploadInput = document.getElementById("back-upload");
  const analyzeBtn = document.getElementById("analyze-btn");
  const statusEl = document.getElementById("upload-status");

  // --- Helper function to update the UI for a file input ---
  function updateFileDisplay(zoneId, file) {
    const uploadContent = document.getElementById(`${zoneId}-upload-content`);
    const fileInfo = document.getElementById(`${zoneId}-file-info`);
    const fileNameEl = document.getElementById(`${zoneId}-file-name`);
    const fileSizeEl = document.getElementById(`${zoneId}-file-size`);
    const zone = document.getElementById(`${zoneId}-drop-zone`);

    if (file) {
      uploadContent.classList.add("hidden");
      fileInfo.classList.remove("hidden");
      zone.classList.remove("border-slate-300", "bg-slate-50");
      zone.classList.add("border-green-500", "bg-green-50");
      fileNameEl.textContent = file.name;
      fileSizeEl.textContent = (file.size / 1024).toFixed(2) + " KB";
    } else {
      uploadContent.classList.remove("hidden");
      fileInfo.classList.add("hidden");
      zone.classList.remove("border-green-500", "bg-green-50");
      zone.classList.add("border-slate-300", "bg-slate-50");
    }
  }

  function checkFormValidity() {
    if (frontUploadInput.files.length > 0 && backUploadInput.files.length > 0) {
      analyzeBtn.disabled = false;
    } else {
      analyzeBtn.disabled = true;
    }
  }

  // Helper to show a transient status message to the user
  function showStatusMessage(msg, timeoutMs = 3000) {
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.classList.remove("text-red-600");
    statusEl.classList.add("text-green-600");
    clearTimeout(showStatusMessage._t);
    showStatusMessage._t = setTimeout(
      () => (statusEl.textContent = ""),
      timeoutMs
    );
  }

  // --- Change listeners ---
  frontUploadInput.addEventListener("change", () => {
    if (frontUploadInput.files.length > 0) {
      const file = frontUploadInput.files[0];
      updateFileDisplay("front", file);
      console.log(
        `✅ Front image accepted: ${file.name} (${(file.size / 1024).toFixed(
          2
        )} KB)`
      );
      showStatusMessage(`Front image "${file.name}" uploaded successfully.`);
    } else {
      updateFileDisplay("front", null);
    }
    checkFormValidity();
  });

  backUploadInput.addEventListener("change", () => {
    if (backUploadInput.files.length > 0) {
      const file = backUploadInput.files[0];
      updateFileDisplay("back", file);
      console.log(
        `✅ Back image accepted: ${file.name} (${(file.size / 1024).toFixed(
          2
        )} KB)`
      );
      showStatusMessage(`Back image "${file.name}" uploaded successfully.`);
    } else {
      updateFileDisplay("back", null);
    }
    checkFormValidity();
  });

  // Handle drag and drop events
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    [frontZone, backZone].forEach((zone) => {
      zone.addEventListener(
        eventName,
        (e) => {
          e.preventDefault();
          e.stopPropagation();
        },
        false
      );
    });
  });

  // Handle file drop
  frontZone.addEventListener("drop", (e) => {
    frontUploadInput.files = e.dataTransfer.files;
    frontUploadInput.dispatchEvent(new Event("change"));
  });

  backZone.addEventListener("drop", (e) => {
    backUploadInput.files = e.dataTransfer.files;
    backUploadInput.dispatchEvent(new Event("change"));
  });

  document.getElementById("upload-form").addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;

    const onclickAttr = button.getAttribute("onclick");

    if (onclickAttr) {
      // Handle inline onclick functions
      e.stopPropagation();

      if (onclickAttr.includes("front-upload")) {
        frontUploadInput.click();
      } else if (onclickAttr.includes("back-upload")) {
        backUploadInput.click();
      } else if (onclickAttr.includes("removeFrontFile")) {
        frontUploadInput.value = "";
        frontUploadInput.dispatchEvent(new Event("change"));
      } else if (onclickAttr.includes("removeBackFile")) {
        backUploadInput.value = "";
        backUploadInput.dispatchEvent(new Event("change"));
      }
      return;
    }
  });

  // Make drop zones clickable, but ignore if clicking on interactive elements
  frontZone.addEventListener("click", (e) => {
    // Ignore clicks on buttons, inputs, or their children
    if (e.target.closest("button") || e.target.closest("input")) {
      return;
    }
    frontUploadInput.click();
  });

  backZone.addEventListener("click", (e) => {
    // Ignore clicks on buttons, inputs, or their children
    if (e.target.closest("button") || e.target.closest("input")) {
      return;
    }
    backUploadInput.click();
  });
});

// Keep these for backward compatibility with HTML onclick attributes
function removeFrontFile(event) {
  event.stopPropagation();
  const input = document.getElementById("front-upload");
  input.value = "";
  input.dispatchEvent(new Event("change"));
}

function removeBackFile(event) {
  event.stopPropagation();
  const input = document.getElementById("back-upload");
  input.value = "";
  input.dispatchEvent(new Event("change"));
}
