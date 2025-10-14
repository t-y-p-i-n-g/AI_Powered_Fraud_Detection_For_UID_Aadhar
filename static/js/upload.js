// File state
let frontFile = null;
let backFile = null;

// Initialize drag and drop
document.addEventListener("DOMContentLoaded", function () {
  const frontZone = document.getElementById("front-drop-zone");
  const backZone = document.getElementById("back-drop-zone");

  // Front card drag and drop
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    frontZone.addEventListener(eventName, preventDefaults, false);
    backZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  // Highlight on drag
  ["dragenter", "dragover"].forEach((eventName) => {
    frontZone.addEventListener(eventName, () => highlightFront(true), false);
    backZone.addEventListener(eventName, () => highlightBack(true), false);
  });

  ["dragleave", "drop"].forEach((eventName) => {
    frontZone.addEventListener(eventName, () => highlightFront(false), false);
    backZone.addEventListener(eventName, () => highlightBack(false), false);
  });

  // Handle drops
  frontZone.addEventListener("drop", handleFrontDrop, false);
  backZone.addEventListener("drop", handleBackDrop, false);

  // Click to upload
  frontZone.addEventListener("click", () =>
    document.getElementById("front-upload").click()
  );
  backZone.addEventListener("click", () =>
    document.getElementById("back-upload").click()
  );
});

function highlightFront(highlight) {
  const zone = document.getElementById("front-drop-zone");
  if (highlight) {
    zone.classList.add("border-blue-500", "bg-blue-50");
    zone.classList.remove("border-slate-300", "bg-slate-50");
  } else if (!frontFile) {
    zone.classList.remove("border-blue-500", "bg-blue-50");
    zone.classList.add("border-slate-300", "bg-slate-50");
  }
}

function highlightBack(highlight) {
  const zone = document.getElementById("back-drop-zone");
  if (highlight) {
    zone.classList.add("border-blue-500", "bg-blue-50");
    zone.classList.remove("border-slate-300", "bg-slate-50");
  } else if (!backFile) {
    zone.classList.remove("border-blue-500", "bg-blue-50");
    zone.classList.add("border-slate-300", "bg-slate-50");
  }
}

function handleFrontDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length > 0) {
    handleFrontFile(files[0]);
  }
}

function handleBackDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length > 0) {
    handleBackFile(files[0]);
  }
}

function handleFrontFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please upload an image file");
    return;
  }

  frontFile = file;
  const zone = document.getElementById("front-drop-zone");
  zone.classList.remove(
    "border-slate-300",
    "bg-slate-50",
    "hover:border-blue-400",
    "hover:bg-blue-50/50"
  );
  zone.classList.add("border-green-500", "bg-green-50");

  document.getElementById("front-upload-content").classList.add("hidden");
  document.getElementById("front-file-info").classList.remove("hidden");
  document.getElementById("front-file-name").textContent = file.name;
  document.getElementById("front-file-size").textContent =
    (file.size / 1024).toFixed(2) + " KB";

  checkFormValidity();
}

function handleBackFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please upload an image file");
    return;
  }

  backFile = file;
  const zone = document.getElementById("back-drop-zone");
  zone.classList.remove(
    "border-slate-300",
    "bg-slate-50",
    "hover:border-blue-400",
    "hover:bg-blue-50/50"
  );
  zone.classList.add("border-green-500", "bg-green-50");

  document.getElementById("back-upload-content").classList.add("hidden");
  document.getElementById("back-file-info").classList.remove("hidden");
  document.getElementById("back-file-name").textContent = file.name;
  document.getElementById("back-file-size").textContent =
    (file.size / 1024).toFixed(2) + " KB";

  checkFormValidity();
}

function removeFrontFile() {
  event.stopPropagation();
  frontFile = null;
  const zone = document.getElementById("front-drop-zone");
  zone.classList.remove("border-green-500", "bg-green-50");
  zone.classList.add(
    "border-slate-300",
    "bg-slate-50",
    "hover:border-blue-400",
    "hover:bg-blue-50/50"
  );

  document.getElementById("front-upload-content").classList.remove("hidden");
  document.getElementById("front-file-info").classList.add("hidden");
  document.getElementById("front-upload").value = "";

  checkFormValidity();
}

function removeBackFile() {
  event.stopPropagation();
  backFile = null;
  const zone = document.getElementById("back-drop-zone");
  zone.classList.remove("border-green-500", "bg-green-50");
  zone.classList.add(
    "border-slate-300",
    "bg-slate-50",
    "hover:border-blue-400",
    "hover:bg-blue-50/50"
  );

  document.getElementById("back-upload-content").classList.remove("hidden");
  document.getElementById("back-file-info").classList.add("hidden");
  document.getElementById("back-upload").value = "";

  checkFormValidity();
}

function checkFormValidity() {
  const analyzeBtn = document.getElementById("analyze-btn");
  if (frontFile && backFile) {
    analyzeBtn.disabled = false;
  } else {
    analyzeBtn.disabled = true;
  }
}
