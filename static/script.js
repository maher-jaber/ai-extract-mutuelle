const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const result = document.getElementById("result");
const spinner = document.getElementById("spinner");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;
    spinner.classList.remove("hidden");
    result.textContent = "";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const res = await fetch("/ocr", { method: "POST", body: formData });
    const data = await res.json();

    spinner.classList.add("hidden");
    result.textContent = JSON.stringify(data, null, 2);
});
