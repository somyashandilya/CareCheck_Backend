document.getElementById("detect-btn").addEventListener("click", function () {
  const fileInput = document.getElementById("ctscan");
  const output = document.getElementById("output");

  if (!fileInput.files || fileInput.files.length === 0) {
    output.textContent = "Please upload an image before detection.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("model", "cnn"); // Always using CNN

  output.textContent = "Processing...";

  fetch("http://localhost:5000/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Prediction failed");
      }
      return response.json();
    })
    .then((data) => {
      const classes = {
        0: "Beginning stage",
        1: "Malignant",
        2: "Normal",
      };
      const prediction = classes[data.prediction] || "Unknown result";
      output.textContent = `Prediction: ${prediction}`;
    })
    .catch((error) => {
      console.error(error);
      output.textContent = "Error during prediction. Try again.";
    });
});
