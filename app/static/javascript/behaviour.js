const termsCheckbox = document.getElementById("terms");
const submitButton = document.getElementById("submitBtn");

termsCheckbox.addEventListener("change", function () {
  if (termsCheckbox.checked) {
    submitButton.disabled = false;
  } else {
    submitButton.disabled = true;
  }
});

document.getElementById("currentYear").innerHTML = new Date().getFullYear();

