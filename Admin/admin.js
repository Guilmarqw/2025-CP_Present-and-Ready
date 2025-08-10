const dropdown = document.getElementById("dropdown");
const profilePic = document.querySelector(".profile-pic");

function toggleDropdown() {
  dropdown.style.display = dropdown.style.display === "block" ? "none" : "block";
}

window.addEventListener("click", function (e) {
  if (!profilePic.contains(e.target) && !dropdown.contains(e.target)) {
    dropdown.style.display = "none";
  }
});

function showSection(sectionId, element) {
  document.querySelectorAll('.main-content > div').forEach(div => {
    div.classList.add('hidden');
  });

  document.getElementById(`${sectionId}-section`).classList.remove('hidden');

  document.querySelectorAll('.menu-item').forEach(item => {
    item.classList.remove('active');
  });

  element.classList.add('active');
}