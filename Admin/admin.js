    const dropdown = document.getElementById("dropdown");
    const profilePic = document.querySelector(".profile-pic");

    function toggleDropdown() {
      dropdown.style.display = dropdown.style.display === "block" ? "none" : "block";
    }

    window.addEventListener("click", function(e) {
      if (!profilePic.contains(e.target) && !dropdown.contains(e.target)) {
        dropdown.style.display = "none";
      }
    });