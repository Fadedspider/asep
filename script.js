// Function to check if an element is in the viewport
function isElementInViewport(el) {
    const rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Function to handle the scroll event
function handleScroll() {
    const teamMembers = document.querySelectorAll('.team-member');
    teamMembers.forEach(member => {
        if (isElementInViewport(member)) {
            // Add the 'visible' class when in view
            member.classList.add('visible');
            member.classList.remove('hidden');
        } else {
            // Add the 'hidden' class when out of view for a fade-out effect
            member.classList.remove('visible');
            member.classList.add('hidden');
        }
    });
}

// Add scroll event listener
window.addEventListener('scroll', handleScroll);

// Initial check in case the section is already in view when the page loads
handleScroll();