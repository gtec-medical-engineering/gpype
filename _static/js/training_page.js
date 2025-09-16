// Add special styling to h2 headings on training index page
document.addEventListener('DOMContentLoaded', function() {
    // Check if we have season-intro containers (unique to training page)
    const seasonContainers = document.querySelectorAll('.season-intro');
    
    if (seasonContainers.length > 0) {
        // Find all h2 elements and add padding to the ones that are season headings
        const h2Elements = document.querySelectorAll('h2');
        h2Elements.forEach(function(h2) {
            // Check if this h2 is followed by a season-intro container
            const nextElement = h2.nextElementSibling;
            if (nextElement && nextElement.classList.contains('season-intro')) {
                h2.style.paddingLeft = '16px';
                h2.style.paddingTop = '16px';
                console.log('Applied padding to:', h2.textContent);
            }
        });
    }
    
    // Make all external links open in new tabs
    const links = document.querySelectorAll('a');
    links.forEach(function(link) {
        const href = link.getAttribute('href');
        if (href) {
            // Check if it's an external link (starts with http:// or https://)
            if (href.startsWith('http://') || href.startsWith('https://')) {
                // Don't modify if target is already set
                if (!link.getAttribute('target')) {
                    link.setAttribute('target', '_blank');
                    link.setAttribute('rel', 'noopener noreferrer');
                    console.log('Made external link open in new tab:', href);
                }
            }
        }
    });
});