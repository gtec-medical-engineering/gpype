// Make all external links open in new tabs
document.addEventListener('DOMContentLoaded', function() {
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