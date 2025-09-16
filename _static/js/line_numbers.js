// Pad line numbers with leading zeros
document.addEventListener('DOMContentLoaded', function() {
    // Find all line number spans
    const lineNumbers = document.querySelectorAll('span.linenos');
    
    lineNumbers.forEach(function(span) {
        // Get the current text content (the line number)
        const lineNum = span.textContent.trim();
        
        // Pad with leading zeros to 2 digits
        const paddedNum = lineNum.padStart(2, '0');
        
        // Update the content
        span.textContent = paddedNum;
    });
});
