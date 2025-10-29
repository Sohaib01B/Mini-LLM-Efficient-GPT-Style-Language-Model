// Update slider values in real-time
document.getElementById('temperature').addEventListener('input', function() {
    document.getElementById('tempValue').textContent = this.value;
});

document.getElementById('top_k').addEventListener('input', function() {
    document.getElementById('topKValue').textContent = this.value;
});

document.getElementById('max_length').addEventListener('input', function() {
    document.getElementById('lengthValue').textContent = this.value;
});

// Handle form submission
document.getElementById('storyForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const prompt = document.getElementById('prompt').value;
    const temperature = parseFloat(document.getElementById('temperature').value);
    const top_k = parseInt(document.getElementById('top_k').value);
    const max_length = parseInt(document.getElementById('max_length').value);
    
    // Show loading state
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').style.display = 'none';
    document.getElementById('generateBtn').disabled = true;
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                temperature: temperature,
                top_k: top_k,
                max_length: max_length
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Display result
            document.getElementById('originalPrompt').textContent = `Prompt: "${data.prompt}"`;
            document.getElementById('generatedStory').textContent = data.story;
            document.getElementById('result').style.display = 'block';
            
            // Reload history
            loadHistory();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error generating story: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('generateBtn').disabled = false;
    }
});

// Copy to clipboard function
function copyToClipboard() {
    const storyText = document.getElementById('generatedStory').textContent;
    navigator.clipboard.writeText(storyText).then(() => {
        // Show temporary success message
        const btn = event.target;
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-check me-2"></i> Copied!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
}

// Load story history
async function loadHistory() {
    try {
        const response = await fetch('/history');
        const history = await response.json();
        
        const historyList = document.getElementById('historyList');
        
        if (history.length === 0) {
            historyList.innerHTML = '<p class="text-muted text-center">No story history yet.</p>';
            return;
        }
        
        historyList.innerHTML = history.reverse().map(item => `
            <div class="card history-item mb-3">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <h6 class="text-muted">Prompt: "${item.prompt}"</h6>
                            <p class="mb-2">${item.story}</p>
                            <small class="text-muted">
                                <i class="fas fa-clock me-1"></i>
                                ${new Date(item.timestamp).toLocaleString()}
                            </small>
                            <small class="text-muted ms-3">
                                <i class="fas fa-sliders-h me-1"></i>
                                Temp: ${item.params.temperature}, Top-K: ${item.params.top_k}, Length: ${item.params.max_length}
                            </small>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Clear history
document.getElementById('clearHistory').addEventListener('click', async function() {
    if (confirm('Are you sure you want to clear your story history?')) {
        try {
            const response = await fetch('/clear_history', {
                method: 'POST'
            });
            
            if (response.ok) {
                loadHistory();
            }
        } catch (error) {
            console.error('Error clearing history:', error);
        }
    }
});

// Load history on page load
document.addEventListener('DOMContentLoaded', loadHistory);