// toy_project/app/static/js/main.js
async function detectJellyfish() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image file');
        return;
    }

    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultSection').style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            alert(result.error);
            return;
        }

        displayResults(result);
    } catch (error) {
        alert('Error processing image: ' + error);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function displayResults(result) {
    const resultSection = document.getElementById('resultSection');
    const detectionList = document.getElementById('detectionList');
    const resultImage = document.getElementById('resultImage');

    // Clear previous results
    detectionList.innerHTML = '';

    // Display detections
    result.detections.forEach(detection => {
        const detectionElement = document.createElement('div');
        detectionElement.className = 'detection-item';
        detectionElement.textContent = 
            `${detection.class} (Confidence: ${(detection.confidence * 100).toFixed(2)}%)`;
        detectionList.appendChild(detectionElement);
    });

    // Display result image
    resultImage.src = result.result_image + '?t=' + new Date().getTime();

    // Show result section
    resultSection.style.display = 'block';
}