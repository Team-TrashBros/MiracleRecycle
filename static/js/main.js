document.addEventListener('DOMContentLoaded', (event) => {
    const video = document.getElementById('video');
    const startCameraButton = document.getElementById('start-camera');
    const captureImageButton = document.getElementById('capture-image');
    const classifyWasteButton = document.getElementById('classify-waste');
    const resetClassificationButton = document.getElementById('reset-classification');
    const cameraContainer = document.getElementById('camera-container');
    const resultContainer = document.getElementById('result-container');
    const classificationResult = document.getElementById('classification-result');
    const capturedImage = document.getElementById('captured-image');

    let stream = null;
    
    startCameraButton.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({video:true})
            video.srcObject = stream;
        } catch (err) {
            alert("Error accessing the camera", err);
        }
    });
    
    captureImageButton.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        capturedImage.src = canvas.toDataURL('image/jpeg');
        cameraContainer.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });

    classifyWasteButton.addEventListener('click', async () => {
        try {
            const response = await fetch('/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: capturedImage.src }),
            });
            const result = await response.json();
            displayClassificationResult(result);
        } catch (error) {
            console.error('Error:', error);
        }
    });

    resetClassificationButton.addEventListener('click', () => {
        cameraContainer.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        classificationResult.classList.add('hidden');
        classificationResult.innerHTML = '';
    });

    function displayClassificationResult(result) {
        classificationResult.innerHTML = `
            <div class="flex items-center space-x-2">
                <i data-feather="${result.icon}" class="h-6 w-6 ${result.icon_color}"></i>
                <h3 class="text-xl font-semibold">${result.name}</h3>
            </div>
            <p class="text-gray-600">${result.description}</p>
            <div class="bg-blue-100 p-4 rounded-md">
                <h4 class="font-semibold text-blue-800 mb-2">처리 방법:</h4>
                <p class="text-blue-700">${result.disposal_method}</p>
            </div>
        `;
        classificationResult.classList.remove('hidden');
        feather.replace();
    }

    feather.replace();
});