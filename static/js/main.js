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
        const imageDataUrl = canvas.toDataURL('image/jpeg');
        capturedImage.src = imageDataUrl;
        cameraContainer.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }

        // // 파일 다운로드 링크 생성 및 클릭
        // const downloadLink = document.createElement('a');
        // downloadLink.href = imageDataUrl;

        // // 원하는 파일명을 설정
        // downloadLink.download = 'captured-image.jpg';

        // // 링크를 DOM에 추가하고 자동으로 클릭한 후 제거
        // document.body.appendChild(downloadLink);
        // downloadLink.click();
        // document.body.removeChild(downloadLink);
        // 서버로 이미지 전송

        const base64Image = imageDataUrl.split(',')[1];

        fetch('/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image }) // Base64 이미지 데이터 전송
        })
        .then(response => response.json())
        .then(data => {
            console.log('이미지 저장 완료:', data);
        })
        .catch(error => {
            console.error('이미지 저장 실패:', error);
        });
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
        var len = Object.keys(result).length;
        for(i = 0; i < len; i++) {
            text += `
            <div class="flex items-center space-x-2">
                <i data-feather="${result[i].icon}" class="h-6 w-6 ${result[i].icon_color}"></i>
                <h3 class="text-xl font-semibold">${result[i].name}</h3>
            </div>
            <p class="text-gray-600">${result[i].description}</p>
            <div class="bg-blue-100 p-4 rounded-md">
                <h4 class="font-semibold text-blue-800 mb-2">처리 방법:</h4>
                <p class="text-blue-700">${result[i].disposal_method}</p>
            </div>
        `
        }
        classificationResult.innerHTML = text;
        // classificationResult.innerHTML = `
        //     <div class="flex items-center space-x-2">
        //         <i data-feather="${result.icon}" class="h-6 w-6 ${result.icon_color}"></i>
        //         <h3 class="text-xl font-semibold">${result.name}</h3>
        //     </div>
        //     <p class="text-gray-600">${result.description}</p>
        //     <div class="bg-blue-100 p-4 rounded-md">
        //         <h4 class="font-semibold text-blue-800 mb-2">처리 방법:</h4>
        //         <p class="text-blue-700">${result.disposal_method}</p>
        //     </div>
        // `;
        classificationResult.classList.remove('hidden');
        feather.replace();
    }

    feather.replace();
});