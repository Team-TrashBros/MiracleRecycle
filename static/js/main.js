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
        text = '';
        for(i = 0; i < len; i++) {
            const imageName = result[i].filename; // 불러올 이미지 파일 이름
            const imagePath = `/img/${imageName}`; // Flask 서버에서 제공하는 경로
            text += `
            <div class="aspect-h-9 bg-black rounded-lg overflow-hidden">
                     <img src="${imagePath}" alt="Captured waste" class="object-cover" style="margin: 0 auto;"/>
                </div>
            <div class="flex items-center space-x-2">
                <i data-feather="${result[i].icon}" class="h-6 w-6 ${result[i].icon_color}"></i>
                <h3 class="text-xl font-semibold">${result[i].name}</h3>
            </div>
            <p class="text-gray-600">${result[i].description}</p>
            <div class="bg-blue-100 p-4 rounded-md">
                <h4 class="font-semibold text-blue-800 mb-2">처리 방법:</h4>
                <p class="text-blue-700">${result[i].disposal_method}</p>
                <div class="flex justify-center space-x-4">
                    <!-- Trigger Button -->
                    <button id="openModal_${i}" class="open-button inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">Detail</button>
                </div>
            </div>

            <!-- Modal Structure -->
            <div id="myModal_${i}" class="modal">
                <div class="modal-content">
                    <span class="close" data-index="${i}">&times;</span>
                    <h2>Modal Title_${i}</h2>
                    <p>This is a simple modal window.</p>
                </div>
            </div>
        `
        }

        classificationResult.innerHTML = text;
        classificationResult.classList.remove('hidden');
        feather.replace();

        // 이벤트 리스너 추가
        for (let i = 0; i < len; i++) {
            // 모달 및 버튼 요소 가져오기
            const modal = document.getElementById(`myModal_${i}`);
            const btn = document.getElementById(`openModal_${i}`);
            const closeBtn = modal.querySelector('.close');

            // 모달 열기 버튼 클릭 시 모달 열기
            btn.addEventListener('click', () => {
                modal.style.display = 'block';
            });

            // 닫기 버튼 클릭 시 모달 닫기
            closeBtn.addEventListener('click', () => {
                modal.style.display = 'none';
            });

            // 모달 외부 클릭 시 모달 닫기
            window.addEventListener('click', (event) => {
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            });
        }

    }

    feather.replace();
});