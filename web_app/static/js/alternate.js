
  const videoElement = document.querySelector('.input_video');
  const canvasElement = document.querySelector('.output_canvas');
  const canvasCtx = canvasElement.getContext('2d');
  const messageOverlay = document.getElementById('message-overlay');
  const loader = document.getElementById('loader');
  const statusMessage = document.getElementById('status-message');
  const toggleCameraButton = document.getElementById('toggle-camera-button');
  const showLandmarksCheckbox = document.getElementById('show-landmarks-checkbox');
  const translationOutput = document.getElementById('translation-output');

  let holistic, camera;
  let isCameraActive = false;
  let showLandmarks = true;
  let inferenceIntervalId = null;
  let lastPredictedWord = '';

  function onResults(results) {
    if (!isCameraActive) return;

    if (messageOverlay.style.display !== 'none') messageOverlay.style.display = 'none';
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (showLandmarks) {
      window.drawConnectors(canvasCtx, results.poseLandmarks, window.POSE_CONNECTIONS, { color: '#0ea5e9', lineWidth: 4 });
      window.drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#0284c7', lineWidth: 2 });
      window.drawConnectors(canvasCtx, results.faceLandmarks, window.FACEMESH_TESSELATION, { color: '#f87171', lineWidth: 1 });
      window.drawConnectors(canvasCtx, results.leftHandLandmarks, window.HAND_CONNECTIONS, { color: '#34d399', lineWidth: 4 });
      window.drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#10b981', lineWidth: 2 });
      window.drawConnectors(canvasCtx, results.rightHandLandmarks, window.HAND_CONNECTIONS, { color: '#fbbf24', lineWidth: 4 });
      window.drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#f59e0b', lineWidth: 2 });
    }

    canvasCtx.restore();
  }

  async function sendFrameToServer() {
    if (!isCameraActive) return;
    const frame = canvasElement.toDataURL('image/png');
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: frame })
      });
      const data = await res.json();
      if (data.word && data.word !== lastPredictedWord) {
        lastPredictedWord = data.word;
        translationOutput.innerText = data.word.toUpperCase();
      }
    } catch (err) { console.error(err); }
  }

  async function startCamera() {
    statusMessage.innerText = 'Please allow camera access.';
    loader.style.display = 'block';

    camera = new window.Camera(videoElement, {
      onFrame: async () => { if (videoElement.readyState === 4) await holistic.send({ image: videoElement }); },
      width: 1280, height: 720
    });
    await camera.start();

    isCameraActive = true;
    updateToggleButton();
    inferenceIntervalId = setInterval(sendFrameToServer, 1000);
  }

  function stopCamera() {
    if (camera) camera.stop();
    isCameraActive = false;
    updateToggleButton();
    if (inferenceIntervalId) clearInterval(inferenceIntervalId);
    inferenceIntervalId = null;
    messageOverlay.style.display = 'flex';
    loader.style.display = 'none';
    statusMessage.innerText = 'Camera is off. Click "Start Camera" to begin.';
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  }

  function updateToggleButton() {
    toggleCameraButton.innerText = isCameraActive ? 'Stop Camera' : 'Start Camera';
    toggleCameraButton.className = isCameraActive
      ? 'bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-6 rounded-lg'
      : 'bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-6 rounded-lg';
  }

  toggleCameraButton.addEventListener('click', () => isCameraActive ? stopCamera() : startCamera());
  showLandmarksCheckbox.addEventListener('change', () => showLandmarks = showLandmarksCheckbox.checked);
  document.getElementById('play-tts-button').addEventListener('click', () => {
    if (lastPredictedWord) speechSynthesis.speak(new SpeechSynthesisUtterance(lastPredictedWord));
  });

  (async () => {
    statusMessage.innerText = 'Loading AI models...';
    loader.style.display = 'block';
    await Promise.all([
      import('https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js'),
      import('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js'),
      import('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js')
    ]);
    holistic = new window.Holistic({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}` });
    holistic.setOptions({ modelComplexity: 1, refineFaceLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    holistic.onResults(onResults);
    loader.style.display = 'none';
    statusMessage.innerText = 'Models loaded. Click "Start Camera".';
  })();

