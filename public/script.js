/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const ageDetectionSection = document.getElementById('age-detection-section');
const enableWebcamButton = document.getElementById('enableWebcamButton');
const disableWebcamButton = document.getElementById('disableWebcamButton');

// This contains the highlighters that would be overlaid on top of the camera feed
let liveViewMarkers = [];
let webcamIsActive = false;

const resizeVideoDimensions = () => {
  video.height = Math.min(window.screen.height * 0.65, 480);
  video.width = Math.min(window.screen.width * 0.85, 640);
}

const faceCascade = new cv.CascadeClassifier();
let faceCascadeFile = 'haarcascade_frontalface_default.xml';
let utils = new Utils('errorMessage');
utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
    console.log('cascade ready to load.');
    faceCascade.load("haarcascade_frontalface_default.xml")
    console.log('cascade loaded.');
});

// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will 
// define in the next step.
if (getUserMediaSupported()) {
  enableWebcamButton.addEventListener('click', enableCam);
  disableWebcamButton.addEventListener('click', disableCam);
} else {
  console.warn('getUserMedia() is not supported by your browser');
}

// Enable the live webcam view and start classification.
function enableCam(event) {

  // Enable this button
  disableWebcamButton.classList.remove("removed");
  webcamIsActive = true;

  // Only continue if the Age Prediction Model has finished loading
  if (!model) {
    return;
  }
  
  // Hide the button once clicked.
  event.target.classList.add('removed');  
  
  // getUsermedia parameters to force video but not audio.
  const constraints = {
    video: true
  };

  video.addEventListener('loadedmetadata', resizeVideoDimensions);

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadedmetadata', predictAge);
  });
}

// Disable the live cam
function disableCam(event){
  
  // Enable this button
  disableWebcamButton.classList.add("removed");
  webcamIsActive = false;

  // Remove the 'removed' class from the enableWebcamButton
  enableWebcamButton.classList.remove('removed');
  
  // Remove the webcam stream from the video element
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
  }

  video.removeEventListener('loadedmetadata', resizeVideoDimensions);
  video.removeEventListener('loadedmetadata', predictAge);
}

// Detect faces from the image
function predictAge() {

  // Remove any highlighting we did previous frame.
  for (let i = 0; i < liveViewMarkers.length; i++) {
    liveView.removeChild(liveViewMarkers[i]);
  }

  liveViewMarkers.splice(0);
  
  let cap = new cv.VideoCapture('webcam');
  const videoTagHeight = video.height;
  const videoTagWidth = video.width;
  const contextWidth = videoTagWidth * 0.05;
  const contextHeight = videoTagHeight * 0.10;

  let src = new cv.Mat(videoTagHeight, videoTagWidth, cv.CV_8UC4);
  let gray = new cv.Mat();
  let faces = new cv.RectVector();

  cap.read(src);
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

  try{
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0);
  } catch(err){
      console.error('Error detecting faces:', err);
  }

  for (let i = 0; i < faces.size(); ++i) {
    let face = faces.get(i);
    
    // For the sake of our age prediction model
    let x = face.x;
    let y = face.y;
    let width = face.width;
    let height = face.height;

    // Ensure the new bounding box stays within image boundaries
    if (x - contextWidth >= 0) {
        x -= contextWidth;
        width += contextWidth;
    }
    if (y - contextHeight >= 0) {
        y -= contextHeight;
        height += contextHeight;
    }
    if (x + width + contextWidth <= src.cols) {
        width += contextWidth;
    }
    if (y + height + contextHeight <= src.rows) {
        height += contextHeight;
    }

    // Crop, convert to RGB, and resize the face region
    let faceRegion = new cv.Mat();
    cv.cvtColor(src, faceRegion, cv.COLOR_RGBA2RGB);
    faceRegion = faceRegion.roi(new cv.Rect(x, y, width, height));
    cv.resize(faceRegion, faceRegion, new cv.Size(128, 128));

    let faceTensor = tf.tensor(faceRegion.data, [1, 128, 128, 3], 'float32');
    faceTensor = faceTensor.div(tf.scalar(255.));
    let age = model.predict(faceTensor).dataSync()[0];
    const ageText = `Age: ${age.toFixed(2)}`;

    const p = document.createElement('p');
    p.innerText = ageText;
    p.style = `margin-left: ${x}px; margin-top: ${y - 10}px; width: ${width - 10}px; top: 0px; left: 0px;`;

    const highlighter = document.createElement('div');
    highlighter.setAttribute('class', 'highlighter');
    highlighter.style = `left: ${x}px; top: ${y}px; width: ${width}px; height: ${height}px;`;

    liveView.appendChild(highlighter);
    liveView.appendChild(p);
    liveViewMarkers.push(highlighter);
    liveViewMarkers.push(p);

    // Clean up Mats
    faceRegion.delete();
    faceTensor.dispose();
  }

  // Clean up Mats
  src.delete();
  gray.delete();

  // Call this function again to keep predicting when the browser is ready.
  if (webcamIsActive){
    window.requestAnimationFrame(predictAge);
  }
}

// Store the resulting model in the global scope of our app.
var model = undefined;

tflite.loadTFLiteModel('age_detection.tflite').then(function (loadedModel) {
    model = loadedModel;
    // Show demo section when model is ready to use.
    ageDetectionSection.classList.remove('invisible');
});
