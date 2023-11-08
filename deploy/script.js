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

const video = document.getElementById('webcam_ip');
const liveView = document.getElementById('liveView');
const ageDetectionSection = document.getElementById('age-detection-section');
const enableWebcamButton = document.getElementById('enableWebcamButton');
const disableWebcamButton = document.getElementById('disableWebcamButton');

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

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', () => predictAge(video, stream));
  });
}

// Disable the live cam
function disableCam(event){
  
  // Enable this button
  disableWebcamButton.classList.add("removed");

  // Remove the 'removed' class from the enableWebcamButton
  enableWebcamButton.classList.remove('removed');
  
  // Remove the webcam stream from the video element
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
  }
}

// Detect faces from the image
async function predictAge(video, frame) {
    
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let gray = new cv.Mat();
    let cap = new cv.VideoCapture('webcam_ip');
    let faces = new cv.RectVector();

    const FPS = 24;
    function processVideo() {
        let begin = Date.now();
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
        try{
            faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0);
            console.log(faces.size());
        }catch(err){
            console.log(err);
        }
        for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
            let point1 = new cv.Point(face.x, face.y);
            let point2 = new cv.Point(face.x + face.width, face.y + face.height);
            cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
        }
        cv.imshow("webcam_op", dst);
        // schedule next one.
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    }

    // schedule first one.
    setTimeout(processVideo, 0);
}

// Store the resulting model in the global scope of our app.
var model = undefined;

tflite.loadTFLiteModel('age_detection.tflite').then(function (loadedModel) {
    model = loadedModel;
    // Show demo section now model is ready to use.
    ageDetectionSection.classList.remove('invisible');
});
