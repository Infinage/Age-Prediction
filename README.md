## Age Detection

The goal of this project is to explore the full end to end life cycle of a Machine learning project. Although the model predictions aren't great, the pipeline to build, train and deploy has been tested to work well. 

A live inference demo is deployed on 
1. [Netlify](https://age-pred.netlify.app/).
2. [AWS with custom Subdomain](https://age-prediction.deesa.space/).

### TODO
1. ~~Train a simple age detection model~~
2. ~~Convert trained model to TFLite~~
3. ~~Using Opencv JS & TFJS create a live age detection UI~~
4. ~~Dockerize development environment~~
5. ~~Deploy the front end only code~~
6. ~~Mobile friendly inference~~
7. Refactor code base to use async await instead of event listeners? 
8. Improve age prediction model accuracy
9. Explore options to train and use Yolo for face detection

### Useful links
- Age prediction dataset: https://www.kaggle.com/datasets/mariafrenti/age-prediction 
- Face detection with Open CV JS: https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html
- Yolo training with Keras_cv: https://keras.io/examples/vision/yolov8/
- Face detection Dataset: https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset/data
- Convert Model to TFLite: https://www.tensorflow.org/lite/models/convert
- Using TFLite with TFJS: https://js.tensorflow.org/api_tflite/0.0.1-alpha.10/
- TFJS Boiler plate copied from Google: https://codelabs.developers.google.com/codelabs/tensorflowjs-object-detection#7
- TFlite Sample Inference: https://codepen.io/jinjingforever/pen/xxgWRaE?editors=1010
- Open CV JS Object detection: https://docs.opencv.org/3.4/df/d6c/tutorial_js_face_detection_camera.html
