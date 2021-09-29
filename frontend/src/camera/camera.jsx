import React, { createRef } from "react";
import { Hands, HAND_CONNECTIONS } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";

class CamComp extends React.Component {
  constructor(props) {
    super(props);
    this.videoElement = createRef();
    this.canvasElement = createRef();
  }

  componentDidMount = () => {
    this.videoElement = this.videoElement.current;
    this.canvasElement = this.canvasElement.current;
    this.canvasCtx = this.canvasElement.getContext('2d');

    const hands = new Hands({locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }});
    hands.setOptions({
      maxNumHands: 2,
      minDetectionConfidence: 0.8,
      minTrackingConfidence: 0.8
    });
    hands.onResults(this.onResults);
    const camera = new Camera(this.videoElement, {
      onFrame: async () => {
        await hands.send({image: this.videoElement});
      },
      width: 1280,
      height: 720
    });
    camera.start();
  };

  onResults = (results) => {
    console.log(results.multiHandLandmarks);
    this.canvasCtx.save();
    this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    this.canvasCtx.drawImage(
        results.image, 0, 0, this.canvasElement.width, this.canvasElement.height);
    if (results.multiHandLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(this.canvasCtx, landmarks, HAND_CONNECTIONS,
                       {color: '#00FF00', lineWidth: 3});
        drawLandmarks(this.canvasCtx, landmarks, {color: '#FF0000', lineWidth: 1});
      }
    }
    this.canvasCtx.restore();
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <video ref={this.videoElement} className="input_video" hidden></video>
          <canvas ref={this.canvasElement} className="output_canvas" width="1280px" height="720px"></canvas>
        </header>
      </div>
    );
  }
}

export default CamComp;
