import React, { createRef } from "react";
import ml5 from "ml5";

class CamComp extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      videoRef:null,
      classifier:null
    }
  }

  componentDidMount() {
    this.videoRef = createRef();
    this.setState({ videoRef: this.videoRef });
    console.log(this.props.modelUrl);
    this.classifier = ml5.imageClassifier(this.props.modelUrl, () => {
      navigator.mediaDevices
        .getUserMedia({ video: true, audio: false })
        .then((stream) => {
          this.videoRef.current.srcObject = stream;
          this.videoRef.current.play();
        });
      this.setState({ classifier: this.classifier });
      //console.log(this.classifier);
    });

    var classifier = this.classifier;
    var videoRef = this.videoRef;
    var callback = this.props.callback;
    console.log(videoRef.current);
    this.intervalID = setInterval(function () {
      //console.log("update", classifier);
      if (classifier && videoRef.current) {
        classifier.classify(videoRef.current, (error, results) => {
          if (error) {
            console.error(error);
            return;
          }
          callback(results);
        });
      }
    }, 500);
  }

  render() {
    return (
      <div>
        <video
          ref={this.videoRef}
          style={{ transform: "scale(-1, 1)" }}
          width="300"
          height="150"
        />
      </div>
    );
  }
}

export default CamComp;
