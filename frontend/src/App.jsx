import CamComp from "./camera/camera"
import './App.css';

function App() {
  let onPredictionsTM_img = (value) => {
    console.log("onPredictionsTM_img :",value)
  };

  let imageModelURL = './model/model.json';
	
  return (
    <CamComp callback={onPredictionsTM_img} modelUrl={imageModelURL}/>
  );
}

export default App;
