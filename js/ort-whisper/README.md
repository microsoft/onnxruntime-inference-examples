# Run the OpenAI whisper in your browser using onnxruntime-web

This example demonstrates how to run [whisper tiny.en](https://github.com/openai/whisper) in your 
browser using [onnxruntime-web](https://github.com/microsoft/onnxruntime) and the browser's audio interfaces.

## Usage

### Installation
First, install the required dependencies by running the following command in your terminal:
```sh
npm install
```

### Build the code
Next, bundle the code using webpack by running:
```sh
npm run build
```
this generates the bundle file `./dist/bundle.min.js`

### Create an ONNX Model
To create an optimized end-to-end ONNX model from the original OpenAI Whisper model, follow these steps:

1. Install Olive:
```sh
pip install olive-ai[cpu]
```

2. Go to the Olive Whisper example at https://github.com/microsoft/Olive/tree/v0.6.0/examples/whisper and install the example requirements:
```sh
pip install -r requirements.txt
```

3. Run the following Olive commands to optimize the Whisper model:
```sh
python prepare_whisper_configs.py --model_name openai/whisper-tiny.en --no_audio_decoder
python -m olive.workflows.run --config whisper_cpu_int8.json --setup
python -m olive.workflows.run --config whisper_cpu_int8.json
```

4. Move the resulting model from `models/whisper_cpu_int8_0_model.onnx` into `js/ort-whisper/`.

### Start a web server
Use NPM package `light-server` to serve the current folder at http://localhost:8888/.
To start the server, run:
```sh
npx light-server -s . -p 8888
```

### Point your browser at the web server
Once the web server is running, open your browser and navigate to http://localhost:8888/. 
You should now be able to run Whisper in your browser.

