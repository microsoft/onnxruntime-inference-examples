# Run Segment-Anything in your browser using webgpu and onnxruntime-web

This example demonstrates how to run [Segment-Anything](https://github.com/facebookresearch/segment-anything) in your 
browser using [onnxruntime-web](https://github.com/microsoft/onnxruntime) and webgpu.

Segment-Anything is a encoder/decoder model. The encoder creates embeddings and using the embeddings the decoder creates the segmentation mask.

One can run the decoder in onnxruntime-web using WebAssembly with  latencies at ~200ms. 

The encoder is much more compute intensive and takes ~45sec using WebAssembly what is not practical.
Using webgpu we can speedup the encoder ~50 times and it becomes visible to run it inside the browser, even on a integrated GPU.

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

We use [samexporter](https://github.com/vietanhdev/samexporter) to export encoder and decoder to onnx.
Install samexporter:
```sh
pip install https://github.com/vietanhdev/samexporter
```
Download the pytorch model from [Segment-Anything](https://github.com/facebookresearch/segment-anything). We use the smallest flavor (vit_b).
```sh
curl -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
Export both encoder and decoder to onnx:
```sh
python -m samexporter.export_encoder --checkpoint models/sam_vit_b_01ec64.pth \
    --output models/sam_vit_b_01ec64.encoder.onnx \
    --model-type vit_b 

python -m samexporter.export_decoder --checkpoint models/sam_vit_b_01ec64.pth \
    --output models/sam_vit_b_01ec64.decoder.onnx \
    --model-type vit_b \
    --return-single-mask
```
### Start a web server
Use NPM package `light-server` to serve the current folder at http://localhost:8888/.
To start the server, run:
```sh
npx light-server -s . -p 8888
```

### Point your browser at the web server
Once the web server is running, open your browser and navigate to http://localhost:8888/. 
You should now be able to run Segment-Anything in your browser.

## TODO
* add support for fp16
* add support for MobileSam
