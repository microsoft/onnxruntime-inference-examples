import * as ort from './dist/esm/ort.webgpu.min.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = 'dist/';

const model_path = 'Xenova/yolov9-c';

const colorMap = [
    "green", "blue", "red", "yellow"
]

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let labels;
let insession = false;
let processed = 0;
let frames = 0;
let latencies = 0;

video.src = 'traffic-480.mp4';

function log(i) {
    console.log(i);
    // document.getElementById('status').innerText += `\n${i}`;
}

function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        model: "https://huggingface.co/Xenova/yolov9-c/resolve/main",
        provider: "webgpu",
        device: "gpu",
        threads: "1",
    };
    let vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        let pair = vars[i].split("=");
        if (pair[0] in config) {
            config[pair[0]] = decodeURIComponent(pair[1]);
        } else if (pair[0].length > 0) {
            throw new Error("unknown argument: " + pair[0]);
        }
    }
    config.threads = parseInt(config.threads);
    config.local = parseInt(config.local);
    return config;
}

const config = getConfig();

function renderBox([xmin, ymin, xmax, ymax, score, id]) {
    const color = colorMap[id % colorMap.length];
    const label = labels[id];
    ctx.beginPath();
    ctx.lineWidth = 1;
    ctx.strokeStyle = color;
    ctx.rect(xmin, ymin, xmax - xmin, ymax - ymin);
    ctx.stroke();
    ctx.font = "14px Comic Sans MS";
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.fillText(label, xmin, ymin);
}

async function fetchAndCache(url, name) {
    const fullurl = `${url}/${name}`;
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(fullurl);
        if (cachedResponse == undefined) {
            await cache.add(fullurl);
            cachedResponse = await cache.match(fullurl);
            log(`${name} (network)`);
        } else {
            log(`${name} (cached)`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`${name} (network)`);
        return await fetch(fullurl).then(response => response.arrayBuffer());
    }
}

async function main() {
    const video = document.getElementById('video');

    const json_bytes = await fetchAndCache(config.model, "config.json");
    const model_bytes = await fetchAndCache(config.model, "onnx/model.onnx");
    const opt = {};
    switch (config.provider) {
        case "wasm":
            break;
        case "webnn":
            opt.executionProviders = [{
                name: "webnn",
                deviceType: config.device,
                powerPreference: 'default',
                numThreads: config.threads,
            }];
            opt.freeDimensionOverrides = {
                batch: 1, height: 360, width: 640
            }
            break;
        case "webgpu":
            opt.executionProviders = [{
                name: "webgpu",
            }];
            break;
    }

    const sess = await ort.InferenceSession.create(model_bytes, opt);
    let textDecoder = new TextDecoder();
    const json_config = JSON.parse(textDecoder.decode(json_bytes));
    labels = json_config.id2label;

    // video upload
    document.getElementById("file-in").onchange = function (evt) {
        let target = evt.target || window.event.src, files = target.files;
        if (FileReader && files && files.length) {
            let fileReader = new FileReader();
            fileReader.onload = () => {
                video.src = fileReader.result;
            }
            fileReader.readAsDataURL(files[0]);
        }
    };

    log("ready.")

    video.addEventListener('loadedmetadata', function () {
        console.log("loadedmetadata");
    });

    video.addEventListener("play", () => {
        let w = video.videoWidth;
        let h = video.videoHeight;
        canvas.width = w;
        canvas.height = h;
        processed = 0;
        frames = 0;
        latencies = 0;

        document.getElementById('resolution').innerText = `${w}x${h}`;

        const frameCallback = (now, metadata) => {
            const data = ctx.getImageData(0, 0, w, h);
            frames = metadata.presentedFrames;
            if (!insession) {
                insession = true;
                ort.Tensor.fromImage(data).then((pixel_values) => {
                    const start = performance.now();
                    sess.run({ images: pixel_values }).then((outputs) => {
                        const end = performance.now();
                        latencies += end - start;
                        processed++;
                        if (processed % 10 == 0) {
                            document.getElementById('latency').innerText = (latencies / processed).toFixed(2) + "ms";
                            document.getElementById('dropped').innerText = (100 * (frames - processed) / frames).toFixed(1) + "%";
                        }
                        ctx.drawImage(video, 0, 0);
                        const t = outputs.outputs;
                        for (let i = 0; i < t.dims[0]; i++) {
                            renderBox(t.data.slice(i * t.dims[1], i * t.dims[1] + t.dims[0]));
                        }
                        insession = false;
                        video.requestVideoFrameCallback(frameCallback);
                    });
                });
            }
        };

        // request the first frame
        video.requestVideoFrameCallback(frameCallback);
    });

}

document.addEventListener("DOMContentLoaded", () => {
    main();
});
