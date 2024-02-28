// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run segment-anything with webgpu in onnxruntime-web.
//

import ort from 'onnxruntime-web/webgpu';

// the image size on canvas
const MAX_WIDTH = 500;
const MAX_HEIGHT = 500;

// the image size supported by the model
const MODEL_WIDTH = 1024;
const MODEL_HEIGHT = 1024;

const MODELS = {
    sam_b: [
        {
            name: "sam-b-encoder",
            url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b_01ec64.encoder-fp16.onnx",
            size: 180,
        },
        {
            name: "sam-b-decoder",
            url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b_01ec64.decoder.onnx",
            size: 17,
        },
    ],
    sam_b_int8: [
        {
            name: "sam-b-encoder-int8",
            url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b-encoder-int8.onnx",
            size: 108,
        },
        {
            name: "sam-b-decoder-int8",
            url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b-decoder-int8.onnx",
            size: 5,
        },
    ],
};

const config = getConfig();

ort.env.wasm.wasmPaths = 'dist/';
ort.env.wasm.numThreads = config.threads;
// ort.env.wasm.proxy = config.provider == "wasm";

let canvas;
let filein;
let decoder_latency;

var image_embeddings;
var points = [];
var labels = [];
var imageImageData;
var isClicked = false;
var maskImageData;

function log(i) {
    document.getElementById('status').innerText += `\n${i}`;
}

/**
 * create config from url
 */
function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        model: "sam_b",
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

/**
 * clone tensor
 */
function cloneTensor(t) {
    return new ort.Tensor(t.type, Float32Array.from(t.data), t.dims);
}

/*
 * create feed for the original facebook model
 */
function feedForSam(emb, points, labels) {
    const maskInput = new ort.Tensor(new Float32Array(256 * 256), [1, 1, 256, 256]);
    const hasMask = new ort.Tensor(new Float32Array([0]), [1,]);
    const origianlImageSize = new ort.Tensor(new Float32Array([MODEL_HEIGHT, MODEL_WIDTH]), [2,]);
    const pointCoords = new ort.Tensor(new Float32Array(points), [1, points.length / 2, 2]);
    const pointLabels = new ort.Tensor(new Float32Array(labels), [1, labels.length]);

    return {
        "image_embeddings": cloneTensor(emb.image_embeddings),
        "point_coords": pointCoords,
        "point_labels": pointLabels,
        "mask_input": maskInput,
        "has_mask_input": hasMask,
        "orig_im_size": origianlImageSize
    }
}

/*
 * Handle cut-out event
 */
async function handleCut(event) {
    if (points.length == 0) {
        return;
    }

    const [w, h] = [canvas.width, canvas.height];

    // canvas for cut-out
    const cutCanvas = new OffscreenCanvas(w, h);
    const cutContext = cutCanvas.getContext('2d');
    const cutPixelData = cutContext.getImageData(0, 0, w, h);

    // need to rescale mask to image size
    const maskCanvas = new OffscreenCanvas(w, h);
    const maskContext = maskCanvas.getContext('2d');
    maskContext.drawImage(await createImageBitmap(maskImageData), 0, 0);
    const maskPixelData = maskContext.getImageData(0, 0, w, h);

    // copy masked pixels to cut-out
    for (let i = 0; i < maskPixelData.data.length; i += 4) {
        if (maskPixelData.data[i] > 0) {
            for (let j = 0; j < 4; ++j) {
                const offset = i + j;
                cutPixelData.data[offset] = imageImageData.data[offset];
            }
        }
    }
    cutContext.putImageData(cutPixelData, 0, 0);

    // Download image 
    const link = document.createElement('a');
    link.download = 'image.png';
    link.href = URL.createObjectURL(await cutCanvas.convertToBlob());
    link.click();
    link.remove();
}

async function decoder(points, labels) {
    let ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imageImageData.width;
    canvas.height = imageImageData.height;
    ctx.putImageData(imageImageData, 0, 0);

    if (points.length > 0) {
        // need to wait for encoder to be ready
        if (image_embeddings === undefined) {
            await MODELS[config.model][0].sess;
        }

        // wait for encoder to deliver embeddings
        const emb = await image_embeddings;

        // the decoder
        const session = MODELS[config.model][1].sess;

        const feed = feedForSam(emb, points, labels);
        const start = performance.now();
        const res = await session.run(feed);
        decoder_latency.innerText = `${(performance.now() - start).toFixed(1)}ms`;

        for (let i = 0; i < points.length; i += 2) {
            ctx.fillStyle = 'blue';
            ctx.fillRect(points[i], points[i + 1], 10, 10);
        }
        const mask = res.masks;
        maskImageData = mask.toImageData();
        ctx.globalAlpha = 0.3;
        ctx.drawImage(await createImageBitmap(maskImageData), 0, 0);
    }
}

function getPoint(event) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.trunc(event.clientX - rect.left);
    const y = Math.trunc(event.clientY - rect.top);
    return [x, y];
}

/**
 * handler mouse move event
 */
async function handleMouseMove(event) {
    if (isClicked) {
        return;
    }
    try {
        isClicked = true;
        canvas.style.cursor = "wait";
        const point = getPoint(event);
        await decoder([...points, point[0], point[1]], [...labels, 1]);
    }
    finally {
        canvas.style.cursor = "default";
        isClicked = false;
    }
}

/**
 * handler to handle click event on canvas
 */
async function handleClick(event) {
    if (isClicked) {
        return;
    }
    try {
        isClicked = true;
        canvas.style.cursor = "wait";

        const point = getPoint(event);
        const label = 1;
        points.push(point[0]);
        points.push(point[1]);
        labels.push(label);
        await decoder(points, labels);
    }
    finally {
        canvas.style.cursor = "default";
        isClicked = false;
    }
}

/**
 * handler called when image available
 */
async function handleImage(img) {
    const encoder_latency = document.getElementById("encoder_latency");
    encoder_latency.innerText = "";
    points = [];
    labels = [];
    filein.disabled = true;
    decoder_latency.innerText = "";
    canvas.style.cursor = "wait";
    image_embeddings = undefined;

    let width = img.width;
    let height = img.height;
    if (width > height) {
        if (width > MAX_WIDTH) {
            height = height * (MAX_WIDTH / width);
            width = MAX_WIDTH;
        }
    } else {
        if (height > MAX_HEIGHT) {
            width = width * (MAX_HEIGHT / height);
            height = MAX_HEIGHT;
        }
    }
    width = Math.round(width);
    height = Math.round(height);
    canvas.width = width;
    canvas.height = height;
    
    var ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, width, height);

    imageImageData = ctx.getImageData(0, 0, width, height);

    const t = await ort.Tensor.fromImage(imageImageData, { resizedWidth: MODEL_WIDTH, resizedHeight: MODEL_HEIGHT });
    const feed = (config.isSlimSam) ? { "pixel_values": t } : { "input_image": t };
    const session = await MODELS[config.model][0].sess;

    const start = performance.now();
    image_embeddings = session.run(feed);
    image_embeddings.then(() => {
        encoder_latency.innerText = `${(performance.now() - start).toFixed(1)}ms`;
        canvas.style.cursor = "default";
    });
    filein.disabled = false;
}

/*
 * fetch and cache url
 */
async function fetchAndCache(url, name) {
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(url);
        if (cachedResponse == undefined) {
            await cache.add(url);
            cachedResponse = await cache.match(url);
            log(`${name} (network)`);
        } else {
            log(`${name} (cached)`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`${name} (network)`);
        return await fetch(url).then(response => response.arrayBuffer());
    }
}

/*
 * load models one at a time
 */
async function load_models(models) {
    const cache = await caches.open("onnx");
    let missing = 0;
    for (const [name, model] of Object.entries(models)) {
        let cachedResponse = await cache.match(model.url);
        if (cachedResponse === undefined) {
            missing += model.size;
        }
    }
    if (missing > 0) {
        log(`downloading ${missing} MB from network ... it might take a while`);
    } else {
        log("loading...");
    }
    const start = performance.now();
    for (const [name, model] of Object.entries(models)) {
        try {
            const opt = {
                executionProviders: [config.provider],
                enableMemPattern: false,
                enableCpuMemArena: false,
                extra: {
                    session: {
                        disable_prepacking: "1",
                        use_device_allocator_for_initializers: "1",
                        use_ort_model_bytes_directly: "1",
                        use_ort_model_bytes_for_initializers: "1"
                    }
                },
            };
            const model_bytes = await fetchAndCache(model.url, model.name);
            const extra_opt = model.opt || {};
            const sess_opt = { ...opt, ...extra_opt };
            model.sess = await ort.InferenceSession.create(model_bytes, sess_opt);
        } catch (e) {
            log(`${model.url} failed, ${e}`);
        }
    }
    const stop = performance.now();
    log(`ready, ${(stop - start).toFixed(1)}ms`);
}

async function main() {
    const model = MODELS[config.model];

    canvas = document.getElementById("img_canvas");
    canvas.style.cursor = "wait";

    filein = document.getElementById("file-in");
    decoder_latency = document.getElementById("decoder_latency");

    document.getElementById("clear-button").addEventListener("click", () => {
        points = [];
        labels = [];
        decoder(points, labels);
    });

    let img = document.getElementById("original-image");

    await load_models(MODELS[config.model]).then(() => {
        canvas.addEventListener("click", handleClick);
        canvas.addEventListener("mousemove", handleMouseMove);
        document.getElementById("cut-button").addEventListener("click", handleCut);

        // image upload
        filein.onchange = function (evt) {
            let target = evt.target || window.event.src, files = target.files;
            if (FileReader && files && files.length) {
                let fileReader = new FileReader();
                fileReader.onload = () => {
                    img.onload = () => handleImage(img);
                    img.src = fileReader.result;
                }
                fileReader.readAsDataURL(files[0]);
            }
        };
        handleImage(img);
    }, (e) => {
        log(e);
    });
}

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter()
        return adapter.features.has('shader-f16')
    } catch (e) {
        return false
    }
}

document.addEventListener("DOMContentLoaded", () => {
    hasFp16().then((fp16) => {
        if (fp16) {
            main();
        } else {
            log("Your GPU or Browser doesn't support webgpu/f16");
        }
    });
});
