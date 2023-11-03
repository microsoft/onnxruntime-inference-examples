// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run segment-anything with webgpu in onnxruntime-web.
//

const ort = require('onnxruntime-web/webgpu');

const MAX_WIDTH = 500;
const MAX_HEIGHT = 500;
const MODEL_WIDTH = 1024;
const MODEL_HEIGHT = 1024;

const MODEL_MAP = {
    sam_b: ["models/sam_vit_b_01ec64.encoder.onnx", "models/sam_vit_b_01ec64.decoder.onnx"],
};

const config = getConfig();

ort.env.wasm.numThreads = config.threads;
ort.env.wasm.proxy = true;

let canvas;
let filein;
let decoder_latency;

var image_embeddings;
var sess = [];
var points = [];
var labels = [];
var imageImageData;
var isClicked = false;

function log(i) {
    document.getElementById('status').innerText += `\n[${performance.now().toFixed(3)}] ` + i;
}

/**
 * get some parameters from url
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
    return config;
}

/**
 * handler to handle click on the image canvas
 *  with ctl: add point
 *  with shift: forground label
 */
async function handleClick(event) {
    if (isClicked) {
        return;
    }
    try {
        isClicked = true;
        canvas.style.cursor = "wait";

        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const label = (event.shiftKey) ? 0 : 1;

        if (image_embeddings === undefined) {
            await sess[0];
        }
        const emb = await image_embeddings;

        if (!event.ctrlKey) {
            points = [];
            labels = [];
        }
        points.push(x, y);
        labels.push(label);

        let ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.width = imageImageData.width;
        canvas.height = imageImageData.height;
        ctx.putImageData(imageImageData, 0, 0);
        ctx.fillStyle = 'blue';
        ctx.fillRect(x, y, 10, 10);

        const pointCoords = new ort.Tensor(new Float32Array(points), [1, points.length / 2, 2]);
        const pointLabels = new ort.Tensor(new Float32Array(labels), [1, labels.length]);
        const maskInput = new ort.Tensor(new Float32Array(256 * 256), [1, 1, 256, 256]);
        const hasMask = new ort.Tensor(new Float32Array([0]), [1,]);
        const origianlImageSize = new ort.Tensor(new Float32Array([MODEL_HEIGHT, MODEL_WIDTH]), [2,]);

        const s = await sess[1];
        const t = new ort.Tensor(emb.image_embeddings.type, Float32Array.from(emb.image_embeddings.data), emb.image_embeddings.dims);
        const feed = {
            "image_embeddings": t,
            "point_coords": pointCoords,
            "point_labels": pointLabels,
            "mask_input": maskInput,
            "has_mask_input": hasMask,
            "orig_im_size": origianlImageSize
        }
        const start = performance.now();
        const res = await s.run(feed);
        decoder_latency.innerText = `${(performance.now() - start).toFixed(1)}ms`;
        const mask = res.masks;
        const maskImageData = mask.toImageData();
        ctx.globalAlpha = 0.3;
        ctx.drawImage(await createImageBitmap(maskImageData), 0, 0);
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
    filein.disabled = true;
    decoder_latency.innerText = "";
    canvas.style.cursor = "wait";
    image_embeddings = undefined;
    var width = img.width;
    var height = img.height;

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

    // eslint-disable-next-line no-undef
    const t = await ort.Tensor.fromImage(imageImageData, options = { resizedWidth: MODEL_WIDTH, resizedHeight: MODEL_HEIGHT });
    const feed = { "input_image": t };
    const s = await sess[0];

    const start = performance.now();
    image_embeddings = s.run(feed);
    image_embeddings.then(() => {
        encoder_latency.innerText = `${(performance.now() - start).toFixed(1)}ms`;
        canvas.style.cursor = "default";
    });
    filein.disabled = false;
}


/**
 * fetch and cache url
 */
async function fetchAndCache(url) {
    try {
        const cache = await caches.open("onnx");
        if (config.clear_cache) {
            cache.delete(url);
        }
        let cachedResponse = await cache.match(url);
        if (cachedResponse == undefined) {
            await cache.add(url);
            cachedResponse = await cache.match(url);
            log(`${url} (from network)`);
        } else {
            log(`${url} (from cache)`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`${url} (from network)`);
        return await fetch(url).then(response => response.arrayBuffer());
    }
}

/*
 * load encoder and decoder sequentially
 */
async function load_model(model, idx, img) {
    let provider = config.provider;

    switch (provider) {
        case "webnn":
            if (!("ml" in navigator)) {
                throw new Error("webnn is NOT supported");
            }
            provider = {
                name: "webnn",
                deviceType: config.device,
                powerPreference: 'default'
            };
            break;
        case "webgpu":
            if (!navigator.gpu) {
                throw new Error("webgpu is NOT supported");
            }
            break;
    }

    const opt = { executionProviders: [provider] };

    fetchAndCache(model[idx]).then((data) => {
        sess[idx] = ort.InferenceSession.create(data, opt);
        sess[idx].then(() => {
            log(`${model[idx]} loaded.`);
            if (idx == 0) {
                load_model(model, 1);
            }
        }, (e) => {
            log(`${model[idx]} failed with ${e}.`);
            throw e;
        });
        if (img !== undefined) {
            handleImage(img);
        }
    })
}

async function main() {
    const model = MODEL_MAP[config.model];

    canvas = document.getElementById("img_canvas");
    canvas.addEventListener("click", handleClick);
    canvas.style.cursor = "wait";

    filein = document.getElementById("file-in");
    decoder_latency = document.getElementById("decoder_latency");

    let img = document.getElementById("original-image");

    load_model(model, 0, img).then(() => {}, (e) => {
        log(e);
    });

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
}

document.addEventListener("DOMContentLoaded", () => { main(); });
