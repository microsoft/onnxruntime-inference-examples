// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run sd-turbo with webgpu in onnxruntime-web.
//

import ort from 'onnxruntime-web/webgpu';

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

/*
 * get configuration from url
*/
function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        // model: "models/onnx-sd-turbo-fp16",
        model: "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main",
        provider: "webgpu",
        device: "gpu",
        threads: "1",
        images: "2",
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
    config.images = parseInt(config.images);
    return config;
}

/*
 * initialize latents with random noise
 */
function randn_latents(shape, noise_sigma) {
    function randn() {
        // Use the Box-Muller transform
        let u = Math.random();
        let v = Math.random();
        let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return z;
    }
    let size = 1;
    shape.forEach(element => {
        size *= element;
    });

    let data = new Float32Array(size);
    // Loop over the shape dimensions
    for (let i = 0; i < size; i++) {
        data[i] = randn() * noise_sigma;
    }
    return data;
}

/*
 * fetch and cache model
 */
async function fetchAndCache(base_url, model_path) {
    const url = `${base_url}/${model_path}`;
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(url);
        if (cachedResponse == undefined) {
            await cache.add(url);
            cachedResponse = await cache.match(url);
            log(`${model_path} (network)`);
        } else {
            log(`${model_path} (cached)`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`${model_path} (network)`);
        return await fetch(url).then(response => response.arrayBuffer());
    }
}

/*
 * load models used in the pipeline
 */
async function load_models(models) {
    const cache = await caches.open("onnx");
    let missing = 0;
    for (const [name, model] of Object.entries(models)) {
        const url = `${config.model}/${model.url}`;
        let cachedResponse = await cache.match(url);
        if (cachedResponse === undefined) {
            missing += model.size;
        }
    }
    if (missing > 0) {
        log(`downloading ${missing} MB from network ... it might take a while`);
    } else {
        log("loading...");
    }
    for (const [name, model] of Object.entries(models)) {
        try {
            const start = performance.now();
            const model_bytes = await fetchAndCache(config.model, model.url);
            const sess_opt = { ...opt, ...model.opt };
            models[name].sess = await ort.InferenceSession.create(model_bytes, sess_opt);
            const stop = performance.now();
            log(`${model.url} in ${(stop - start).toFixed(1)}ms`);
        } catch (e) {
            log(`${model.url} failed, ${e}`);
        }
    }
    log("ready.");
}

const config = getConfig();

const models = {
    "unet": {
        url: "unet/model.onnx", size: 640,
        // should have 'steps: 1' but will fail to create the session
        opt: { freeDimensionOverrides: { batch_size: 1, num_channels: 4, height: 64, width: 64, sequence_length: 77, } }
    },
    "text_encoder": {
        url: "text_encoder/model.onnx", size: 1700,
        // should have 'sequence_length: 77' but produces a bad image
        opt: { freeDimensionOverrides: { batch_size: 1, } },
    },
    "vae_decoder": {
        url: "vae_decoder/model.onnx", size: 95,
        opt: { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } }
    }
}

ort.env.wasm.wasmPaths = 'dist/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let tokenizer;
let loading;
const sigma = 14.6146;
const gamma = 0;
const vae_scaling_factor = 0.18215;
const text = document.getElementById("user-input");

text.value = "Paris with the river in the background";

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

switch (config.provider) {
    case "webgpu":
        if (!("gpu" in navigator)) {
            throw new Error("webgpu is NOT supported");
        }
        opt.preferredOutputLocation = { last_hidden_state: "gpu-buffer" };
        break;
    case "webnn":
        if (!("ml" in navigator)) {
            throw new Error("webnn is NOT supported");
        }
        opt.executionProviders = [{
            name: "webnn",
            deviceType: config.device,
            powerPreference: 'default'
        }];
        break;
}

// Event listener for Ctrl + Enter or CMD + Enter
document.getElementById('user-input').addEventListener('keydown', function (e) {
    if (e.ctrlKey && e.key === 'Enter') {
        generate_image();
    }
});
document.getElementById('send-button').addEventListener('click', function (e) {
    generate_image()
});

/*
 * scale the latents
*/
function scale_model_inputs(t) {
    const d_i = t.data;
    const d_o = new Float32Array(d_i.length);

    const divi = (sigma ** 2 + 1) ** 0.5;
    for (let i = 0; i < d_i.length; i++) {
        d_o[i] = d_i[i] / divi;
    }
    return new ort.Tensor(d_o, t.dims);
}

/*
 * Poor mens EulerA step
 * Since this example is just sd-turbo, implement the absolute minimum needed to create an image
 * Maybe next step is to support all sd flavors and create a small helper model in onnx can deal
 * much more efficient with latents.
 */
function step(model_output, sample) {
    const d_o = new Float32Array(model_output.data.length);
    const prev_sample = new ort.Tensor(d_o, model_output.dims);
    const sigma_hat = sigma * (gamma + 1);

    for (let i = 0; i < model_output.data.length; i++) {
        const pred_original_sample = sample.data[i] - sigma_hat * model_output.data[i];
        const derivative = (sample.data[i] - pred_original_sample) / sigma_hat;
        const dt = 0 - sigma_hat;
        d_o[i] = (sample.data[i] + derivative * dt) / vae_scaling_factor;
    }
    return prev_sample;
}

/**
 * draw an image from tensor
 * @param {ort.Tensor} t
 * @param {number} image_nr
*/
function draw_image(t, image_nr) {
    let pix = t.data;
    for (var i = 0; i < pix.length; i++) {
        let x = pix[i];
        x = x / 2 + 0.5
        if (x < 0.) x = 0.;
        if (x > 1.) x = 1.;
        pix[i] = x;
    }
    const imageData = t.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
    const canvas = document.getElementById(`img_canvas_${image_nr}`);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);
    const div = document.getElementById(`img_div_${image_nr}`);
    div.style.opacity = 1.
}

async function generate_image() {
    try {
        document.getElementById('status').innerText = "generating ...";

        if (tokenizer === undefined) {
            tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
            tokenizer.pad_token_id = 0;
        }
        let canvases = [];
        await loading;

        for (let j = 0; j < config.images; j++) {
            const div = document.getElementById(`img_div_${j}`);
            div.style.opacity = 0.5
        }

        const { input_ids } = await tokenizer(text.value, { padding: true, max_length: 77, truncation: true, return_tensor: false });

        // text-encoder
        let start = performance.now();
        const { last_hidden_state } = await models.text_encoder.sess.run(
            { "input_ids": new ort.Tensor("int32", input_ids, [1, input_ids.length]) });

        let perf_info = [`text_encoder: ${(performance.now() - start).toFixed(1)}ms`];

        for (let j = 0; j < config.images; j++) {
            const latent_shape = [1, 4, 64, 64];
            let latent = new ort.Tensor(randn_latents(latent_shape, sigma), latent_shape);
            const latent_model_input = scale_model_inputs(latent);

            // unet
            start = performance.now();
            let feed = {
                "sample": latent_model_input,
                "timestep": new ort.Tensor("int64", [999n], [1]),
                "encoder_hidden_states": last_hidden_state,
            };
            let { out_sample } = await models.unet.sess.run(feed);
            perf_info.push(`unet: ${(performance.now() - start).toFixed(1)}ms`);

            // scheduler
            const new_latents = step(out_sample, latent)

            // vae_decoder
            start = performance.now();
            const { sample } = await models.vae_decoder.sess.run({ "latent_sample": new_latents });
            perf_info.push(`vae_decoder: ${(performance.now() - start).toFixed(1)}ms`);

            draw_image(sample, j);
            log(perf_info.join(", "))
            perf_info = [];
        }
        // this is a gpu-buffer we own, so we need to dispose it
        last_hidden_state.dispose();
        log("done");
    } catch (e) {
        log(e);
    }
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
            loading = load_models(models);
        } else {
            log("Your GPU or Browser doesn't support webgpu/f16");
        }
    });
});
