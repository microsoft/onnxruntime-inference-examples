// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run whisper in onnxruntime-web.
//

const ort = require('onnxruntime-web');

function log(i) { document.getElementById('status').innerText += `\n[${performance.now().toFixed(2)}] ` + i; }

const kSampleRate = 16000;
const kIntervalAudio_ms = 1000;
const kSteps = kSampleRate * 30;
const kDelay = 100;
const kModel = "whisper_cpu_int8_0_model.onnx";

// ort session
let sess;

// audio context
var context = null;
let mediaRecorder;

// stats
let total_processing_time = 0;
let total_processing_count = 0;

// some dom shortcuts
let record;
let transcribe;
let progress;
let audio_src;

// transcribe active
function busy() {
    transcribe.disabled = true;
    progress.parentNode.style.display = "block";
    document.getElementById("outputText").value = "";
    document.getElementById('latency').innerText = "";
}

// transcribe done
function ready() {
    transcribe.disabled = false;
    progress.style.width = "0%";
    progress.parentNode.style.display = "none";
}

// called when document is loaded
document.addEventListener("DOMContentLoaded", function () {
    audio_src = document.querySelector('audio');
    record = document.getElementById('record');
    transcribe = document.getElementById('transcribe');
    progress = document.getElementById('progress');
    transcribe.disabled = true;
    progress.parentNode.style.display = "none";

    // audio_src.src = "colorado_river_crisis_is_hitting_home.mp3";

    // click on Record
    record.addEventListener("click", (e) => {
        if (e.currentTarget.innerText == "Record") {
            e.currentTarget.innerText = "Stop Recording";
            startRecord(0);
        }
        else {
            e.currentTarget.innerText = "Record";
            stopRecord();
        }
    });

    // click on Transcribe
    transcribe.addEventListener("click", () => {
        transcribe_file();
    });

    // drop file
    document.getElementById("file-upload").onchange = function (evt) {
        let target = evt.target || window.event.src, files = target.files;
        audio_src.src = URL.createObjectURL(files[0]);
    }

    log("loading model");
    try {
        sess = new Whisper(kModel, (e) => {
            if (e === undefined) {
                log(`${kModel} loaded, ${ort.env.wasm.numThreads} threads`);
                ready();
            } else {
                log(`Error: ${e}`);
            }
        });

        context = new AudioContext({
            sampleRate: kSampleRate,
            channelCount: 1,
            echoCancellation: false,
            autoGainControl: true,
            noiseSuppression: true,
        });
        if (!context) {
            throw new Error("no AudioContext, make sure domain has access to Microphone");
        }
    } catch (e) {
        log(`Error: ${e}`);
    }
});

// wrapper around onnxruntime and model
class Whisper {
    constructor(url, cb) {
        ort.env.logLevel = "error";
        this.sess = null;

        // semi constants that we initialize once and pass to every run() call
        this.min_length = Int32Array.from({ length: 1 }, () => 1);
        this.max_length = Int32Array.from({ length: 1 }, () => 448);
        this.num_return_sequences = Int32Array.from({ length: 1 }, () => 1);
        this.length_penalty = Float32Array.from({ length: 1 }, () => 1.);
        this.repetition_penalty = Float32Array.from({ length: 1 }, () => 1.);
        this.attention_mask = Int32Array.from({ length: 1 * 80 * 3000 }, () => 0);

        const opt = {
            executionProviders: ["wasm"],
            logSeverityLevel: 3,
            logVerbosityLevel: 3,
        };
        ort.InferenceSession.create(url, opt).then((s) => {
            this.sess = s;
            cb();
        }, (e) => { cb(e); })
    }

    async run(audio_pcm, beams = 1) {
        // clone semi constants into feed. The clone is needed if we run with ort.env.wasm.proxy=true
        const feed = {
            "audio_pcm": audio_pcm,
            "max_length": new ort.Tensor(new Int32Array(this.max_length), [1]),
            "min_length": new ort.Tensor(new Int32Array(this.min_length), [1]),
            "num_beams": new ort.Tensor(Int32Array.from({ length: 1 }, () => beams), [1]),
            "num_return_sequences": new ort.Tensor(new Int32Array(this.num_return_sequences), [1]),
            "length_penalty": new ort.Tensor(new Float32Array(this.length_penalty), [1]),
            "repetition_penalty": new ort.Tensor(new Float32Array(this.repetition_penalty), [1]),
            "attention_mask": new ort.Tensor(new Int32Array(this.attention_mask), [1, 80, 3000]),
        }

        return this.sess.run(feed);
    }
}

// report progress
function update_status(t) {
    total_processing_time += t;
    total_processing_count += 1;
    const avg = 1000 * 30 * total_processing_count / total_processing_time;
    document.getElementById('latency').innerText = `${avg.toFixed(1)} x realtime`;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// process audio buffer
async function process_audio(audio, starttime, idx, pos) {
    if (idx < audio.length) {
        // not done
        try {
            // update progress bar
            progress.style.width = (idx * 100 / audio.length).toFixed(1) + "%";
            progress.textContent = progress.style.width;
            await sleep(kDelay);

            // run inference for 30 sec
            const xa = audio.slice(idx, idx + kSteps);
            const start = performance.now();
            const ret = await sess.run(new ort.Tensor(xa, [1, xa.length]));
            const diff = performance.now() - start;
            update_status(diff);

            // append results to textarea 
            const textarea = document.getElementById('outputText');
            textarea.value += `${ret.str.data[0]}\n`;
            textarea.scrollTop = textarea.scrollHeight;
            await sleep(kDelay);
            process_audio(audio, starttime, idx + kSteps, pos + 30);
        } catch (e) {
            log(`Error: ${e}`);
            ready();
        }
    } else {
        // done with audio buffer
        const processing_time = ((performance.now() - starttime) / 1000);
        const total = (audio.length / kSampleRate);
        log(`${document.getElementById('latency').innerText}, total ${processing_time.toFixed(1)}sec for ${total.toFixed(1)}sec`);
        ready();
    }
}

// transcribe audio source
async function transcribe_file() {
    if (audio_src.src == "") {
        log("Error: set some Audio input");
        return;
    }

    busy();
    log("start transcribe ...");
    try {
        const buffer = await (await fetch(audio_src.src)).arrayBuffer();
        const audioBuffer = await context.decodeAudioData(buffer);
        var offlineContext = new OfflineAudioContext(audioBuffer.numberOfChannels, audioBuffer.length, audioBuffer.sampleRate);
        var source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start();
        const renderedBuffer = await offlineContext.startRendering();
        const audio = renderedBuffer.getChannelData(0);
        process_audio(audio, performance.now(), 0, 0);
    }
    catch (e) {
        log(`Error: ${e}`);
        ready();
    }
}

// start recording
async function startRecord() {
    if (mediaRecorder === undefined) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            mediaRecorder = new MediaRecorder(stream);
        } catch (e) {
            record.innerText = "Record";
            log(`Access to Microphone, ${e}`);
        }
    }
    let recording_start = performance.now();
    let chunks = [];

    mediaRecorder.ondataavailable = (e) => {
        chunks.push(e.data);
        document.getElementById('latency').innerText = `recorded: ${((performance.now() - recording_start) / 1000).toFixed(1)}sec`;
    }

    mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { 'type': 'audio/ogg; codecs=opus' });
        log(`recorded ${((performance.now() - recording_start) / 1000).toFixed(1)}sec audio`);
        audio_src.src = window.URL.createObjectURL(blob);
    };
    mediaRecorder.start(kIntervalAudio_ms);
}

// stop recording
function stopRecord() {
    if (mediaRecorder) {
        mediaRecorder.stop();
        mediaRecorder = undefined;
    }
}
