import { env, AutoTokenizer } from '@xenova/transformers';
import { LLM } from './llm.js';
import { marked } from 'marked';


const MODELS = {
  "phi3": { name: "phi3", path: "microsoft/Phi-3-mini-4k-instruct-onnx-web", externaldata: true },
  "phi3dev": { name: "phi3dev", path: "schmuell/Phi-3-mini-4k-instruct-onnx-web", externaldata: true },
}

const preCannedQueries = {
  "1": "Tell me about the lighthouse of Alexandria.",
  "2": "Did the lighthouse of Alexandria existed at the same time the library of Alexandria existed?",
  "3": "How did the Pharos lighthouse impact ancient maritime trade?",
  "4": "Tell me about Constantinople.",
};

const clipboardIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard" viewBox="0 0 16 16">
<path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
<path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
</svg>`

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

marked.use({ mangle: false, headerIds: false });

const sendButton = document.getElementById('send-button');
const scrollWrapper = document.getElementById('scroll-wrapper');

//
// auto scroll the content area until a user scrolls up
//
let isAutoScrollOn = true;
let lastKnownScrollPosition = 0;
let ticking = false;

const autoScroller = new ResizeObserver(() => {
  if (isAutoScrollOn) {
    scrollWrapper.scrollIntoView({ behavior: "smooth", block: "end" });
  }
});

document.addEventListener("scroll", () => {
  if (!ticking && isAutoScrollOn && window.scrollY < lastKnownScrollPosition) {
    window.requestAnimationFrame(() => {
      isAutoScrollOn = false;
      ticking = false;
    });
    ticking = true;
  }
  else if (!ticking && !isAutoScrollOn && window.scrollY > lastKnownScrollPosition &&
    window.scrollY >= document.documentElement.scrollHeight - window.innerHeight - 30) {
    window.requestAnimationFrame(() => {
      isAutoScrollOn = true;
      ticking = false;
    });
    ticking = true;
  }
  lastKnownScrollPosition = window.scrollY;
});


//
// make response available for copying to clipboard
//
function copyTextToClipboard(responseDiv) {
  let elem = responseDiv;
  const copyButton = document.createElement('button');
  copyButton.className = 'btn btn-secondary copy-button';
  copyButton.innerHTML = clipboardIcon;
  elem = copyButton;
  elem.onclick = () => {
    navigator.clipboard.writeText(responseDiv.innerText);
  };
  responseDiv.appendChild(elem);
}

// 
// user hits send, enter or ctl enter
//
async function submitRequest(e) {
  if (sendButton.innerHTML == "Stop") {
    llm.abort();
    return;
  }

  // enter clears the chat history, ctl enter will continue the conversation
  const continuation = e.ctrlKey && e.key === 'Enter';

  document.getElementById('chat-container').style.display = 'block';

  let input = document.getElementById('user-input').value;
  if (input.length == 0) {
    document.getElementById('chat-history').context = "";
    let chatHistory = document.getElementById('chat-history');
    while (chatHistory.firstChild) {
      chatHistory.firstChild.remove();
    }
    return;
  }
  let context = document.getElementById('chat-history').context;
  if (context === undefined) {
    context = "";
  }

  // append to chat history
  let chatHistory = document.getElementById('chat-history');
  let userMessageDiv = document.createElement('div');
  userMessageDiv.className = 'mb-2 user-message';
  userMessageDiv.innerText = input;
  chatHistory.appendChild(userMessageDiv);

  // container for llm response
  let responseDiv = document.createElement('div');
  responseDiv.className = 'response-message mb-2 text-start';
  responseDiv.style.minHeight = '3em';
  let spinner = document.createElement('div');
  spinner.className = 'spinner-border text-light';
  spinner.setAttribute('role', 'status');
  responseDiv.appendChild(spinner);
  chatHistory.appendChild(responseDiv);

  // toggle button to stop text generation
  sendButton.innerHTML = "Stop";

  // change autoScroller to keep track of our new responseDiv
  autoScroller.observe(responseDiv);

  if (continuation) {
    input = context + " " + input;
  }

  Query(continuation, input, (word) => {
    responseDiv.innerHTML = marked.parse(word);
  }).then(() => {
    chatHistory.context = responseDiv.innerHTML;
    copyTextToClipboard(responseDiv, true);
    sendButton.innerHTML = "Send";
    spinner.remove();
  }).catch(error => {
    console.error(error);
    sendButton.innerHTML = "Send";
    spinner.remove();
  });

  // Clear user input
  document.getElementById('user-input').value = '';
}


// 
// event listener for Ctrl+Enter or Enter
//
document.getElementById('user-input').addEventListener('keydown', function (e) {
  if (e.ctrlKey) {
    if (e.key === 'Enter') {
      submitRequest(e);
    } else {
      const query = preCannedQueries[e.key];
      if (query) {
        document.getElementById('user-input').value = query;
        submitRequest(e);
      }
    }
  } else if (e.key === 'Enter') {
    e.preventDefault();
    submitRequest(e);
  }
});

function getConfig() {
  const query = window.location.search.substring(1);
  var config = {
    model: "phi3",
    provider: "webgpu",
    profiler: 0,
    verbose: 0,
    threads: 1,
    show_special: 0,
    csv: 0,
    max_tokens: 9999,
    local: 0,
  }
  let vars = query.split("&");
  for (var i = 0; i < vars.length; i++) {
    let pair = vars[i].split("=");
    if (pair[0] in config) {
      const key = pair[0];
      const value = decodeURIComponent(pair[1]);
      if (typeof config[key] == "number") {
        config[key] = parseInt(value);
      }
      else {
        config[key] = value;
      }
    } else if (pair[0].length > 0) {
      throw new Error("unknown argument: " + pair[0]);
    }
  }
  if (MODELS[config.model] !== undefined) {
    config.model = MODELS[config.model];
  }
  return config;
}

const config = getConfig();

// setup for transformers.js tokenizer
env.localModelPath = 'models';
env.allowRemoteModels = config.local == 0;
env.allowLocalModels = config.local == 1;

let tokenizer;

const llm = new LLM();

function token_to_text(tokenizer, tokens, startidx) {
  const txt = tokenizer.decode(tokens.slice(startidx), { skip_special_tokens: config.show_special != 1, });
  return txt;
}

async function Query(continuation, query, cb) {
  let prompt = (continuation) ? query : `<|system|>\nYou are a friendly assistant.<|end|>\n<|user|>\n${query}<|end|>\n<|assistant|>\n`;

  const { input_ids } = await tokenizer(prompt, { return_tensor: false, padding: true, truncation: true });

  // clear caches 
  // TODO: use kv_cache for continuation
  llm.initilize_feed();

  const start_timer = performance.now();
  const output_index = llm.output_tokens.length + input_ids.length;
  const output_tokens = await llm.generate(input_ids, (output_tokens) => {
    if (output_tokens.length == input_ids.length + 1) {
      // time to first token
      const took = (performance.now() - start_timer) / 1000;
      console.log(`time to first token in ${took.toFixed(1)}sec, ${input_ids.length} tokens`);
    }
    cb(token_to_text(tokenizer, output_tokens, output_index));
  }, { max_tokens: config.max_tokens });

  const took = (performance.now() - start_timer) / 1000;
  cb(token_to_text(tokenizer, output_tokens, output_index));
  const seqlen = output_tokens.length - output_index;
  console.log(`${seqlen} tokens in ${took.toFixed(1)}sec, ${(seqlen / took).toFixed(2)} tokens/sec`);
}

//
// Load the model and tokenizer
//
async function Init(hasFP16) {
  try {
    tokenizer = await AutoTokenizer.from_pretrained(config.model.path);

    log("Loading model...");
    await llm.load(config.model, {
      provider: config.provider,
      profiler: config.profiler,
      verbose: config.verbose,
      local: config.local,
      max_tokens: config.max_tokens,
      hasFP16: hasFP16,
    });
    log("Ready.");
  } catch (error) {
    log(error);
  }
}

//
// Check if we have webgpu and fp16
//
async function hasWebGPU() {
  // returns 0 for webgpu with f16, 1 for webgpu without f16, 2 for no webgpu
  if (!("gpu" in navigator)) {
    return 2;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (adapter.features.has('shader-f16')) {
      return 0;
    }
    return 1;
  } catch (e) {
    return 2;
  }
}

window.onload = () => {
  hasWebGPU().then((supported) => {
    if (supported < 2) {
      if (supported == 1) {
        log("Your GPU or Browser does not support webgpu with fp16, using fp32 instead.");
      }
      Init(supported === 0).then(() => {
        // adjustPadding();
        sendButton.addEventListener('click', submitRequest);
        const userInput = document.getElementById('user-input');
        document.getElementById("status").style.display = "none";
        userInput.focus();
      });
    } else {
      log("Your GPU or Browser does not support webgpu");
    }
  });
}
