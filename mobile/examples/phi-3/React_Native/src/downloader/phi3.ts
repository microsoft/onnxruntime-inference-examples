/** assets that need to be downloaded before init phi3 model */
export type Asset = {
    prefix: string
    url: string
}
export const ASSETS: Asset[] = [
  {
    prefix: "engle/phi3/add_tokens.json",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/added_tokens.json?download=true",
  },
  {
    prefix: "engle/phi3/config.json",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/config.json?download=true",
  },
  {
    prefix: "engle/phi3/configuration_phi3.py",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/configuration_phi3.py?download=true",
  },
  {
    prefix: "engle/phi3/genai_config.json",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/genai_config.json?download=true",
  },
  {
    prefix:
      "engle/phi3/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx?download=true",
  },
  {
    prefix:
      "engle/phi3/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data?download=true",
  },
  {
    prefix: "engle/phi3/special_tokens_map.json",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/special_tokens_map.json?download=true",
  },
  {
    prefix: "engle/phi3/tokenizer.json",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer.json?download=true",
  },
  {
    prefix: "engle/phi3/tokenizer.model",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer.model?download=true",
  },
  {
    prefix: "engle/phi3/tokenizer_config.json",
    url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer_config.json?download=true",
  }
];
