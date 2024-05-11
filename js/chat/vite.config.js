import { defineConfig } from "vite";
import * as path from "path";
import { fileURLToPath } from "node:url";
import { viteStaticCopy } from 'vite-plugin-static-copy'

const filesNeedToExclude = ["models", "dist/transformers.js"];
const filesPathToExclude = filesNeedToExclude.map((src) => {
    return fileURLToPath(new URL(src, import.meta.url));
});

export default defineConfig({
    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: 'node_modules/onnxruntime-web/dist/*jsep*.wasm',
                    dest: path.join(__dirname, 'dist/dist')
                },
                {
                    src: 'node_modules/@xenova/transformers/dist/transformers.js',
                    dest: path.join(__dirname, 'dist/dist')
                }

            ]
        })
    ],
    build: {
        outDir: "dist",
        rollupOptions: {
            external: [
                ...filesPathToExclude
            ],
        },
    },
});