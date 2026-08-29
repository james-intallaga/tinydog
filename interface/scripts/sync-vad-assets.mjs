import { copyFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const destination = path.join(root, "public", "vad");
await mkdir(destination, { recursive: true });

const assets = [
  ["node_modules/@ricky0123/vad-web/dist/silero_vad_v5.onnx", "silero_vad_v5.onnx"],
  ["node_modules/@ricky0123/vad-web/dist/vad.worklet.bundle.min.js", "vad.worklet.bundle.min.js"],
  ["node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.mjs"],
  ["node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.wasm"],
];

for (const [source, filename] of assets) {
  await copyFile(path.join(root, source), path.join(destination, filename));
}

await writeFile(
  path.join(destination, "NOTICE.txt"),
  [
    "Silero VAD model: https://github.com/snakers4/silero-vad (MIT)",
    "Browser integration: https://github.com/ricky0123/vad (ISC)",
    "ONNX Runtime: https://github.com/microsoft/onnxruntime (MIT)",
    "Complete permission notices: LICENSES.txt",
    "These files are a replaceable microphone gate and are not MLM parameters.",
    "",
  ].join("\n"),
  "utf8",
);
