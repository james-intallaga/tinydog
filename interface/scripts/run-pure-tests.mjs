import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";

const project = process.cwd();
const output = await mkdtemp(join(tmpdir(), "mlm-browser-tests-"));
const tsc = join(project, "node_modules", "typescript", "bin", "tsc");

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: project, stdio: "inherit" });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} exited with ${code ?? signal ?? "unknown status"}`));
    });
  });
}

try {
  await run(process.execPath, [
    tsc,
    "--module", "commonjs",
    "--moduleResolution", "node",
    "--target", "ES2020",
    "--lib", "ES2020,DOM,DOM.Iterable",
    "--types", "node",
    "--esModuleInterop",
    "--skipLibCheck",
    "--outDir", output,
    "src/lib/realtime-webrtc.ts",
    "src/lib/realtime-webrtc.test.ts",
    "src/lib/realtime-gateway.ts",
    "src/lib/realtime-gateway.test.ts",
    "src/lib/realtime-audio-telemetry.ts",
    "src/lib/realtime-audio-telemetry.test.ts",
    "src/lib/frame-difference.ts",
    "src/lib/frame-difference.test.ts",
    "src/lib/companion-gaze.ts",
    "src/lib/companion-gaze.test.ts",
    "src/lib/pets.ts",
    "src/lib/pets.test.ts",
    "src/lib/local-request.ts",
    "src/lib/local-request.test.ts",
  ]);
  await run(process.execPath, [
    "--test",
    join(output, "realtime-webrtc.test.js"),
    join(output, "realtime-gateway.test.js"),
    join(output, "realtime-audio-telemetry.test.js"),
    join(output, "frame-difference.test.js"),
    join(output, "companion-gaze.test.js"),
    join(output, "pets.test.js"),
    join(output, "local-request.test.js"),
  ]);
} finally {
  await rm(output, { recursive: true, force: true });
}
