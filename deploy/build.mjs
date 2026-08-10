import { mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
const REPO = "Renato1212/Quantitative";
const REF = "b4673d900aca2ee631947925db20cefad2d168bf";
const ARTEFACTS = {
  "index.html": "a4ac6e3d8bc546b727e3cdee34e477a3a1c1c87c5805fb3f9b740566d40305e1",
  "app.js": "e26e1cb60ffa53377f1d996b1d73c0a7ddcd1ca3711444092bf6b3b7d02a431f",
  "app.css": "762f80e71115d4ac064649313d46c0fdf8c8deb2f7658816adde9b5dd14055dc",
};
await mkdir("dist", { recursive: true });
for (const [name, expected] of Object.entries(ARTEFACTS)) {
  const url = `https://raw.githubusercontent.com/${REPO}/${REF}/app/${name}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${name}: ${response.status} from ${url}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== expected) throw new Error(`${name}: sha256 ${actual} does not match the pinned ${expected}`);
  await writeFile(`dist/${name}`, bytes);
  console.log(`${name.padEnd(11)} ${String(bytes.length).padStart(7)} bytes  sha256 ok`);
}
console.log(`\nbayline: 3 artefacts verified at ${REF.slice(0, 12)}`);
