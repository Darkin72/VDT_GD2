const path = require("node:path");
const fs = require("node:fs/promises");
const { bundle } = require("@remotion/bundler");
const {
  getCompositions,
  renderStill,
  openBrowser,
} = require("@remotion/renderer");

async function main() {
  const root = path.resolve(__dirname, "..");
  const out = path.join(root, "out", "hazeclear-check");
  await fs.mkdir(out, { recursive: true });
  const serveUrl = await bundle({
    entryPoint: path.join(root, "src/index.ts"),
    publicDir: path.join(root, "public"),
  });
  const browser = await openBrowser("chrome");
  try {
    const compositions = await getCompositions(serveUrl, {
      puppeteerInstance: browser,
    });
    const composition = compositions.find((c) => c.id === "HazeClearExplainer");
    if (!composition || composition.durationInFrames !== 2220)
      throw new Error("Unexpected explainer duration");
    if (!compositions.some((c) => c.id === "MyComp"))
      throw new Error("Original composition missing");
    for (const frame of [
      60, 180, 330, 470, 600, 720, 840, 960, 1110, 1260, 1410, 1530, 1680,
      1860, 2040, 2219,
    ]) {
      await renderStill({
        serveUrl,
        composition,
        frame,
        output: path.join(out, `frame-${frame}.png`),
        imageFormat: "png",
        puppeteerInstance: browser,
      });
      console.log("Checked frame", frame);
    }
    for (const id of [
      "HazeClear-Wavelet",
      "HazeClear-Mamba",
      "HazeClear-Detail",
      "HazeClear-Optimization",
    ]) {
      await renderStill({
        serveUrl,
        composition: compositions.find((c) => c.id === id),
        frame: 90,
        output: path.join(out, `${id}.png`),
        puppeteerInstance: browser,
      });
    }
    console.log("Validated main timeline and standalone scenes");
  } finally {
    await browser.close({ silent: true });
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
