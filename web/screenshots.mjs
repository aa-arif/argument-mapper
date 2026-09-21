/**
 * Capture README screenshots.
 *
 *   node screenshots.mjs
 *
 * Requires the API on :8000 and `vite preview` on :4173.
 *
 * Preview rather than dev: the dev server keeps an HMR websocket open, so
 * Playwright's "networkidle" never fires, and the preview build is what
 * actually ships anyway.
 *
 * Only the synthetic sample passage is ever typed in. The Argument Annotated
 * Essays licence forbids displaying the corpus, and a screenshot in a public
 * README is about as displayed as text gets -- so the gold-overlay feature is
 * exercised against a fixture, not against a real annotated essay.
 */
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const APP = process.env.APP_URL ?? "http://localhost:4173";
const OUT = "../docs/screenshots";
const VIEWPORT = { width: 1680, height: 1000 };

async function shoot(page, name) {
  await page.screenshot({ path: `${OUT}/${name}.png` });
  console.log(`  wrote ${OUT}/${name}.png`);
}

async function main() {
  await mkdir(OUT, { recursive: true });

  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: VIEWPORT,
    deviceScaleFactor: 2,
    colorScheme: "dark",
  });

  page.on("console", (message) => {
    if (message.type() === "error") console.log(`  [console] ${message.text()}`);
  });

  await page.goto(APP, { waitUntil: "domcontentloaded" });
  await page.locator(".app__header h1").waitFor({ timeout: 30_000 });
  await page.waitForTimeout(500);
  await shoot(page, "01-empty");

  // Sample passage, then a real extraction.
  await page.getByRole("button", { name: "Sample passage" }).click();
  await page.getByRole("button", { name: "Extract" }).click();

  // The graph appears when the first node renders; waiting on the node rather
  // than a fixed delay keeps this stable on a slow API call.
  await page.locator(".component-node").first().waitFor({ timeout: 120_000 });
  await page.waitForTimeout(800); // let the layout settle before capturing
  await shoot(page, "02-graph");

  // Selecting a component links the text, the graph and the inspector.
  await page.locator(".component-node").first().click();
  await page.waitForTimeout(400);
  await shoot(page, "03-inspector");

  await browser.close();
  console.log("done");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
