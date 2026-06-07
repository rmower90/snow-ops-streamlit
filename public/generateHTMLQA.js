// generateHTMLQA.js
//
// Generates per-basin Pillow QA Investigation HTML pages.
//
// Usage:
//   node generateHTMLQA.js USCASJ [USCATM ...]
//
// For each basin, discovers the pillow list from the basin's qa_viz_2026/
// directory (filenames minus the all-pillows overview), renders the template
// at templates/qa_template.html with {{BASIN}} and {{PILLOW_OPTIONS}} filled
// in, and writes <basin>_qa.html next to this script.

const fs = require("fs");
const path = require("path");

// --------------------------
// Config
// --------------------------
const SCRIPT_DIR = __dirname;
const TEMPLATE_PATH = path.join(SCRIPT_DIR, "templates", "qa_template.html");
const OUT_DIR = SCRIPT_DIR;

// where each basin's per-pillow timeline PNGs live on hydro-c2.
// {BASIN} is substituted at lookup time.
const QA_VIZ_DIR_PATTERN =
  "/home/rossamower/work/aso/data/insitu/{BASIN}/qa/qa_viz_2026/";

// PNGs that are NOT individual pillows and so should be excluded from
// the dropdown (compared case-insensitively to the filename stem).
const NON_PILLOW_PNG_STEMS = new Set(["all_pils"]);

// --------------------------
// Helpers
// --------------------------
function qaVizDirFor(basin) {
  return QA_VIZ_DIR_PATTERN.replace("{BASIN}", basin);
}

function discoverPillows(qaVizDir) {
  if (!fs.existsSync(qaVizDir)) {
    throw new Error(
      "QA viz directory not found (run insitu_qa.py first): " + qaVizDir
    );
  }
  const stems = fs
    .readdirSync(qaVizDir)
    .filter((f) => f.toLowerCase().endsWith(".png"))
    .map((f) => f.slice(0, -4))
    .filter((stem) => !NON_PILLOW_PNG_STEMS.has(stem.toLowerCase()));
  stems.sort();
  return stems;
}

function buildOptionsBlock(pillows) {
  return pillows
    .map((p) => `        <option value="${p}">${p}</option>`)
    .join("\n");
}

function renderTemplate(template, basin, pillows) {
  const options = buildOptionsBlock(pillows);
  return template
    .replace(/{{BASIN}}/g, basin)
    .replace(/{{PILLOW_OPTIONS}}/g, options);
}

// --------------------------
// Main
// --------------------------
const args = process.argv.slice(2).filter(Boolean);
const basins = args.filter((a) => !a.startsWith("--"));

if (basins.length === 0) {
  console.error("Usage: node generateHTMLQA.js <basin> [<basin>...]");
  process.exit(1);
}

if (!fs.existsSync(TEMPLATE_PATH)) {
  console.error("Template not found: " + TEMPLATE_PATH);
  process.exit(1);
}

const template = fs.readFileSync(TEMPLATE_PATH, "utf8");

let failed = 0;
for (const basin of basins) {
  try {
    const qaVizDir = qaVizDirFor(basin);
    const pillows = discoverPillows(qaVizDir);
    if (pillows.length === 0) {
      console.error(`[${basin}] no pillow PNGs found in ${qaVizDir}; skipping`);
      failed++;
      continue;
    }
    const html = renderTemplate(template, basin, pillows);
    const outPath = path.join(OUT_DIR, `${basin}_qa.html`);
    fs.writeFileSync(outPath, html);
    console.log(`[${basin}] wrote ${outPath}  (${pillows.length} pillows)`);
  } catch (err) {
    console.error(`[${basin}] failed: ${err.message}`);
    failed++;
  }
}

process.exit(failed > 0 ? 1 : 0);
