// generateHTMLSWE.js
//
// Generates per-basin "SWE Volume by Elevation (ASO Dates)" HTML pages.
//
// Usage:
//   node generateHTMLSWE.js USCASJ [USCATM ...]
//
// For each basin, scans the basin's swe_volume/ directory for PNGs of the form
//   <basin>_<model>_<seasonality>_<YYYY-MM-DD>.png
// extracts the unique ASO dates, and renders the template at
// templates/swe_volume_template.html with {{BASIN}} and {{ASO_DATES_JSON}}
// substituted. Writes <basin>_swe_volume.html next to this script.
//
// If no PNGs exist (no ASO flights yet), the page is still generated; the
// in-page JS renders an empty-state message.

const fs = require("fs");
const path = require("path");

// --------------------------
// Config
// --------------------------
const SCRIPT_DIR = __dirname;
const TEMPLATE_PATH = path.join(SCRIPT_DIR, "templates", "swe_volume_template.html");
const OUT_DIR = SCRIPT_DIR;

// where each basin's per-(model,seasonality,date) SWE volume PNGs live on hydro-c2.
// {BASIN} substituted at lookup time.
const SWE_VOLUME_DIR_PATTERN =
  "/home/rossamower/work/aso/data/insitu/{BASIN}/swe_volume/";

// filename pattern emitted by plotting.plot_swe_volume_by_elevation:
//   <basin>_<model>_<seasonality>_<YYYY-MM-DD>.png
const DATE_RE = /(\d{4}-\d{2}-\d{2})\.png$/;

// --------------------------
// Helpers
// --------------------------
function sweVolumeDirFor(basin) {
  return SWE_VOLUME_DIR_PATTERN.replace("{BASIN}", basin);
}

function discoverAsoDates(sweVolumeDir) {
  if (!fs.existsSync(sweVolumeDir)) {
    return []; // directory doesn't exist yet — treated as "no flights"
  }
  const dates = new Set();
  for (const f of fs.readdirSync(sweVolumeDir)) {
    const m = f.match(DATE_RE);
    if (m) dates.add(m[1]);
  }
  return [...dates].sort();
}

function renderTemplate(template, basin, asoDates) {
  return template
    .replace(/{{BASIN}}/g, basin)
    .replace(/{{ASO_DATES_JSON}}/g, JSON.stringify(asoDates));
}

// --------------------------
// Main
// --------------------------
const args = process.argv.slice(2).filter(Boolean);
const basins = args.filter((a) => !a.startsWith("--"));

if (basins.length === 0) {
  console.error("Usage: node generateHTMLSWE.js <basin> [<basin>...]");
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
    const sweDir = sweVolumeDirFor(basin);
    const asoDates = discoverAsoDates(sweDir);
    const html = renderTemplate(template, basin, asoDates);
    const outPath = path.join(OUT_DIR, `${basin}_swe_volume.html`);
    fs.writeFileSync(outPath, html);
    const dateNote = asoDates.length
      ? `${asoDates.length} ASO date(s)`
      : "no ASO dates (empty-state page)";
    console.log(`[${basin}] wrote ${outPath}  (${dateNote})`);
  } catch (err) {
    console.error(`[${basin}] failed: ${err.message}`);
    failed++;
  }
}

process.exit(failed > 0 ? 1 : 0);
