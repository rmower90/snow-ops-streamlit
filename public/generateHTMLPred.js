
/**
 * generateHTMLPreds.js (drop-in)
 *
 * Usage:
 *   node generateHTMLPreds.js USCASJ [USCATM ...]
 *   node generateHTMLPreds.js USCASJ --dropTrainInfer="drop NaNs"
 *
 * What it does:
 *  - Recursively finds prediction_acreFt_wy*_combination.csv under ROOT/<BASIN>/
 *  - Infers modelName + seasonality from path segments
 *  - Picks lexicographically latest WY per (modelName, seasonality)
 *  - Reads last 2 rows from each CSV and concatenates into one table
 *  - Drops display columns in DROP_COLS
 *  - Optionally drops rows where "Training Infer NaNs" == --dropTrainInfer value
 *  - Appends a Timeseries image: <BASIN>_timeseries_plot.png
 *  - Writes <BASIN>.html into the same directory as this script (so relative image works)
 *  - If the PNG exists elsewhere under basinRoot, it copies it next to the HTML
 */

const fs = require("fs");
const path = require("path");

// --------------------------
// Config
// --------------------------
const ROOT = "/home/rossamower/work/aso/data/mlr_prediction";
const SEASONALITIES = ["season", "accum", "melt"];
const SCRIPT_DIR = __dirname;
const OUT_DIR = SCRIPT_DIR;

// match file pattern; if you truly only ever have one, you can hardcode instead
function isPredictionCsv(filename) {
  // e.g. prediction_acreFt_wy2026_combination.csv
  return (
    filename.indexOf("prediction_acreFt_wy") === 0 &&
    filename.indexOf("_combination.csv") > -1
  );
}

// columns to drop (exact matches) from DISPLAY (still available for filtering)
const DROP_COLS = {
  "Prediction QA": true,
  "Model Type": true,
  "Training Infer NaNs": true,
};

// --------------------------
// Args
// --------------------------
let dropTrainInferValue = null;
let basins = process.argv
  .slice(2)
  .filter(Boolean)
  .filter((arg) => {
    if (arg.indexOf("--dropTrainInfer=") === 0) {
      dropTrainInferValue = arg.split("=").slice(1).join("=");
      dropTrainInferValue = dropTrainInferValue.replace(/^["']|["']$/g, "");
      return false;
    }
    return true;
  })
  .map((b) => (b || "").trim())
  .filter(Boolean);

if (basins.length === 0) {
  console.error(
    'Usage: node generateHTMLPreds.js USCASJ [USCATM ...] [--dropTrainInfer="drop NaNs"]'
  );
  process.exit(1);
}

// --------------------------
// CSV parsing (no deps)
// --------------------------
function parseCSVLine(line) {
  var out = [];
  var cur = "";
  var inQuotes = false;

  for (var i = 0; i < line.length; i++) {
    var ch = line[i];

    if (ch === '"') {
      if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function readLastTwoRows(csvPath, cb) {
  fs.readFile(csvPath, "utf8", function (err, data) {
    if (err) return cb(err);

    var lines = data.split(/\r?\n/).filter(function (l) {
      return l.trim() !== "";
    });

    if (lines.length < 3) {
      return cb(
        new Error(
          "CSV must contain a header row + at least 2 data rows: " + csvPath
        )
      );
    }

    var headerRaw = parseCSVLine(lines[0]);
    var lastTwoLines = lines.slice(-2);
    var rows = lastTwoLines.map(function (l) {
      return parseCSVLine(l);
    });

    cb(null, { headerRaw: headerRaw, rows: rows });
  });
}

// --------------------------
// Recursive discovery (no withFileTypes dependency)
// --------------------------
function walkDir(dirPath, fileCb, doneCb) {
  fs.readdir(dirPath, function (err, entries) {
    if (err) return doneCb(err);

    var pending = entries.length;
    if (!pending) return doneCb(null);

    entries.forEach(function (entry) {
      var fullPath = path.join(dirPath, entry);

      fs.stat(fullPath, function (err2, st) {
        if (err2) {
          pending--;
          if (pending === 0) doneCb(null);
          return;
        }

        if (st.isDirectory()) {
          walkDir(fullPath, fileCb, function () {
            pending--;
            if (pending === 0) doneCb(null);
          });
        } else {
          fileCb(fullPath);
          pending--;
          if (pending === 0) doneCb(null);
        }
      });
    });
  });
}

// --------------------------
// Infer modelName & seasonality from path
// Expected tree anywhere under basin root:
//   .../<MODELNAME>/<seasonality>/acreFt/prediction_acreFt_wyXXXX_combination.csv
// --------------------------
function inferModelAndSeasonality(csvPath) {
  var parts = csvPath.split(path.sep);

  for (var i = 0; i < parts.length; i++) {
    var p = parts[i];
    if (SEASONALITIES.indexOf(p) > -1) {
      var seasonality = p;
      var modelName = i > 0 ? parts[i - 1] : "UNKNOWN_MODEL";
      return { modelName: modelName, seasonality: seasonality };
    }
  }
  return null;
}

// --------------------------
// Timeseries PNG discovery
// Priority:
//  1) next to this JS file
//  2) anywhere under basinRoot
// --------------------------
function findTimeseriesPng(basin, basinRoot, cb) {
  var targetName = basin + "_timeseries_plot.png";

  var scriptDirCandidate = path.join(SCRIPT_DIR, targetName);
  fs.stat(scriptDirCandidate, function (err, st) {
    if (!err && st && st.isFile()) {
      return cb(null, scriptDirCandidate);
    }

    var found = null;
    walkDir(
      basinRoot,
      function (fp) {
        if (found) return;
        if (path.basename(fp) === targetName) found = fp;
      },
      function () {
        cb(null, found); // may be null
      }
    );
  });
}

function buildHTMLForBasin(basin, basinRoot, cb) {
  var discovered = []; // { csvPath, modelName, seasonality }

  walkDir(
    basinRoot,
    function (fp) {
      var base = path.basename(fp);
      if (!isPredictionCsv(base)) return;

      var inferred = inferModelAndSeasonality(fp);
      if (!inferred) return;

      discovered.push({
        csvPath: fp,
        modelName: inferred.modelName,
        seasonality: inferred.seasonality,
      });
    },
    function (err) {
      if (err) return cb(err);

      if (discovered.length === 0) {
        return cb(new Error("No prediction CSVs found under: " + basinRoot));
      }

      // Pick one CSV per (modelName, seasonality): choose lexicographically last filename
      var pickedMap = {};
      discovered.forEach(function (d) {
        var key = d.modelName + "||" + d.seasonality;
        if (!pickedMap[key]) {
          pickedMap[key] = d;
        } else {
          if (path.basename(d.csvPath) > path.basename(pickedMap[key].csvPath)) {
            pickedMap[key] = d;
          }
        }
      });

      var picked = Object.keys(pickedMap).map(function (k) {
        return pickedMap[k];
      });

      var pending = picked.length;
      var results = []; // {modelName, seasonality, headerRaw, rows}
      var firstHeaderRaw = null;

      picked.forEach(function (item) {
        readLastTwoRows(item.csvPath, function (err2, res) {
          if (err2) return cb(err2);

          if (!firstHeaderRaw) firstHeaderRaw = res.headerRaw;

          results.push({
            modelName: item.modelName,
            seasonality: item.seasonality,
            headerRaw: res.headerRaw,
            rows: res.rows,
          });

          pending--;
          if (pending === 0) {
            findTimeseriesPng(basin, basinRoot, function (_err3, pngAbsPath) {
              var html = renderConcatenatedTableHTML(
                basin,
                results,
                firstHeaderRaw,
                pngAbsPath,
                dropTrainInferValue
              );
              cb(null, { html: html, pngAbsPath: pngAbsPath });
            });
          }
        });
      });
    }
  );
}

function renderConcatenatedTableHTML(
  basin,
  results,
  headerRaw,
  pngAbsPathOrNull,
  dropTrainInferValue
) {
  // Keep original headers except drops (DISPLAY only)
  var keepHeaders = headerRaw.filter(function (h) {
    return !DROP_COLS[h];
  });

  // Final headers include new cols
  var finalHeaders = ["Model Name", "Seasonality"].concat(keepHeaders);

  // header -> index map
  var headerIndexMap = {};
  for (var i = 0; i < headerRaw.length; i++) headerIndexMap[headerRaw[i]] = i;

  var trainInferIdx = headerIndexMap["Training Infer NaNs"]; // may be undefined

  // Stable ordering: group by model name then seasonality (season, accum, melt)
  results.sort(function (a, b) {
    if (a.modelName < b.modelName) return -1;
    if (a.modelName > b.modelName) return 1;
    return (
      SEASONALITIES.indexOf(a.seasonality) -
      SEASONALITIES.indexOf(b.seasonality)
    );
  });

  function parseYYYYMMDD(s) {
  if (!s) return null;
  var parts = String(s).trim().split("-");
  if (parts.length !== 3) return null;
  var y = parseInt(parts[0], 10);
  var m = parseInt(parts[1], 10);
  var d = parseInt(parts[2], 10);
  if (!y || !m || !d) return null;
  return new Date(Date.UTC(y, m - 1, d)); // UTC avoids local TZ shifts
  }


  // Flatten all rows (with optional drop)
  var finalRows = [];
  results.forEach(function (block) {
    block.rows.forEach(function (row) {
      // Optionally drop rows by Training Infer NaNs
      if (dropTrainInferValue && trainInferIdx !== undefined) {
        var v = row[trainInferIdx];
        if ((v || "").trim() === dropTrainInferValue) return; // skip
      }

      var out = [];
      out.push(block.modelName);
      out.push(block.seasonality);

      for (var c = 0; c < keepHeaders.length; c++) {
        var h = keepHeaders[c];
        var idx = headerIndexMap[h];
        out.push(row[idx] !== undefined ? row[idx] : "");
      }
      finalRows.push(out);
    });
  });

  // Determine the latest prediction date appearing in the table (based on CSV "Date" column)
var dateIdx = headerIndexMap["Date"]; // this is the original CSV column index
var latestDt = null;

results.forEach(function (block) {
  block.rows.forEach(function (row) {
    // Respect the same drop filter applied above
    if (dropTrainInferValue && trainInferIdx !== undefined) {
      var v = row[trainInferIdx];
      if ((v || "").trim() === dropTrainInferValue) return;
    }

    var dt = parseYYYYMMDD(row[dateIdx]);
    if (!dt) return;
    if (!latestDt || dt.getTime() > latestDt.getTime()) latestDt = dt;
  });
});

var formattedDate = latestDt
  ? latestDt.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" })
  : new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });


  var pngRel = basin + "_timeseries_plot.png";
  var hasPng = !!pngAbsPathOrNull;

  var html = [
    "<!DOCTYPE html>",
    '<html lang="en">',
    "<head>",
    '  <meta charset="UTF-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
    "  <title>" + basin + " SWE Prediction</title>",
    "  <style>",
    "    body { font-family: Arial, sans-serif; }",
    "    table { border-collapse: collapse; margin-top: 20px; }",
    "    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }",
    "    th { background-color: #f4f4f4; }",
    "    tr:nth-child(even) { background-color: #f9f9f9; }",
    "    tr:hover { background-color: #f1f1f1; }",
    "    .layout { display: flex; flex-direction: column; align-items: center; gap: 18px; }",
    "    .timeseries { width: 100%; max-width: 1100px; text-align: center; }",
    "    .timeseries img { width: 100%; height: auto; border: 1px solid #ddd; }",
    "    .missing { color: #a00; font-style: italic; }",
    "  </style>",
    "</head>",
    "<body>",
    '  <h1 style="text-align: center;">' +
      basin +
      " Predictions on " +
      formattedDate +
      "</h1>",
    '  <div class="layout">',
    "    <table>",
    "      <thead>",
    "        <tr>" +
      finalHeaders
        .map(function (h) {
          return "<th>" + h + "</th>";
        })
        .join("") +
      "</tr>",
    "      </thead>",
    "      <tbody>",
  ];

  for (var r = 0; r < finalRows.length; r++) {
    html.push("        <tr>");
    for (var c2 = 0; c2 < finalHeaders.length; c2++) {
      var v2 = finalRows[r][c2] !== undefined ? finalRows[r][c2] : "";
      html.push("          <td>" + v2 + "</td>");
    }
    html.push("        </tr>");
  }

  html.push(
    "      </tbody>",
    "    </table>",
    '    <div class="timeseries">',
    '      <h2 style="margin: 0 0 10px 0;">Timeseries</h2>',
    hasPng
      ? '      <img src="' + pngRel + '" alt="' + basin + ' timeseries plot" />'
      : '      <div class="missing">Timeseries PNG not found: expected ' +
          pngRel +
          "</div>",
    "    </div>",
    "  </div>",
    "</body>",
    "</html>"
  );

  return html.join("\n");
}

// --------------------------
// Run for each basin -> write <BASIN>.html into script directory
// Ensure PNG is alongside HTML (copy if found elsewhere)
// --------------------------
basins.forEach(function (basin) {
  var basinRoot = path.join(ROOT, basin);

  buildHTMLForBasin(basin, basinRoot, function (err, out) {
    if (err) {
      console.error("[" + basin + "] ERROR:", err.message || err);
      return;
    }

    var html = out.html;
    var pngAbsPath = out.pngAbsPath;

    var outPath = path.join(OUT_DIR, basin + ".html");
    fs.writeFile(outPath, html, function (err2) {
      if (err2) {
        console.error("[" + basin + "] Error writing HTML:", err2);
        return;
      }

      // Ensure PNG exists next to HTML
      var pngName = basin + "_timeseries_plot.png";
      var dstPng = path.join(OUT_DIR, pngName);

      fs.stat(dstPng, function (eStat, st) {
        if (!eStat && st && st.isFile()) {
          console.log("[" + basin + "] Wrote " + outPath + " (PNG already present)");
          return;
        }

        if (pngAbsPath) {
          fs.copyFile(pngAbsPath, dstPng, function (err3) {
            if (err3) {
              console.warn(
                "[" +
                  basin +
                  "] Wrote " +
                  outPath +
                  " (PNG copy failed: " +
                  (err3.message || err3) +
                  ")"
              );
            } else {
              console.log("[" + basin + "] Wrote " + outPath + " + " + dstPng);
            }
          });
        } else {
          console.log("[" + basin + "] Wrote " + outPath + " (no PNG found)");
        }
      });
    });
  });
});

