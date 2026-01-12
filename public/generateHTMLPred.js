const fs = require("fs");
const path = require("path");

// --------------------------
// Config
// --------------------------
const ROOT = "/home/rossamower/work/aso/data/mlr_prediction";
const SEASONALITIES = ["season", "accum", "melt"];

// match file pattern; if you truly only ever have one, you can hardcode instead
function isPredictionCsv(filename) {
  // e.g. prediction_acreFt_wy2026_combination.csv
  return (
    filename.indexOf("prediction_acreFt_wy") === 0 &&
    filename.indexOf("_combination.csv") > -1
  );
}

// columns to drop (exact matches)
const DROP_COLS = {
  "Prediction QA": true,
  "Model Type": true,
};

// --------------------------
// Args
// --------------------------
const basins = process.argv.slice(2).map((b) => (b || "").trim()).filter(Boolean);
if (basins.length === 0) {
  console.error("Usage: node generateHTMLPred.js USCASJ [USCATM ...]");
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
      return cb(new Error("CSV must contain a header row + at least 2 data rows: " + csvPath));
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
// Find prediction CSVs + infer modelName & seasonality from path
// Expected tree anywhere under basin root:
//   .../<MODELNAME>/<seasonality>/acreFt/prediction_acreFt_wyXXXX_combination.csv
// --------------------------
function inferModelAndSeasonality(csvPath) {
  // Normalize split
  var parts = csvPath.split(path.sep);

  // Find seasonality segment
  for (var i = 0; i < parts.length; i++) {
    var p = parts[i];
    if (SEASONALITIES.indexOf(p) > -1) {
      var seasonality = p;
      // model name is the directory right before seasonality
      var modelName = (i > 0) ? parts[i - 1] : "UNKNOWN_MODEL";
      return { modelName: modelName, seasonality: seasonality };
    }
  }
  return null;
}

function buildHTMLForBasin(basin, basinRoot, cb) {
  // Collect discovered csvs
  var discovered = []; // { csvPath, modelName, seasonality }

  walkDir(
    basinRoot,
    function (fp) {
      var base = path.basename(fp);
      if (!isPredictionCsv(base)) return;

      var inferred = inferModelAndSeasonality(fp);
      if (!inferred) return;

      // also ensure it is under ".../<seasonality>/acreFt/<file>"
      // (optional guard; keep it light)
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

      // For each (modelName, seasonality) we only want ONE csv:
      // if multiple years exist, pick lexicographically last filename (usually highest WY)
      var pickedMap = {}; // key -> {csvPath, modelName, seasonality}
      discovered.forEach(function (d) {
        var key = d.modelName + "||" + d.seasonality;
        if (!pickedMap[key]) {
          pickedMap[key] = d;
        } else {
          // choose later filename as "most recent"
          if (path.basename(d.csvPath) > path.basename(pickedMap[key].csvPath)) {
            pickedMap[key] = d;
          }
        }
      });

      var picked = Object.keys(pickedMap).map(function (k) { return pickedMap[k]; });

      // Read last two rows for each picked csv
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
            // build a single concatenated table
            var html = renderConcatenatedTableHTML(basin, results, firstHeaderRaw);
            cb(null, html);
          }
        });
      });
    }
  );
}

function renderConcatenatedTableHTML(basin, results, headerRaw) {
  // Keep original headers except drops
  var keepHeaders = headerRaw.filter(function (h) {
    return !DROP_COLS[h];
  });

  // Final headers include new cols
  var finalHeaders = ["Model Name", "Seasonality"].concat(keepHeaders);

  // header -> index map
  var headerIndexMap = {};
  for (var i = 0; i < headerRaw.length; i++) headerIndexMap[headerRaw[i]] = i;

  // Stable ordering: group by model name then seasonality (season, accum, melt)
  results.sort(function (a, b) {
    if (a.modelName < b.modelName) return -1;
    if (a.modelName > b.modelName) return 1;
    return SEASONALITIES.indexOf(a.seasonality) - SEASONALITIES.indexOf(b.seasonality);
  });

  // Flatten all rows
  var finalRows = [];
  results.forEach(function (block) {
    block.rows.forEach(function (row) {
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

  // Title date = today
  var today = new Date();
  var formattedDate = today.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  // Table-only HTML
  var html = [
    "<!DOCTYPE html>",
    '<html lang="en">',
    "<head>",
    '  <meta charset="UTF-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
    "  <title>" + basin + " SWE Prediction</title>",
    "  <style>",
    "    table { border-collapse: collapse; margin-top: 20px; }",
    "    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }",
    "    th { background-color: #f4f4f4; }",
    "    tr:nth-child(even) { background-color: #f9f9f9; }",
    "    tr:hover { background-color: #f1f1f1; }",
    "    .layout { display: flex; justify-content: center; align-items: flex-start; gap: 10px; }",
    "  </style>",
    "</head>",
    "<body>",
    '  <h1 style="text-align: center;">' + basin + " Predictions on " + formattedDate + "</h1>",
    '  <div class="layout">',
    "    <table>",
    "      <thead>",
    "        <tr>" +
      finalHeaders.map(function (h) { return "<th>" + h + "</th>"; }).join("") +
      "</tr>",
    "      </thead>",
    "      <tbody>",
  ];

  for (var r = 0; r < finalRows.length; r++) {
    html.push("        <tr>");
    for (var c2 = 0; c2 < finalHeaders.length; c2++) {
      var v = finalRows[r][c2] !== undefined ? finalRows[r][c2] : "";
      html.push("          <td>" + v + "</td>");
    }
    html.push("        </tr>");
  }

  html.push(
    "      </tbody>",
    "    </table>",
    "  </div>",
    "</body>",
    "</html>"
  );

  return html.join("\n");
}

// --------------------------
// Run for each basin -> write <BASIN>.html
// --------------------------
basins.forEach(function (basin) {
  var basinRoot = path.join(ROOT, basin);

  buildHTMLForBasin(basin, basinRoot, function (err, html) {
    if (err) {
      console.error("[" + basin + "] ERROR:", err.message || err);
      return;
    }

    var outPath = basin + ".html";
    fs.writeFile(outPath, html, function (err2) {
      if (err2) console.error("[" + basin + "] Error writing HTML:", err2);
      else console.log("[" + basin + "] Wrote " + outPath);
    });
  });
});
