import {
  CleaningReport,
  ColumnStat,
  ColumnType,
  DroppedColumnInfo,
  ParsedDataset,
  TaskType,
} from './types';

const MISSING_TOKENS = new Set([
  '',
  'na',
  'n/a',
  'null',
  'none',
  'nan',
  '?',
  '-',
  '#n/a',
  '#na',
]);

const MAX_ROWS_FOR_BROWSER = 2500;
const MAX_TOP_CATEGORIES = 20;

export function isMissingValue(val: string | undefined | null): boolean {
  if (val === undefined || val === null) return true;
  return MISSING_TOKENS.has(val.trim().toLowerCase());
}

function isNumericToken(val: string): boolean {
  const cleaned = val.trim().replace(/[$,%]/g, '');
  if (cleaned === '') return false;
  return Number.isFinite(Number(cleaned));
}

function isSentinelMinusOne(val: string): boolean {
  const t = val.trim();
  return t === '-1' || t === '-1.0';
}

/**
 * RFC-4180 compliant CSV parser supporting quoted fields, escaped quotes, and CRLF/LF.
 */
export function parseCSVText(text: string): { headers: string[]; rawRows: string[][] } {
  const cleanText = text.replace(/^\uFEFF/, ''); // strip UTF-8 BOM
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentField = '';
  let inQuotes = false;

  for (let i = 0; i < cleanText.length; i++) {
    const char = cleanText[i];
    const nextChar = cleanText[i + 1];

    if (inQuotes) {
      if (char === '"' && nextChar === '"') {
        currentField += '"';
        i++;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        currentField += char;
      }
    } else {
      if (char === '"') {
        inQuotes = true;
      } else if (char === ',') {
        currentRow.push(currentField.trim());
        currentField = '';
      } else if (char === '\r' && nextChar === '\n') {
        currentRow.push(currentField.trim());
        if (currentRow.some((cell) => cell !== '')) {
          rows.push(currentRow);
        }
        currentRow = [];
        currentField = '';
        i++;
      } else if (char === '\n' || char === '\r') {
        currentRow.push(currentField.trim());
        if (currentRow.some((cell) => cell !== '')) {
          rows.push(currentRow);
        }
        currentRow = [];
        currentField = '';
      } else {
        currentField += char;
      }
    }
  }

  if (currentField.length > 0 || currentRow.length > 0) {
    currentRow.push(currentField.trim());
    if (currentRow.some((cell) => cell !== '')) {
      rows.push(currentRow);
    }
  }

  if (rows.length < 2) {
    throw new Error('CSV must contain a header row and at least one data row.');
  }

  const rawHeaders = rows[0].map((h, idx) => (h.trim() ? h.trim() : `column_${idx + 1}`));
  const seenHeaders = new Map<string, number>();
  const headers = rawHeaders.map((h) => {
    const count = seenHeaders.get(h) ?? 0;
    seenHeaders.set(h, count + 1);
    return count === 0 ? h : `${h}_${count + 1}`;
  });

  const dataRows = rows.slice(1).map((r) => {
    const padded = [...r];
    while (padded.length < headers.length) padded.push('');
    return padded.slice(0, headers.length);
  });

  return { headers, rawRows: dataRows };
}

/**
 * Detect if Row 1 (index 0 of data rows) is a descriptive metadata sub-header row
 * (e.g., survey question prompts like Kaggle/Qualtrics exports) where Rows 2-20 are numeric.
 */
function detectMetadataSubHeaderRow(
  headers: string[],
  rawRows: string[][]
): { shouldStrip: boolean; sampleText?: string } {
  if (rawRows.length < 3) {
    return { shouldStrip: false };
  }

  const row1 = rawRows[0];
  const sampleWindow = rawRows.slice(1, 20); // Rows 2-20
  let numericColsInWindow = 0;
  let descriptiveMismatchCols = 0;
  let sampleText: string | undefined;

  for (let c = 0; c < headers.length; c++) {
    const windowVals = sampleWindow
      .map((r) => r[c] ?? '')
      .filter((v) => !isMissingValue(v));

    if (windowVals.length < 2) continue;

    const numericCount = windowVals.filter((v) => isNumericToken(v)).length;
    const isWindowNumeric = numericCount / windowVals.length >= 0.75;

    if (isWindowNumeric) {
      numericColsInWindow++;
      const row1Val = (row1[c] ?? '').trim();
      if (!isMissingValue(row1Val) && !isNumericToken(row1Val)) {
        descriptiveMismatchCols++;
        if (!sampleText) {
          sampleText =
            row1Val.length > 64 ? `${row1Val.slice(0, 61)}...` : row1Val;
        }
      }
    }
  }

  const shouldStrip =
    numericColsInWindow >= 1 &&
    (descriptiveMismatchCols >= 2 ||
      (numericColsInWindow <= 2 && descriptiveMismatchCols >= 1));

  return { shouldStrip, sampleText };
}

function seededSampleRows<T>(arr: T[], maxRows: number, seed = 4207): T[] {
  if (arr.length <= maxRows) return arr;
  // Reservoir-style deterministic shuffle of indices
  const indices = arr.map((_, i) => i);
  let s = seed >>> 0;
  for (let i = indices.length - 1; i > 0; i--) {
    s = (1664525 * s + 1013904223) >>> 0;
    const j = s % (i + 1);
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
  const chosen = indices.slice(0, maxRows).sort((a, b) => a - b);
  return chosen.map((i) => arr[i]);
}

function computeMedian(sortedNums: number[]): number {
  if (sortedNums.length === 0) return 0;
  const mid = Math.floor(sortedNums.length / 2);
  return sortedNums.length % 2 === 0
    ? (sortedNums[mid - 1] + sortedNums[mid]) / 2
    : sortedNums[mid];
}

function suggestStep(min: number, max: number, allIntegers: boolean): number {
  const range = Math.abs(max - min);
  if (range === 0) return 1;
  if (allIntegers && range <= 100) return 1;
  if (allIntegers && range <= 1000) return 5;
  if (allIntegers && range <= 20000) return 50;
  if (allIntegers) return Math.max(1, Math.round(range / 100));
  const rawStep = range / 100;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  return Number((Math.ceil(rawStep / magnitude) * magnitude).toPrecision(2));
}

function detectIdColumn(
  name: string,
  nonMissingValues: string[],
  isNumeric: boolean
): { isLikelyId: boolean; reason?: string } {
  if (nonMissingValues.length < 5) {
    return { isLikelyId: false };
  }
  const uniqueCount = new Set(nonMissingValues).size;
  const uniqueRatio = uniqueCount / nonMissingValues.length;
  const lower = name.toLowerCase();

  const matchesIdName =
    lower === 'id' ||
    lower.endsWith('_id') ||
    (lower.endsWith('id') && lower.length <= 14) ||
    lower === 'uuid' ||
    lower === 'guid' ||
    lower === 'row_id' ||
    lower === 'index' ||
    lower === 'record_id';

  if (matchesIdName && uniqueRatio >= 0.85) {
    return {
      isLikelyId: true,
      reason: `Identifier name ("${name}") with ${Math.round(uniqueRatio * 100)}% unique values`,
    };
  }

  if (!isNumeric && uniqueCount === nonMissingValues.length && nonMissingValues.length >= 12) {
    return {
      isLikelyId: true,
      reason: `100% unique categorical values (${uniqueCount}/${nonMissingValues.length} rows)`,
    };
  }

  if (isNumeric && uniqueCount === nonMissingValues.length && nonMissingValues.length >= 15) {
    const nums = nonMissingValues.map(Number);
    let isSequential = true;
    for (let i = 1; i < Math.min(nums.length, 25); i++) {
      if (nums[i] - nums[i - 1] !== 1) {
        isSequential = false;
        break;
      }
    }
    if (isSequential) {
      return {
        isLikelyId: true,
        reason: 'Sequential row index (monotonic +1 integer series)',
      };
    }
  }

  return { isLikelyId: false };
}

export function buildDatasetFromCSV(
  csvText: string,
  fileName: string,
  label?: string,
  preferredTarget?: string
): ParsedDataset {
  const { headers: rawHeaders, rawRows: parsedRawRows } = parseCSVText(csvText);
  const originalRowCount = parsedRawRows.length;
  const originalColumnCount = rawHeaders.length;

  // 1. Strip Metadata Sub-Header Rows (Row 1 descriptive text when Rows 2-20 are numeric)
  let workingRows = parsedRawRows;
  let strippedSubHeaderRows = 0;
  let subHeaderSampleText: string | undefined;

  const subHeaderCheck = detectMetadataSubHeaderRow(rawHeaders, workingRows);
  if (subHeaderCheck.shouldStrip) {
    workingRows = workingRows.slice(1);
    strippedSubHeaderRows = 1;
    subHeaderSampleText = subHeaderCheck.sampleText;
  }

  if (workingRows.length === 0) {
    throw new Error('CSV contains no usable data rows after stripping metadata sub-header.');
  }

  // 2. Sample up to 2,500 rows for large datasets to keep browser responsive
  const cleanedRowCount = workingRows.length;
  const wasSampled = workingRows.length > MAX_ROWS_FOR_BROWSER;
  if (wasSampled) {
    workingRows = seededSampleRows(workingRows, MAX_ROWS_FOR_BROWSER);
  }
  const sampledRowCount = workingRows.length;

  // 3. Convert Sentinel Values (-1, N/A, ?) to Nulls ("")
  let sentinelValuesConverted = 0;
  const sentinelColumnsSet = new Set<string>();

  const colCount = rawHeaders.length;
  const rowCount = workingRows.length;

  // Determine per-column whether standalone -1 should be treated as null
  const treatMinusOneAsNull = rawHeaders.map((header, colIdx) => {
    const isOtherTextCol = /_other_text$/i.test(header) || /other_text/i.test(header);
    let missingOrSentinel = 0;
    let minusOneCount = 0;
    let nonMissingNonMinusOneCount = 0;
    let hasNegativeOtherThanMinusOne = false;

    for (let r = 0; r < rowCount; r++) {
      const cell = (workingRows[r][colIdx] ?? '').trim();
      if (isMissingValue(cell)) {
        missingOrSentinel++;
      } else if (isSentinelMinusOne(cell)) {
        minusOneCount++;
      } else {
        nonMissingNonMinusOneCount++;
        if (isNumericToken(cell)) {
          const n = Number(cell.replace(/[$,%]/g, ''));
          if (n < 0) hasNegativeOtherThanMinusOne = true;
        }
      }
    }

    if (minusOneCount === 0) return false;
    if (isOtherTextCol) return true;

    const missingRate = missingOrSentinel / rowCount;
    const isSparse = missingRate > 0.2 || (missingOrSentinel + minusOneCount) / rowCount > 0.35;
    if (isSparse && !hasNegativeOtherThanMinusOne) {
      return true;
    }
    // Also if column is primarily non-numeric strings (e.g. text survey responses with -1 skipped code)
    if (nonMissingNonMinusOneCount > 0) {
      let textCount = 0;
      for (let r = 0; r < rowCount; r++) {
        const cell = (workingRows[r][colIdx] ?? '').trim();
        if (!isMissingValue(cell) && !isSentinelMinusOne(cell) && !isNumericToken(cell)) {
          textCount++;
        }
      }
      if (textCount / nonMissingNonMinusOneCount >= 0.5) {
        return true;
      }
    }
    return false;
  });

  // Apply sentinel normalization in-place on a cloned matrix
  const normalizedMatrix: string[][] = workingRows.map((row) =>
    row.map((rawCell, colIdx) => {
      const trimmed = (rawCell ?? '').trim();
      if (trimmed !== '' && MISSING_TOKENS.has(trimmed.toLowerCase())) {
        sentinelValuesConverted++;
        sentinelColumnsSet.add(rawHeaders[colIdx]);
        return '';
      }
      if (treatMinusOneAsNull[colIdx] && isSentinelMinusOne(trimmed)) {
        sentinelValuesConverted++;
        sentinelColumnsSet.add(rawHeaders[colIdx]);
        return '';
      }
      return trimmed;
    })
  );

  // 4. Prune Sparse (>80% missing) & Constant (<= 1 unique value) Columns
  const keptColIndices: number[] = [];
  const droppedColumns: DroppedColumnInfo[] = [];
  let droppedSparseCount = 0;
  let droppedConstantCount = 0;

  for (let c = 0; c < colCount; c++) {
    const colName = rawHeaders[c];
    let missing = 0;
    const uniqueNonMissing = new Set<string>();

    for (let r = 0; r < rowCount; r++) {
      const val = normalizedMatrix[r][c];
      if (val === '') {
        missing++;
      } else {
        uniqueNonMissing.add(val);
      }
    }

    const missingPct = missing / rowCount;
    if (missingPct > 0.8) {
      droppedSparseCount++;
      droppedColumns.push({
        name: colName,
        reason: 'sparse',
        detail: `${Math.round(missingPct * 100)}% missing (>80% threshold)`,
      });
      continue;
    }

    if (uniqueNonMissing.size <= 1) {
      droppedConstantCount++;
      droppedColumns.push({
        name: colName,
        reason: 'constant',
        detail:
          uniqueNonMissing.size === 0
            ? '0 non-missing values'
            : `Only 1 unique value ("${Array.from(uniqueNonMissing)[0]}")`,
      });
      continue;
    }

    keptColIndices.push(c);
  }

  if (keptColIndices.length < 2) {
    throw new Error(
      'Fewer than 2 usable columns remained after pruning sparse (>80% missing) and constant columns.'
    );
  }

  const headers = keptColIndices.map((c) => rawHeaders[c]);
  const rows: Record<string, string>[] = normalizedMatrix.map((arr) => {
    const obj: Record<string, string> = {};
    keptColIndices.forEach((origIdx, newIdx) => {
      obj[headers[newIdx]] = arr[origIdx] ?? '';
    });
    return obj;
  });

  // 5. Audit remaining columns & cap categorical features to Top 20 + "Other"
  let missingCells = 0;
  let numericCount = 0;
  let categoricalCount = 0;
  let idCount = 0;
  const cappedCategoricalColumns: string[] = [];

  const columnStats: ColumnStat[] = headers.map((header) => {
    const allVals = rows.map((r) => r[header]);
    const nonMissing: string[] = [];
    let missing = 0;

    for (const v of allVals) {
      if (isMissingValue(v)) {
        missing++;
      } else {
        nonMissing.push(v.trim());
      }
    }
    missingCells += missing;

    const numericValues: number[] = [];
    let allIntegers = true;
    for (const v of nonMissing) {
      const cleaned = v.replace(/[$,%]/g, '');
      const n = Number(cleaned);
      if (cleaned !== '' && Number.isFinite(n)) {
        numericValues.push(n);
        if (!Number.isInteger(n)) allIntegers = false;
      }
    }

    const isNumeric =
      nonMissing.length > 0 && numericValues.length / nonMissing.length >= 0.85;

    const idCheck = detectIdColumn(header, nonMissing, isNumeric);
    let inferredType: ColumnType = isNumeric ? 'numeric' : 'categorical';
    if (idCheck.isLikelyId) {
      inferredType = 'id';
      idCount++;
    } else if (inferredType === 'numeric') {
      numericCount++;
    } else {
      categoricalCount++;
    }

    const uniqueValues = Array.from(new Set(nonMissing));
    const sampleValues = nonMissing.slice(0, 4);

    if (isNumeric && numericValues.length > 0) {
      const sorted = [...numericValues].sort((a, b) => a - b);
      const min = sorted[0];
      const max = sorted[sorted.length - 1];
      const mean = numericValues.reduce((s, v) => s + v, 0) / numericValues.length;
      const median = computeMedian(sorted);
      return {
        name: header,
        inferredType,
        missingCount: missing,
        uniqueCount: uniqueValues.length,
        min,
        max,
        mean,
        median,
        step: suggestStep(min, max, allIntegers),
        sampleValues,
        isLikelyId: idCheck.isLikelyId,
        idReason: idCheck.reason,
      };
    }

    // Categorical stats — cap to Top 20 categories + "Other"
    const freq = new Map<string, number>();
    for (const v of nonMissing) {
      freq.set(v, (freq.get(v) ?? 0) + 1);
    }
    const sortedCats = Array.from(freq.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([cat]) => cat);

    const hasOverflow = !idCheck.isLikelyId && sortedCats.length > MAX_TOP_CATEGORIES;
    const topCategories = hasOverflow
      ? [...sortedCats.slice(0, MAX_TOP_CATEGORIES), 'Other']
      : sortedCats.slice(0, MAX_TOP_CATEGORIES);

    if (hasOverflow) {
      cappedCategoricalColumns.push(header);
      const topSet = new Set(sortedCats.slice(0, MAX_TOP_CATEGORIES));
      for (const r of rows) {
        const val = r[header];
        if (val !== '' && !topSet.has(val)) {
          r[header] = 'Other';
        }
      }
    }

    return {
      name: header,
      inferredType,
      missingCount: missing,
      uniqueCount: uniqueValues.length,
      mode: sortedCats[0] ?? '',
      categories: topCategories,
      hasOtherBucket: hasOverflow,
      sampleValues,
      isLikelyId: idCheck.isLikelyId,
      idReason: idCheck.reason,
    };
  });

  const nonIdHeaders = columnStats
    .filter((c) => c.inferredType !== 'id')
    .map((c) => c.name);
  const defaultTarget =
    preferredTarget && headers.includes(preferredTarget)
      ? preferredTarget
      : nonIdHeaders[nonIdHeaders.length - 1] ?? headers[headers.length - 1];

  const cleaningReport: CleaningReport = {
    originalRowCount,
    cleanedRowCount,
    sampledRowCount,
    wasSampled,
    originalColumnCount,
    cleanedColumnCount: headers.length,
    strippedSubHeaderRows,
    subHeaderSampleText,
    sentinelValuesConverted,
    sentinelColumns: Array.from(sentinelColumnsSet),
    droppedColumns,
    droppedSparseCount,
    droppedConstantCount,
    cappedCategoricalColumns,
  };

  return {
    id: `${fileName}-${Date.now()}`,
    fileName,
    label: label ?? fileName.replace(/\.csv$/i, ''),
    headers,
    rows,
    columnStats,
    totalRows: rows.length,
    totalColumns: headers.length,
    missingCells,
    numericCount,
    categoricalCount,
    idCount,
    defaultTarget,
    cleaningReport,
  };
}

export function inferTaskType(dataset: ParsedDataset, targetColumn: string): TaskType {
  const stat = dataset.columnStats.find((c) => c.name === targetColumn);
  if (!stat) return 'regression';

  if (stat.inferredType === 'categorical') {
    return 'classification';
  }

  const nonMissing = dataset.rows
    .map((r) => r[targetColumn])
    .filter((v) => !isMissingValue(v));
  const uniqueVals = new Set(nonMissing.map((v) => Number(v.replace(/[$,%]/g, ''))));
  const allIntegers = Array.from(uniqueVals).every((n) => Number.isInteger(n));

  if (allIntegers && uniqueVals.size >= 2 && uniqueVals.size <= 6 && nonMissing.length >= 20) {
    return 'classification';
  }

  return 'regression';
}
