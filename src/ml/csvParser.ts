import { ColumnStat, ColumnType, ParsedDataset, TaskType } from './types';

const MISSING_TOKENS = new Set(['', 'na', 'n/a', 'null', 'none', 'nan', '?', '-']);

export function isMissingValue(val: string | undefined | null): boolean {
  if (val === undefined || val === null) return true;
  return MISSING_TOKENS.has(val.trim().toLowerCase());
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
  // Ensure unique headers
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
    lower.endsWith('id') && lower.length <= 14 ||
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

  // Non-numeric column with 100% unique values across >= 12 rows (e.g. CUST-1001, PROP-204)
  if (!isNumeric && uniqueCount === nonMissingValues.length && nonMissingValues.length >= 12) {
    return {
      isLikelyId: true,
      reason: `100% unique categorical values (${uniqueCount}/${nonMissingValues.length} rows)`,
    };
  }

  // Monotonically increasing integer sequence 1..N
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
  const { headers, rawRows } = parseCSVText(csvText);
  const rows: Record<string, string>[] = rawRows.map((arr) => {
    const obj: Record<string, string> = {};
    headers.forEach((h, i) => {
      obj[h] = arr[i] ?? '';
    });
    return obj;
  });

  let missingCells = 0;
  let numericCount = 0;
  let categoricalCount = 0;
  let idCount = 0;

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

    // Categorical stats
    const freq = new Map<string, number>();
    for (const v of nonMissing) {
      freq.set(v, (freq.get(v) ?? 0) + 1);
    }
    const sortedCats = Array.from(freq.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([cat]) => cat);

    return {
      name: header,
      inferredType,
      missingCount: missing,
      uniqueCount: uniqueValues.length,
      mode: sortedCats[0] ?? '',
      categories: sortedCats.slice(0, 20),
      sampleValues,
      isLikelyId: idCheck.isLikelyId,
      idReason: idCheck.reason,
    };
  });

  // Choose default target: preferredTarget if present, else last non-id column
  const nonIdHeaders = columnStats
    .filter((c) => c.inferredType !== 'id')
    .map((c) => c.name);
  const defaultTarget =
    preferredTarget && headers.includes(preferredTarget)
      ? preferredTarget
      : nonIdHeaders[nonIdHeaders.length - 1] ?? headers[headers.length - 1];

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
  };
}

export function inferTaskType(dataset: ParsedDataset, targetColumn: string): TaskType {
  const stat = dataset.columnStats.find((c) => c.name === targetColumn);
  if (!stat) return 'regression';

  if (stat.inferredType === 'categorical') {
    return 'classification';
  }

  // If numeric target has very few unique values (e.g., 0/1 or 1/2/3), treat as classification
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
