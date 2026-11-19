import { isMissingValue } from './csvParser';
import {
  FeatureEncoderSpec,
  ParsedDataset,
  PreprocessorState,
  TaskType,
} from './types';

export interface SplitPreparedData {
  preprocessor: PreprocessorState;
  XTrain: number[][];
  yTrain: number[];
  XTest: number[][];
  yTest: number[];
}

function seededShuffle<T>(arr: T[], seed: number): T[] {
  const copy = [...arr];
  let s = seed >>> 0;
  for (let i = copy.length - 1; i > 0; i--) {
    s = (1664525 * s + 1013904223) >>> 0;
    const j = s % (i + 1);
    const tmp = copy[i];
    copy[i] = copy[j];
    copy[j] = tmp;
  }
  return copy;
}

function parseNumericClean(val: string | undefined): number | null {
  if (isMissingValue(val)) return null;
  const cleaned = val!.trim().replace(/[$,%]/g, '');
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function prepareTrainingData(
  dataset: ParsedDataset,
  targetColumn: string,
  taskType: TaskType,
  droppedIdColumns: string[]
): SplitPreparedData {
  // Filter rows where target is valid
  const validRows = dataset.rows.filter((row) => {
    const tVal = row[targetColumn];
    if (isMissingValue(tVal)) return false;
    if (taskType === 'regression') {
      return parseNumericClean(tVal) !== null;
    }
    return true;
  });

  if (validRows.length < 10) {
    throw new Error(
      `Need at least 10 valid rows with non-missing "${targetColumn}" values to train.`
    );
  }

  const droppedSet = new Set([...droppedIdColumns, targetColumn]);
  const featureStats = dataset.columnStats.filter((c) => !droppedSet.has(c.name));

  if (featureStats.length === 0) {
    throw new Error('No input feature columns remain after excluding target and ID columns.');
  }

  // Build 80/20 split (stratified for classification)
  const indices = validRows.map((_, i) => i);
  let trainIndices: number[] = [];
  let testIndices: number[] = [];

  let classLabels: string[] | undefined;
  const rawTargets = validRows.map((r) => r[targetColumn].trim());

  if (taskType === 'classification') {
    const uniqueLabels = Array.from(new Set(rawTargets)).sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true })
    );
    if (uniqueLabels.length < 2) {
      throw new Error(`Target "${targetColumn}" must have at least 2 distinct classes.`);
    }
    classLabels = uniqueLabels;

    const byClass = new Map<string, number[]>();
    for (const label of uniqueLabels) byClass.set(label, []);
    indices.forEach((idx) => {
      byClass.get(rawTargets[idx])!.push(idx);
    });

    for (const label of uniqueLabels) {
      const shuffled = seededShuffle(byClass.get(label)!, 42 + label.length);
      const splitPoint = Math.max(1, Math.floor(shuffled.length * 0.8));
      trainIndices.push(...shuffled.slice(0, splitPoint));
      testIndices.push(...shuffled.slice(splitPoint));
    }

    if (testIndices.length === 0) {
      // Fallback if dataset is tiny
      const splitPoint = Math.max(1, Math.floor(indices.length * 0.8));
      const allShuffled = seededShuffle(indices, 42);
      trainIndices = allShuffled.slice(0, splitPoint);
      testIndices = allShuffled.slice(splitPoint);
    } else {
      trainIndices = seededShuffle(trainIndices, 101);
      testIndices = seededShuffle(testIndices, 202);
    }
  } else {
    const shuffled = seededShuffle(indices, 42);
    const splitPoint = Math.max(2, Math.floor(shuffled.length * 0.8));
    trainIndices = shuffled.slice(0, splitPoint);
    testIndices = shuffled.slice(splitPoint);
  }

  const trainRows = trainIndices.map((i) => validRows[i]);

  // Fit preprocessing specs on trainRows only
  const specs: FeatureEncoderSpec[] = [];
  const encodedColumnToRaw: string[] = [];
  let encodedOffset = 0;

  for (const stat of featureStats) {
    if (stat.inferredType === 'numeric') {
      const nums: number[] = [];
      for (const r of trainRows) {
        const parsed = parseNumericClean(r[stat.name]);
        if (parsed !== null) nums.push(parsed);
      }
      const median = nums.length > 0 ? medianOf(nums) : stat.median ?? 0;
      const imputed = trainRows.map((r) => parseNumericClean(r[stat.name]) ?? median);
      const mean = imputed.reduce((a, b) => a + b, 0) / Math.max(1, imputed.length);
      const variance =
        imputed.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, imputed.length);
      const std = Math.sqrt(variance) > 1e-8 ? Math.sqrt(variance) : 1;

      specs.push({
        rawName: stat.name,
        type: 'numeric',
        median,
        mean,
        std,
        min: stat.min ?? median,
        max: stat.max ?? median,
        step: stat.step ?? 1,
        encodedStartIndex: encodedOffset,
        encodedLength: 1,
      });
      encodedColumnToRaw.push(stat.name);
      encodedOffset += 1;
    } else {
      // Categorical
      const freq = new Map<string, number>();
      for (const r of trainRows) {
        const raw = r[stat.name];
        if (!isMissingValue(raw)) {
          const val = raw.trim();
          freq.set(val, (freq.get(val) ?? 0) + 1);
        }
      }
      const sortedCats = Array.from(freq.entries())
        .sort((a, b) => b[1] - a[1])
        .map(([k]) => k);
      const mode = sortedCats[0] ?? stat.mode ?? 'Unknown';
      // Keep up to 12 categories to keep browser vector compact
      const keptCategories = sortedCats.slice(0, 12);
      const encLen = Math.max(1, keptCategories.length);

      specs.push({
        rawName: stat.name,
        type: 'categorical',
        mode,
        categories: keptCategories.length > 0 ? keptCategories : [mode],
        encodedStartIndex: encodedOffset,
        encodedLength: encLen,
      });
      for (let k = 0; k < encLen; k++) {
        encodedColumnToRaw.push(stat.name);
      }
      encodedOffset += encLen;
    }
  }

  let targetMin: number | undefined;
  let targetMax: number | undefined;
  let targetMean: number | undefined;

  const encodeTarget = (row: Record<string, string>): number => {
    const raw = row[targetColumn].trim();
    if (taskType === 'regression') {
      return parseNumericClean(raw) ?? 0;
    }
    return classLabels!.indexOf(raw);
  };

  const yTrain = trainIndices.map((i) => encodeTarget(validRows[i]));
  const yTest = testIndices.map((i) => encodeTarget(validRows[i]));

  if (taskType === 'regression') {
    const allY = [...yTrain, ...yTest];
    targetMin = Math.min(...allY);
    targetMax = Math.max(...allY);
    targetMean = allY.reduce((a, b) => a + b, 0) / allY.length;
  }

  const preprocessor: PreprocessorState = {
    taskType,
    targetColumn,
    droppedIdColumns,
    featureNames: featureStats.map((s) => s.name),
    encodedFeatureCount: encodedOffset,
    specs,
    encodedColumnToRaw,
    classLabels,
    targetMin,
    targetMax,
    targetMean,
    trainCount: trainIndices.length,
    testCount: testIndices.length,
  };

  const XTrain = trainIndices.map((i) => transformSingleRow(validRows[i], preprocessor));
  const XTest = testIndices.map((i) => transformSingleRow(validRows[i], preprocessor));

  return {
    preprocessor,
    XTrain,
    yTrain,
    XTest,
    yTest,
  };
}

export function transformSingleRow(
  row: Record<string, string | number>,
  preprocessor: PreprocessorState
): number[] {
  const vec = new Array<number>(preprocessor.encodedFeatureCount).fill(0);

  for (const spec of preprocessor.specs) {
    const rawVal = row[spec.rawName];

    if (spec.type === 'numeric') {
      let num: number | null = null;
      if (typeof rawVal === 'number') {
        num = Number.isFinite(rawVal) ? rawVal : null;
      } else {
        num = parseNumericClean(rawVal);
      }
      const imputed = num ?? spec.median ?? 0;
      const mean = spec.mean ?? 0;
      const std = spec.std ?? 1;
      vec[spec.encodedStartIndex] = (imputed - mean) / std;
    } else {
      const strVal =
        rawVal === undefined || rawVal === null || isMissingValue(String(rawVal))
          ? spec.mode ?? ''
          : String(rawVal).trim();
      const cats = spec.categories ?? [];
      const idx = cats.indexOf(strVal);
      if (idx >= 0) {
        vec[spec.encodedStartIndex + idx] = 1;
      }
    }
  }

  return vec;
}
