import { SplitPreparedData } from './preprocessor';
import {
  ConfusionMatrixData,
  FeatureImportanceItem,
  ModelAlgorithmKey,
  ModelMetrics,
  PredictionOutput,
  PreprocessorState,
  RegressionPoint,
  SerializedModelWeights,
  SerializedTreeNode,
  TrainedModelResult,
} from './types';

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function createRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function sigmoid(z: number): number {
  if (z > 30) return 1 - 1e-13;
  if (z < -30) return 1e-13;
  return 1 / (1 + Math.exp(-z));
}

function softmax(logits: number[]): number[] {
  const maxL = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxL));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Solve linear system A * x = b using Gaussian elimination with partial pivoting.
 */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > maxVal) {
        maxVal = Math.abs(M[r][col]);
        pivotRow = r;
      }
    }
    if (pivotRow !== col) {
      const tmp = M[col];
      M[col] = M[pivotRow];
      M[pivotRow] = tmp;
    }

    const pivot = Math.abs(M[col][col]) < 1e-10 ? 1e-10 : M[col][col];
    for (let j = col; j <= n; j++) {
      M[col][j] /= pivot;
    }

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (factor === 0) continue;
      for (let j = col; j <= n; j++) {
        M[r][j] -= factor * M[col][j];
      }
    }
  }

  return M.map((row) => row[n]);
}

function aggregateImportances(
  encodedImportances: number[],
  preprocessor: PreprocessorState
): FeatureImportanceItem[] {
  const byRaw = new Map<string, number>();
  for (const name of preprocessor.featureNames) {
    byRaw.set(name, 0);
  }
  encodedImportances.forEach((val, idx) => {
    const rawName = preprocessor.encodedColumnToRaw[idx];
    if (rawName) {
      byRaw.set(rawName, (byRaw.get(rawName) ?? 0) + Math.abs(val));
    }
  });

  const total = Array.from(byRaw.values()).reduce((a, b) => a + b, 0);
  const items = Array.from(byRaw.entries()).map(([feature, rawVal]) => ({
    feature,
    importance: total > 1e-12 ? Number(((rawVal / total) * 100).toFixed(1)) : 0,
  }));

  items.sort((a, b) => b.importance - a.importance);
  return items;
}

// ============================================================================
// 1. Ridge Regression & Logistic Regression
// ============================================================================

function fitRidgeRegression(X: number[][], y: number[], alpha = 1.5) {
  const n = X.length;
  const p = X[0].length;
  // Augmented with bias at index 0
  const dim = p + 1;
  const XtX: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  const Xty: number[] = new Array(dim).fill(0);

  for (let i = 0; i < n; i++) {
    const xi = X[i];
    const yi = y[i];
    Xty[0] += yi;
    XtX[0][0] += 1;
    for (let j = 0; j < p; j++) {
      const xij = xi[j];
      XtX[0][j + 1] += xij;
      XtX[j + 1][0] += xij;
      Xty[j + 1] += xij * yi;
      for (let k = j; k < p; k++) {
        XtX[j + 1][k + 1] += xij * xi[k];
      }
    }
  }

  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) {
      XtX[j + 1][k + 1] = XtX[k + 1][j + 1];
    }
    XtX[j + 1][j + 1] += alpha; // L2 penalty on weights, unpenalized bias
  }

  const sol = solveLinearSystem(XtX, Xty);
  const bias = sol[0];
  const weights = sol.slice(1);
  return { bias, weights };
}

async function fitLogisticRegression(
  X: number[][],
  y: number[],
  numClasses: number,
  l2 = 0.01,
  epochs = 260
) {
  const n = X.length;
  const p = X[0].length;

  if (numClasses === 2) {
    const weights = new Array(p).fill(0);
    let bias = 0;
    const lr = 0.25;

    for (let ep = 0; ep < epochs; ep++) {
      if (ep > 0 && ep % 50 === 0) {
        await yieldToBrowser();
      }
      const gradW = new Array(p).fill(0);
      let gradB = 0;

      for (let i = 0; i < n; i++) {
        let z = bias;
        const xi = X[i];
        for (let j = 0; j < p; j++) z += weights[j] * xi[j];
        const pred = sigmoid(z);
        const err = pred - y[i];
        gradB += err;
        for (let j = 0; j < p; j++) gradW[j] += err * xi[j];
      }

      const step = lr / (1 + ep * 0.006);
      bias -= step * (gradB / n);
      for (let j = 0; j < p; j++) {
        weights[j] -= step * (gradW[j] / n + l2 * weights[j]);
      }
    }

    return {
      kind: 'linear' as const,
      weights,
      bias,
    };
  }

  // Multiclass softmax logistic regression
  const multiWeights: number[][] = Array.from({ length: numClasses }, () =>
    new Array(p).fill(0)
  );
  const multiBias = new Array(numClasses).fill(0);
  const lr = 0.25;

  for (let ep = 0; ep < epochs; ep++) {
    if (ep > 0 && ep % 40 === 0) {
      await yieldToBrowser();
    }
    const gradW: number[][] = Array.from({ length: numClasses }, () =>
      new Array(p).fill(0)
    );
    const gradB = new Array(numClasses).fill(0);

    for (let i = 0; i < n; i++) {
      const xi = X[i];
      const logits = new Array(numClasses);
      for (let c = 0; c < numClasses; c++) {
        let z = multiBias[c];
        for (let j = 0; j < p; j++) z += multiWeights[c][j] * xi[j];
        logits[c] = z;
      }
      const probs = softmax(logits);
      for (let c = 0; c < numClasses; c++) {
        const target = y[i] === c ? 1 : 0;
        const err = probs[c] - target;
        gradB[c] += err;
        for (let j = 0; j < p; j++) gradW[c][j] += err * xi[j];
      }
    }

    const step = lr / (1 + ep * 0.006);
    for (let c = 0; c < numClasses; c++) {
      multiBias[c] -= step * (gradB[c] / n);
      for (let j = 0; j < p; j++) {
        multiWeights[c][j] -= step * (gradW[c][j] / n + l2 * multiWeights[c][j]);
      }
    }
  }

  return {
    kind: 'linear' as const,
    weights: [],
    bias: 0,
    multiWeights,
    multiBias,
  };
}

// ============================================================================
// 2. CART Decision Tree (Regression & Classification)
// ============================================================================

interface TreeBuildOptions {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  maxFeatures?: number;
  rng?: () => number;
}

function buildCartTree(
  X: number[][],
  y: number[],
  indices: number[],
  taskType: 'regression' | 'classification',
  numClasses: number,
  options: TreeBuildOptions,
  featureGainAccum: number[],
  depth = 0
): SerializedTreeNode {
  const n = indices.length;

  const makeLeaf = (): SerializedTreeNode => {
    if (taskType === 'regression') {
      let sum = 0;
      for (const idx of indices) sum += y[idx];
      return { isLeaf: true, value: sum / Math.max(1, n) };
    }
    const counts = new Array(numClasses).fill(0);
    for (const idx of indices) counts[y[idx]]++;
    const probs = counts.map((c) => c / Math.max(1, n));
    return { isLeaf: true, probs };
  };

  if (
    depth >= options.maxDepth ||
    n < options.minSamplesSplit ||
    n < options.minSamplesLeaf * 2
  ) {
    return makeLeaf();
  }

  // Check purity
  if (taskType === 'classification') {
    const first = y[indices[0]];
    if (indices.every((idx) => y[idx] === first)) {
      return makeLeaf();
    }
  } else {
    const first = y[indices[0]];
    if (indices.every((idx) => Math.abs(y[idx] - first) < 1e-9)) {
      return makeLeaf();
    }
  }

  const totalFeatures = X[0].length;
  let candidateFeatures: number[];
  if (
    options.maxFeatures &&
    options.maxFeatures < totalFeatures &&
    options.rng
  ) {
    const all = Array.from({ length: totalFeatures }, (_, i) => i);
    for (let i = all.length - 1; i > 0; i--) {
      const j = Math.floor(options.rng() * (i + 1));
      const tmp = all[i];
      all[i] = all[j];
      all[j] = tmp;
    }
    candidateFeatures = all.slice(0, options.maxFeatures);
  } else {
    candidateFeatures = Array.from({ length: totalFeatures }, (_, i) => i);
  }

  // Precompute parent stats
  let parentImpurity = 0;
  let totalSum = 0;
  const totalCounts = new Array(numClasses).fill(0);

  if (taskType === 'regression') {
    for (const idx of indices) totalSum += y[idx];
    const mean = totalSum / n;
    let sse = 0;
    for (const idx of indices) sse += (y[idx] - mean) ** 2;
    parentImpurity = sse / n;
  } else {
    for (const idx of indices) totalCounts[y[idx]]++;
    let gini = 1;
    for (let c = 0; c < numClasses; c++) {
      const p = totalCounts[c] / n;
      gini -= p * p;
    }
    parentImpurity = gini;
  }

  let bestGain = 1e-9;
  let bestFeature = -1;
  let bestThreshold = 0;

  for (const featIdx of candidateFeatures) {
    // Sort indices by feature value
    const sorted = [...indices].sort((a, b) => X[a][featIdx] - X[b][featIdx]);

    if (taskType === 'regression') {
      let leftSum = 0;
      let leftSqSum = 0;
      let totalSqSum = 0;
      for (const idx of sorted) totalSqSum += y[idx] * y[idx];

      for (let pos = 0; pos < n - 1; pos++) {
        const idx = sorted[pos];
        const yv = y[idx];
        leftSum += yv;
        leftSqSum += yv * yv;

        const nextIdx = sorted[pos + 1];
        if (Math.abs(X[idx][featIdx] - X[nextIdx][featIdx]) < 1e-9) continue;

        const leftCount = pos + 1;
        const rightCount = n - leftCount;
        if (leftCount < options.minSamplesLeaf || rightCount < options.minSamplesLeaf) {
          continue;
        }

        const rightSum = totalSum - leftSum;
        const rightSqSum = totalSqSum - leftSqSum;

        const leftSSE = leftSqSum - (leftSum * leftSum) / leftCount;
        const rightSSE = rightSqSum - (rightSum * rightSum) / rightCount;
        const weightedImpurity = (leftSSE + rightSSE) / n;
        const gain = parentImpurity - weightedImpurity;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = featIdx;
          bestThreshold = (X[idx][featIdx] + X[nextIdx][featIdx]) / 2;
        }
      }
    } else {
      const leftCounts = new Array(numClasses).fill(0);
      for (let pos = 0; pos < n - 1; pos++) {
        const idx = sorted[pos];
        leftCounts[y[idx]]++;

        const nextIdx = sorted[pos + 1];
        if (Math.abs(X[idx][featIdx] - X[nextIdx][featIdx]) < 1e-9) continue;

        const leftCount = pos + 1;
        const rightCount = n - leftCount;
        if (leftCount < options.minSamplesLeaf || rightCount < options.minSamplesLeaf) {
          continue;
        }

        let leftGini = 1;
        let rightGini = 1;
        for (let c = 0; c < numClasses; c++) {
          const pl = leftCounts[c] / leftCount;
          const pr = (totalCounts[c] - leftCounts[c]) / rightCount;
          leftGini -= pl * pl;
          rightGini -= pr * pr;
        }
        const weightedGini = (leftCount * leftGini + rightCount * rightGini) / n;
        const gain = parentImpurity - weightedGini;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = featIdx;
          bestThreshold = (X[idx][featIdx] + X[nextIdx][featIdx]) / 2;
        }
      }
    }
  }

  if (bestFeature === -1) {
    return makeLeaf();
  }

  featureGainAccum[bestFeature] += bestGain * n;

  const leftIndices: number[] = [];
  const rightIndices: number[] = [];
  for (const idx of indices) {
    if (X[idx][bestFeature] <= bestThreshold) {
      leftIndices.push(idx);
    } else {
      rightIndices.push(idx);
    }
  }

  if (leftIndices.length === 0 || rightIndices.length === 0) {
    return makeLeaf();
  }

  return {
    isLeaf: false,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildCartTree(
      X,
      y,
      leftIndices,
      taskType,
      numClasses,
      options,
      featureGainAccum,
      depth + 1
    ),
    right: buildCartTree(
      X,
      y,
      rightIndices,
      taskType,
      numClasses,
      options,
      featureGainAccum,
      depth + 1
    ),
  };
}

function predictTreeNode(node: SerializedTreeNode, x: number[]): SerializedTreeNode {
  let curr = node;
  while (!curr.isLeaf) {
    const val = x[curr.featureIndex!];
    curr = val <= curr.threshold! ? curr.left! : curr.right!;
  }
  return curr;
}

// ============================================================================
// 3. Random Forest (20 Trees)
// ============================================================================

async function fitRandomForest(
  X: number[][],
  y: number[],
  taskType: 'regression' | 'classification',
  numClasses: number,
  nTrees = 20
) {
  const n = X.length;
  const p = X[0].length;
  const maxFeatures = Math.max(1, Math.round(Math.sqrt(p)));
  const rng = createRng(7331);
  const trees: SerializedTreeNode[] = [];
  const gainAccum = new Array(p).fill(0);

  for (let t = 0; t < nTrees; t++) {
    if (t > 0 && t % 3 === 0) {
      await yieldToBrowser();
    }
    const sampleIndices: number[] = [];
    for (let i = 0; i < n; i++) {
      sampleIndices.push(Math.floor(rng() * n));
    }
    const tree = buildCartTree(
      X,
      y,
      sampleIndices,
      taskType,
      numClasses,
      {
        maxDepth: 7,
        minSamplesSplit: 4,
        minSamplesLeaf: 2,
        maxFeatures,
        rng,
      },
      gainAccum
    );
    trees.push(tree);
  }

  return { trees, gainAccum };
}

// ============================================================================
// 4. Gradient Boosting (Regression, Binary & Multiclass)
// ============================================================================

async function fitGradientBoosting(
  X: number[][],
  y: number[],
  taskType: 'regression' | 'classification',
  numClasses: number
) {
  const n = X.length;
  const p = X[0].length;
  const allIndices = Array.from({ length: n }, (_, i) => i);
  const gainAccum = new Array(p).fill(0);

  if (taskType === 'regression') {
    const learningRate = 0.1;
    const nEstimators = 36;
    const baseScore = y.reduce((a, b) => a + b, 0) / n;
    const preds = new Array(n).fill(baseScore);
    const trees: SerializedTreeNode[] = [];

    for (let m = 0; m < nEstimators; m++) {
      if (m > 0 && m % 5 === 0) {
        await yieldToBrowser();
      }
      const residuals = y.map((yi, i) => yi - preds[i]);
      const tree = buildCartTree(
        X,
        residuals,
        allIndices,
        'regression',
        1,
        { maxDepth: 3, minSamplesSplit: 5, minSamplesLeaf: 2 },
        gainAccum
      );
      trees.push(tree);
      for (let i = 0; i < n; i++) {
        preds[i] += learningRate * (predictTreeNode(tree, X[i]).value ?? 0);
      }
    }

    return {
      weights: {
        kind: 'gbdt' as const,
        learningRate,
        baseScore,
        trees,
      },
      gainAccum,
    };
  }

  if (numClasses === 2) {
    const learningRate = 0.12;
    const nEstimators = 35;
    const posRate = Math.min(
      0.99,
      Math.max(0.01, y.reduce((a, b) => a + b, 0) / n)
    );
    const baseScore = Math.log(posRate / (1 - posRate));
    const rawF = new Array(n).fill(baseScore);
    const trees: SerializedTreeNode[] = [];

    for (let m = 0; m < nEstimators; m++) {
      if (m > 0 && m % 5 === 0) {
        await yieldToBrowser();
      }
      const pseudoResiduals = y.map((yi, i) => yi - sigmoid(rawF[i]));
      const tree = buildCartTree(
        X,
        pseudoResiduals,
        allIndices,
        'regression',
        1,
        { maxDepth: 3, minSamplesSplit: 5, minSamplesLeaf: 2 },
        gainAccum
      );
      trees.push(tree);
      for (let i = 0; i < n; i++) {
        // Newton-step approximation scaled for stability
        rawF[i] += learningRate * 2.2 * (predictTreeNode(tree, X[i]).value ?? 0);
      }
    }

    return {
      weights: {
        kind: 'gbdt' as const,
        learningRate: learningRate * 2.2,
        baseScore,
        trees,
      },
      gainAccum,
    };
  }

  // Multiclass GBDT
  const learningRate = 0.15;
  const nEstimators = 24;
  const multiBaseScores = new Array(numClasses).fill(0);
  const rawF: number[][] = Array.from({ length: n }, () =>
    new Array(numClasses).fill(0)
  );
  const multiTrees: SerializedTreeNode[][] = [];

  for (let m = 0; m < nEstimators; m++) {
    if (m > 0 && m % 3 === 0) {
      await yieldToBrowser();
    }
    const roundTrees: SerializedTreeNode[] = [];
    const probsMatrix = rawF.map((row) => softmax(row));

    for (let c = 0; c < numClasses; c++) {
      const residuals = y.map((yi, i) => (yi === c ? 1 : 0) - probsMatrix[i][c]);
      const tree = buildCartTree(
        X,
        residuals,
        allIndices,
        'regression',
        1,
        { maxDepth: 3, minSamplesSplit: 5, minSamplesLeaf: 2 },
        gainAccum
      );
      roundTrees.push(tree);
      for (let i = 0; i < n; i++) {
        rawF[i][c] += learningRate * 1.8 * (predictTreeNode(tree, X[i]).value ?? 0);
      }
    }
    multiTrees.push(roundTrees);
  }

  return {
    weights: {
      kind: 'gbdt' as const,
      learningRate: learningRate * 1.8,
      baseScore: 0,
      multiBaseScores,
      multiTrees,
    },
    gainAccum,
  };
}

// ============================================================================
// Inference & Evaluation Helpers
// ============================================================================

export function predictEncodedVector(
  weights: SerializedModelWeights,
  x: number[],
  taskType: 'regression' | 'classification',
  numClasses: number
): { value: number; probs?: number[] } {
  if (weights.kind === 'linear') {
    if (taskType === 'regression') {
      let sum = weights.bias;
      for (let j = 0; j < x.length; j++) sum += weights.weights[j] * x[j];
      return { value: sum };
    }
    if (weights.multiWeights && weights.multiBias) {
      const logits = weights.multiWeights.map((wRow, c) => {
        let z = weights.multiBias![c];
        for (let j = 0; j < x.length; j++) z += wRow[j] * x[j];
        return z;
      });
      const probs = softmax(logits);
      const bestClass = probs.indexOf(Math.max(...probs));
      return { value: bestClass, probs };
    }
    let z = weights.bias;
    for (let j = 0; j < x.length; j++) z += weights.weights[j] * x[j];
    const p1 = sigmoid(z);
    const probs = [1 - p1, p1];
    return { value: p1 >= 0.5 ? 1 : 0, probs };
  }

  if (weights.kind === 'tree') {
    const leaf = predictTreeNode(weights.tree, x);
    if (taskType === 'regression') {
      return { value: leaf.value ?? 0 };
    }
    const probs = leaf.probs ?? new Array(numClasses).fill(1 / numClasses);
    return { value: probs.indexOf(Math.max(...probs)), probs };
  }

  if (weights.kind === 'forest') {
    if (taskType === 'regression') {
      let sum = 0;
      for (const tree of weights.trees) {
        sum += predictTreeNode(tree, x).value ?? 0;
      }
      return { value: sum / weights.trees.length };
    }
    const avgProbs = new Array(numClasses).fill(0);
    for (const tree of weights.trees) {
      const leafProbs = predictTreeNode(tree, x).probs ?? [];
      for (let c = 0; c < numClasses; c++) {
        avgProbs[c] += (leafProbs[c] ?? 0) / weights.trees.length;
      }
    }
    return { value: avgProbs.indexOf(Math.max(...avgProbs)), probs: avgProbs };
  }

  // GBDT
  if (taskType === 'regression') {
    let score = weights.baseScore;
    for (const tree of weights.trees ?? []) {
      score += weights.learningRate * (predictTreeNode(tree, x).value ?? 0);
    }
    return { value: score };
  }

  if (weights.multiTrees && weights.multiBaseScores) {
    const logits = [...weights.multiBaseScores];
    for (const round of weights.multiTrees) {
      for (let c = 0; c < numClasses; c++) {
        logits[c] += weights.learningRate * (predictTreeNode(round[c], x).value ?? 0);
      }
    }
    const probs = softmax(logits);
    return { value: probs.indexOf(Math.max(...probs)), probs };
  }

  let logOdds = weights.baseScore;
  for (const tree of weights.trees ?? []) {
    logOdds += weights.learningRate * (predictTreeNode(tree, x).value ?? 0);
  }
  const p1 = sigmoid(logOdds);
  const probs = [1 - p1, p1];
  return { value: p1 >= 0.5 ? 1 : 0, probs };
}

export function predictSingleRow(
  weights: SerializedModelWeights,
  x: number[],
  preprocessor: PreprocessorState
): PredictionOutput {
  const numClasses = preprocessor.classLabels?.length ?? 2;
  const out = predictEncodedVector(weights, x, preprocessor.taskType, numClasses);

  if (preprocessor.taskType === 'regression') {
    const val = out.value;
    const absMax = Math.max(
      Math.abs(preprocessor.targetMin ?? 0),
      Math.abs(preprocessor.targetMax ?? 0)
    );
    const formatted =
      absMax >= 1000
        ? val.toLocaleString(undefined, { maximumFractionDigits: 0 })
        : val.toLocaleString(undefined, { maximumFractionDigits: 2 });

    return {
      taskType: 'regression',
      rawPrediction: val,
      formattedPrediction: formatted,
    };
  }

  const labels = preprocessor.classLabels ?? ['0', '1'];
  const probs = out.probs ?? labels.map((_, i) => (i === out.value ? 1 : 0));
  const bestIdx = out.value;
  const bestLabel = labels[bestIdx] ?? String(bestIdx);
  const confidence = probs[bestIdx] ?? 1;

  return {
    taskType: 'classification',
    rawPrediction: bestLabel,
    formattedPrediction: bestLabel,
    confidence,
    classProbabilities: labels.map((label, idx) => ({
      label,
      probability: probs[idx] ?? 0,
    })),
  };
}

function evaluateRegression(yTrue: number[], yPred: number[]): ModelMetrics {
  const n = yTrue.length;
  const meanTrue = yTrue.reduce((a, b) => a + b, 0) / n;
  let ssRes = 0;
  let ssTot = 0;
  let absErr = 0;

  for (let i = 0; i < n; i++) {
    const err = yTrue[i] - yPred[i];
    ssRes += err * err;
    ssTot += (yTrue[i] - meanTrue) ** 2;
    absErr += Math.abs(err);
  }

  const r2 = ssTot > 1e-12 ? 1 - ssRes / ssTot : 0;
  const rmse = Math.sqrt(ssRes / n);
  const mae = absErr / n;

  return {
    r2: Number(r2.toFixed(4)),
    rmse: Number(rmse.toFixed(2)),
    mae: Number(mae.toFixed(2)),
  };
}

function evaluateClassification(
  yTrue: number[],
  yPred: number[],
  labels: string[]
): { metrics: ModelMetrics; confusionMatrix: ConfusionMatrixData } {
  const k = labels.length;
  const matrix: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  let correct = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const a = yTrue[i];
    const p = yPred[i];
    if (a >= 0 && a < k && p >= 0 && p < k) {
      matrix[a][p]++;
    }
    if (a === p) correct++;
  }

  const accuracy = correct / Math.max(1, yTrue.length);

  // Macro Precision, Recall, F1 across classes
  let sumPrec = 0;
  let sumRec = 0;
  let sumF1 = 0;

  for (let c = 0; c < k; c++) {
    const tp = matrix[c][c];
    let fp = 0;
    let fn = 0;
    for (let r = 0; r < k; r++) {
      if (r !== c) {
        fp += matrix[r][c];
        fn += matrix[c][r];
      }
    }
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
    sumPrec += prec;
    sumRec += rec;
    sumF1 += f1;
  }

  return {
    metrics: {
      accuracy: Number(accuracy.toFixed(4)),
      precision: Number((sumPrec / k).toFixed(4)),
      recall: Number((sumRec / k).toFixed(4)),
      f1: Number((sumF1 / k).toFixed(4)),
    },
    confusionMatrix: {
      labels,
      matrix,
    },
  };
}

export async function trainSingleAlgorithm(
  algoKey: ModelAlgorithmKey,
  data: SplitPreparedData
): Promise<TrainedModelResult> {
  const { preprocessor, XTrain, yTrain, XTest, yTest } = data;
  const taskType = preprocessor.taskType;
  const numClasses = preprocessor.classLabels?.length ?? 2;
  const t0 = performance.now();

  let name = '';
  let sklearnName = '';
  let family = '';
  let weights: SerializedModelWeights;
  let encodedImportances: number[] = new Array(preprocessor.encodedFeatureCount).fill(0);

  if (algoKey === 'ridge') {
    name = 'Ridge Regression';
    sklearnName = 'Ridge';
    family = 'Linear (L2)';
    const res = fitRidgeRegression(XTrain, yTrain, 1.5);
    weights = { kind: 'linear', weights: res.weights, bias: res.bias };
    encodedImportances = res.weights.map((w) => Math.abs(w));
  } else if (algoKey === 'logistic') {
    name = 'Logistic Regression';
    sklearnName = 'LogisticRegression';
    family = 'Linear (L2)';
    const res = await fitLogisticRegression(XTrain, yTrain, numClasses);
    weights = res;
    if (res.multiWeights) {
      encodedImportances = new Array(preprocessor.encodedFeatureCount).fill(0);
      for (const wRow of res.multiWeights) {
        wRow.forEach((w, idx) => {
          encodedImportances[idx] += Math.abs(w);
        });
      }
    } else {
      encodedImportances = res.weights.map((w) => Math.abs(w));
    }
  } else if (algoKey === 'decision_tree') {
    name = 'Decision Tree';
    sklearnName =
      taskType === 'regression' ? 'DecisionTreeRegressor' : 'DecisionTreeClassifier';
    family = 'CART Tree';
    const gainAccum = new Array(preprocessor.encodedFeatureCount).fill(0);
    const allIdx = Array.from({ length: XTrain.length }, (_, i) => i);
    await yieldToBrowser();
    const tree = buildCartTree(
      XTrain,
      yTrain,
      allIdx,
      taskType,
      numClasses,
      { maxDepth: 6, minSamplesSplit: 5, minSamplesLeaf: 2 },
      gainAccum
    );
    weights = { kind: 'tree', tree };
    encodedImportances = gainAccum;
  } else if (algoKey === 'random_forest') {
    name = 'Random Forest';
    sklearnName =
      taskType === 'regression' ? 'RandomForestRegressor' : 'RandomForestClassifier';
    family = 'Bagged Ensemble (20 Trees)';
    const res = await fitRandomForest(XTrain, yTrain, taskType, numClasses, 20);
    weights = { kind: 'forest', trees: res.trees };
    encodedImportances = res.gainAccum;
  } else {
    name = 'Gradient Boosting';
    sklearnName =
      taskType === 'regression'
        ? 'GradientBoostingRegressor'
        : 'GradientBoostingClassifier';
    family = 'Boosted Ensemble (35 Stages)';
    const res = await fitGradientBoosting(XTrain, yTrain, taskType, numClasses);
    weights = res.weights;
    encodedImportances = res.gainAccum;
  }

  const trainingTimeMs = Math.max(1, Math.round(performance.now() - t0));

  // Evaluate on holdout 20% test set
  const yPred = XTest.map(
    (x) => predictEncodedVector(weights, x, taskType, numClasses).value
  );

  const featureImportances = aggregateImportances(encodedImportances, preprocessor);

  if (taskType === 'regression') {
    const metrics = evaluateRegression(yTest, yPred);
    const regressionPoints: RegressionPoint[] = yTest.map((actual, i) => ({
      actual,
      predicted: yPred[i],
      rowIndex: i + 1,
    }));

    return {
      key: algoKey,
      name,
      sklearnName,
      family,
      trainingTimeMs,
      primaryScore: metrics.r2 ?? 0,
      metrics,
      featureImportances,
      regressionPoints,
      weights,
      isWinner: false,
    };
  }

  const { metrics, confusionMatrix } = evaluateClassification(
    yTest,
    yPred,
    preprocessor.classLabels!
  );

  return {
    key: algoKey,
    name,
    sklearnName,
    family,
    trainingTimeMs,
    primaryScore: metrics.accuracy ?? 0,
    metrics,
    featureImportances,
    confusionMatrix,
    weights,
    isWinner: false,
  };
}
