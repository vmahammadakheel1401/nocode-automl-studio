export type ColumnType = 'numeric' | 'categorical' | 'id';
export type TaskType = 'regression' | 'classification';

export interface ColumnStat {
  name: string;
  inferredType: ColumnType;
  missingCount: number;
  uniqueCount: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  step?: number;
  mode?: string;
  categories?: string[];
  hasOtherBucket?: boolean;
  sampleValues: string[];
  isLikelyId: boolean;
  idReason?: string;
}

export interface DroppedColumnInfo {
  name: string;
  reason: 'sparse' | 'constant';
  detail: string;
}

export interface CleaningReport {
  originalRowCount: number;
  cleanedRowCount: number;
  sampledRowCount: number;
  wasSampled: boolean;
  originalColumnCount: number;
  cleanedColumnCount: number;
  strippedSubHeaderRows: number;
  subHeaderSampleText?: string;
  sentinelValuesConverted: number;
  sentinelColumns: string[];
  droppedColumns: DroppedColumnInfo[];
  droppedSparseCount: number;
  droppedConstantCount: number;
  cappedCategoricalColumns: string[];
}

export interface ParsedDataset {
  id: string;
  fileName: string;
  label: string;
  headers: string[];
  rows: Record<string, string>[];
  columnStats: ColumnStat[];
  totalRows: number;
  totalColumns: number;
  missingCells: number;
  numericCount: number;
  categoricalCount: number;
  idCount: number;
  defaultTarget: string;
  cleaningReport: CleaningReport;
}

export interface FeatureEncoderSpec {
  rawName: string;
  type: 'numeric' | 'categorical';
  // For numeric
  median?: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  step?: number;
  // For categorical
  mode?: string;
  categories?: string[];
  hasOtherBucket?: boolean;
  encodedStartIndex: number;
  encodedLength: number;
}

export interface PreprocessorState {
  taskType: TaskType;
  targetColumn: string;
  droppedIdColumns: string[];
  featureNames: string[];
  encodedFeatureCount: number;
  specs: FeatureEncoderSpec[];
  encodedColumnToRaw: string[];
  classLabels?: string[];
  targetMin?: number;
  targetMax?: number;
  targetMean?: number;
  trainCount: number;
  testCount: number;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number; // 0..100 percentage
}

export interface RegressionPoint {
  actual: number;
  predicted: number;
  rowIndex: number;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][]; // [actualIdx][predIdx]
}

export type ModelAlgorithmKey =
  | 'ridge'
  | 'logistic'
  | 'decision_tree'
  | 'random_forest'
  | 'gradient_boosting';

export interface SerializedTreeNode {
  isLeaf: boolean;
  value?: number; // regression prediction or binary log-odds / class prob
  probs?: number[]; // multiclass probabilities
  featureIndex?: number;
  threshold?: number;
  left?: SerializedTreeNode;
  right?: SerializedTreeNode;
}

export type SerializedModelWeights =
  | {
      kind: 'linear';
      weights: number[]; // for regression or binary logistic
      bias: number;
      multiWeights?: number[][]; // [classIdx][featIdx] for multiclass logistic
      multiBias?: number[];
    }
  | {
      kind: 'tree';
      tree: SerializedTreeNode;
    }
  | {
      kind: 'forest';
      trees: SerializedTreeNode[];
    }
  | {
      kind: 'gbdt';
      learningRate: number;
      baseScore: number;
      trees?: SerializedTreeNode[]; // regression or binary classification
      multiBaseScores?: number[];
      multiTrees?: SerializedTreeNode[][]; // [round][class]
    };

export interface ModelMetrics {
  // Regression
  r2?: number;
  rmse?: number;
  mae?: number;
  // Classification
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
}

export interface TrainedModelResult {
  key: ModelAlgorithmKey;
  name: string;
  sklearnName: string;
  family: string;
  trainingTimeMs: number;
  primaryScore: number; // R2 for regression, Accuracy for classification
  metrics: ModelMetrics;
  featureImportances: FeatureImportanceItem[];
  regressionPoints?: RegressionPoint[];
  confusionMatrix?: ConfusionMatrixData;
  weights: SerializedModelWeights;
  isWinner: boolean;
}

export interface AutoMLRunResult {
  runId: string;
  datasetName: string;
  fileName: string;
  taskType: TaskType;
  targetColumn: string;
  preprocessor: PreprocessorState;
  models: TrainedModelResult[];
  winner: TrainedModelResult;
  totalDurationMs: number;
  createdAt: string;
}

export interface PredictionOutput {
  taskType: TaskType;
  rawPrediction: string | number;
  formattedPrediction: string;
  confidence?: number;
  classProbabilities?: { label: string; probability: number }[];
}
