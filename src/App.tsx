import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, Cpu, Moon, Sun } from 'lucide-react';
import {
  StepConfigureTrain,
  TrainingStageStatus,
} from './components/StepConfigureTrain';
import { StepLeaderboard } from './components/StepLeaderboard';
import { StepPredictExport } from './components/StepPredictExport';
import { StepUploadAudit } from './components/StepUploadAudit';
import { trainSingleAlgorithm } from './ml/algorithms';
import { buildDatasetFromCSV, inferTaskType } from './ml/csvParser';
import { prepareTrainingData } from './ml/preprocessor';
import {
  loadSampleDataset,
  SAMPLE_DATASET_META,
  SampleDatasetKey,
} from './ml/sampleDatasets';
import {
  AutoMLRunResult,
  ModelAlgorithmKey,
  ParsedDataset,
  TaskType,
  TrainedModelResult,
} from './ml/types';

type WorkflowStep = 1 | 2 | 3 | 4;

const STEPS: { step: WorkflowStep; label: string }[] = [
  { step: 1, label: '1. Upload & Audit' },
  { step: 2, label: '2. Configure & Train' },
  { step: 3, label: '3. Model Leaderboard' },
  { step: 4, label: '4. Predict & Export' },
];

function buildInitialStages(taskType: TaskType): TrainingStageStatus[] {
  const linearLabel =
    taskType === 'regression' ? 'Training Ridge Regression' : 'Training Logistic Regression';
  return [
    {
      id: 'preprocess',
      label: 'Preprocessing',
      detail: 'Median/mode imputation, scaling & one-hot encoding',
      status: 'pending',
    },
    {
      id: 'split',
      label: 'Splitting 80/20',
      detail:
        taskType === 'classification'
          ? 'Stratified 80% train / 20% holdout test split'
          : 'Seeded 80% train / 20% holdout test split',
      status: 'pending',
    },
    {
      id: 'linear',
      label: linearLabel,
      detail: 'L2-regularized linear baseline model',
      status: 'pending',
    },
    {
      id: 'tree',
      label: 'Training Decision Tree',
      detail: 'CART recursive variance/Gini impurity splitter',
      status: 'pending',
    },
    {
      id: 'forest',
      label: 'Training Random Forest',
      detail: '20-tree bagged ensemble with feature subsampling',
      status: 'pending',
    },
    {
      id: 'gbdt',
      label: 'Training Gradient Boosting',
      detail: '35-stage residual boosted shallow trees',
      status: 'pending',
    },
  ];
}

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

export default function App() {
  const initialDataset = useMemo(() => loadSampleDataset('house_prices'), []);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedTheme = window.localStorage.getItem('automl-theme');
    return savedTheme
      ? savedTheme === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    window.localStorage.setItem('automl-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const [activeStep, setActiveStep] = useState<WorkflowStep>(1);
  const [dataset, setDataset] = useState<ParsedDataset | null>(initialDataset);
  const [activeSampleKey, setActiveSampleKey] = useState<SampleDatasetKey | null>(
    'house_prices'
  );
  const [parseError, setParseError] = useState<string | null>(null);

  const [targetColumn, setTargetColumn] = useState<string>(
    initialDataset.defaultTarget
  );
  const [droppedIdColumns, setDroppedIdColumns] = useState<string[]>(() =>
    initialDataset.columnStats
      .filter((c) => c.isLikelyId && c.name !== initialDataset.defaultTarget)
      .map((c) => c.name)
  );

  const [isTraining, setIsTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainingStages, setTrainingStages] = useState<TrainingStageStatus[]>(() =>
    buildInitialStages('regression')
  );
  const [runResult, setRunResult] = useState<AutoMLRunResult | null>(null);

  const [sampleMenuOpen, setSampleMenuOpen] = useState(false);
  const sampleMenuRef = useRef<HTMLDivElement>(null);

  const taskType: TaskType = useMemo(() => {
    if (!dataset) return 'regression';
    return inferTaskType(dataset, targetColumn);
  }, [dataset, targetColumn]);

  const applyNewDataset = useCallback(
    (nextDataset: ParsedDataset, sampleKey: SampleDatasetKey | null) => {
      setDataset(nextDataset);
      setActiveSampleKey(sampleKey);
      setParseError(null);
      setTrainError(null);
      setRunResult(null);

      const nextTarget = nextDataset.defaultTarget;
      setTargetColumn(nextTarget);
      const autoDrops = nextDataset.columnStats
        .filter((c) => c.isLikelyId && c.name !== nextTarget)
        .map((c) => c.name);
      setDroppedIdColumns(autoDrops);

      const nextTask = inferTaskType(nextDataset, nextTarget);
      setTrainingStages(buildInitialStages(nextTask));
      setActiveStep(1);
    },
    []
  );

  const handleSelectSample = useCallback(
    (key: SampleDatasetKey) => {
      const loaded = loadSampleDataset(key);
      applyNewDataset(loaded, key);
      setSampleMenuOpen(false);
    },
    [applyNewDataset]
  );

  const handleUploadFile = useCallback(
    (file: File) => {
      if (!file.name.toLowerCase().endsWith('.csv') && file.type !== 'text/csv') {
        setParseError('Please upload a valid .csv file.');
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const text = String(reader.result ?? '');
          const parsed = buildDatasetFromCSV(text, file.name);
          applyNewDataset(parsed, null);
        } catch (err) {
          setParseError(
            err instanceof Error ? err.message : 'Unable to parse CSV file.'
          );
        }
      };
      reader.onerror = () => {
        setParseError('Failed to read the selected CSV file.');
      };
      reader.readAsText(file);
    },
    [applyNewDataset]
  );

  const handleChangeTarget = useCallback(
    (nextTarget: string) => {
      if (!dataset) return;
      setTargetColumn(nextTarget);
      setTrainError(null);
      setRunResult(null);
      setDroppedIdColumns((prev) => prev.filter((c) => c !== nextTarget));
      const nextTask = inferTaskType(dataset, nextTarget);
      setTrainingStages(buildInitialStages(nextTask));
    },
    [dataset]
  );

  const handleToggleDropColumn = useCallback((colName: string) => {
    setDroppedIdColumns((prev) =>
      prev.includes(colName)
        ? prev.filter((c) => c !== colName)
        : [...prev, colName]
    );
    setRunResult(null);
  }, []);

  const updateStage = (
    stageId: string,
    status: 'pending' | 'running' | 'done'
  ) => {
    setTrainingStages((prev) =>
      prev.map((s) => (s.id === stageId ? { ...s, status } : s))
    );
  };

  const handleStartTraining = async () => {
    if (!dataset || isTraining) return;
    setIsTraining(true);
    setTrainError(null);

    const currentTask = inferTaskType(dataset, targetColumn);
    setTrainingStages(buildInitialStages(currentTask));
    const overallStart = performance.now();

    try {
      // Stage 1: Preprocessing
      updateStage('preprocess', 'running');
      await sleep(160);
      updateStage('preprocess', 'done');

      // Stage 2: Splitting 80/20 + fitting preprocessor on train
      updateStage('split', 'running');
      await sleep(140);
      const prepared = prepareTrainingData(
        dataset,
        targetColumn,
        currentTask,
        droppedIdColumns.filter((c) => c !== targetColumn)
      );
      updateStage('split', 'done');

      const algoQueue: { stageId: string; key: ModelAlgorithmKey }[] = [
        {
          stageId: 'linear',
          key: currentTask === 'regression' ? 'ridge' : 'logistic',
        },
        { stageId: 'tree', key: 'decision_tree' },
        { stageId: 'forest', key: 'random_forest' },
        { stageId: 'gbdt', key: 'gradient_boosting' },
      ];

      const trainedModels: TrainedModelResult[] = [];

      for (const item of algoQueue) {
        updateStage(item.stageId, 'running');
        await sleep(100);
        const result = await trainSingleAlgorithm(item.key, prepared);
        trainedModels.push(result);
        updateStage(item.stageId, 'done');
      }

      // Rank models by primaryScore descending (tie-break by faster trainingTimeMs)
      const sorted = [...trainedModels].sort((a, b) => {
        if (Math.abs(b.primaryScore - a.primaryScore) > 1e-6) {
          return b.primaryScore - a.primaryScore;
        }
        return a.trainingTimeMs - b.trainingTimeMs;
      });

      const ranked = sorted.map((m, idx) => ({
        ...m,
        isWinner: idx === 0,
      }));

      const winner = ranked[0];
      const totalDurationMs = Math.max(1, Math.round(performance.now() - overallStart));

      const completedRun: AutoMLRunResult = {
        runId: `run-${Date.now()}`,
        datasetName: dataset.label,
        fileName: dataset.fileName,
        taskType: currentTask,
        targetColumn,
        preprocessor: prepared.preprocessor,
        models: ranked,
        winner,
        totalDurationMs,
        createdAt: new Date().toISOString(),
      };

      setRunResult(completedRun);
      await sleep(180);
      setActiveStep(3);
    } catch (err) {
      setTrainError(
        err instanceof Error ? err.message : 'Training failed unexpectedly.'
      );
    } finally {
      setIsTraining(false);
    }
  };

  const canAccessStep = (step: WorkflowStep): boolean => {
    if (step === 1) return true;
    if (step === 2) return dataset !== null;
    if (step === 3 || step === 4) return runResult !== null;
    return false;
  };

  return (
    <div className="app-shell min-h-screen flex flex-col">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-xs border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-6">
          {/* Left Zone: Brand Title + Subtle Status Badge */}
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0">
              <Cpu className="w-4 h-4" />
            </div>
            <div className="flex items-center gap-3 min-w-0">
              <span className="text-base sm:text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0">
                AutoML Studio
              </span>
              <span className="hidden md:inline-flex items-center px-2.5 py-0.5 rounded-md border border-slate-200 bg-slate-50 text-xs font-medium text-slate-600 whitespace-nowrap shrink-0">
                100% Client-Side • Zero APIs
              </span>
            </div>
          </div>

          {/* Right Zone: Appearance and Sample Data Controls */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setIsDarkMode((dark) => !dark)}
              aria-pressed={isDarkMode}
              aria-label="Toggle dark mode"
              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-xs sm:text-sm font-medium text-slate-800 transition-colors whitespace-nowrap cursor-pointer"
            >
              {isDarkMode ? (
                <Sun className="w-4 h-4 text-amber-500" />
              ) : (
                <Moon className="w-4 h-4 text-slate-500" />
              )}
              <span className="hidden sm:inline">Dark mode</span>
            </button>

            <div className="relative shrink-0" ref={sampleMenuRef}>
              <button
                type="button"
                disabled={isTraining}
                onClick={() => setSampleMenuOpen((o) => !o)}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-xs sm:text-sm font-medium text-slate-800 transition-colors whitespace-nowrap shrink-0 disabled:opacity-50 cursor-pointer"
              >
                <span>Load Sample Data</span>
                <ChevronDown className="w-4 h-4 text-slate-500" />
              </button>

              {sampleMenuOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setSampleMenuOpen(false)}
                  />
                  <div className="absolute right-0 mt-2 w-72 rounded-xl border border-slate-200 bg-white shadow-lg py-1.5 z-20">
                    <div className="px-3.5 py-1.5 text-[11px] font-semibold text-slate-400">
                      Benchmark Datasets
                    </div>
                    {(['house_prices', 'customer_churn'] as SampleDatasetKey[]).map(
                      (key) => {
                        const meta = SAMPLE_DATASET_META[key];
                        const isCurrent = activeSampleKey === key;
                        return (
                          <button
                            key={key}
                            type="button"
                            onClick={() => handleSelectSample(key)}
                            className="w-full text-left px-3.5 py-2.5 hover:bg-slate-50 flex items-center justify-between gap-3 transition-colors cursor-pointer"
                          >
                            <div>
                              <div className="text-xs font-semibold text-slate-900">
                                {meta.dropdownLabel}
                              </div>
                              <div className="text-[11px] text-slate-500 font-mono tabular-nums mt-0.5">
                                {meta.rowCountHint} rows · Target: {meta.targetColumn}
                              </div>
                            </div>
                            {isCurrent && (
                              <Check className="w-4 h-4 text-indigo-600 shrink-0" />
                            )}
                          </button>
                        );
                      }
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Horizontal 4-Step Progress Stepper */}
        <nav
          aria-label="AutoML workflow steps"
          className="bg-white border border-slate-200 rounded-xl p-1.5 flex items-center gap-1.5 overflow-x-auto"
        >
          {STEPS.map((item) => {
            const enabled = canAccessStep(item.step);
            const isActive = activeStep === item.step;
            const isCompleted =
              (item.step === 1 && dataset !== null && activeStep > 1) ||
              (item.step === 2 && runResult !== null && activeStep > 2) ||
              (item.step === 3 && runResult !== null && activeStep > 3);

            return (
              <button
                key={item.step}
                type="button"
                disabled={!enabled || isTraining}
                onClick={() => {
                  if (enabled) setActiveStep(item.step);
                }}
                className={`flex-1 min-w-[160px] inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-xs sm:text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:cursor-not-allowed ${
                  isActive
                    ? 'bg-indigo-600 text-white'
                    : enabled
                    ? 'text-slate-700 hover:bg-slate-100 hover:text-slate-900'
                    : 'text-slate-400 opacity-60'
                }`}
              >
                {isCompleted && !isActive && (
                  <Check className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                )}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Active Step View */}
        {activeStep === 1 && (
          <StepUploadAudit
            dataset={dataset}
            activeSampleKey={activeSampleKey}
            parseError={parseError}
            onUploadFile={handleUploadFile}
            onSelectSample={handleSelectSample}
            onContinue={() => setActiveStep(2)}
          />
        )}

        {activeStep === 2 && dataset && (
          <StepConfigureTrain
            dataset={dataset}
            targetColumn={targetColumn}
            taskType={taskType}
            droppedIdColumns={droppedIdColumns}
            isTraining={isTraining}
            trainingStages={trainingStages}
            trainError={trainError}
            hasTrainedResult={runResult !== null}
            onChangeTarget={handleChangeTarget}
            onToggleDropColumn={handleToggleDropColumn}
            onStartTraining={handleStartTraining}
            onViewLeaderboard={() => setActiveStep(3)}
          />
        )}

        {activeStep === 3 && runResult && (
          <StepLeaderboard
            run={runResult}
            onProceedToPredict={() => setActiveStep(4)}
          />
        )}

        {activeStep === 4 && runResult && <StepPredictExport run={runResult} />}
      </main>
    </div>
  );
}
