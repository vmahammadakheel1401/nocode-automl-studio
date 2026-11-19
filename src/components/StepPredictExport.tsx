import React, { useMemo, useState } from 'react';
import {
  Check,
  Code2,
  Copy,
  Download,
  FileJson,
  RotateCcw,
  Sliders,
} from 'lucide-react';
import { predictSingleRow } from '../ml/algorithms';
import {
  generateModelStateJson,
  generatePythonScript,
} from '../ml/codeGenerator';
import { transformSingleRow } from '../ml/preprocessor';
import { AutoMLRunResult } from '../ml/types';

interface StepPredictExportProps {
  run: AutoMLRunResult;
}

function buildBaselineInputs(run: AutoMLRunResult): Record<string, string | number> {
  const init: Record<string, string | number> = {};
  for (const spec of run.preprocessor.specs) {
    if (spec.type === 'numeric') {
      init[spec.rawName] = spec.median ?? 0;
    } else {
      init[spec.rawName] = spec.mode ?? spec.categories?.[0] ?? '';
    }
  }
  return init;
}

function downloadTextFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export const StepPredictExport: React.FC<StepPredictExportProps> = ({ run }) => {
  const [inputs, setInputs] = useState<Record<string, string | number>>(() =>
    buildBaselineInputs(run)
  );
  const [activeTab, setActiveTab] = useState<'python' | 'json'>('python');
  const [copied, setCopied] = useState(false);

  // Reset inputs if user trains a new run
  const [lastRunId, setLastRunId] = useState(run.runId);
  if (run.runId !== lastRunId) {
    setLastRunId(run.runId);
    setInputs(buildBaselineInputs(run));
  }

  const prediction = useMemo(() => {
    const encoded = transformSingleRow(inputs, run.preprocessor);
    return predictSingleRow(run.winner.weights, encoded, run.preprocessor);
  }, [inputs, run]);

  const pythonScript = useMemo(() => generatePythonScript(run), [run]);
  const modelJson = useMemo(() => generateModelStateJson(run), [run]);
  const activeCode = activeTab === 'python' ? pythonScript : modelJson;

  const baseName = run.fileName.replace(/\.csv$/i, '').replace(/[^a-z0-9_-]+/gi, '_');

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(activeCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Fallback ignored if clipboard blocked
    }
  };

  // Gauge position for regression
  const regressionGaugePct = useMemo(() => {
    if (prediction.taskType !== 'regression') return 0;
    const min = run.preprocessor.targetMin ?? 0;
    const max = run.preprocessor.targetMax ?? 1;
    const val = Number(prediction.rawPrediction);
    if (max <= min) return 50;
    return Math.min(100, Math.max(0, ((val - min) / (max - min)) * 100));
  }, [prediction, run.preprocessor]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
      {/* Left Column: Live What-If Predictor */}
      <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-600" />
              <h2 className="text-base font-semibold text-slate-900">
                Live What-If Predictor
              </h2>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Powered in real time by {run.winner.name}
            </p>
          </div>

          <button
            type="button"
            onClick={() => setInputs(buildBaselineInputs(run))}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-xs font-medium text-slate-700 transition-colors cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Baseline
          </button>
        </div>

        {/* Real-Time Predicted Output Card */}
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-semibold text-emerald-800">
              Predicted {run.targetColumn}
            </span>
            <span className="text-[11px] font-mono text-emerald-800">
              {run.winner.name}
            </span>
          </div>

          <div className="mt-2 flex items-baseline justify-between gap-4">
            <div className="text-3xl font-bold font-mono tabular-nums text-slate-900">
              {prediction.formattedPrediction}
            </div>
            {prediction.taskType === 'classification' &&
              prediction.confidence !== undefined && (
                <div className="text-sm font-semibold font-mono tabular-nums text-emerald-800">
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                </div>
              )}
          </div>

          {prediction.taskType === 'regression' && (
            <div className="mt-4">
              <div className="h-2 w-full rounded-full bg-emerald-200/70 overflow-hidden">
                <div
                  className="h-full rounded-full bg-emerald-600 transition-all duration-150"
                  style={{ width: `${regressionGaugePct}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-[11px] font-mono tabular-nums text-emerald-900/75 mt-1.5">
                <span>
                  Dataset Min:{' '}
                  {(run.preprocessor.targetMin ?? 0).toLocaleString(undefined, {
                    maximumFractionDigits: 1,
                  })}
                </span>
                <span>
                  Dataset Max:{' '}
                  {(run.preprocessor.targetMax ?? 0).toLocaleString(undefined, {
                    maximumFractionDigits: 1,
                  })}
                </span>
              </div>
            </div>
          )}

          {prediction.taskType === 'classification' &&
            prediction.classProbabilities && (
              <div className="mt-4 space-y-2">
                {prediction.classProbabilities.map((cp) => (
                  <div key={cp.label} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-800">
                        {cp.label}
                      </span>
                      <span className="font-mono tabular-nums text-slate-700">
                        {(cp.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-emerald-200/70 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-150 ${
                          cp.label === prediction.formattedPrediction
                            ? 'bg-emerald-600'
                            : 'bg-slate-400'
                        }`}
                        style={{
                          width: `${Math.max(2, cp.probability * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
        </div>

        {/* Auto-Generated Interactive Feature Sliders & Dropdowns */}
        <div className="space-y-4 max-h-[520px] overflow-y-auto pr-1">
          {run.preprocessor.specs.map((spec) => {
            if (spec.type === 'numeric') {
              const min = spec.min ?? 0;
              const max = spec.max ?? 100;
              const step = spec.step ?? 1;
              const current = Number(inputs[spec.rawName] ?? spec.median ?? min);

              return (
                <div
                  key={spec.rawName}
                  className="rounded-lg border border-slate-200 bg-slate-50/50 px-4 py-3"
                >
                  <div className="flex items-center justify-between gap-3 mb-2">
                    <label
                      htmlFor={`feat-${spec.rawName}`}
                      className="text-xs font-semibold text-slate-800 truncate"
                    >
                      {spec.rawName}
                    </label>
                    <input
                      type="number"
                      value={current}
                      step={step}
                      onChange={(e) => {
                        const val = Number(e.target.value);
                        if (Number.isFinite(val)) {
                          setInputs((prev) => ({
                            ...prev,
                            [spec.rawName]: val,
                          }));
                        }
                      }}
                      className="w-24 rounded-md border border-slate-300 bg-white px-2 py-1 text-right text-xs font-mono tabular-nums text-slate-900 focus:border-indigo-500 focus:outline-none"
                    />
                  </div>
                  <input
                    id={`feat-${spec.rawName}`}
                    type="range"
                    min={min}
                    max={max === min ? min + 1 : max}
                    step={step}
                    value={Math.min(max, Math.max(min, current))}
                    onChange={(e) =>
                      setInputs((prev) => ({
                        ...prev,
                        [spec.rawName]: Number(e.target.value),
                      }))
                    }
                    className="w-full accent-indigo-600 cursor-pointer"
                  />
                  <div className="flex items-center justify-between text-[11px] font-mono tabular-nums text-slate-400 mt-1">
                    <span>{min.toLocaleString()}</span>
                    <span>{max.toLocaleString()}</span>
                  </div>
                </div>
              );
            }

            const categories = spec.categories ?? [];
            const currentCat = String(inputs[spec.rawName] ?? spec.mode ?? '');

            return (
              <div
                key={spec.rawName}
                className="rounded-lg border border-slate-200 bg-slate-50/50 px-4 py-3"
              >
                <label
                  htmlFor={`feat-${spec.rawName}`}
                  className="block text-xs font-semibold text-slate-800 mb-2"
                >
                  {spec.rawName}
                </label>
                <select
                  id={`feat-${spec.rawName}`}
                  value={currentCat}
                  onChange={(e) =>
                    setInputs((prev) => ({
                      ...prev,
                      [spec.rawName]: e.target.value,
                    }))
                  }
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                >
                  {categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right Column: Dual Code & Model Artifact Export Viewer */}
      <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Export Model & Training Code
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Download the trained browser weights (.json) or a complete Python scikit-learn script (.py).
            </p>
          </div>
        </div>

        {/* Segmented Tabs + Quick Download Actions */}
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-200">
          <div className="inline-flex items-center p-1 rounded-lg bg-slate-100">
            <button
              type="button"
              onClick={() => setActiveTab('python')}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                activeTab === 'python'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              Python Script (.py)
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('json')}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                activeTab === 'json'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <FileJson className="w-3.5 h-3.5" />
              Model State (.json)
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-xs font-medium text-slate-700 transition-colors whitespace-nowrap cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Copy
                </>
              )}
            </button>
          </div>
        </div>

        {/* Code / JSON Viewer */}
        <div className="mt-4 flex-1 min-h-[380px] max-h-[520px] rounded-lg border border-slate-200 bg-slate-950 text-slate-100 overflow-auto">
          <pre className="p-4 text-xs leading-relaxed font-mono">
            <code>{activeCode}</code>
          </pre>
        </div>

        {/* Dual 1-Click Download Buttons */}
        <div className="mt-4 pt-4 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs text-slate-500">
            Tailored to <span className="font-mono text-slate-700">{run.fileName}</span>
          </span>
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              type="button"
              onClick={() =>
                downloadTextFile(
                  modelJson,
                  `${baseName}_model_state.json`,
                  'application/json'
                )
              }
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-xs font-semibold text-slate-800 transition-colors whitespace-nowrap cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Download .json
            </button>

            <button
              type="button"
              onClick={() =>
                downloadTextFile(
                  pythonScript,
                  `${baseName}_train_pipeline.py`,
                  'text/x-python'
                )
              }
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold transition-colors whitespace-nowrap cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Download .py Script
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
