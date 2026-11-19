import React, { useState } from 'react';
import { ArrowRight, Award, Clock, Layers } from 'lucide-react';
import { AutoMLRunResult, ModelAlgorithmKey } from '../ml/types';
import {
  ActualVsPredictedChart,
  ConfusionMatrixHeatmap,
  FeatureImportanceChart,
} from './Charts';

interface StepLeaderboardProps {
  run: AutoMLRunResult;
  onProceedToPredict: () => void;
}

export const StepLeaderboard: React.FC<StepLeaderboardProps> = ({
  run,
  onProceedToPredict,
}) => {
  const [selectedModelKey, setSelectedModelKey] = useState<ModelAlgorithmKey>(
    run.winner.key
  );

  // Sync selected model when a new run finishes
  const [lastRunId, setLastRunId] = useState(run.runId);
  if (run.runId !== lastRunId) {
    setLastRunId(run.runId);
    setSelectedModelKey(run.winner.key);
  }

  const inspectedModel =
    run.models.find((m) => m.key === selectedModelKey) ?? run.winner;

  const isRegression = run.taskType === 'regression';
  const winnerScoreLabel = isRegression
    ? `R² = ${run.winner.primaryScore.toFixed(4)}`
    : `Accuracy = ${(run.winner.primaryScore * 100).toFixed(1)}%`;

  return (
    <div className="space-y-6">
      {/* Emerald Winner Banner */}
      <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 px-6 py-5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="w-11 h-11 rounded-lg bg-emerald-600 text-white flex items-center justify-center shrink-0">
            <Award className="w-5 h-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-emerald-600 text-white text-[11px] font-semibold">
                Winning Model
              </span>
              <span className="text-xs text-emerald-900/75 font-mono">
                {run.winner.sklearnName}
              </span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 mt-1">
              {run.winner.name}
            </h2>
            <p className="text-xs text-slate-600 mt-0.5">
              Target <span className="font-mono font-medium text-slate-800">{run.targetColumn}</span>{' '}
              · Evaluated on {run.preprocessor.testCount} holdout test samples
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-6">
          <div>
            <div className="text-[11px] font-medium text-emerald-800">
              Primary Score
            </div>
            <div className="text-xl font-bold font-mono tabular-nums text-emerald-950">
              {winnerScoreLabel}
            </div>
          </div>

          <div className="h-9 w-px bg-emerald-200 hidden sm:block" />

          <div>
            <div className="text-[11px] font-medium text-emerald-800">
              Train Time
            </div>
            <div className="text-lg font-semibold font-mono tabular-nums text-emerald-950 flex items-center gap-1">
              <Clock className="w-4 h-4 text-emerald-700" />
              {run.winner.trainingTimeMs} ms
            </div>
          </div>

          <button
            type="button"
            onClick={onProceedToPredict}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Predict & Export
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 4-Model Comparison Table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-600" />
            <h3 className="text-sm font-semibold text-slate-900">
              Model Leaderboard
            </h3>
          </div>
          <span className="text-xs text-slate-500">
            Click any model row to inspect its diagnostics and feature importances below
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/70 text-xs font-semibold text-slate-600">
                <th className="py-3 px-5 w-16">Rank</th>
                <th className="py-3 px-5">Algorithm</th>
                <th className="py-3 px-5">Family</th>
                {isRegression ? (
                  <>
                    <th className="py-3 px-5 text-right">R² Score</th>
                    <th className="py-3 px-5 text-right">RMSE</th>
                    <th className="py-3 px-5 text-right">MAE</th>
                  </>
                ) : (
                  <>
                    <th className="py-3 px-5 text-right">Accuracy</th>
                    <th className="py-3 px-5 text-right">Precision</th>
                    <th className="py-3 px-5 text-right">Recall</th>
                    <th className="py-3 px-5 text-right">F1 Score</th>
                  </>
                )}
                <th className="py-3 px-5 text-right">Train Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm">
              {run.models.map((model, idx) => {
                const isSelected = model.key === inspectedModel.key;
                return (
                  <tr
                    key={model.key}
                    onClick={() => setSelectedModelKey(model.key)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-indigo-50/50'
                        : 'hover:bg-slate-50/80'
                    }`}
                  >
                    <td className="py-3.5 px-5 font-mono tabular-nums text-xs text-slate-600">
                      #{idx + 1}
                    </td>
                    <td className="py-3.5 px-5">
                      <div className="flex items-center gap-2.5">
                        <span className="font-semibold text-slate-900">
                          {model.name}
                        </span>
                        {model.isWinner && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-800 text-[11px] font-semibold">
                            Best
                          </span>
                        )}
                        {isSelected && !model.isWinner && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-indigo-100 text-indigo-800 text-[11px] font-semibold">
                            Inspecting
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3.5 px-5 text-xs text-slate-600">
                      {model.family}
                    </td>

                    {isRegression ? (
                      <>
                        <td className="py-3.5 px-5 text-right font-mono tabular-nums font-semibold text-slate-900">
                          {(model.metrics.r2 ?? 0).toFixed(4)}
                        </td>
                        <td className="py-3.5 px-5 text-right font-mono tabular-nums text-slate-700">
                          {(model.metrics.rmse ?? 0).toLocaleString(undefined, {
                            maximumFractionDigits: 2,
                          })}
                        </td>
                        <td className="py-3.5 px-5 text-right font-mono tabular-nums text-slate-700">
                          {(model.metrics.mae ?? 0).toLocaleString(undefined, {
                            maximumFractionDigits: 2,
                          })}
                        </td>
                      </>
                    ) : (
                      <>
                        <td className="py-3.5 px-5 text-right font-mono tabular-nums font-semibold text-slate-900">
                          {((model.metrics.accuracy ?? 0) * 100).toFixed(1)}%
                        </td>
                        <td className="py-3.5 px-5 text-right font-mono tabular-nums text-slate-700">
                          {((model.metrics.precision ?? 0) * 100).toFixed(1)}%
                        </td>
                        <td className="py-3.5 px-5 text-right font-mono tabular-nums text-slate-700">
                          {((model.metrics.recall ?? 0) * 100).toFixed(1)}%
                        </td>
                        <td className="py-3.5 px-5 text-right font-mono tabular-nums text-slate-700">
                          {((model.metrics.f1 ?? 0) * 100).toFixed(1)}%
                        </td>
                      </>
                    )}

                    <td className="py-3.5 px-5 text-right font-mono tabular-nums text-xs text-slate-600">
                      {model.trainingTimeMs} ms
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2 Interactive Charts Side-by-Side */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6">
          {isRegression ? (
            <ActualVsPredictedChart
              points={inspectedModel.regressionPoints ?? []}
              targetColumn={run.targetColumn}
              modelName={inspectedModel.name}
            />
          ) : (
            <ConfusionMatrixHeatmap
              data={
                inspectedModel.confusionMatrix ?? {
                  labels: run.preprocessor.classLabels ?? [],
                  matrix: [],
                }
              }
              modelName={inspectedModel.name}
            />
          )}
        </div>

        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6">
          <FeatureImportanceChart
            importances={inspectedModel.featureImportances}
            modelName={inspectedModel.name}
          />
        </div>
      </div>
    </div>
  );
};
