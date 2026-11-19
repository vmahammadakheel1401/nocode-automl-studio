import React, { useMemo } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Check,
  Loader2,
  Play,
  ShieldAlert,
  SlidersHorizontal,
} from 'lucide-react';
import { isMissingValue } from '../ml/csvParser';
import { ParsedDataset, TaskType } from '../ml/types';

export interface TrainingStageStatus {
  id: string;
  label: string;
  detail: string;
  status: 'pending' | 'running' | 'done';
}

interface StepConfigureTrainProps {
  dataset: ParsedDataset;
  targetColumn: string;
  taskType: TaskType;
  droppedIdColumns: string[];
  isTraining: boolean;
  trainingStages: TrainingStageStatus[];
  trainError: string | null;
  hasTrainedResult: boolean;
  onChangeTarget: (target: string) => void;
  onToggleDropColumn: (colName: string) => void;
  onStartTraining: () => void;
  onViewLeaderboard: () => void;
}

export const StepConfigureTrain: React.FC<StepConfigureTrainProps> = ({
  dataset,
  targetColumn,
  taskType,
  droppedIdColumns,
  isTraining,
  trainingStages,
  trainError,
  hasTrainedResult,
  onChangeTarget,
  onToggleDropColumn,
  onStartTraining,
  onViewLeaderboard,
}) => {
  const targetStat = dataset.columnStats.find((c) => c.name === targetColumn);

  const targetSummary = useMemo(() => {
    const validVals = dataset.rows
      .map((r) => r[targetColumn])
      .filter((v) => !isMissingValue(v));

    if (taskType === 'regression') {
      const nums = validVals
        .map((v) => Number(v.replace(/[$,%]/g, '')))
        .filter((n) => Number.isFinite(n));
      if (nums.length === 0) return null;
      const min = Math.min(...nums);
      const max = Math.max(...nums);
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      return {
        kind: 'regression' as const,
        validCount: nums.length,
        min,
        max,
        mean,
      };
    }

    const counts = new Map<string, number>();
    for (const v of validVals) {
      const clean = v.trim();
      counts.set(clean, (counts.get(clean) ?? 0) + 1);
    }
    const classes = Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([label, count]) => ({
        label,
        count,
        pct: (count / Math.max(1, validVals.length)) * 100,
      }));
    return {
      kind: 'classification' as const,
      validCount: validVals.length,
      classes,
    };
  }, [dataset, targetColumn, taskType]);

  const activeFeatures = dataset.columnStats.filter(
    (c) => c.name !== targetColumn && !droppedIdColumns.includes(c.name)
  );
  const autoDetectedIds = dataset.columnStats.filter(
    (c) => c.isLikelyId && c.name !== targetColumn
  );
  const trainCount = Math.floor((targetSummary?.validCount ?? dataset.totalRows) * 0.8);
  const testCount = Math.max(
    1,
    (targetSummary?.validCount ?? dataset.totalRows) - trainCount
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
      {/* Left Column: Target & Feature Schema Configuration */}
      <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between space-y-6">
        <div className="space-y-6">
          <div>
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-indigo-600" />
              <h2 className="text-base font-semibold text-slate-900">
                Target & Feature Configuration
              </h2>
            </div>
            <p className="text-sm text-slate-500 mt-1">
              Select the column you want to predict. Problem type and identifier exclusions are inferred automatically.
            </p>
          </div>

          {/* Target Column Selector + Auto-Detected Problem Type */}
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <label
                htmlFor="target-column-select"
                className="text-xs font-semibold text-slate-700"
              >
                Target Column (Prediction Output)
              </label>
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-semibold border ${
                  taskType === 'regression'
                    ? 'bg-indigo-50 text-indigo-700 border-indigo-200'
                    : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                }`}
              >
                {taskType === 'regression'
                  ? 'Regression · Continuous'
                  : 'Classification · Discrete'}
              </span>
            </div>

            <select
              id="target-column-select"
              value={targetColumn}
              disabled={isTraining}
              onChange={(e) => onChangeTarget(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3.5 py-2.5 text-sm font-medium text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:opacity-60"
            >
              {dataset.columnStats.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.inferredType} · {col.uniqueCount} unique)
                </option>
              ))}
            </select>

            {/* Target Distribution Summary */}
            {targetSummary?.kind === 'regression' && (
              <div className="grid grid-cols-3 gap-3 pt-2">
                <div className="rounded-lg bg-slate-50 border border-slate-200/80 px-3 py-2">
                  <div className="text-[11px] text-slate-500">Min Value</div>
                  <div className="text-sm font-semibold font-mono tabular-nums text-slate-900">
                    {targetSummary.min.toLocaleString(undefined, {
                      maximumFractionDigits: 2,
                    })}
                  </div>
                </div>
                <div className="rounded-lg bg-slate-50 border border-slate-200/80 px-3 py-2">
                  <div className="text-[11px] text-slate-500">Mean Value</div>
                  <div className="text-sm font-semibold font-mono tabular-nums text-slate-900">
                    {targetSummary.mean.toLocaleString(undefined, {
                      maximumFractionDigits: 2,
                    })}
                  </div>
                </div>
                <div className="rounded-lg bg-slate-50 border border-slate-200/80 px-3 py-2">
                  <div className="text-[11px] text-slate-500">Max Value</div>
                  <div className="text-sm font-semibold font-mono tabular-nums text-slate-900">
                    {targetSummary.max.toLocaleString(undefined, {
                      maximumFractionDigits: 2,
                    })}
                  </div>
                </div>
              </div>
            )}

            {targetSummary?.kind === 'classification' && (
              <div className="pt-2 space-y-2">
                <div className="text-xs font-medium text-slate-600">
                  Class Balance ({targetSummary.classes.length} classes ·{' '}
                  <span className="font-mono tabular-nums">
                    {targetSummary.validCount}
                  </span>{' '}
                  valid rows)
                </div>
                <div className="space-y-1.5">
                  {targetSummary.classes.slice(0, 5).map((cls) => (
                    <div key={cls.label} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-slate-800">
                          {cls.label}
                        </span>
                        <span className="font-mono tabular-nums text-slate-600">
                          {cls.count} ({cls.pct.toFixed(1)}%)
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-emerald-500"
                          style={{ width: `${Math.max(3, cls.pct)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Auto-Dropped Unique ID Columns */}
          <div className="pt-4 border-t border-slate-200">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
                <ShieldAlert className="w-3.5 h-3.5 text-amber-600" />
                Auto-Dropped Unique ID Columns
              </h3>
              <span className="text-xs font-mono tabular-nums text-slate-500">
                {droppedIdColumns.filter((c) => c !== targetColumn).length} excluded
              </span>
            </div>

            {autoDetectedIds.length === 0 ? (
              <p className="text-xs text-slate-500 mt-2">
                No unique identifier columns detected. All non-target columns will be used as input features.
              </p>
            ) : (
              <div className="mt-2.5 space-y-2">
                {autoDetectedIds.map((col) => {
                  const isDropped = droppedIdColumns.includes(col.name);
                  return (
                    <label
                      key={col.name}
                      className="flex items-start justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50/70 px-3.5 py-2.5 cursor-pointer hover:bg-slate-50 transition-colors"
                    >
                      <div className="min-w-0">
                        <div className="text-xs font-semibold font-mono text-slate-900">
                          {col.name}
                        </div>
                        <div className="text-[11px] text-slate-500 mt-0.5">
                          {col.idReason ?? 'High-cardinality unique identifier'}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0 pt-0.5">
                        <span className="text-[11px] font-medium text-slate-600">
                          {isDropped ? 'Dropped' : 'Included'}
                        </span>
                        <input
                          type="checkbox"
                          checked={isDropped}
                          disabled={isTraining}
                          onChange={() => onToggleDropColumn(col.name)}
                          className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                        />
                      </div>
                    </label>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Active Features Summary Footer */}
        <div className="pt-4 border-t border-slate-200 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
          <span>
            Active input features:{' '}
            <strong className="font-mono tabular-nums text-slate-900">
              {activeFeatures.length}
            </strong>
          </span>
          <span className="font-mono tabular-nums text-slate-500 truncate max-w-full">
            {activeFeatures
              .slice(0, 4)
              .map((f) => f.name)
              .join(', ')}
            {activeFeatures.length > 4
              ? ` +${activeFeatures.length - 4} more`
              : ''}
          </span>
        </div>
      </div>

      {/* Right Column: Auto-Train Execution & Animated Pipeline Progress */}
      <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between space-y-6">
        <div className="space-y-5">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Automated Training Pipeline
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Runs leak-free preprocessing, an 80/20 holdout split, and trains 4 competing ML algorithms locally in your browser.
            </p>
          </div>

          {/* Pipeline Plan Metadata */}
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
              <div className="text-[11px] text-slate-500">Train Split (80%)</div>
              <div className="text-base font-bold font-mono tabular-nums text-slate-900 mt-0.5">
                {trainCount} rows
              </div>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
              <div className="text-[11px] text-slate-500">Test Split (20%)</div>
              <div className="text-base font-bold font-mono tabular-nums text-slate-900 mt-0.5">
                {testCount} rows
              </div>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
              <div className="text-[11px] text-slate-500">Target Type</div>
              <div className="text-sm font-semibold text-slate-900 mt-1 truncate">
                {targetStat?.inferredType === 'numeric' && taskType === 'regression'
                  ? 'Numeric'
                  : 'Categorical'}
              </div>
            </div>
          </div>

          {/* Animated Step-by-Step Progress Indicator */}
          <div className="space-y-2.5 pt-1">
            {trainingStages.map((stage, index) => {
              const isDone = stage.status === 'done';
              const isRunning = stage.status === 'running';
              return (
                <div
                  key={stage.id}
                  className={`flex items-center justify-between gap-3 rounded-lg border px-3.5 py-2.5 transition-colors ${
                    isRunning
                      ? 'border-indigo-300 bg-indigo-50/60'
                      : isDone
                      ? 'border-emerald-200 bg-emerald-50/40'
                      : 'border-slate-200 bg-white'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-mono shrink-0 ${
                        isDone
                          ? 'bg-emerald-500 text-white'
                          : isRunning
                          ? 'bg-indigo-600 text-white'
                          : 'bg-slate-100 text-slate-500'
                      }`}
                    >
                      {isDone ? (
                        <Check className="w-3.5 h-3.5" />
                      ) : isRunning ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        index + 1
                      )}
                    </div>
                    <div className="min-w-0">
                      <div className="text-xs font-semibold text-slate-900 truncate">
                        {stage.label}
                      </div>
                      <div className="text-[11px] text-slate-500 truncate">
                        {stage.detail}
                      </div>
                    </div>
                  </div>
                  <span
                    className={`text-[11px] font-medium shrink-0 ${
                      isDone
                        ? 'text-emerald-700'
                        : isRunning
                        ? 'text-indigo-700'
                        : 'text-slate-400'
                    }`}
                  >
                    {isDone ? 'Complete' : isRunning ? 'Running' : 'Queued'}
                  </span>
                </div>
              );
            })}
          </div>

          {trainError && (
            <div className="flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-700">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{trainError}</span>
            </div>
          )}
        </div>

        {/* Action CTA */}
        <div className="pt-4 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            disabled={isTraining || activeFeatures.length === 0}
            onClick={onStartTraining}
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap cursor-pointer"
          >
            {isTraining ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Training 4 Models...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Auto-Train 4 Models
              </>
            )}
          </button>

          {hasTrainedResult && !isTraining && (
            <button
              type="button"
              onClick={onViewLeaderboard}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 text-sm font-medium transition-colors whitespace-nowrap cursor-pointer"
            >
              View Leaderboard
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
