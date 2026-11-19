import React, { useState } from 'react';
import {
  ConfusionMatrixData,
  FeatureImportanceItem,
  RegressionPoint,
} from '../ml/types';

interface ScatterChartProps {
  points: RegressionPoint[];
  targetColumn: string;
  modelName: string;
}

function formatCompactNumber(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (abs >= 10_000) return `${Math.round(n / 1000)}k`;
  if (abs >= 1000) return `${(n / 1000).toFixed(1)}k`;
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(1);
}

export const ActualVsPredictedChart: React.FC<ScatterChartProps> = ({
  points,
  targetColumn,
  modelName,
}) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (points.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-sm text-slate-500">
        No holdout test predictions available.
      </div>
    );
  }

  const allVals = points.flatMap((p) => [p.actual, p.predicted]);
  const rawMin = Math.min(...allVals);
  const rawMax = Math.max(...allVals);
  const pad = Math.max(1e-6, (rawMax - rawMin) * 0.08);
  const minVal = rawMin - pad;
  const maxVal = rawMax + pad;
  const span = Math.max(1e-6, maxVal - minVal);

  const width = 520;
  const height = 310;
  const margin = { top: 20, right: 24, bottom: 44, left: 60 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const toX = (v: number) => margin.left + ((v - minVal) / span) * plotW;
  const toY = (v: number) => margin.top + plotH - ((v - minVal) / span) * plotH;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => minVal + t * span);
  const hovered = hoveredIdx !== null ? points[hoveredIdx] : null;

  return (
    <div className="relative">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">
            Actual vs. Predicted ({targetColumn})
          </h3>
          <p className="text-xs text-slate-500">
            Holdout 20% test split for {modelName} · Diagonal dashed line marks ideal y = x
          </p>
        </div>
      </div>

      <div className="relative w-full">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto overflow-visible select-none"
          role="img"
          aria-label="Actual versus predicted scatter plot"
        >
          {/* Grid lines */}
          {ticks.map((tick, idx) => {
            const x = toX(tick);
            const y = toY(tick);
            return (
              <g key={idx}>
                <line
                  x1={margin.left}
                  y1={y}
                  x2={width - margin.right}
                  y2={y}
                  stroke="#E2E8F0"
                  strokeWidth={1}
                />
                <line
                  x1={x}
                  y1={margin.top}
                  x2={x}
                  y2={height - margin.bottom}
                  stroke="#F1F5F9"
                  strokeWidth={1}
                />
                <text
                  x={margin.left - 8}
                  y={y + 4}
                  textAnchor="end"
                  className="fill-slate-500 text-[10px] font-mono tabular-nums"
                >
                  {formatCompactNumber(tick)}
                </text>
                <text
                  x={x}
                  y={height - margin.bottom + 16}
                  textAnchor="middle"
                  className="fill-slate-500 text-[10px] font-mono tabular-nums"
                >
                  {formatCompactNumber(tick)}
                </text>
              </g>
            );
          })}

          {/* Axes */}
          <line
            x1={margin.left}
            y1={margin.top}
            x2={margin.left}
            y2={height - margin.bottom}
            stroke="#CBD5E1"
            strokeWidth={1.25}
          />
          <line
            x1={margin.left}
            y1={height - margin.bottom}
            x2={width - margin.right}
            y2={height - margin.bottom}
            stroke="#CBD5E1"
            strokeWidth={1.25}
          />

          {/* Ideal y = x diagonal */}
          <line
            x1={toX(minVal)}
            y1={toY(minVal)}
            x2={toX(maxVal)}
            y2={toY(maxVal)}
            stroke="#10B981"
            strokeWidth={1.75}
            strokeDasharray="5 4"
          />

          {/* Data Points */}
          {points.map((pt, idx) => {
            const cx = toX(pt.actual);
            const cy = toY(pt.predicted);
            const isHovered = hoveredIdx === idx;
            return (
              <circle
                key={idx}
                cx={cx}
                cy={cy}
                r={isHovered ? 6 : 4.2}
                fill={isHovered ? '#4F46E5' : '#6366F1'}
                fillOpacity={isHovered ? 1 : 0.72}
                stroke="#FFFFFF"
                strokeWidth={1.25}
                className="cursor-pointer transition-transform duration-150"
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
              />
            );
          })}

          {/* Axis Labels */}
          <text
            x={margin.left + plotW / 2}
            y={height - 6}
            textAnchor="middle"
            className="fill-slate-600 text-[11px] font-medium"
          >
            Actual {targetColumn}
          </text>
          <text
            x={14}
            y={margin.top + plotH / 2}
            textAnchor="middle"
            transform={`rotate(-90 14 ${margin.top + plotH / 2})`}
            className="fill-slate-600 text-[11px] font-medium"
          >
            Predicted {targetColumn}
          </text>
        </svg>

        {/* Interactive Tooltip */}
        {hovered && (
          <div className="mt-2 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700">
            <span className="font-medium text-slate-900">
              Test Sample #{hovered.rowIndex}
            </span>
            <div className="flex items-center gap-4 font-mono tabular-nums">
              <span>
                Actual:{' '}
                <strong className="text-slate-900">
                  {hovered.actual.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </strong>
              </span>
              <span>
                Predicted:{' '}
                <strong className="text-indigo-600">
                  {hovered.predicted.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </strong>
              </span>
              <span>
                Error:{' '}
                <strong className="text-slate-700">
                  {(hovered.predicted - hovered.actual >= 0 ? '+' : '') +
                    (hovered.predicted - hovered.actual).toLocaleString(undefined, {
                      maximumFractionDigits: 2,
                    })}
                </strong>
              </span>
            </div>
          </div>
        )}
        {!hovered && (
          <div className="mt-2 rounded-lg border border-transparent px-3 py-2 text-xs text-slate-400">
            Hover any point to inspect actual vs. predicted values and residual error.
          </div>
        )}
      </div>
    </div>
  );
};

interface ConfusionMatrixProps {
  data: ConfusionMatrixData;
  modelName: string;
}

export const ConfusionMatrixHeatmap: React.FC<ConfusionMatrixProps> = ({
  data,
  modelName,
}) => {
  const [hoveredCell, setHoveredCell] = useState<{
    actual: string;
    predicted: string;
    count: number;
    pct: string;
    isDiagonal: boolean;
  } | null>(null);

  const { labels, matrix } = data;
  const maxCell = Math.max(1, ...matrix.flat());
  const totalSamples = Math.max(
    1,
    matrix.flat().reduce((a, b) => a + b, 0)
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">
            Confusion Matrix Heatmap
          </h3>
          <p className="text-xs text-slate-500">
            Holdout 20% test split ({totalSamples} samples) for {modelName}
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          <div className="flex items-center justify-center mb-2 text-xs font-medium text-slate-600">
            Predicted Class
          </div>

          <div className="flex items-stretch gap-3">
            <div className="flex items-center justify-center">
              <span className="text-xs font-medium text-slate-600 [writing-mode:vertical-rl] rotate-180">
                Actual Class
              </span>
            </div>

            <div className="flex-1">
              <div
                className="grid gap-2"
                style={{
                  gridTemplateColumns: `minmax(76px, auto) repeat(${labels.length}, minmax(72px, 1fr))`,
                }}
              >
                <div />
                {labels.map((predLabel) => (
                  <div
                    key={predLabel}
                    className="text-center text-xs font-semibold text-slate-700 truncate px-1 py-1"
                    title={predLabel}
                  >
                    {predLabel}
                  </div>
                ))}

                {labels.map((actualLabel, rIdx) => {
                  const rowSum = Math.max(
                    1,
                    matrix[rIdx].reduce((a, b) => a + b, 0)
                  );
                  return (
                    <React.Fragment key={actualLabel}>
                      <div
                        className="flex items-center justify-end pr-2 text-xs font-semibold text-slate-700 truncate"
                        title={actualLabel}
                      >
                        {actualLabel}
                      </div>

                      {labels.map((predLabel, cIdx) => {
                        const count = matrix[rIdx][cIdx];
                        const intensity = count / maxCell;
                        const rowPct = ((count / rowSum) * 100).toFixed(1);
                        const isDiag = rIdx === cIdx;

                        const bgStyle = isDiag
                          ? `rgba(16, 185, 129, ${0.12 + intensity * 0.72})`
                          : count > 0
                          ? `rgba(79, 70, 229, ${0.08 + intensity * 0.45})`
                          : '#F8FAFC';

                        const darkText = isDiag
                          ? intensity > 0.62
                            ? 'text-white'
                            : 'text-emerald-950'
                          : intensity > 0.65 && count > 0
                          ? 'text-white'
                          : 'text-slate-800';

                        return (
                          <button
                            key={`${rIdx}-${cIdx}`}
                            type="button"
                            onMouseEnter={() =>
                              setHoveredCell({
                                actual: actualLabel,
                                predicted: predLabel,
                                count,
                                pct: rowPct,
                                isDiagonal: isDiag,
                              })
                            }
                            onMouseLeave={() => setHoveredCell(null)}
                            style={{ backgroundColor: bgStyle }}
                            className={`rounded-lg border px-3 py-4 text-center transition-all focus-visible:outline-2 focus-visible:outline-indigo-600 ${
                              isDiag
                                ? 'border-emerald-300/80'
                                : 'border-slate-200'
                            } ${darkText}`}
                          >
                            <div className="text-lg font-bold font-mono tabular-nums leading-tight">
                              {count}
                            </div>
                            <div className="text-[11px] font-mono tabular-nums opacity-80 mt-0.5">
                              {rowPct}%
                            </div>
                          </button>
                        );
                      })}
                    </React.Fragment>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
        {hoveredCell ? (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>
              Actual <strong className="text-slate-900">{hoveredCell.actual}</strong>{' '}
              → Predicted{' '}
              <strong className="text-slate-900">{hoveredCell.predicted}</strong>
            </span>
            <span className="font-mono tabular-nums">
              <strong className="text-slate-900">{hoveredCell.count}</strong> samples (
              {hoveredCell.pct}% of actual class ·{' '}
              {hoveredCell.isDiagonal ? 'Correct classification' : 'Misclassified'})
            </span>
          </div>
        ) : (
          <span>
            Diagonal emerald cells indicate correct predictions; off-diagonal cells show misclassifications.
          </span>
        )}
      </div>
    </div>
  );
};

interface FeatureImportanceChartProps {
  importances: FeatureImportanceItem[];
  modelName: string;
}

export const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({
  importances,
  modelName,
}) => {
  const [hoveredFeat, setHoveredFeat] = useState<string | null>(null);
  const topItems = importances.slice(0, 8);
  const maxVal = Math.max(1, ...topItems.map((i) => i.importance));

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">
            Feature Importance
          </h3>
          <p className="text-xs text-slate-500">
            Relative contribution (%) computed from {modelName}
          </p>
        </div>
      </div>

      {topItems.length === 0 ? (
        <div className="h-60 flex items-center justify-center text-sm text-slate-500">
          No feature importances available.
        </div>
      ) : (
        <div className="space-y-3 mt-4">
          {topItems.map((item, idx) => {
            const barWidth = Math.max(2, (item.importance / maxVal) * 100);
            const isHovered = hoveredFeat === item.feature;
            const isTop = idx === 0;
            return (
              <div
                key={item.feature}
                onMouseEnter={() => setHoveredFeat(item.feature)}
                onMouseLeave={() => setHoveredFeat(null)}
                className={`rounded-lg px-2.5 py-1.5 transition-colors ${
                  isHovered ? 'bg-slate-50' : ''
                }`}
              >
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span
                    className="font-medium text-slate-800 truncate pr-3"
                    title={item.feature}
                  >
                    {item.feature}
                  </span>
                  <span className="font-mono tabular-nums font-semibold text-slate-900 shrink-0">
                    {item.importance.toFixed(1)}%
                  </span>
                </div>
                <div className="h-2.5 w-full rounded-full bg-slate-100 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      isTop ? 'bg-indigo-600' : 'bg-indigo-400'
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
