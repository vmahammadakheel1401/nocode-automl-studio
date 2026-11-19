import React, { useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Database,
  FileSpreadsheet,
  Filter,
  Upload,
} from 'lucide-react';
import { SAMPLE_DATASET_META, SampleDatasetKey } from '../ml/sampleDatasets';
import { ParsedDataset } from '../ml/types';

interface StepUploadAuditProps {
  dataset: ParsedDataset | null;
  activeSampleKey: SampleDatasetKey | null;
  parseError: string | null;
  onUploadFile: (file: File) => void;
  onSelectSample: (key: SampleDatasetKey) => void;
  onContinue: () => void;
}

const ROWS_PER_PAGE = 8;

export const StepUploadAudit: React.FC<StepUploadAuditProps> = ({
  dataset,
  activeSampleKey,
  parseError,
  onUploadFile,
  onSelectSample,
  onContinue,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [page, setPage] = useState(0);

  // Reset pagination when dataset changes
  const datasetId = dataset?.id ?? '';
  const [lastDatasetId, setLastDatasetId] = useState(datasetId);
  if (datasetId !== lastDatasetId) {
    setLastDatasetId(datasetId);
    setPage(0);
  }

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    onUploadFile(files[0]);
  };

  const totalPages = dataset
    ? Math.max(1, Math.ceil(dataset.rows.length / ROWS_PER_PAGE))
    : 1;
  const safePage = Math.min(page, totalPages - 1);
  const visibleRows = dataset
    ? dataset.rows.slice(
        safePage * ROWS_PER_PAGE,
        safePage * ROWS_PER_PAGE + ROWS_PER_PAGE
      )
    : [];

  return (
    <div className="space-y-6">
      {/* Top Row: Upload Dropzone + 1-Click Sample Dataset Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* CSV Drag-and-Drop Zone */}
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Upload CSV Dataset
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Drop any tabular CSV file here. All parsing, cleaning, and training execute locally inside your browser.
            </p>
          </div>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              handleFiles(e.dataTransfer.files);
            }}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            className={`mt-5 flex-1 min-h-[168px] rounded-xl border-2 border-dashed flex flex-col items-center justify-center px-6 py-8 text-center cursor-pointer transition-colors focus-visible:outline-2 focus-visible:outline-indigo-600 ${
              isDragging
                ? 'border-indigo-500 bg-indigo-50/70'
                : 'border-slate-300 bg-slate-50/60 hover:border-indigo-400 hover:bg-indigo-50/30'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => {
                handleFiles(e.target.files);
                e.target.value = '';
              }}
            />
            <div className="w-11 h-11 rounded-lg bg-white border border-slate-200 flex items-center justify-center text-indigo-600 mb-3">
              <Upload className="w-5 h-5" />
            </div>
            <p className="text-sm font-medium text-slate-900">
              Drag & drop your CSV file, or{' '}
              <span className="text-indigo-600 font-semibold">browse files</span>
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports standard comma-separated CSV files with a header row
            </p>
          </div>

          {parseError && (
            <div className="mt-4 flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-700">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{parseError}</span>
            </div>
          )}
        </div>

        {/* 1-Click Sample Dataset Cards */}
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Instant Sample Datasets
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Load a curated benchmark dataset in one click to explore the full AutoML pipeline immediately.
            </p>
          </div>

          <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 gap-4 flex-1">
            {(['house_prices', 'customer_churn'] as SampleDatasetKey[]).map(
              (key) => {
                const meta = SAMPLE_DATASET_META[key];
                const isActive = activeSampleKey === key;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => onSelectSample(key)}
                    className={`text-left rounded-xl border p-4 flex flex-col justify-between transition-all focus-visible:outline-2 focus-visible:outline-indigo-600 ${
                      isActive
                        ? 'border-indigo-600 bg-indigo-50/40 ring-1 ring-indigo-600/20'
                        : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/70'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2">
                        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-indigo-700">
                          <Database className="w-3.5 h-3.5" />
                          {meta.taskBadge}
                        </span>
                        {isActive && (
                          <span className="text-[11px] font-semibold text-indigo-700">
                            Loaded
                          </span>
                        )}
                      </div>
                      <h3 className="text-sm font-semibold text-slate-900 mt-2">
                        {meta.title}
                      </h3>
                      <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                        {meta.description}
                      </p>
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-600 font-mono tabular-nums">
                      <span>{meta.rowCountHint} rows · {meta.colCountHint} cols</span>
                      <span className="text-indigo-600 font-sans font-semibold">
                        Load →
                      </span>
                    </div>
                  </button>
                );
              }
            )}
          </div>
        </div>
      </div>

      {/* Loaded Dataset Audit & Preview */}
      {dataset ? (
        <>
          {/* 4 Compact KPI Summary Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">Total Rows</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {dataset.totalRows.toLocaleString()}
              </div>
              <div className="text-xs text-slate-500 mt-1 truncate">
                {dataset.fileName}
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">
                Total Columns
              </div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {dataset.totalColumns}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Default target: <span className="font-mono text-slate-700">{dataset.defaultTarget}</span>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">
                Missing Cells
              </div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {dataset.missingCells.toLocaleString()}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {dataset.missingCells > 0
                  ? 'Auto-imputed via median / mode'
                  : 'Complete dataset · zero nulls'}
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">
                Auto-Detected Feature Types
              </div>
              <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-1.5">
                {dataset.numericCount} Num · {dataset.categoricalCount} Cat
                {dataset.idCount > 0 ? ` · ${dataset.idCount} ID` : ''}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {dataset.idCount > 0
                  ? `${dataset.idCount} unique ID col${dataset.idCount > 1 ? 's' : ''} flagged`
                  : 'All columns ready for modeling'}
              </div>
            </div>
          </div>

          {/* Data Cleaning Report Card */}
          {(() => {
            const report = dataset.cleaningReport;
            const totalDropped = report.droppedColumns.length;
            const hasAnyCleaningAction =
              report.strippedSubHeaderRows > 0 ||
              report.sentinelValuesConverted > 0 ||
              totalDropped > 0 ||
              report.wasSampled ||
              report.cappedCategoricalColumns.length > 0;

            return (
              <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5">
                    <Filter className="w-4 h-4 text-indigo-600" />
                    <div>
                      <h3 className="text-sm font-semibold text-slate-900">
                        Data Cleaning Report
                      </h3>
                      <p className="text-xs text-slate-500">
                        Automated hygiene pass applied on upload ·{' '}
                        <span className="font-mono tabular-nums">
                          {report.originalColumnCount} → {report.cleanedColumnCount}
                        </span>{' '}
                        columns retained
                      </p>
                    </div>
                  </div>

                  <span
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold border ${
                      hasAnyCleaningAction
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : 'bg-slate-50 text-slate-600 border-slate-200'
                    }`}
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    {hasAnyCleaningAction
                      ? 'Auto-Cleaned & Verified'
                      : 'Verified Clean Input'}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
                  <div className="p-4">
                    <div className="text-xs font-medium text-slate-500">
                      Sub-Header Rows Stripped
                    </div>
                    <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-1">
                      {report.strippedSubHeaderRows}
                    </div>
                    <div className="text-xs text-slate-500 mt-1 truncate" title={report.subHeaderSampleText}>
                      {report.strippedSubHeaderRows > 0
                        ? `Removed Row 1 ("${report.subHeaderSampleText ?? 'metadata text'}")`
                        : 'No descriptive sub-header row detected'}
                    </div>
                  </div>

                  <div className="p-4">
                    <div className="text-xs font-medium text-slate-500">
                      Sentinels Converted to Null
                    </div>
                    <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-1">
                      {report.sentinelValuesConverted.toLocaleString()}
                    </div>
                    <div className="text-xs text-slate-500 mt-1 truncate">
                      {report.sentinelValuesConverted > 0
                        ? `Normalized -1, N/A, ? across ${report.sentinelColumns.length} col${report.sentinelColumns.length > 1 ? 's' : ''}`
                        : 'No sentinel tokens (-1, N/A, ?) found'}
                    </div>
                  </div>

                  <div className="p-4">
                    <div className="text-xs font-medium text-slate-500">
                      Dropped Sparse & Constant Cols
                    </div>
                    <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-1">
                      {totalDropped}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      {totalDropped > 0
                        ? `${report.droppedSparseCount} sparse (>80% null) · ${report.droppedConstantCount} constant`
                        : '0 columns exceeded 80% null or constant'}
                    </div>
                  </div>

                  <div className="p-4">
                    <div className="text-xs font-medium text-slate-500">
                      Browser Performance Caps
                    </div>
                    <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-1">
                      {report.wasSampled
                        ? `${report.sampledRowCount.toLocaleString()} / ${report.cleanedRowCount.toLocaleString()}`
                        : `${report.sampledRowCount.toLocaleString()} rows`}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      {report.cappedCategoricalColumns.length > 0
                        ? `${report.cappedCategoricalColumns.length} cat col${report.cappedCategoricalColumns.length > 1 ? 's' : ''} capped to Top 20 + Other`
                        : report.wasSampled
                        ? 'Sampled to 2,500 rows max'
                        : 'Top 20 categories · ≤2,500 row cap'}
                    </div>
                  </div>
                </div>

                {(totalDropped > 0 || report.cappedCategoricalColumns.length > 0) && (
                  <div className="px-6 py-3.5 border-t border-slate-200 bg-slate-50/70 space-y-2">
                    {totalDropped > 0 && (
                      <div className="text-xs text-slate-600">
                        <span className="font-semibold text-slate-800">
                          Pruned columns ({totalDropped}):
                        </span>{' '}
                        <span className="font-mono text-slate-700">
                          {report.droppedColumns
                            .slice(0, 8)
                            .map((d) => `${d.name} (${d.detail})`)
                            .join(', ')}
                          {report.droppedColumns.length > 8
                            ? ` +${report.droppedColumns.length - 8} more`
                            : ''}
                        </span>
                      </div>
                    )}
                    {report.cappedCategoricalColumns.length > 0 && (
                      <div className="text-xs text-slate-600">
                        <span className="font-semibold text-slate-800">
                          Top-20 category grouping (&ldquo;Other&rdquo;):
                        </span>{' '}
                        <span className="font-mono text-slate-700">
                          {report.cappedCategoricalColumns.join(', ')}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })()}

          {/* Paginated Data Preview Table (8 rows/page) */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-2.5">
                <FileSpreadsheet className="w-4 h-4 text-indigo-600" />
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">
                    Dataset Preview
                  </h3>
                  <p className="text-xs text-slate-500">
                    Showing rows{' '}
                    <span className="font-mono tabular-nums">
                      {safePage * ROWS_PER_PAGE + 1}–
                      {Math.min(
                        (safePage + 1) * ROWS_PER_PAGE,
                        dataset.rows.length
                      )}
                    </span>{' '}
                    of{' '}
                    <span className="font-mono tabular-nums">
                      {dataset.rows.length}
                    </span>
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={safePage === 0}
                  className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  Prev
                </button>
                <span className="text-xs font-mono tabular-nums text-slate-600 px-2">
                  Page {safePage + 1} / {totalPages}
                </span>
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={safePage >= totalPages - 1}
                  className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                >
                  Next
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/70">
                    <th className="py-2.5 px-4 text-[11px] font-semibold text-slate-500 font-mono w-12">
                      #
                    </th>
                    {dataset.columnStats.map((col) => (
                      <th
                        key={col.name}
                        className={`py-2.5 px-4 text-xs font-semibold text-slate-700 whitespace-nowrap ${
                          col.inferredType === 'numeric' ? 'text-right' : 'text-left'
                        }`}
                      >
                        <div
                          className={`inline-flex items-center gap-1.5 ${
                            col.inferredType === 'numeric' ? 'flex-row-reverse' : ''
                          }`}
                        >
                          <span>{col.name}</span>
                          <span className="text-[10px] font-mono font-medium text-slate-400">
                            {col.inferredType === 'numeric'
                              ? 'num'
                              : col.inferredType === 'id'
                              ? 'id'
                              : 'cat'}
                          </span>
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-sm">
                  {visibleRows.map((row, idx) => {
                    const rowNumber = safePage * ROWS_PER_PAGE + idx + 1;
                    return (
                      <tr
                        key={rowNumber}
                        className="hover:bg-slate-50/80 transition-colors"
                      >
                        <td className="py-2.5 px-4 text-xs font-mono tabular-nums text-slate-400">
                          {rowNumber}
                        </td>
                        {dataset.columnStats.map((col) => {
                          const val = row[col.name];
                          const isEmpty = !val || val.trim() === '';
                          return (
                            <td
                              key={col.name}
                              className={`py-2.5 px-4 text-xs whitespace-nowrap ${
                                col.inferredType === 'numeric'
                                  ? 'text-right font-mono tabular-nums text-slate-800'
                                  : col.inferredType === 'id'
                                  ? 'font-mono text-slate-500'
                                  : 'text-slate-800'
                              }`}
                            >
                              {isEmpty ? (
                                <span className="text-amber-600 font-mono italic">
                                  null
                                </span>
                              ) : (
                                val
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="px-6 py-4 border-t border-slate-200 bg-slate-50/50 flex items-center justify-between gap-4">
              <div className="text-xs text-slate-500">
                Audit verified · Ready for target selection and model configuration.
              </div>
              <button
                type="button"
                onClick={onContinue}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                Continue to Configure & Train
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl p-10 text-center">
          <p className="text-sm font-medium text-slate-800">
            No dataset loaded yet
          </p>
          <p className="text-xs text-slate-500 mt-1">
            Upload a CSV file or select one of the instant sample datasets above to audit columns and preview rows.
          </p>
        </div>
      )}
    </div>
  );
};
