import { useCallback, useEffect, useRef, useState } from "react";
import { useDocumentUpload } from "../hooks/useDocumentUpload";
import type { DocumentInfo, DocumentList } from "../types";

const SIZE_UNITS = ["B", "KB", "MB", "GB"] as const;

export function formatSize(bytes?: number | null): string {
  if (bytes == null) return "—";
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < SIZE_UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const rounded = unit === 0 ? String(value) : value.toFixed(1);
  return `${rounded} ${SIZE_UNITS[unit]}`;
}

export function typeLabel(doc: DocumentInfo): string {
  const type = doc.contentType.toLowerCase();
  if (type.endsWith("pdf")) return "PDF";
  if (type === "text/markdown" || /\.(md|markdown)$/i.test(doc.filename)) {
    return "Markdown";
  }
  return "Text";
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString();
}

interface Preview {
  kind: "text" | "pdf";
  text?: string;
  url?: string;
}

const SIDEBAR_MIN_FRACTION = 0.4;
const SIDEBAR_MAX_FRACTION = 0.6;

function clampSidebarWidth(width: number): number {
  const min = window.innerWidth * SIDEBAR_MIN_FRACTION;
  const max = window.innerWidth * SIDEBAR_MAX_FRACTION;
  return Math.min(Math.max(width, min), max);
}

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [listError, setListError] = useState<string | null>(null);
  const [loadingList, setLoadingList] = useState(true);

  const [selected, setSelected] = useState<DocumentInfo | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const [sidebarWidth, setSidebarWidth] = useState(() =>
    clampSidebarWidth(window.innerWidth * SIDEBAR_MIN_FRACTION),
  );
  const [resizing, setResizing] = useState(false);

  const startResize = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      setResizing(true);

      const onPointerMove = (move: PointerEvent) => {
        setSidebarWidth(clampSidebarWidth(window.innerWidth - move.clientX));
      };
      const onPointerUp = () => {
        setResizing(false);
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
      };
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
    },
    [],
  );

  // Keep the sidebar within bounds if the window shrinks.
  useEffect(() => {
    const onWindowResize = () =>
      setSidebarWidth((width) => clampSidebarWidth(width));
    window.addEventListener("resize", onWindowResize);
    return () => window.removeEventListener("resize", onWindowResize);
  }, []);

  const loadDocuments = useCallback(async () => {
    try {
      const response = await fetch("/v1/documents");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as DocumentList;
      setDocuments(data.items);
      setTotal(data.total);
      setListError(null);
    } catch (error) {
      setListError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  const {
    upload,
    status: uploadStatus,
    uploading,
  } = useDocumentUpload(loadDocuments);

  const releaseObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  useEffect(() => releaseObjectUrl, [releaseObjectUrl]);

  const openPreview = useCallback(
    async (doc: DocumentInfo) => {
      setSelected(doc);
      setPreview(null);
      setPreviewError(null);
      setPreviewLoading(true);
      releaseObjectUrl();
      try {
        const response = await fetch(`/v1/documents/${doc.id}/content`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const contentType = response.headers.get("content-type") ?? "";
        if (contentType.includes("pdf")) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          objectUrlRef.current = url;
          setPreview({ kind: "pdf", url });
        } else {
          setPreview({ kind: "text", text: await response.text() });
        }
      } catch (error) {
        setPreviewError(error instanceof Error ? error.message : String(error));
      } finally {
        setPreviewLoading(false);
      }
    },
    [releaseObjectUrl],
  );

  return (
    <div className="flex h-screen bg-[#0f1419] text-slate-100">
      <div className="flex min-w-0 flex-1 flex-col gap-3 p-4">
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700 pb-3">
          <div>
            <h1 className="text-xl font-semibold text-white">Documents</h1>
            <p className="text-sm text-slate-400">
              {loadingList
                ? "Loading..."
                : `${total} uploaded file${total === 1 ? "" : "s"}`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <label
              className={`flex items-center gap-2 rounded-lg border border-slate-600 px-3 py-1.5 text-sm ${
                uploading
                  ? "cursor-wait opacity-60"
                  : "cursor-pointer hover:bg-slate-800"
              }`}
            >
              {uploading ? "Uploading..." : "Upload"}
              <input
                type="file"
                accept=".md,.txt,.markdown,.pdf"
                className="hidden"
                disabled={uploading}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void upload(file);
                  event.target.value = "";
                }}
              />
            </label>
            <a
              href="#/"
              className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm hover:bg-slate-800"
            >
              ← Chat
            </a>
          </div>
        </header>

        {uploadStatus && (
          <p className="text-sm text-emerald-400">{uploadStatus}</p>
        )}

        {listError && (
          <p className="text-sm text-red-400">
            Failed to load documents: {listError}
          </p>
        )}

        {!loadingList && !listError && documents.length === 0 && (
          <p className="text-sm text-slate-400">
            No documents yet — use the Upload button to add your first file.
          </p>
        )}

        {documents.length > 0 && (
          <div className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-slate-700">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-slate-900 text-slate-400">
                <tr>
                  <th className="px-4 py-2.5 font-medium">File name</th>
                  <th className="px-4 py-2.5 font-medium">Type</th>
                  <th className="px-4 py-2.5 font-medium">Size</th>
                  <th className="px-4 py-2.5 font-medium">Uploaded</th>
                  <th className="px-4 py-2.5 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr
                    key={doc.id}
                    tabIndex={0}
                    onClick={() => void openPreview(doc)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        void openPreview(doc);
                      }
                    }}
                    className={`cursor-pointer border-t border-slate-800 hover:bg-slate-800/60 focus:bg-slate-800 focus:outline-none ${
                      selected?.id === doc.id ? "bg-slate-800" : ""
                    }`}
                  >
                    <td className="px-4 py-2.5 font-medium text-slate-100">
                      {doc.filename}
                    </td>
                    <td className="px-4 py-2.5 text-slate-300">
                      {typeLabel(doc)}
                    </td>
                    <td className="px-4 py-2.5 text-slate-300">
                      {formatSize(doc.sizeBytes)}
                    </td>
                    <td className="px-4 py-2.5 text-slate-300">
                      {formatDate(doc.createdAt)}
                    </td>
                    <td className="px-4 py-2.5">
                      <span
                        className={`rounded-full px-2 py-0.5 text-xs ${
                          doc.status === "ingested"
                            ? "bg-emerald-900/60 text-emerald-300"
                            : "bg-slate-700 text-slate-300"
                        }`}
                      >
                        {doc.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize preview sidebar"
        tabIndex={0}
        onPointerDown={startResize}
        onKeyDown={(event) => {
          const step = 32;
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            setSidebarWidth((width) => clampSidebarWidth(width + step));
          } else if (event.key === "ArrowRight") {
            event.preventDefault();
            setSidebarWidth((width) => clampSidebarWidth(width - step));
          }
        }}
        className={`w-1.5 shrink-0 cursor-col-resize bg-slate-700/40 transition-colors hover:bg-sky-500/70 focus:bg-sky-500 focus:outline-none ${
          resizing ? "bg-sky-500" : ""
        }`}
      />
      <aside
        style={{ width: sidebarWidth, minWidth: "40vw", maxWidth: "60vw" }}
        className={`flex shrink-0 flex-col border-l border-slate-700 bg-slate-900/50 ${
          resizing ? "select-none" : ""
        }`}
      >
        <div className="border-b border-slate-700 px-4 py-3">
          <h2 className="text-sm font-semibold text-white">Preview</h2>
          <p className="truncate text-xs text-slate-400">
            {selected ? selected.filename : "Select a file to preview it"}
          </p>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {previewLoading && (
            <p className="text-sm text-slate-400">Loading preview...</p>
          )}
          {previewError && (
            <p className="text-sm text-red-400">
              Failed to load preview: {previewError}
            </p>
          )}
          {!previewLoading && preview?.kind === "pdf" && preview.url && (
            <iframe
              title={`Preview of ${selected?.filename ?? "PDF"}`}
              src={preview.url}
              className={`h-full w-full rounded-lg border border-slate-700 bg-white ${
                resizing ? "pointer-events-none" : ""
              }`}
            />
          )}
          {!previewLoading && preview?.kind === "text" && (
            <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-slate-200">
              {preview.text}
            </pre>
          )}
          {!previewLoading && !preview && !previewError && (
            <p className="text-sm text-slate-500">
              Click a row to preview its contents — text and Markdown render
              inline, PDFs open in an embedded viewer.
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}
