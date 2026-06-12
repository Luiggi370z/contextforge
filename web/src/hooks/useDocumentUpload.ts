import { useCallback, useState } from "react";
import type { IngestionJob } from "../types";

/** Upload a document and poll its ingestion job until it reaches a terminal state. */
export function useDocumentUpload(onCompleted?: () => void) {
  const [status, setStatus] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const upload = useCallback(
    async (file: File) => {
      setUploading(true);
      setStatus(`Uploading ${file.name}...`);
      try {
        const form = new FormData();
        form.append("file", file);
        const response = await fetch("/v1/documents/upload", {
          method: "POST",
          body: form,
        });
        if (!response.ok) {
          setStatus(`Upload failed: ${response.status}`);
          return;
        }
        const job = (await response.json()) as IngestionJob;
        for (let attempt = 0; attempt < 60; attempt++) {
          const statusResponse = await fetch(`/v1/documents/jobs/${job.id}`);
          if (!statusResponse.ok) break;
          const current = (await statusResponse.json()) as IngestionJob;
          setStatus(
            `${current.filename}: ${current.status} (${current.progress}%)`,
          );
          if (current.status === "completed" || current.status === "failed") {
            if (current.status === "completed") onCompleted?.();
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      } finally {
        setUploading(false);
      }
    },
    [onCompleted],
  );

  return { upload, status, uploading };
}
