import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DocumentsPage, formatSize, typeLabel } from "./DocumentsPage";

const documents = [
  {
    id: "11111111-1111-1111-1111-111111111111",
    filename: "policy_pto.md",
    contentType: "text/markdown",
    status: "ingested",
    sizeBytes: 2048,
    createdAt: "2026-06-10T15:30:00.000Z",
  },
  {
    id: "22222222-2222-2222-2222-222222222222",
    filename: "handbook.pdf",
    contentType: "application/pdf",
    status: "ingested",
    sizeBytes: 1048576,
    createdAt: "2026-06-11T09:00:00.000Z",
  },
];

function mockFetch(contentBody = "# PTO\n20 days") {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/v1/documents") {
      return new Response(
        JSON.stringify({ items: documents, total: documents.length }),
        {
          headers: { "content-type": "application/json" },
        },
      );
    }
    if (url.endsWith("/content")) {
      return new Response(contentBody, {
        headers: { "content-type": "text/markdown" },
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DocumentsPage", () => {
  it("lists uploaded files with name, type, size, and upload date", async () => {
    vi.stubGlobal("fetch", mockFetch());
    render(<DocumentsPage />);

    expect(await screen.findByText("policy_pto.md")).toBeInTheDocument();
    expect(screen.getByText("handbook.pdf")).toBeInTheDocument();
    expect(screen.getByText("Markdown")).toBeInTheDocument();
    expect(screen.getByText("PDF")).toBeInTheDocument();
    expect(screen.getByText("2.0 KB")).toBeInTheDocument();
    expect(screen.getByText("1.0 MB")).toBeInTheDocument();
    expect(screen.getByText("2 uploaded files")).toBeInTheDocument();
  });

  it("shows a text preview in the sidebar when a row is clicked", async () => {
    const fetchMock = mockFetch("# PTO\n20 days per year");
    vi.stubGlobal("fetch", fetchMock);
    render(<DocumentsPage />);

    fireEvent.click(await screen.findByText("policy_pto.md"));

    await waitFor(() => {
      expect(screen.getByText(/20 days per year/)).toBeInTheDocument();
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `/v1/documents/${documents[0].id}/content`,
    );
  });

  it("uploads a file from the documents page and refreshes the list", async () => {
    const job = {
      id: "33333333-3333-3333-3333-333333333333",
      filename: "new.md",
      status: "completed",
      progress: 100,
    };
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url === "/v1/documents/upload" && init?.method === "POST") {
          return new Response(JSON.stringify(job), {
            status: 202,
            headers: { "content-type": "application/json" },
          });
        }
        if (url === `/v1/documents/jobs/${job.id}`) {
          return new Response(JSON.stringify(job), {
            headers: { "content-type": "application/json" },
          });
        }
        if (url === "/v1/documents") {
          return new Response(
            JSON.stringify({ items: documents, total: documents.length }),
            {
              headers: { "content-type": "application/json" },
            },
          );
        }
        throw new Error(`unexpected fetch: ${url}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<DocumentsPage />);
    await screen.findByText("policy_pto.md");

    const input = screen.getByLabelText(/Upload/) as HTMLInputElement;
    const file = new File(["# hello"], "new.md", { type: "text/markdown" });
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByText("new.md: completed (100%)")).toBeInTheDocument();
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/v1/documents/upload",
      expect.objectContaining({ method: "POST" }),
    );
    // List is re-fetched after the job completes (initial load + refresh).
    const listCalls = fetchMock.mock.calls.filter(
      ([url]) => String(url) === "/v1/documents",
    );
    expect(listCalls.length).toBeGreaterThanOrEqual(2);
  });

  it("shows the empty state when no documents exist", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ items: [], total: 0 }), {
            headers: { "content-type": "application/json" },
          }),
      ),
    );
    render(<DocumentsPage />);

    expect(await screen.findByText(/No documents yet/)).toBeInTheDocument();
  });
});

describe("formatSize", () => {
  it("formats byte counts into readable units", () => {
    expect(formatSize(null)).toBe("—");
    expect(formatSize(512)).toBe("512 B");
    expect(formatSize(2048)).toBe("2.0 KB");
    expect(formatSize(1048576)).toBe("1.0 MB");
  });
});

describe("typeLabel", () => {
  it("derives a friendly label from content type and filename", () => {
    expect(
      typeLabel({
        ...documents[0],
        contentType: "text/plain",
        filename: "notes.txt",
      }),
    ).toBe("Text");
    expect(typeLabel(documents[0])).toBe("Markdown");
    expect(typeLabel(documents[1])).toBe("PDF");
  });
});
