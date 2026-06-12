import { useEffect, useState } from "react";
import { Chat } from "./components/Chat";
import { DocumentsPage } from "./components/DocumentsPage";

/** Minimal hash router: "#/documents" → documents page, anything else → chat. */
function useHashRoute(): string {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  return hash;
}

export default function App() {
  const hash = useHashRoute();
  return hash.startsWith("#/documents") ? <DocumentsPage /> : <Chat />;
}
