import type { DemoPrompt } from "./types";

/** Quick prompts to exercise each LangGraph route in the demo UI. */
export const DEMO_PROMPTS: DemoPrompt[] = [
  { label: "Direct", text: "hi", hint: "direct" },
  {
    label: "RAG",
    text: "How many PTO days do employees get?",
    hint: "single_hop_rag",
  },
  {
    label: "Multi-hop",
    text: "Compare PTO policy steps with remote work approval steps",
    hint: "multi_hop",
  },
  {
    label: "Abstain",
    text: "What is the lunar landing budget for 2099?",
    hint: "low retrieval / abstain",
  },
];
