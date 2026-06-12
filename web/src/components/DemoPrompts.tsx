import { DEMO_PROMPTS } from "../constants";

interface Props {
  disabled: boolean;
  onSelect: (text: string) => void;
}

export function DemoPrompts({ disabled, onSelect }: Props) {
  return (
    <section className="space-y-2">
      <p className="text-sm text-slate-500">
        Use these suggested prompts to test each route, or ask about your
        uploaded documents.
      </p>
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        {DEMO_PROMPTS.map((prompt) => (
          <button
            key={prompt.label}
            type="button"
            disabled={disabled}
            title={prompt.hint}
            onClick={() => onSelect(prompt.text)}
            className="flex flex-col gap-1 rounded-xl border border-slate-700 bg-slate-800/80 px-3 py-2 text-left transition hover:border-sky-600 hover:bg-slate-800 disabled:opacity-50"
          >
            <span className="text-sm font-medium text-slate-100">
              {prompt.label}
            </span>
            <span className="line-clamp-2 text-xs text-slate-400">
              {prompt.text}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
