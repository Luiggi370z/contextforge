import { DEMO_PROMPTS } from "../constants";

interface Props {
  disabled: boolean;
  onSelect: (text: string) => void;
}

export function DemoPrompts({ disabled, onSelect }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {DEMO_PROMPTS.map((prompt) => (
        <button
          key={prompt.label}
          type="button"
          disabled={disabled}
          title={prompt.hint}
          onClick={() => onSelect(prompt.text)}
          className="rounded-full border border-slate-600 bg-slate-800 px-3 py-1 text-xs hover:bg-slate-700 disabled:opacity-50"
        >
          {prompt.label}
        </button>
      ))}
    </div>
  );
}
