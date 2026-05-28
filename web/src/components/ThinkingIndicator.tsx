interface Props {
  stage?: string | null;
}

export function ThinkingIndicator({ stage }: Props) {
  return (
    <div
      className="max-w-[90%] rounded-xl bg-slate-800 px-4 py-3"
      aria-live="polite"
      aria-busy="true"
    >
      <p className="thinking-shimmer text-sm font-medium tracking-wide">
        Thinking
      </p>
      {stage && (
        <p className="mt-1.5 font-mono text-xs text-slate-500">
          Graph: {stage}
        </p>
      )}
    </div>
  );
}
