import type { LucideIcon } from "lucide-react";
import {
  Boxes,
  Database,
  FlaskConical,
  Recycle,
  Rocket,
  Sparkles,
} from "lucide-react";
import type { StageId } from "../types";

export interface StageMeta {
  id: StageId;
  label: string;
  blurb: string;
  icon: LucideIcon;
}

export const STAGES: StageMeta[] = [
  { id: "data", label: "Synthetic Data", blurb: "Generate & inspect corpus", icon: Database },
  { id: "finetune", label: "Fine-Tune", blurb: "SFT · DPO · RFT", icon: Sparkles },
  { id: "deploy", label: "Deploy", blurb: "Publish a model", icon: Rocket },
  { id: "eval", label: "Foundry Eval", blurb: "Scoreboard", icon: FlaskConical },
  { id: "agent", label: "Agent Service", blurb: "Create · test · capture", icon: Boxes },
  { id: "distill", label: "Distill & Retrain", blurb: "Close the loop", icon: Recycle },
];

export function Sidebar({
  active,
  onSelect,
}: {
  active: StageId;
  onSelect: (id: StageId) => void;
}) {
  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-white/5 bg-ink-900/60">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-accent to-indigo-700 shadow-lg shadow-accent/30">
          <Sparkles className="h-5 w-5 text-white" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-slate-100">Alignment Studio</div>
          <div className="text-xs text-slate-500">Fine-tuning control plane</div>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-2">
        {STAGES.map((stage, i) => {
          const Icon = stage.icon;
          const isActive = stage.id === active;
          return (
            <button
              key={stage.id}
              onClick={() => onSelect(stage.id)}
              className={`group flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition ${
                isActive
                  ? "bg-accent/15 text-slate-100 ring-1 ring-accent/30"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
              }`}
            >
              <span
                className={`flex h-7 w-7 items-center justify-center rounded-md text-xs font-semibold ${
                  isActive ? "bg-accent text-white" : "bg-ink-800 text-slate-400"
                }`}
              >
                {i + 1}
              </span>
              <span className="flex-1">
                <span className="flex items-center gap-2 text-sm font-medium">
                  <Icon className="h-4 w-4" />
                  {stage.label}
                </span>
                <span className="text-xs text-slate-500">{stage.blurb}</span>
              </span>
            </button>
          );
        })}
      </nav>

      <div className="px-5 py-4 text-xs text-slate-600">
        Live Azure AI Foundry · single-tenant demo
      </div>
    </aside>
  );
}
