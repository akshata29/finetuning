import { useState } from "react";
import { ConfigBar } from "./components/ConfigBar";
import { JobConsole } from "./components/JobConsole";
import { Sidebar, STAGES } from "./components/Sidebar";
import { AgentStage } from "./stages/AgentStage";
import { DataStage } from "./stages/DataStage";
import { DeployStage } from "./stages/DeployStage";
import { DistillStage } from "./stages/DistillStage";
import { EvalStage } from "./stages/EvalStage";
import { FineTuneStage } from "./stages/FineTuneStage";
import type { StageId } from "./types";

const STAGE_VIEWS: Record<StageId, () => JSX.Element> = {
  data: DataStage,
  finetune: FineTuneStage,
  deploy: DeployStage,
  eval: EvalStage,
  agent: AgentStage,
  distill: DistillStage,
};

export default function App() {
  const [active, setActive] = useState<StageId>("data");
  const meta = STAGES.find((s) => s.id === active)!;
  const StageView = STAGE_VIEWS[active];

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar active={active} onSelect={setActive} />

      <div className="flex min-w-0 flex-1 flex-col">
        <ConfigBar />
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-7xl px-6 py-6 pb-32">
            <header className="mb-6">
              <h2 className="flex items-center gap-2.5 text-xl font-semibold text-slate-100">
                <meta.icon className="h-5 w-5 text-accent-soft" />
                {meta.label}
              </h2>
              <p className="mt-1 text-sm text-slate-400">{meta.blurb}</p>
            </header>
            <StageView />
          </div>
        </main>
      </div>

      <JobConsole />
    </div>
  );
}
