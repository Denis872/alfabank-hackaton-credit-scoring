"use client";

import { ChartsPanel } from "../ChartsPanel";
import { Card } from "../ui";
import { Header } from "./Header";
import { ScoringData } from "./ScoringData";

type Props = {
  clientId?: number | null;
  prediction?: number | null;
};

export function ProfileCard({ clientId, prediction }: Props) {
  return (
    <Card className="col-span-1">
      <div className="flex flex-col items-start gap-6">
        <Header />
        <ScoringData clientId={clientId} prediction={prediction} />
        <ChartsPanel />
      </div>
    </Card>
  );
}
