"use client";

import { useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { Header, OffersPanel, ProfileCard, SalesScripts, ScriptModal } from "./components";
import { Button } from "./components/ui";
import { FromValues } from "./types";
import { IdentityKind, resolveDemoClientId } from "@/app/lib/mockIdentity";

type PredictResponse = { id: number; prediction: number };

export default function Page() {
  const [dialogOpened, setDialogOpened] = useState<boolean>(false);
  const [dialogData, setDialogData] = useState<FromValues | undefined>(undefined);

  const [clientId, setClientId] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  const form = useForm<FromValues>();

  const onSubmit = (formValues: FromValues) => {
    setDialogData(formValues);
    setDialogOpened(true);
  };

  const handleSearch = async (kind: IdentityKind, value: string) => {
    setPredictError(null);
    setPrediction(null);

    const resolved = resolveDemoClientId(kind, value);
    if (resolved == null) {
      setClientId(null);
      setPredictError("Не найдено в демо-наборе. Выбери значение из подсказок.");
      return;
    }

    setClientId(resolved);
    setLoadingPredict(true);
    try {
      const res = await fetch(`/api/predict/${resolved}`, { cache: "no-store" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `API error ${res.status}`);
      }
      const json = (await res.json()) as PredictResponse;
      setPrediction(json.prediction);
    } catch (e: any) {
      setPredictError(e?.message ?? "Ошибка запроса к API");
    } finally {
      setLoadingPredict(false);
    }
  };

  return (
    <FormProvider {...form}>
      <Header onSearch={handleSearch} loading={loadingPredict} />

      {predictError ? (
        <div className="mt-3 text-sm text-red-600">{predictError}</div>
      ) : null}

      <div className="grid grid-cols-2 gap-6 mt-6">
        <ProfileCard clientId={clientId} prediction={prediction} />
        <div className="col-span-1 flex flex-col gap-4">
          <OffersPanel />
          <SalesScripts />
          <Button
            className="w-[250px] h-[52px] bg-alfa-red text-white font-semibold rounded-[12px] hover:bg-red-600"
            disabled={!form.formState.isDirty || !form.formState.isValid}
            onClick={form.handleSubmit(onSubmit)}
            type="submit"
          >
            Получить скрипт
          </Button>
        </div>
      </div>

      {dialogData ? (
        <ScriptModal onOpenChange={setDialogOpened} open={dialogOpened} {...dialogData} />
      ) : null}
    </FormProvider>
  );
}
