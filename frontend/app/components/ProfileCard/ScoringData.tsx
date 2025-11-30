type Props = {
  clientId?: number | null;
  prediction?: number | null;
};

const formatRub = (value: number) =>
  Math.round(value).toLocaleString("ru-RU") + " ₽";

export const ScoringData = ({ clientId, prediction }: Props) => (
  <div className="w-full mt-4 flex items-center gap-12">
    <div>
      <div className="text-green-500 font-semibold text-3xl font-bold">
        {prediction != null ? "↑ потенциал" : "—"}
      </div>
      <div className="text-sm text-gray-400">
        {clientId != null ? `ID клиента: ${clientId}` : "Клиент не выбран"}
      </div>
    </div>

    <div>
      <div className="font-bold text-3xl font-bold">
        {prediction != null ? formatRub(prediction) : "—"}
      </div>
      <div className="text-sm text-gray-400">Потенциальный доход</div>
    </div>

    <div>
      <div className="font-bold text-3xl font-bold">
        {prediction != null ? formatRub(prediction * 0.25) : "—"}
      </div>
      <div className="text-sm text-gray-400">Upsell (25%)</div>
    </div>
  </div>
);
