import { FromValues } from "@/app/types";
import { Controller, useFormContext } from "react-hook-form";
import { Card } from "../ui";
import { Script } from "./Script";

const scripts = [
  { name: "Скрипт продажи товара", id: 1 },
  { name: "Скрипт удержания", id: 2 },
  { name: "Скрипт прогрева", id: 3 },
];

export function SalesScripts() {
  const { control } = useFormContext<FromValues>();

  return (
    <Card>
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold">Скрипты продаж</h3>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {scripts.map(({ name, id }) => (
            <Controller
              control={control}
              name="scriptId"
              render={({ field: { value, onChange } }) => (
                <Script
                  onChange={() => onChange(id)}
                  selected={value === id}
                  name={name}
                />
              )}
              key={id}
            />
          ))}
        </div>
      </div>
    </Card>
  );
}
