import { cn } from "@/app/utils";

interface ScriptProps {
  onChange: () => void;
  selected: boolean;
  name: string;
}

export const Script = ({ name, selected, onChange }: ScriptProps) => (
  <div
    className={cn(
      "cursor-pointer transition duration-200 flex col-span-1 items-center gap-2 p-3 border border-gray-100 rounded-lg",
      {
        "border-gray-100": !selected,
        "border-alfa-red": selected,
      }
    )}
    onClick={onChange}
  >
    <div
      className={cn("size-5 transition duration-200 flex items-center justify-center rounded-full shrink-0", {
        "bg-alfa-red": selected,
        "bg-gray-200": !selected,
      })}
    >
      {selected ? <div className="size-2 rounded-full bg-white" /> : null}
    </div>
    <div className="flex-1 text-sm font-semibold whitespace-nowrap">{name}</div>
  </div>
);
