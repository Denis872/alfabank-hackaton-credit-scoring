import { cn } from "@/app/utils";
import { Badge } from "../ui";

interface OfferItemProps {
  onChange: () => void;
  selected: boolean;
  subtitle: string;
  tags: string[];
  title: string;
}

export const OfferItem = ({
  onChange,
  subtitle,
  selected,
  title,
  tags,
}: OfferItemProps) => (
  <div
    onClick={onChange}
    className={cn("cursor-pointer transition duration-100 flex items-center justify-between border rounded-lg p-3", {
      "border-gray-100": !selected,
      "border-alfa-red": selected,
    })}
  >
    <div className="flex items-center gap-3">
      <div className="w-12 h-12 rounded-lg bg-red-100 flex items-center justify-center font-semibold text-red-600">
        {title.split(" ")[0][0]}
      </div>
      <div>
        <div className="font-medium">{title}</div>
        <div className="text-xs text-gray-400">{subtitle}</div>
      </div>
    </div>
    <div className="flex gap-2">
      {tags.map((t, i) => (
        <Badge key={i}>{t}</Badge>
      ))}
    </div>
  </div>
);
