"use client";

import { cn } from "@/app/utils";
import { Search } from "lucide-react";
import { useId, useMemo, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { DEMO_CLIENTS, IdentityKind, normalizeCard, normalizePhone } from "@/app/lib/mockIdentity";

type Props = {
  onSearch?: (kind: IdentityKind, value: string) => void;
  loading?: boolean;
};

export function SearchInput({ onSearch, loading }: Props) {
  const [mode, setMode] = useState<IdentityKind>("phone");
  const [query, setQuery] = useState<string>("");

  const listId = useId();

  const demoValues = useMemo(() => {
    return mode === "phone"
      ? DEMO_CLIENTS.map((c) => c.phone)
      : DEMO_CLIENTS.map((c) => c.card);
  }, [mode]);

  const placeholder = mode === "phone" ? "+7 999 000 00 01" : "2200 1102 0011 1212";

  const normalized =
    mode === "phone" ? normalizePhone(query) : normalizeCard(query);

  const canSearch =
    (mode === "phone" && normalized.length >= 12) ||
    (mode === "card" && normalized.length === 16);

  const handleSearch = () => {
    if (!onSearch) return;
    onSearch(mode, query);
  };

  return (
    <div className="w-full">
      <div className="w-full flex items-center p-3 pl-4 pr-4 rounded-2xl border border-gray-200 bg-white">
        <Search className="w-5 h-5 text-gray-400 mr-3 shrink-0" />
        <Input
          className="w-full shadow-none border-none focus-visible:ring-0 text-base"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSearch();
          }}
          list={listId}
          inputMode={mode === "card" ? "numeric" : "text"}
        />
        <datalist id={listId}>
          {demoValues.map((v) => (
            <option key={v} value={v} />
          ))}
        </datalist>

        <div className="flex items-center gap-1 bg-gray-100 rounded-full p-1 ml-3">
          <button
            type="button"
            className={cn(
              "px-3 py-1 text-sm rounded-full transition",
              mode === "phone" ? "bg-white shadow font-medium" : "text-gray-500"
            )}
            onClick={() => setMode("phone")}
          >
            Телефон
          </button>
          <button
            type="button"
            className={cn(
              "px-3 py-1 text-sm rounded-full transition",
              mode === "card" ? "bg-white shadow font-medium" : "text-gray-500"
            )}
            onClick={() => setMode("card")}
          >
            Карта
          </button>
        </div>

        <Button
          className="ml-4 bg-red-500 hover:bg-red-600 text-white rounded-[12px] px-6"
          onClick={handleSearch}
          disabled={loading || !canSearch}
        >
          {loading ? "..." : "Найти"}
        </Button>
      </div>

      <div className="mt-1 text-xs text-gray-400">
        Демо: {mode === "phone" ? "например " : "например "}
        <span className="font-mono">
          {mode === "phone" ? DEMO_CLIENTS[1].phone : DEMO_CLIENTS[1].card}
        </span>
      </div>
    </div>
  );
}
