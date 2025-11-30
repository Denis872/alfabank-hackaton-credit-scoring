import Image from "next/image";
import { SearchInput } from "./SearchInput";
import { IdentityKind } from "@/app/lib/mockIdentity";

type Props = {
  onSearch?: (kind: IdentityKind, value: string) => void;
  loading?: boolean;
};

export function Header({ onSearch, loading }: Props) {
  return (
    <header className="flex items-center gap-6">
      <Image src="/logo.svg" width={158} height={28} alt="Alfa Score" />
      <SearchInput onSearch={onSearch} loading={loading} />
    </header>
  );
}
