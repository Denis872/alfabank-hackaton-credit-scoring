export type IdentityKind = "phone" | "card";

export type DemoClient = {
  id: number;
  phone: string; // +7XXXXXXXXXX
  card: string;  // 16 digits
};

export const DEMO_CLIENTS: DemoClient[] = [
  { id: 0, phone: "+79990000000", card: "2200110200111200" },
  { id: 1, phone: "+79990000001", card: "2200110200111212" }, // requested mapping
  { id: 3, phone: "+79990000003", card: "2200110200111203" },
  { id: 9, phone: "+79990000009", card: "2200110200111209" },
  { id: 11, phone: "+79990000011", card: "2200110200111211" },
  { id: 12, phone: "+79990000012", card: "2200110200111222" },
  { id: 13, phone: "+79990000013", card: "2200110200111233" },
  { id: 15, phone: "+79990000015", card: "2200110200111255" },
  { id: 16, phone: "+79990000016", card: "2200110200111266" },
  { id: 17, phone: "+79990000017", card: "2200110200111277" },
];

export function normalizeCard(input: string) {
  return input.replace(/[\s-]/g, "").trim();
}

export function normalizePhone(input: string) {
  const digits = input.replace(/\D/g, "");
  if (!digits) return "";
  // accept 8XXXXXXXXXX, 7XXXXXXXXXX, XXXXXXXXXX
  if (digits.length === 11 && digits.startsWith("8")) return "+7" + digits.slice(1);
  if (digits.length === 11 && digits.startsWith("7")) return "+7" + digits.slice(1);
  if (digits.length === 10) return "+7" + digits;
  return input.trim().startsWith("+") ? input.trim() : "+" + digits;
}

export function resolveDemoClientId(kind: IdentityKind, value: string): number | null {
  if (kind === "card") {
    const v = normalizeCard(value);
    return DEMO_CLIENTS.find((c) => c.card === v)?.id ?? null;
  }
  const v = normalizePhone(value);
  return DEMO_CLIENTS.find((c) => c.phone === v)?.id ?? null;
}
