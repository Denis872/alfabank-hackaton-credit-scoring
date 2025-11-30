import type { NextRequest } from "next/server";

export async function GET(
  _req: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  const apiBase =
    process.env.API_BASE_URL?.trim() || "http://94.103.13.98:8080";

  const upstream = await fetch(`${apiBase}/predict/${encodeURIComponent(id)}`, {
    cache: "no-store",
  });

  const body = await upstream.text();

  return new Response(body, {
    status: upstream.status,
    headers: {
      "content-type":
        upstream.headers.get("content-type") ?? "application/json; charset=utf-8",
    },
  });
}
