import { proxyRealtimeGateway } from "../_gateway";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return proxyRealtimeGateway(request, "/status");
}
