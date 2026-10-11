// Vercel function: MARTA's live bus positions (GTFS-realtime) as JSON for the
// mesh map. The feed is public but sends no CORS headers, so the browser
// can't read it directly. Cached at the edge for 15 s.

import GtfsRealtimeBindings from "gtfs-realtime-bindings";

const VEHICLES_URL = "https://gtfs-rt.itsmarta.com/TMGTFSRealTimeWebService/vehicle/vehiclepositions.pb";

export async function GET(): Promise<Response> {
  const headers = { "Content-Type": "application/json", "Cache-Control": "public, s-maxage=15, stale-while-revalidate=60" };
  try {
    const resp = await fetch(VEHICLES_URL);
    if (!resp.ok) throw new Error(`MARTA feed ${resp.status}`);
    const feed = GtfsRealtimeBindings.transit_realtime.FeedMessage.decode(new Uint8Array(await resp.arrayBuffer()));
    const vehicles = feed.entity
      .filter((e) => e.vehicle?.position)
      .map((e) => ({
        id: e.vehicle!.vehicle?.label || e.vehicle!.vehicle?.id || e.id,
        route: e.vehicle!.trip?.routeId ?? null,
        lat: e.vehicle!.position!.latitude,
        lng: e.vehicle!.position!.longitude,
        bearing: e.vehicle!.position!.bearing ?? null,
        ts: Number(e.vehicle!.timestamp ?? 0) * 1000,
      }));
    return new Response(JSON.stringify({ source: "MARTA GTFS-realtime", fetchedAt: Date.now(), vehicles }), { headers });
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err), vehicles: [] }), { status: 502, headers });
  }
}
