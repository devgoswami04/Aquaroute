// Thin API client. In dev, Vite proxies /api -> http://localhost:8000 (vite.config.js).
const BASE = "/api";

export async function getHealth() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

// Major road classes by default — 121k residential segments would choke Leaflet.
export const MAJOR_CLASSES = [
  "motorway", "trunk", "primary", "secondary", "tertiary",
  "primary_link", "secondary_link", "tertiary_link", "trunk_link",
  "unclassified", "busway",
];

export async function getSegments({ classes = MAJOR_CLASSES, bbox, limit } = {}) {
  const params = new URLSearchParams();
  if (classes) params.set("classes", classes.join(","));
  if (bbox) params.set("bbox", bbox.join(","));
  if (limit) params.set("limit", String(limit));
  const r = await fetch(`${BASE}/segments?${params.toString()}`);
  if (!r.ok) throw new Error(`segments ${r.status}`);
  return r.json();
}

export async function getEvents() {
  const r = await fetch(`${BASE}/events`);
  if (!r.ok) throw new Error(`events ${r.status}`);
  return r.json();
}

// Flooded-segment overlay for a labelled event. Scoped to major classes to match
// the base layer and keep the canvas responsive (residential stays in the store).
export async function getLabels(event, classes = MAJOR_CLASSES) {
  const params = new URLSearchParams({ event, only_flooded: "true" });
  if (classes) params.set("classes", classes.join(","));
  const r = await fetch(`${BASE}/labels?${params.toString()}`);
  if (!r.ok) throw new Error(`labels ${r.status}`);
  return r.json();
}

// FRF forecast overlay: per-segment predicted peak depth for a scenario
// ("live" or a historical event slug).
export async function getPredict(scenario = "live", classes = MAJOR_CLASSES) {
  const params = new URLSearchParams({ scenario });
  if (classes) params.set("classes", classes.join(","));
  const r = await fetch(`${BASE}/predict?${params.toString()}`);
  if (!r.ok) throw new Error(`predict ${r.status}`);
  return r.json();
}

export async function getSegmentCurve(segmentId, scenario = "live") {
  const r = await fetch(`${BASE}/segment/${encodeURIComponent(segmentId)}/curve?scenario=${scenario}`);
  if (!r.ok) throw new Error(`curve ${r.status}`);
  return r.json();
}

export async function getVehicles() {
  const r = await fetch(`${BASE}/vehicles`);
  if (!r.ok) throw new Error(`vehicles ${r.status}`);
  return r.json();
}

export async function getCivicSummary(topN = 15) {
  const r = await fetch(`${BASE}/civic/summary?top_n=${topN}`);
  if (!r.ok) throw new Error(`civic ${r.status}`);
  return r.json();
}

// origin/dest are [lon, lat]. depart is an hour 0..23 into the forecast.
export async function postRoute({ origin, dest, vehicle, depart, scenario }) {
  const r = await fetch(`${BASE}/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ origin, dest, vehicle, depart_time: depart, scenario }),
  });
  if (!r.ok) throw new Error(`route ${r.status}`);
  return r.json();
}
