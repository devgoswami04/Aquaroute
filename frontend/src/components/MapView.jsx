import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { getSegments, getLabels, getPredict } from "../api";

function riskColor(s) {
  if (s == null || Number.isNaN(s)) return "#9aa0a6";
  const x = Math.max(0, Math.min(1, s));
  return `hsl(${(1 - x) * 120}, 85%, 45%)`;
}
function depthColor(d) {
  if (d == null) return "#9aa0a6";
  if (d < 0.1) return "#2ecc71";
  if (d < 0.15) return "#f1c40f";
  if (d < 0.25) return "#e67e22";
  if (d < 0.4) return "#e74c3c";
  return "#8e1a1a";
}

const CENTER = [12.945, 80.175];

export default function MapView({ onSelect, event, baseMode, scenario,
                                 origin, dest, route, pickMode, onMapClick }) {
  const mapRef = useRef(null);
  const containerRef = useRef(null);
  const baseLayerRef = useRef(null);
  const labelLayerRef = useRef(null);
  const markerLayerRef = useRef(null);
  const routeLayerRef = useRef(null);
  const rendererRef = useRef(null);
  const fittedRef = useRef(false);
  const pickModeRef = useRef(pickMode);
  const onMapClickRef = useRef(onMapClick);
  const onSelectRef = useRef(onSelect);
  pickModeRef.current = pickMode;
  onMapClickRef.current = onMapClick;
  onSelectRef.current = onSelect;
  const [status, setStatus] = useState("loading");
  const [count, setCount] = useState(0);
  const [floodCount, setFloodCount] = useState(null);

  useEffect(() => {
    if (mapRef.current) return;
    const map = L.map(containerRef.current, { preferCanvas: true }).setView(CENTER, 13);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors", maxZoom: 19,
    }).addTo(map);
    map.on("click", (e) => {
      if (pickModeRef.current) onMapClickRef.current && onMapClickRef.current(e.latlng);
    });
    mapRef.current = map;
    rendererRef.current = L.canvas({ padding: 0.5 });
  }, []);

  // Base layer (susceptibility or forecast).
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    setStatus("loading");
    const fetcher = baseMode === "forecast" ? getPredict(scenario) : getSegments();
    fetcher
      .then((fc) => {
        if (baseLayerRef.current) { map.removeLayer(baseLayerRef.current); baseLayerRef.current = null; }
        const layer = L.geoJSON(fc, {
          renderer: rendererRef.current,
          style: (f) => ({
            color: baseMode === "forecast" ? depthColor(f.properties.peak_depth)
                                           : riskColor(f.properties.susceptibility),
            weight: 2, opacity: 0.85,
          }),
          onEachFeature: (f, lyr) => {
            lyr.on("click", () => { if (!pickModeRef.current) onSelectRef.current && onSelectRef.current(f.properties); });
          },
        }).addTo(map);
        baseLayerRef.current = layer;
        setCount(fc.features.length);
        setStatus("ready");
        if (!fittedRef.current) {
          try { map.fitBounds(layer.getBounds(), { padding: [20, 20] }); fittedRef.current = true; } catch (_) {}
        }
      })
      .catch((e) => setStatus("error: " + e.message));
  }, [baseMode, scenario]);

  // Observed flood-label overlay.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (labelLayerRef.current) { map.removeLayer(labelLayerRef.current); labelLayerRef.current = null; }
    setFloodCount(null);
    if (!event) return;
    getLabels(event)
      .then((fc) => {
        const layer = L.geoJSON(fc, {
          renderer: rendererRef.current,
          style: () => ({ color: "#1565c0", weight: 4, opacity: 0.95 }),
        }).addTo(map);
        labelLayerRef.current = layer;
        setFloodCount(fc.features.length);
      })
      .catch(() => setFloodCount(0));
  }, [event]);

  // Origin/destination markers.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (markerLayerRef.current) { map.removeLayer(markerLayerRef.current); }
    const g = L.layerGroup();
    const dot = (ll, color, label) => L.circleMarker(ll, { radius: 7, color, fillColor: color, fillOpacity: 1 })
      .bindTooltip(label, { permanent: false });
    if (origin) dot(origin, "#27ae60", "Origin").addTo(g);
    if (dest) dot(dest, "#e67e22", "Destination").addTo(g);
    g.addTo(map);
    markerLayerRef.current = g;
  }, [origin, dest]);

  // Route polylines (safe = solid blue, shortest = dashed grey).
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (routeLayerRef.current) { map.removeLayer(routeLayerRef.current); routeLayerRef.current = null; }
    if (!route) return;
    const g = L.layerGroup();
    const toLL = (coords) => coords.map(([lng, lat]) => [lat, lng]);
    if (route.shortest_route?.geometry)
      L.polyline(toLL(route.shortest_route.geometry.coordinates), { color: "#888", weight: 4, dashArray: "6 6", opacity: 0.8 }).addTo(g);
    if (route.safe_route?.geometry)
      L.polyline(toLL(route.safe_route.geometry.coordinates), { color: "#1565c0", weight: 4, opacity: 0.9 }).addTo(g);
    g.addTo(map);
    routeLayerRef.current = g;
    try { map.fitBounds(g.getBounds(), { padding: [30, 30] }); } catch (_) {}
  }, [route]);

  const forecast = baseMode === "forecast";
  return (
    <div style={{ position: "relative", height: "100%" }}>
      <div ref={containerRef} style={{ height: "100%", width: "100%", cursor: pickMode ? "crosshair" : "" }} />
      <div style={legendStyle}>
        <strong>{forecast ? "Predicted peak depth" : "Flood susceptibility"}</strong>
        <div style={{ color: "#666", fontSize: 11, margin: "2px 0 6px" }}>
          {forecast ? `FRF · ${scenario === "live" ? "live forecast" : scenario}` : "heuristic (pre-model)"} · {count.toLocaleString()} segs
        </div>
        {forecast ? <DepthLegend /> : <SuscLegend />}
        {event && (
          <div style={{ marginTop: 8, fontSize: 12 }}>
            <span style={{ display: "inline-block", width: 12, height: 4, background: "#1565c0", marginRight: 6, verticalAlign: "middle" }} />
            observed flooded ({floodCount == null ? "…" : floodCount.toLocaleString()})
          </div>
        )}
        {status !== "ready" && <div style={{ fontSize: 11, marginTop: 4 }}>{status}</div>}
      </div>
    </div>
  );
}

function SuscLegend() {
  return (<>
    <div style={{ height: 10, borderRadius: 5, background: `linear-gradient(90deg, ${riskColor(0)}, ${riskColor(0.5)}, ${riskColor(1)})` }} />
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}><span>low</span><span>high</span></div>
  </>);
}
function DepthLegend() {
  const stops = [["<0.1", "#2ecc71"], ["0.15", "#f1c40f"], ["0.25", "#e67e22"], ["0.4", "#e74c3c"], [">0.4", "#8e1a1a"]];
  return (
    <div style={{ display: "flex", gap: 2, fontSize: 10 }}>
      {stops.map(([lbl, c]) => (
        <div key={lbl} style={{ flex: 1, textAlign: "center" }}>
          <div style={{ height: 10, background: c, borderRadius: 2 }} />{lbl}
        </div>
      ))}
    </div>
  );
}

const legendStyle = {
  position: "absolute", bottom: 18, right: 12, zIndex: 1000, background: "white",
  padding: "8px 10px", borderRadius: 8, boxShadow: "0 1px 4px rgba(0,0,0,0.3)",
  width: 190, fontFamily: "system-ui, sans-serif", fontSize: 12,
};
