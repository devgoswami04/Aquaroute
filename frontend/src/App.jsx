import { useEffect, useState } from "react";
import MapView from "./components/MapView";
import DepthChart from "./components/DepthChart";
import RoutePanel from "./components/RoutePanel";
import CivicDashboard from "./components/CivicDashboard";
import { getHealth, getEvents, getVehicles, postRoute } from "./api";

const slugify = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

// Phase 6 UI: choose a base layer (static susceptibility, or live/replayed FRF
// forecast coloured by predicted peak depth); optionally overlay observed flood
// labels; click a segment to inspect features and its predicted depth-vs-time curve.
export default function App() {
  const [health, setHealth] = useState(null);
  const [selected, setSelected] = useState(null);
  const [events, setEvents] = useState([]);
  const [event, setEvent] = useState("");        // observed-label overlay
  const [view, setView] = useState("map");                    // "map" | "dashboard"
  const [baseMode, setBaseMode] = useState("susceptibility"); // or "forecast"
  const [scenario, setScenario] = useState("live");

  // Routing state (Phase 8)
  const [vehicles, setVehicles] = useState({});
  const [vehicle, setVehicle] = useState("two_wheeler");
  const [depart, setDepart] = useState(18);
  const [routeScenario, setRouteScenario] = useState("2021_chennai_floods");
  const [origin, setOrigin] = useState(null);
  const [dest, setDest] = useState(null);
  const [pickMode, setPickMode] = useState(null);
  const [route, setRoute] = useState(null);
  const [routing, setRouting] = useState(false);
  const [routeError, setRouteError] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getEvents().then(setEvents).catch(() => {});
    getVehicles().then(setVehicles).catch(() => {});
  }, []);

  const handleMapClick = (ll) => {
    if (pickMode === "origin") setOrigin(ll);
    else if (pickMode === "dest") setDest(ll);
    setPickMode(null);
  };

  const findRoute = async () => {
    if (!origin || !dest) return;
    setRouting(true); setRouteError(null);
    try {
      const r = await postRoute({
        origin: [origin.lng, origin.lat], dest: [dest.lng, dest.lat],
        vehicle, depart, scenario: routeScenario,
      });
      setRoute(r);
    } catch (e) { setRouteError(String(e)); }
    finally { setRouting(false); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", fontFamily: "system-ui, sans-serif" }}>
      <header style={{ padding: "10px 16px", background: "#0b3d5c", color: "white" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
          <h1 style={{ margin: 0, fontSize: 20 }}>AquaRoute</h1>
          <span style={{ opacity: 0.8, fontSize: 12 }}>Chennai corridor · predictive flood intelligence</span>

          <div style={{ display: "flex", gap: 0, borderRadius: 6, overflow: "hidden", border: "1px solid rgba(255,255,255,0.4)" }}>
            {["map", "dashboard"].map((v) => (
              <button key={v} onClick={() => setView(v)}
                style={{ fontSize: 12, padding: "4px 10px", border: "none", cursor: "pointer",
                         background: view === v ? "white" : "transparent",
                         color: view === v ? "#0b3d5c" : "white", textTransform: "capitalize" }}>
                {v}
              </button>
            ))}
          </div>

          {view === "map" && (
          <label style={{ fontSize: 13 }}>
            Layer:{" "}
            <select value={baseMode} onChange={(e) => setBaseMode(e.target.value)} style={sel}>
              <option value="susceptibility">Susceptibility (static)</option>
              <option value="forecast">Forecast (FRF)</option>
            </select>
          </label>
          )}

          {view === "map" && baseMode === "forecast" && (
            <label style={{ fontSize: 13 }}>
              Scenario:{" "}
              <select value={scenario} onChange={(e) => setScenario(e.target.value)} style={sel}>
                <option value="live">Live forecast</option>
                {events.map((e) => (
                  <option key={e.name} value={slugify(e.name)}>replay {e.name}</option>
                ))}
              </select>
            </label>
          )}

          {view === "map" && (
          <label style={{ fontSize: 13 }}>
            Observed overlay:{" "}
            <select value={event} onChange={(e) => setEvent(e.target.value)} style={sel}>
              <option value="">— none —</option>
              {events.map((e) => (
                <option key={e.name} value={e.name} disabled={!e.labelled}>
                  {e.name}{e.labelled ? "" : " (no labels)"}
                </option>
              ))}
            </select>
          </label>
          )}

          <span style={{ marginLeft: "auto", fontSize: 12, opacity: 0.8 }}>
            {health ? `API ${health.version} · ok` : "API …"}
          </span>
        </div>
      </header>

      {view === "dashboard" ? (
        <div style={{ flex: 1, minHeight: 0 }}><CivicDashboard /></div>
      ) : (
      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <main style={{ flex: 1, minWidth: 0 }}>
          <MapView onSelect={setSelected} event={event} baseMode={baseMode} scenario={scenario}
            origin={origin} dest={dest} route={route} pickMode={pickMode} onMapClick={handleMapClick} />
        </main>

        <aside style={{ width: 340, borderLeft: "1px solid #ddd", padding: 16, overflowY: "auto" }}>
          <h2 style={{ marginTop: 0, fontSize: 16 }}>Segment inspector</h2>
          {!selected && <p style={{ color: "#777", fontSize: 14 }}>Click a road segment on the map.</p>}
          {selected && baseMode === "forecast" && (
            <DepthChart segmentId={selected.segment_id} scenario={scenario} />
          )}
          {selected && <SegmentDetails p={selected} />}

          <RoutePanel
            vehicles={vehicles} vehicle={vehicle} setVehicle={setVehicle}
            depart={depart} setDepart={setDepart}
            scenario={routeScenario} setScenario={setRouteScenario} events={events}
            origin={origin} dest={dest} pickMode={pickMode} setPickMode={setPickMode}
            onFindRoute={findRoute} route={route} routing={routing} error={routeError} />
        </aside>
      </div>
      )}
    </div>
  );
}

const sel = { fontSize: 13, padding: "2px 6px" };

function SegmentDetails({ p }) {
  const rows = [
    ["Segment ID", p.segment_id],
    ["Road class", p.road_class],
    ["Length (m)", fmt(p.length_m, 1)],
    ["Underpass", p.is_underpass ? "yes" : p.is_underpass === false ? "no" : "—"],
    ["Elevation (m)", fmt(p.elevation, 1)],
    ["Slope (rad)", fmt(p.slope, 3)],
    ["TWI", fmt(p.twi, 2)],
    ["Depression depth (m)", fmt(p.depression_depth, 2)],
    ["Upstream area (m²)", fmt(p.upstream_area, 0)],
    ["Imperviousness", fmt(p.imperviousness, 2)],
    ["Susceptibility", fmt(p.susceptibility, 2)],
    ...(p.peak_depth !== undefined ? [["— forecast —", ""]] : []),
    ...(p.peak_depth !== undefined ? [["Peak depth (m)", fmt(p.peak_depth, 2)]] : []),
    ...(p.onset !== undefined ? [["Onset (h)", p.onset >= 0 ? p.onset : "—"]] : []),
    ...(p.clearance !== undefined ? [["Clearance (h)", p.clearance >= 0 ? p.clearance : "—"]] : []),
    ...(p.flooded !== undefined ? [["Observed flooded", p.flooded ? "yes" : "no"]] : []),
    ...(p.depth_proxy !== undefined ? [["Observed depth proxy (m)", fmt(p.depth_proxy, 2)]] : []),
    ...(p.source !== undefined ? [["Label source", p.source]] : []),
  ];
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, marginTop: 8 }}>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td style={{ color: "#555", padding: "4px 6px", borderBottom: "1px solid #eee" }}>{k}</td>
            <td style={{ padding: "4px 6px", borderBottom: "1px solid #eee", textAlign: "right" }}>{v ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function fmt(v, d) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(d);
}
