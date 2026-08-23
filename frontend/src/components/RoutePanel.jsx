// Vehicle-aware routing controls + results (Phase 8). Origin/destination are set
// by clicking the map (pick mode); the panel calls /route and shows the safe vs
// shortest comparison and advisory.
export default function RoutePanel({
  vehicles, vehicle, setVehicle, depart, setDepart, scenario, setScenario, events,
  origin, dest, pickMode, setPickMode, onFindRoute, route, routing, error,
}) {
  const fmtLL = (ll) => (ll ? `${ll.lat.toFixed(4)}, ${ll.lng.toFixed(4)}` : "—");
  return (
    <div style={{ borderTop: "2px solid #0b3d5c", marginTop: 16, paddingTop: 12 }}>
      <h2 style={{ fontSize: 16, margin: "0 0 8px" }}>Routing</h2>

      <div style={{ display: "grid", gap: 8, fontSize: 13 }}>
        <label>Vehicle:{" "}
          <select value={vehicle} onChange={(e) => setVehicle(e.target.value)} style={sel}>
            {Object.entries(vehicles).map(([k, v]) => (
              <option key={k} value={k}>{k.replace("_", "-")} ({v} m)</option>
            ))}
          </select>
        </label>

        <label>Scenario:{" "}
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} style={sel}>
            <option value="live">Live forecast</option>
            {events.map((e) => (
              <option key={e.name} value={slugify(e.name)}>replay {e.name}</option>
            ))}
          </select>
        </label>

        <label>Depart hour: <b>{depart}:00</b>
          <input type="range" min={0} max={23} value={depart}
            onChange={(e) => setDepart(Number(e.target.value))} style={{ width: "100%" }} />
        </label>

        <div style={{ display: "flex", gap: 6 }}>
          <button onClick={() => setPickMode("origin")}
            style={{ ...btn, outline: pickMode === "origin" ? "2px solid #27ae60" : "none" }}>
            Origin: {fmtLL(origin)}
          </button>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <button onClick={() => setPickMode("dest")}
            style={{ ...btn, outline: pickMode === "dest" ? "2px solid #e67e22" : "none" }}>
            Dest: {fmtLL(dest)}
          </button>
        </div>
        {pickMode && <div style={{ fontSize: 12, color: "#0b3d5c" }}>Click the map to set the {pickMode}.</div>}

        <button onClick={onFindRoute} disabled={!origin || !dest || routing}
          style={{ ...btn, background: "#0b3d5c", color: "white", opacity: !origin || !dest ? 0.5 : 1 }}>
          {routing ? "Routing…" : "Find safe route"}
        </button>
        {error && <div style={{ color: "#b00", fontSize: 12 }}>{error}</div>}
      </div>

      {route && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 13, background: "#eef4f8", padding: "6px 8px", borderRadius: 6 }}>
            {route.advisory}
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, marginTop: 8 }}>
            <thead>
              <tr><th></th><th style={th}>safe</th><th style={th}>shortest</th></tr>
            </thead>
            <tbody>
              <Row k="Distance (m)" a={route.safe_route?.distance_m} b={route.shortest_route?.distance_m} />
              <Row k="Time (min)" a={route.safe_route?.time_min} b={route.shortest_route?.time_min} />
              <Row k="Impassable (m)" a={route.safe_route?.blocked_m} b={route.shortest_route?.blocked_m} />
              <Row k="Max depth (m)" a={route.safe_route?.max_depth_m} b={route.shortest_route?.max_depth_m} />
            </tbody>
          </table>
          <div style={{ fontSize: 11, color: "#666", marginTop: 4 }}>
            <span style={{ color: "#1565c0" }}>━</span> safe route ·
            <span style={{ color: "#888" }}> ┄</span> shortest route
          </div>
        </div>
      )}
    </div>
  );
}

const slugify = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
const sel = { fontSize: 13, padding: "2px 6px" };
const btn = { fontSize: 12, padding: "5px 8px", border: "1px solid #bbb", borderRadius: 6, background: "#f7f7f7", cursor: "pointer", textAlign: "left", width: "100%" };
const th = { textAlign: "right", padding: "2px 6px", color: "#555" };

function Row({ k, a, b }) {
  return (
    <tr>
      <td style={{ color: "#555", padding: "3px 6px", borderBottom: "1px solid #eee" }}>{k}</td>
      <td style={{ textAlign: "right", padding: "3px 6px", borderBottom: "1px solid #eee" }}>{fmt(a)}</td>
      <td style={{ textAlign: "right", padding: "3px 6px", borderBottom: "1px solid #eee", color: "#888" }}>{fmt(b)}</td>
    </tr>
  );
}
const fmt = (v) => (v == null ? "—" : Number(v).toLocaleString());
