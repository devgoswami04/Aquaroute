import { useEffect, useState } from "react";
import { getCivicSummary } from "../api";

// Phase 9: civic decision-support dashboard — chronic-flood ranking (drainage
// prioritisation evidence), predicted-vs-observed agreement, and self-calibration
// status. Not a navigation clone: a planner's view.
export default function CivicDashboard() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    getCivicSummary(15).then(setData).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div style={pad}><p style={{ color: "#b00" }}>Failed to load: {err}</p></div>;
  if (!data) return <div style={pad}><p style={{ color: "#777" }}>Building civic summary… (first load runs the model — ~30 s)</p></div>;

  const { corridor, chronic, predicted_vs_observed: pvo, calibration: cal } = data;

  return (
    <div style={{ ...pad, overflowY: "auto", height: "100%" }}>
      <h2 style={{ margin: "0 0 4px" }}>Civic decision dashboard</h2>
      <div style={{ color: "#555", fontSize: 13, marginBottom: 16 }}>
        {corridor.place} · {corridor.segments.toLocaleString()} road segments · events: {corridor.events.join(", ")}
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 20 }}>
        <Card title="Self-calibration status">
          <div style={{ fontSize: 14, marginBottom: 8 }}>{cal.message}</div>
          <Stat label="Segments tracked" value={cal.segments_tracked} />
          <Stat label="Predictions retired" value={cal.segments_retired} />
          <Stat label="Mean correction α" value={cal.mean_alpha} />
        </Card>

        <Card title="Predicted vs observed">
          {!pvo.available && <div style={{ fontSize: 13, color: "#777" }}>Model not available.</div>}
          {pvo.available && (
            <>
              <div style={{ fontSize: 22, fontWeight: 700, color: "#0b3d5c" }}>
                F1 {pvo.mean_f1}
              </div>
              <div style={{ fontSize: 12, color: "#666", marginBottom: 8 }}>mean across events (FRF vs SAR/report labels)</div>
              <table style={{ fontSize: 12, borderCollapse: "collapse", width: "100%" }}>
                <thead><tr><th></th><th style={th}>F1</th><th style={th}>Prec</th><th style={th}>Rec</th></tr></thead>
                <tbody>
                  {Object.entries(pvo.per_event).map(([ev, m]) => (
                    <tr key={ev}>
                      <td style={{ padding: "2px 4px" }}>{ev.replace(/_/g, " ")}</td>
                      <td style={td}>{m.f1}</td><td style={td}>{m.precision}</td><td style={td}>{m.recall}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </Card>
      </div>

      <h3 style={{ margin: "0 0 8px" }}>Chronic-flood segments — drainage prioritisation</h3>
      <div style={{ color: "#666", fontSize: 12, marginBottom: 8 }}>
        Ranked by how many of the {chronic.n_events} historical events each road flooded in. These are the
        evidence-based candidates for drainage works.
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr>
            <th style={thL}>#</th><th style={thL}>Segment</th><th style={thL}>Class</th>
            <th style={thL}>Flood frequency</th><th style={th}>Depth proxy (m)</th>
          </tr>
        </thead>
        <tbody>
          {chronic.segments.map((s, i) => (
            <tr key={s.segment_id}>
              <td style={tdc}>{i + 1}</td>
              <td style={{ ...tdc, fontFamily: "monospace", fontSize: 11 }}>{s.segment_id}</td>
              <td style={tdc}>{s.road_class}</td>
              <td style={tdc}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ background: "#e9eef2", borderRadius: 4, height: 10, width: 90 }}>
                    <div style={{ width: `${s.chronic_score * 100}%`, background: "#0b3d5c", height: 10, borderRadius: 4 }} />
                  </div>
                  <span style={{ fontSize: 12 }}>{s.events_flooded}/{s.n_events}</span>
                </div>
              </td>
              <td style={{ ...tdc, textAlign: "right" }}>{s.mean_depth_proxy.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const pad = { padding: 20, fontFamily: "system-ui, sans-serif" };
const th = { textAlign: "right", padding: "2px 6px", color: "#555" };
const thL = { textAlign: "left", padding: "4px 6px", color: "#555", borderBottom: "2px solid #ddd" };
const td = { textAlign: "right", padding: "2px 6px" };
const tdc = { padding: "5px 6px", borderBottom: "1px solid #eee" };

function Card({ title, children }) {
  return (
    <div style={{ flex: "1 1 300px", background: "#fafbfc", border: "1px solid #e3e7ea", borderRadius: 10, padding: 14 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: "#0b3d5c", marginBottom: 8 }}>{title}</div>
      {children}
    </div>
  );
}
function Stat({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, padding: "2px 0" }}>
      <span style={{ color: "#555" }}>{label}</span><b>{value ?? "—"}</b>
    </div>
  );
}
