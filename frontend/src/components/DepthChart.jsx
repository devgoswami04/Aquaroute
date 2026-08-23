import { useEffect, useState } from "react";
import {
  Bar, CartesianGrid, ComposedChart, Line, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { getSegmentCurve } from "../api";

// Predicted depth-vs-time curve for a segment, with the driving rainfall
// hyetograph and onset/peak/clearance markers (Phase 6).
export default function DepthChart({ segmentId, scenario }) {
  const [curve, setCurve] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!segmentId) return;
    setCurve(null); setErr(null);
    getSegmentCurve(segmentId, scenario).then(setCurve).catch((e) => setErr(String(e)));
  }, [segmentId, scenario]);

  if (err) return <p style={{ color: "#b00", fontSize: 12 }}>No curve: {err}</p>;
  if (!curve) return <p style={{ color: "#777", fontSize: 12 }}>Loading curve…</p>;

  const data = curve.t.map((t, i) => ({ t, depth: curve.depth[i], rain: curve.hyetograph[i] }));

  return (
    <div>
      <div style={{ fontSize: 13, margin: "8px 0" }}>
        <b>Predicted depth-vs-time</b> ({scenario === "live" ? "live forecast" : scenario})
      </div>
      <div style={{ fontSize: 12, color: "#444", marginBottom: 6 }}>
        onset <b>{fmtH(curve.onset)}</b> · peak <b>{curve.peak}h</b> @ <b>{curve.peak_depth.toFixed(2)} m</b> · clearance <b>{fmtH(curve.clearance)}</b>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 4, left: -18 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="t" tick={{ fontSize: 11 }} label={{ value: "hours", position: "insideBottom", fontSize: 11, dy: 10 }} />
          <YAxis yAxisId="d" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="r" orientation="right" tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v, n) => [Number(v).toFixed(2), n === "depth" ? "depth (m)" : "rain (mm)"]} labelFormatter={(l) => `t=${l}h`} />
          <Bar yAxisId="r" dataKey="rain" fill="#bcd4f0" name="rain" />
          <Line yAxisId="d" type="monotone" dataKey="depth" stroke="#0b3d5c" strokeWidth={2} dot={false} name="depth" />
          <ReferenceLine yAxisId="d" y={0.1} stroke="#c0392b" strokeDasharray="4 4" />
          {curve.onset != null && <ReferenceLine yAxisId="d" x={curve.onset} stroke="#27ae60" />}
          {curve.clearance != null && <ReferenceLine yAxisId="d" x={curve.clearance} stroke="#e67e22" />}
        </ComposedChart>
      </ResponsiveContainer>
      <div style={{ fontSize: 11, color: "#888" }}>
        red dashed = 0.1 m onset threshold · green = onset · orange = clearance · bars = rainfall
      </div>
    </div>
  );
}

function fmtH(h) {
  return h == null ? "—" : `${h}h`;
}
