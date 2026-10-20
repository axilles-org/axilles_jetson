import React, { useEffect, useRef } from "react";
import uPlot from "uplot";

/** Static multi-series line chart. series: [{ label, data, color, dash? }] */
export default function LineChart({ x, series, xLabel, yLabel, height = 260 }) {
  const containerRef = useRef(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || x.length === 0) return undefined;
    const plot = new uPlot(
      {
        width: el.clientWidth,
        height,
        scales: { x: { time: false } },
        series: [
          { label: xLabel },
          ...series.map((s) => ({
            label: s.label,
            stroke: s.color,
            width: 2,
            dash: s.dash,
            points: { show: false },
          })),
        ],
        axes: [
          { stroke: "#8b93a7", grid: { stroke: "#232838" }, label: xLabel },
          { stroke: "#8b93a7", grid: { stroke: "#232838" }, label: yLabel },
        ],
      },
      [x, ...series.map((s) => s.data)],
      el
    );
    return () => plot.destroy();
  }, [x, series, xLabel, yLabel, height]);

  return <div ref={containerRef} />;
}
