import React, { useEffect, useRef } from "react";
import uPlot from "uplot";

/**
 * Rolling live chart. Keeps the last `windowSize` samples per series and
 * redraws efficiently via uPlot (handles high-frequency data far better
 * than SVG-based chart libraries at 100-200Hz).
 *
 * seriesData: { t: number[], values: { [label]: number[] } }
 */
export default function LiveChart({ title, seriesData, colors, unit }) {
  const containerRef = useRef(null);
  const plotRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const labels = Object.keys(seriesData.values);
    const data = [seriesData.t, ...labels.map((l) => seriesData.values[l])];

    if (!plotRef.current) {
      const opts = {
        title: "",
        width: containerRef.current.clientWidth,
        height: 220,
        series: [
          {},
          ...labels.map((label, i) => ({
            label,
            stroke: colors?.[i] ?? "#5b9dff",
            width: 1.5,
            points: { show: false },
          })),
        ],
        axes: [
          { stroke: "#8b93a7", grid: { stroke: "#232838" } },
          { stroke: "#8b93a7", grid: { stroke: "#232838" }, label: unit },
        ],
        legend: { show: true },
        scales: { x: { time: false } },
      };
      plotRef.current = new uPlot(opts, data, containerRef.current);
    } else {
      plotRef.current.setData(data);
    }

    return () => {};
  }, [seriesData, colors, unit]);

  // Follow the card's width, e.g. when the assistance panel appears beside
  // the charts after they were first drawn
  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      const plot = plotRef.current;
      const width = el.clientWidth;
      if (plot && width > 0 && width !== plot.width) {
        plot.setSize({ width, height: plot.height });
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    return () => {
      if (plotRef.current) {
        plotRef.current.destroy();
        plotRef.current = null;
      }
    };
  }, []);

  return (
    <div className="card">
      <h3>{title}</h3>
      <div ref={containerRef} style={{ overflow: "hidden" }} />
    </div>
  );
}
