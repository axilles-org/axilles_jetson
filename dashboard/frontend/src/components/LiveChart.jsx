import React, { useEffect, useRef } from "react";
import uPlot from "uplot";

/**
 * Rolling live chart (uPlot handles 100-200 Hz far better than SVG charts).
 *
 * series: [{ label, color, right? }]  — right: true puts it on a second y-axis
 * data:   [t[], ...one array per series]
 * The plot is rebuilt when the series list changes (e.g. a line appears once a
 * controller starts sending it); otherwise only its data is updated.
 */
export default function LiveChart({ title, subtitle, series, data, unit, rightUnit }) {
  const containerRef = useRef(null);
  const plotRef = useRef(null);
  const signatureRef = useRef("");

  const signature = series.map((s) => `${s.label}|${s.right ? 1 : 0}`).join(",");

  useEffect(() => {
    if (!containerRef.current) return;

    if (plotRef.current && signatureRef.current !== signature) {
      plotRef.current.destroy();
      plotRef.current = null;
    }

    if (!plotRef.current) {
      const hasRight = series.some((s) => s.right);
      const axes = [
        { stroke: "#8b93a7", grid: { stroke: "#232838" } },
        { stroke: "#8b93a7", grid: { stroke: "#232838" }, label: unit, scale: "y" },
      ];
      if (hasRight) {
        axes.push({ stroke: "#c792ea", grid: { show: false }, label: rightUnit, side: 1, scale: "y2" });
      }
      const opts = {
        title: "",
        width: containerRef.current.clientWidth,
        height: 220,
        series: [
          {},
          ...series.map((s) => ({
            label: s.label,
            stroke: s.color ?? "#5b9dff",
            width: 1.5,
            points: { show: false },
            scale: s.right ? "y2" : "y",
          })),
        ],
        axes,
        legend: { show: true },
        scales: { x: { time: false } },
      };
      plotRef.current = new uPlot(opts, data, containerRef.current);
      signatureRef.current = signature;
    } else {
      plotRef.current.setData(data);
    }
  }, [signature, data, unit, rightUnit]); // eslint-disable-line react-hooks/exhaustive-deps

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
      <h3>
        {title}
        {subtitle && <span className="chart-subtitle"> · {subtitle}</span>}
      </h3>
      <div ref={containerRef} style={{ overflow: "hidden" }} />
    </div>
  );
}
