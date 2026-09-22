import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import LivePage from "./pages/LivePage.jsx";
import RunsPage from "./pages/RunsPage.jsx";
import RunDetailPage from "./pages/RunDetailPage.jsx";
import LaunchPage from "./pages/LaunchPage.jsx";
import "./style.css";

function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <nav className="topnav">
          <span className="brand">Axilles Exo Dashboard</span>
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
            Live
          </NavLink>
          <NavLink to="/launch" className={({ isActive }) => (isActive ? "active" : "")}>
            Launch
          </NavLink>
          <NavLink to="/runs" className={({ isActive }) => (isActive ? "active" : "")}>
            Runs
          </NavLink>
        </nav>
        <main className="content">
          <Routes>
            <Route path="/" element={<LivePage />} />
            <Route path="/launch" element={<LaunchPage />} />
            <Route path="/runs" element={<RunsPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
