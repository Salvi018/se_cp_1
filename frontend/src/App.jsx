import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Home    from "./pages/Home";
import History from "./pages/History";
import Charts  from "./pages/Charts";

export default function App() {
  const linkClass = ({ isActive }) =>
    `px-4 py-2 rounded-lg text-sm font-medium transition ${
      isActive ? "bg-indigo-600 text-white" : "text-gray-600 hover:bg-gray-100"
    }`;

  return (
    <BrowserRouter>
      <nav className="bg-white border-b shadow-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-2">
          <span className="font-extrabold text-indigo-700 text-lg mr-4">⚙ SDLC Selector</span>
          <NavLink to="/"        className={linkClass}>Predict</NavLink>
          <NavLink to="/charts"  className={linkClass}>Charts</NavLink>
          <NavLink to="/history" className={linkClass}>History</NavLink>
        </div>
      </nav>
      <main className="min-h-screen bg-gray-50">
        <Routes>
          <Route path="/"        element={<Home />} />
          <Route path="/charts"  element={<Charts />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
