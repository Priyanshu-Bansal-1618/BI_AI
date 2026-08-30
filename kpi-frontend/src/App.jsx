import { useState } from "react";
import KPIStoryCard from "./KPIStoryCard";

export default function App() {
  const [story, setStory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchStory = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/kpi-story", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kpi_id: "kpi.revenue",
          as_of_date: (() => {
          const d = new Date();
          return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
          })(), // local date, matches Python's date.today()
          user_role: "Executive",
        }),
      });
      const data = await res.json();
      if (data.status === "no_material_anomaly") {
        setError("No anomaly found for today's date — check the date matches what seed_data.py printed.");
      } else {
        setStory(data);
      }
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  const handleFeedback = async (payload) => {
    await fetch("http://127.0.0.1:8000/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center gap-6 p-8">
      <button
        onClick={fetchStory}
        className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700"
      >
        {loading ? "Loading..." : "Generate KPI Story"}
      </button>
      {error && <p className="text-rose-600 text-sm">{error}</p>}
      {story && <KPIStoryCard story={story} onFeedback={handleFeedback} />}
    </div>
  );
}