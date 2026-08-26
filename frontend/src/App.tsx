import DetectionPage from "./pages/DetectionPage";
import HistoryPage from "./pages/HistoryPage";
import VideoReportPage from "./pages/VideoReportPage";

export default function App() {
  const reportJobId = new URLSearchParams(window.location.search).get("report");
  if (reportJobId) return <VideoReportPage jobId={reportJobId} />;
  if (new URLSearchParams(window.location.search).has("history")) return <HistoryPage />;
  return <DetectionPage />;
}
