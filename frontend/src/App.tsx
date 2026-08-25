import DetectionPage from "./pages/DetectionPage";
import VideoReportPage from "./pages/VideoReportPage";

export default function App() {
  const reportJobId = new URLSearchParams(window.location.search).get("report");
  if (reportJobId) return <VideoReportPage jobId={reportJobId} />;
  return <DetectionPage />;
}
