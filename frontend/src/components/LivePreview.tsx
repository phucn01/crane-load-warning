import { apiUrl } from "../services/api";

export default function LivePreview({ streamUrl }: { streamUrl: string }) {
  return (
    <figure className="live-preview">
      <div className="video-frame">
        <img src={apiUrl(streamUrl) || undefined} alt="Latest annotated processing frame" />
        <span className="preview-label">Processing Preview</span>
      </div>
      <figcaption>
        This shows the latest completed inference frame, not original-FPS playback.
      </figcaption>
    </figure>
  );
}
