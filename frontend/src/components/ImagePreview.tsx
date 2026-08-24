interface ImagePreviewProps {
  file: File;
  previewUrl: string;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function ImagePreview({ file, previewUrl }: ImagePreviewProps) {
  return (
    <figure className="preview-card">
      <div className="preview-frame">
        <img src={previewUrl} alt={`Local preview of ${file.name}`} />
        <span className="preview-label">Local preview</span>
      </div>
      <figcaption>
        <span className="file-name" title={file.name}>{file.name}</span>
        <span className="file-size">{formatBytes(file.size)}</span>
      </figcaption>
    </figure>
  );
}
