import { useRef, useState, type ChangeEvent, type DragEvent } from "react";

export type MediaType = "image" | "video";

const IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png"]);
const VIDEO_EXTENSIONS = new Set(["mp4", "mov", "avi", "mkv", "webm"]);
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
const MAX_VIDEO_BYTES = 500 * 1024 * 1024;

interface Props {
  disabled: boolean;
  onSelect: (file: File, mediaType: MediaType) => void;
  onInvalid: (message: string) => void;
}

export default function MediaUploader({ disabled, onSelect, onInvalid }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const unavailable = disabled || detecting;

  const chooseFile = async (file: File | undefined) => {
    if (!file || unavailable) return;
    if (file.size === 0) {
      onInvalid("The selected file is empty.");
      return;
    }
    if (file.size > MAX_VIDEO_BYTES) {
      onInvalid("The selected file exceeds the 500 MB upload limit.");
      return;
    }

    setDetecting(true);
    try {
      const mediaType = await detectMediaType(file);
      if (mediaType === "image" && file.size > MAX_IMAGE_BYTES) {
        onInvalid("The selected image exceeds the 20 MB upload limit.");
        return;
      }
      onSelect(file, mediaType);
    } catch {
      onInvalid(
        "Could not identify this file. Choose JPG, PNG, MP4, MOV, AVI, MKV, or WebM.",
      );
    } finally {
      setDetecting(false);
    }
  };

  const handleInput = (event: ChangeEvent<HTMLInputElement>) => {
    void chooseFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (!unavailable) void chooseFile(event.dataTransfer.files?.[0]);
  };

  return (
    <div
      className={`drop-zone smart-drop ${dragging ? "is-dragging" : ""} ${unavailable ? "is-disabled" : ""}`}
      onDragEnter={(event) => {
        event.preventDefault();
        if (!unavailable) setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        className="visually-hidden"
        aria-label="Choose an image or video"
        type="file"
        accept=".jpg,.jpeg,.png,.mp4,.mov,.avi,.mkv,.webm,image/*,video/*"
        onChange={handleInput}
        disabled={unavailable}
      />
      <div className="upload-mark" aria-hidden="true">+</div>
      <div>
        <p className="drop-title">Drop one worksite image or video</p>
        <p className="drop-copy">
          Auto-detect JPG, PNG, MP4, MOV, AVI, MKV, or WebM
        </p>
      </div>
      <div className="media-capabilities" aria-label="Supported media">
        <span>Image assessment</span>
        <span>Video processing</span>
      </div>
      <button
        className="button button-primary"
        type="button"
        disabled={unavailable}
        onClick={() => inputRef.current?.click()}
      >
        {detecting ? "Detecting media..." : "Choose image or video"}
      </button>
    </div>
  );
}

export async function detectMediaType(file: File): Promise<MediaType> {
  const header = await readHeader(file);
  if (isJpeg(header) || isPng(header)) return "image";
  if (isIsoBaseMedia(header) || isAvi(header) || isEbml(header)) return "video";

  const mime = file.type.toLowerCase();
  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("video/")) return "video";

  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  if (IMAGE_EXTENSIONS.has(extension)) return "image";
  if (VIDEO_EXTENSIONS.has(extension)) return "video";
  throw new TypeError("unsupported media type");
}

async function readHeader(file: File): Promise<Uint8Array> {
  const blob = file.slice(0, 16);
  try {
    if (typeof blob.arrayBuffer === "function") {
      return new Uint8Array(await blob.arrayBuffer());
    }
    return await readWithFileReader(blob);
  } catch {
    return new Uint8Array();
  }
}

function readWithFileReader(blob: Blob): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.readAsArrayBuffer(blob);
  });
}

function isJpeg(bytes: Uint8Array): boolean {
  return bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
}

function isPng(bytes: Uint8Array): boolean {
  return bytes.length >= 8 && [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
    .every((value, index) => bytes[index] === value);
}

function isIsoBaseMedia(bytes: Uint8Array): boolean {
  return bytes.length >= 8 && String.fromCharCode(...bytes.slice(4, 8)) === "ftyp";
}

function isAvi(bytes: Uint8Array): boolean {
  return bytes.length >= 12
    && String.fromCharCode(...bytes.slice(0, 4)) === "RIFF"
    && String.fromCharCode(...bytes.slice(8, 12)) === "AVI ";
}

function isEbml(bytes: Uint8Array): boolean {
  return bytes.length >= 4
    && bytes[0] === 0x1a
    && bytes[1] === 0x45
    && bytes[2] === 0xdf
    && bytes[3] === 0xa3;
}
