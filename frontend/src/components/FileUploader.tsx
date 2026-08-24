import { useRef, useState, type ChangeEvent, type DragEvent } from "react";

const SUPPORTED_TYPES = new Set(["image/jpeg", "image/png"]);
const MAX_FILE_BYTES = 20 * 1024 * 1024;

interface FileUploaderProps {
  disabled: boolean;
  selectedFile: File | null;
  onSelect: (file: File) => void;
  onInvalid: (message: string) => void;
}

export default function FileUploader({
  disabled,
  selectedFile,
  onSelect,
  onInvalid,
}: FileUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const chooseFile = (file: File | undefined) => {
    if (!file) return;
    if (!SUPPORTED_TYPES.has(file.type)) {
      onInvalid("Please choose a JPG, JPEG, or PNG image.");
      return;
    }
    if (file.size === 0) {
      onInvalid("The selected image is empty.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      onInvalid("The selected image exceeds the 20 MB upload limit.");
      return;
    }
    onSelect(file);
  };

  const handleInput = (event: ChangeEvent<HTMLInputElement>) => {
    chooseFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (!disabled) chooseFile(event.dataTransfer.files?.[0]);
  };

  return (
    <div
      className={`drop-zone ${dragging ? "is-dragging" : ""} ${disabled ? "is-disabled" : ""}`}
      onDragEnter={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        onChange={handleInput}
        disabled={disabled}
      />
      <div className="upload-mark" aria-hidden="true">＋</div>
      <div>
        <p className="drop-title">
          {selectedFile ? "Replace inspection image" : "Drop an inspection image"}
        </p>
        <p className="drop-copy">JPG or PNG · up to 20 MB</p>
      </div>
      <button
        className="button button-secondary"
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
      >
        Browse files
      </button>
    </div>
  );
}
