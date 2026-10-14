import { useRef, useState } from "react";
import Spinner from "./Spinner";

/**
 * File input, used for both a dataset zip and a single ECG image.
 *
 * Generalised rather than duplicated: the two uses differ only in what they
 * accept and what they say, and a second near-identical dropzone is a second
 * place for the drag handling to drift.
 *
 * On the dataset screen this takes a .zip only. The design says "folder or
 * .zip", but POST /datasets takes a zip and building one in the browser needs
 * a compression library we have not added — so the copy says what works. A
 * dropzone that silently ignores a dropped folder is worse than one that never
 * offered.
 */
export default function Dropzone({
  onFile,
  busy,
  disabled,
  accept = ".zip,application/zip",
  match = /\.zip$/i,
  title = "Drop a .zip here",
  busyTitle = "Reading your images",
  hint = "One folder per class inside the zip · up to 1000 images · JPG or PNG",
  busyHint = "Unzipping, grouping by folder, and rendering four previews",
  rejectHint = "That is not a .zip. Compress the folder first.",
  compact = false,
}) {
  const input = useRef(null);
  const [over, setOver] = useState(false);
  const [reject, setReject] = useState(null);

  const take = (file) => {
    if (!file) return;
    if (match && !match.test(file.name)) {
      setReject(`${file.name} — ${rejectHint}`);
      return;
    }
    setReject(null);
    onFile(file);
  };

  return (
    <div>
      <button
        type="button"
        disabled={busy || disabled}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files?.[0]);
        }}
        className={`flex w-full flex-col items-center justify-center gap-2 rounded-[10px]
          border border-dashed transition-colors
          ${compact ? "px-5 py-8" : "px-6 py-12"}
          ${over ? "border-quantum bg-quantum-soft" : "border-quantum bg-white"}
          ${busy || disabled ? "cursor-not-allowed opacity-60" : "hover:bg-quantum-soft/50"}`}
      >
        <span className="grid h-11 w-11 place-items-center rounded-full bg-quantum-soft">
          {busy ? (
            <Spinner size={19} className="text-quantum" />
          ) : (
            <svg width="19" height="19" viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M12 18V6M6 12l6-6 6 6"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-quantum"
              />
            </svg>
          )}
        </span>
        <span className="text-[15px] font-semibold text-ink">
          {busy ? busyTitle : title}
        </span>
        <span className="max-w-[42ch] text-center text-[11.5px] text-muted">
          {busy ? busyHint : hint}
        </span>
        {!busy && (
          <span className="mt-1.5 rounded-[7px] border border-rule px-3 py-1.5 text-[12.5px] text-body">
            Browse files
          </span>
        )}
      </button>

      <input
        ref={input}
        type="file"
        accept={accept}
        className="sr-only"
        onChange={(e) => take(e.target.files?.[0])}
      />

      {reject && (
        <p role="alert" className="mt-2 text-[11.5px] text-alert">
          {reject}
        </p>
      )}
    </div>
  );
}
