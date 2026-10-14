import { useCallback, useEffect, useState } from "react";

import { ApiError, listDatasets, uploadDataset } from "../api";
import Card from "../components/Card";
import ClassChips from "../components/ClassChips";
import DatasetList from "../components/DatasetList";
import Dropzone from "../components/Dropzone";
import PreviewGrid from "../components/PreviewGrid";
import Spinner from "../components/Spinner";

/**
 * Screen 1 of 5.
 *
 * The whole screen renders from ONE response. `POST /datasets` returns class
 * names, per-class counts and four base64 thumbnails together, so there is no
 * second request and no loading shimmer after the upload resolves.
 *
 * Props
 *   onContinue(dataset)  called with the chosen DatasetMeta. Wire it to your
 *                        router; this component holds no routing opinion.
 */
export default function Upload({ onContinue }) {
  const [meta, setMeta] = useState(null); // the dataset just uploaded
  const [previews, setPreviews] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [existing, setExisting] = useState(null);
  const [listError, setListError] = useState(null);
  const [chosen, setChosen] = useState(null); // picked from the list instead

  const refresh = useCallback(async () => {
    try {
      setListError(null);
      setExisting(await listDatasets());
    } catch (e) {
      // Do NOT fall back to an empty array. An empty list and a failed request
      // look identical on screen but mean opposite things — "this machine has
      // no datasets" versus "I could not ask". Swallowing the error here sent
      // someone hunting for a missing dataset that was on disk the whole time.
      setExisting([]);
      setListError(
        e instanceof ApiError
          ? e.detail
          : "Could not reach the backend to list datasets.",
      );
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleFile = async (file) => {
    setBusy(true);
    setError(null);
    setChosen(null);
    try {
      const res = await uploadDataset(file, {
        name: file.name.replace(/\.zip$/i, ""),
      });
      const { preview_b64 = [], ...rest } = res;
      setMeta(rest);
      setPreviews(preview_b64);
      refresh();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.detail
          : "Something went wrong reading that file.",
      );
      setMeta(null);
      setPreviews([]);
    } finally {
      setBusy(false);
    }
  };

  const active = meta ?? chosen;

  return (
    <div className="mx-auto max-w-[1440px] px-9 py-7">
      <header>
        <h1 className="text-[25px] font-semibold text-ink">Upload a dataset</h1>
        <p className="mt-1 text-[13px] text-body">
          One folder per class inside the zip. Nothing leaves this machine.
        </p>
      </header>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_360px]">
        {/* ---- left: upload, then what was found ---- */}
        <div className="space-y-6">
          <Dropzone onFile={handleFile} busy={busy} />

          {error && (
            <p
              role="alert"
              className="rounded-lg border-l-[3px] border-alert bg-alert/5 px-4 py-3 text-[12px] text-body"
            >
              <span className="font-semibold text-ink">
                That upload failed.
              </span>{" "}
              {error}
            </p>
          )}

          {active && !busy && <ClassChips meta={active} />}

          {!active && !busy && !error && (
            <p className="text-[12px] text-muted">
              Class names are read from the folder names inside the zip. Up to
              1000 images are kept, sampled to preserve the class balance, with
              a floor of ten per class so nothing disappears.
            </p>
          )}
        </div>

        {/* ---- right: summary and samples ---- */}
        <div className="space-y-6">
          <Card title="Dataset summary">
            {active ? (
              <dl className="divide-y divide-hairline">
                {[
                  ["Name", active.name],
                  ["Images", active.n_samples.toLocaleString()],
                  ["Classes", String(active.n_classes)],
                  ["Split", "80 / 20, stratified"],
                  ["Seed", "42, fixed"],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-4 py-2.5">
                    <dt className="text-[11.5px] text-muted">{k}</dt>
                    <dd className="text-right text-[11.5px] font-medium text-ink">
                      {v}
                    </dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="text-[12px] text-muted">
                Upload a zip, or pick a dataset already on this machine.
              </p>
            )}
          </Card>

          {previews.length > 0 && (
            <Card
              title="Sample images"
              sub="One drawn from each class, after ingest."
            >
              <PreviewGrid
                previews={previews}
                classNames={meta?.class_names ?? []}
              />
            </Card>
          )}
        </div>
      </div>

      <Card
        className="mt-6"
        title="Datasets already on this machine"
        sub="Images are on disk, so these skip the upload entirely."
      >
        <DatasetList
          datasets={existing}
          loading={existing === null}
          error={listError}
          onRetry={refresh}
          selectedId={chosen?.dataset_id ?? meta?.dataset_id}
          onSelect={(d) => {
            setChosen(d);
            setMeta(null);
            setPreviews([]);
            setError(null);
          }}
        />
      </Card>

      <footer className="mt-6 flex items-center justify-between gap-6">
        <p className="text-[11px] text-muted">
          Unreadable files are dropped during ingest rather than failing a run
          later.
        </p>
        <button
          type="button"
          disabled={!active || busy}
          onClick={() => onContinue?.(active)}
          className="flex items-center gap-2 rounded-[7px] bg-quantum px-5 py-2.5 text-[13px]
            font-semibold text-white transition-opacity disabled:cursor-not-allowed
            disabled:opacity-40"
        >
          {busy && <Spinner size={14} />}
          Continue to Configure
        </button>
      </footer>
    </div>
  );
}
