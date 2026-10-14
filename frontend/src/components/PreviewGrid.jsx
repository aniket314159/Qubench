/**
 * The four thumbnails that came back with the upload.
 *
 * `preview_b64` is already a list of data URIs, so these render with no second
 * request and no object URLs to revoke. They arrive one per class where
 * possible, which is why they are worth showing at all: the user can see the
 * classes differ before spending a minute on a benchmark.
 */
export default function PreviewGrid({ previews = [], classNames = [] }) {
  if (!previews.length) return null;
  return (
    <ul className="grid grid-cols-4 gap-3">
      {previews.map((src, i) => (
        <li key={i}>
          <img
            src={src}
            alt={
              classNames[i]
                ? `Sample ECG from class ${classNames[i]}`
                : `Sample image ${i + 1}`
            }
            className="aspect-square w-full rounded-md border border-rule object-cover"
          />
          {classNames[i] && (
            <p className="mt-1.5 text-center text-[10.5px] text-muted">
              {classNames[i]}
            </p>
          )}
        </li>
      ))}
    </ul>
  );
}
