"""Print native calibration evidence without importing KBMOD or changing data.

Run inside a Rubin Science Pipelines environment with access to the repository.
This checks input calibration provenance, not source recovery or search depth.
"""

import argparse
import datetime
import json
import uuid

from lsst.daf.butler import Butler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dataset-id", nargs="+", required=True)
    args = parser.parse_args()
    butler = Butler(args.repo, writeable=False)
    records = []
    for dataset_id in args.dataset_id:
        ref = butler.get_dataset(uuid.UUID(dataset_id))
        # These are the same persisted Exposure components returned by
        # exp.photoCalib and exp.info.getSummaryStats(), without reading pixels.
        photo_calib = butler.get(ref.makeComponentRef("photoCalib"))
        summary = butler.get(ref.makeComponentRef("summaryStats"))
        metadata = butler.get(ref.makeComponentRef("metadata"))
        records.append(
            {
                "dataset_id": str(ref.id),
                "dataset_type": ref.datasetType.name,
                "run": ref.run,
                "data_id": dict(ref.dataId.mapping),
                "bunit": metadata.get("BUNIT", None),
                "photo_calib_mean": float(photo_calib.getCalibrationMean()),
                "pixel_zero_point": float(photo_calib.instFluxToMagnitude(1.0)),
                "summary_zero_point": float(summary.zeroPoint),
            }
        )
    print(
        json.dumps(
            {
                "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "repository": args.repo,
                "boundary": "Read-only native PhotoCalib, summaryStats and metadata components.",
                "images": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
