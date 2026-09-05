#!/usr/bin/env python3
"""Build deterministic, leakage-safe JEPA manifests for a TartanAir-v2 subset.

Chay (vi du tren Kaggle, sau khi da resize):
    python qwt_jepa/scripts/build_manifests.py \
        --data-root /kaggle/working/tartanair-v2 \
        --output-dir /kaggle/working/tartanair-v2-jepa

Voi 7 environment goc (AmericanDiner, ArchVizTinyHouseDay, CountryHouse,
DesertGasStation, OldTownNight, RetroOffice, WaterMillDay), holdout
val/test trajectory duoc co dinh trong FIXED_VAL_GROUPS/FIXED_TEST_GROUPS
de giu dung split da dung de tinh norm_stats truoc do.

Voi bat ky environment nao khac (khong nam trong hai dict tren), holdout
duoc CHON TU DONG: trajectory dau tien (sap xep theo ten) -> test,
trajectory thu hai -> val, con lai -> train. Nho vay co the them
environment moi ma khong can biet truoc ten trajectory cua no.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DIFFICULTIES = ("Data_easy", "Data_hard")

# Chi giu co dinh cho 7 environment goc - KHONG doi, de tuong thich nguoc
# voi split/norm_stats da tinh truoc do.
FIXED_VAL_GROUPS = {
    "AmericanDiner": "P002",
    "ArchVizTinyHouseDay": "P003",
    "CountryHouse": "P000",
    "DesertGasStation": "P000",
    "OldTownNight": "P001",
    "RetroOffice": "P003",
    "WaterMillDay": "P003",
}

FIXED_TEST_GROUPS = {
    "AmericanDiner": "P000",
    "ArchVizTinyHouseDay": "P006",
    "CountryHouse": "P005",
    "DesertGasStation": "P001",
    "OldTownNight": "P000",
    "RetroOffice": "P004",
    "WaterMillDay": "P004",
}

TRAJECTORY_FIELDS = (
    "split",
    "group_id",
    "environment",
    "difficulty",
    "trajectory",
    "trajectory_path",
    "num_images",
    "num_imu_rows",
    "num_samples",
)

SAMPLE_FIELDS = (
    "sample_id",
    "environment",
    "difficulty",
    "trajectory",
    "previous_image_path",
    "image_path",
    "imu_acc_path",
    "imu_gyro_path",
    "imu_time_path",
    "imu_start_row",
    "imu_end_row_exclusive",
    "pose_path",
    "previous_pose_row",
    "pose_row",
)


def count_lines(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def environment_dirs(data_root: Path) -> list[Path]:
    result = []
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and all((path / difficulty).is_dir() for difficulty in DIFFICULTIES):
            result.append(path)
    return result


def holdouts_for(environment: str, easy_ids_sorted: list[str]) -> tuple[str | None, str | None]:
    """Tra ve (val_trajectory, test_trajectory) cho mot environment."""
    if environment in FIXED_VAL_GROUPS or environment in FIXED_TEST_GROUPS:
        return FIXED_VAL_GROUPS.get(environment), FIXED_TEST_GROUPS.get(environment)
    test_traj = easy_ids_sorted[0] if easy_ids_sorted else None
    val_traj = easy_ids_sorted[1] if len(easy_ids_sorted) > 1 else None
    return val_traj, test_traj


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="thu muc chua cac environment (vd tartanair-v2)")
    ap.add_argument("--output-dir", required=True, help="thu muc ghi train.csv/val.csv/test.csv/summary.json")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    def relative(path: Path) -> str:
        return path.relative_to(data_root).as_posix()

    environments = environment_dirs(data_root)
    if not environments:
        raise RuntimeError(f"khong tim thay environment nao trong {data_root}")

    missing_fixed = [env for env in (set(FIXED_VAL_GROUPS) | set(FIXED_TEST_GROUPS))
                      if env not in {p.name for p in environments}]
    if missing_fixed:
        raise RuntimeError(
            f"Cac environment co holdout co dinh nhung khong co trong data_root: {sorted(missing_fixed)}"
        )

    trajectory_rows: list[dict[str, object]] = []
    sample_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    group_assignments: dict[str, str] = {}

    for environment_dir in environments:
        environment = environment_dir.name
        easy_ids = {path.name for path in (environment_dir / "Data_easy").glob("P*") if path.is_dir()}
        hard_ids = {path.name for path in (environment_dir / "Data_hard").glob("P*") if path.is_dir()}
        if easy_ids != hard_ids:
            raise RuntimeError(
                f"Difficulty mismatch in {environment}: easy={sorted(easy_ids)}, hard={sorted(hard_ids)}"
            )

        easy_ids_sorted = sorted(easy_ids)
        val_traj, test_traj = holdouts_for(environment, easy_ids_sorted)
        if val_traj is not None and val_traj not in easy_ids:
            raise RuntimeError(f"Configured val trajectory missing in {environment}: {val_traj}")
        if test_traj is not None and test_traj not in easy_ids:
            raise RuntimeError(f"Configured test trajectory missing in {environment}: {test_traj}")
        if val_traj is not None and val_traj == test_traj:
            raise RuntimeError(f"Validation and test overlap in {environment}")

        def split_for(trajectory: str) -> str:
            if val_traj == trajectory:
                return "val"
            if test_traj == trajectory:
                return "test"
            return "train"

        for trajectory in easy_ids_sorted:
            split = split_for(trajectory)
            group_id = f"{environment}/{trajectory}"
            previous_assignment = group_assignments.setdefault(group_id, split)
            if previous_assignment != split:
                raise RuntimeError(f"Group leakage detected for {group_id}")

            for difficulty in DIFFICULTIES:
                trajectory_dir = environment_dir / difficulty / trajectory
                image_dir = trajectory_dir / "image_lcam_front"
                imu_dir = trajectory_dir / "imu"
                pose_path = trajectory_dir / "pose_lcam_front.txt"
                images = sorted(image_dir.glob("*_lcam_front.png"))

                num_images = len(images)
                num_imu = count_lines(imu_dir / "imu_time.txt")
                num_acc = count_lines(imu_dir / "acc.txt")
                num_gyro = count_lines(imu_dir / "gyro.txt")
                num_cam = count_lines(imu_dir / "cam_time.txt")
                num_pose = count_lines(pose_path)
                expected_imu = (num_images - 1) * 10

                if num_images < 2:
                    raise RuntimeError(f"Not enough images in {trajectory_dir}")
                if not (num_imu == num_acc == num_gyro == expected_imu):
                    raise RuntimeError(
                        f"IMU alignment failed in {trajectory_dir}: images={num_images}, "
                        f"imu={num_imu}, acc={num_acc}, gyro={num_gyro}, expected={expected_imu}"
                    )
                if num_cam != num_images or num_pose != num_images:
                    raise RuntimeError(
                        f"Camera/pose alignment failed in {trajectory_dir}: "
                        f"images={num_images}, cam={num_cam}, pose={num_pose}"
                    )

                trajectory_rows.append(
                    {
                        "split": split,
                        "group_id": group_id,
                        "environment": environment,
                        "difficulty": difficulty,
                        "trajectory": trajectory,
                        "trajectory_path": relative(trajectory_dir),
                        "num_images": num_images,
                        "num_imu_rows": num_imu,
                        "num_samples": num_images - 1,
                    }
                )

                acc_path = relative(imu_dir / "acc.txt")
                gyro_path = relative(imu_dir / "gyro.txt")
                imu_time_path = relative(imu_dir / "imu_time.txt")
                relative_pose_path = relative(pose_path)

                # Image i is paired with the 10 IMU rows in [t_(i-1), t_i).
                for image_index in range(1, num_images):
                    sample_rows[split].append(
                        {
                            "sample_id": (
                                f"{environment}__{difficulty}__{trajectory}__{image_index:06d}"
                            ),
                            "environment": environment,
                            "difficulty": difficulty,
                            "trajectory": trajectory,
                            "previous_image_path": relative(images[image_index - 1]),
                            "image_path": relative(images[image_index]),
                            "imu_acc_path": acc_path,
                            "imu_gyro_path": gyro_path,
                            "imu_time_path": imu_time_path,
                            "imu_start_row": (image_index - 1) * 10,
                            "imu_end_row_exclusive": image_index * 10,
                            "pose_path": relative_pose_path,
                            "previous_pose_row": image_index - 1,
                            "pose_row": image_index,
                        }
                    )

    # Assert disjoint group membership before writing anything.
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in trajectory_rows:
        split_groups[str(row["split"])].add(str(row["group_id"]))
    if split_groups["train"] & split_groups["val"]:
        raise RuntimeError("Train/validation group overlap")
    if split_groups["train"] & split_groups["test"]:
        raise RuntimeError("Train/test group overlap")
    if split_groups["val"] & split_groups["test"]:
        raise RuntimeError("Validation/test group overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trajectories.csv", TRAJECTORY_FIELDS, trajectory_rows)
    for split in ("train", "val", "test"):
        write_csv(output_dir / f"{split}.csv", SAMPLE_FIELDS, sample_rows[split])

    total_samples = sum(len(rows) for rows in sample_rows.values())
    summary: dict[str, object] = {
        "data_root": str(data_root),
        "grouping_rule": "environment + trajectory; Data_easy/Data_hard stay together",
        "imu_window_rule": "10 rows in [t_(i-1), t_i), end row is exclusive",
        "total_groups": len(group_assignments),
        "total_trajectories": len(trajectory_rows),
        "total_samples": total_samples,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        rows = [row for row in trajectory_rows if row["split"] == split]
        summary["splits"][split] = {
            "groups": len(split_groups[split]),
            "trajectories": len(rows),
            "environments": sorted({str(row["environment"]) for row in rows}),
            "images": sum(int(row["num_images"]) for row in rows),
            "imu_rows": sum(int(row["num_imu_rows"]) for row in rows),
            "samples": len(sample_rows[split]),
            "sample_ratio": len(sample_rows[split]) / total_samples if total_samples else 0.0,
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
