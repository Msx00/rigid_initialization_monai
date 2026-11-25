#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MuReg NIfTI adapter for the author's pca_icp registration implementation.

MR channel-0 mask -> regularized author PCA frames -> author rigid ICP -> US grid.
Configurable final rotation acceptance (default 30 degrees); per-case fitting uses only channel0.
See docs/METHOD.md for attribution, preprocessing and test-set tuning disclosure.
"""

import os
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from .adapter import register_masks, registration_parameters, rotation_angle_deg, configure_registration, configure_rotation_limit, configure_selection_dataset


# =========================
# Config
# =========================
SUBSETS = ["train", "val", "test"]
DATA_ROOT = "."
OUT_DIR = "./rigid_output"

TARGET_SHAPE = (128, 128, 128)           # NIfTI spatial axes;
FORCE_RAS = True

# Optimizer settings are defined in the author's RegistrationConfig.
ICP_MAX_ROT_DEG = registration_parameters()["max_rot_deg"]
LABEL_CH = 0

# Save options
SAVE_LABEL_6CH_255 = True
SAVE_LABEL_MAP = True
SAVE_MR_BEFORE_RIGID = True


# =========================
# Basic utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def print_axcodes(path: str, tag: str = "img"):
    img = nib.load(path)
    ax = nib.aff2axcodes(img.affine)
    zooms = img.header.get_zooms()
    print(f"[Orient] {tag}: {os.path.basename(path)} axcodes={ax}, zooms={zooms}")


def normalize_01(img_np: np.ndarray) -> np.ndarray:
    img_np = img_np.astype(np.float32, copy=False)
    mn = float(img_np.min())
    mx = float(img_np.max())
    return (img_np - mn) / (mx - mn + 1e-5)


def seg_to_onehot_6(seg_np: np.ndarray) -> np.ndarray:
    """
    seg_np: (D,H,W,6), value 0/255 or float 0/255
    return: (6,D,H,W), float32 0/1
    """
    if seg_np.ndim != 4 or seg_np.shape[-1] != 6:
        raise ValueError(f"Expected seg shape (D,H,W,6), got {seg_np.shape}")
    if not np.isfinite(seg_np).all() or not np.isin(np.unique(seg_np), [0, 1, 255]).all():
        raise ValueError("Labels must be binary six-channel arrays encoded as 0/1 or 0/255")
    out = (seg_np > 0).astype(np.uint8)
    out = np.transpose(out, (3, 0, 1, 2))
    return out.astype(np.float32)


def onehot_to_label(seg_onehot: np.ndarray) -> np.ndarray:
    """
    seg_onehot: (6,D,H,W) 0/1 -> (D,H,W) uint8 0..5
    """
    if seg_onehot.ndim != 4 or seg_onehot.shape[0] != 6:
        raise ValueError(f"Expected (6,D,H,W), got {seg_onehot.shape}")
    return np.argmax(seg_onehot, axis=0).astype(np.uint8)


# =========================
# Affine helpers
# =========================
def affine_fix_spacing(A: np.ndarray, zoom3):
    A = np.array(A, dtype=np.float64, copy=True)
    zoom3 = np.array(zoom3, dtype=np.float64)

    col_norm = np.linalg.norm(A[:3, :3], axis=0)
    if np.allclose(col_norm, [1.0, 1.0, 1.0], atol=1e-6) and (
        not np.allclose(zoom3, [1.0, 1.0, 1.0], atol=1e-6)
    ):
        u = A[:3, :3]
        if np.allclose(col_norm, 0):
            u = np.eye(3)
        else:
            u = u / col_norm
        A[:3, :3] = u * zoom3
    return A


def zooms_from_affine(A: np.ndarray):
    return tuple(np.linalg.norm(A[:3, :3], axis=0).tolist())


# =========================
# RAS orientation, no interpolation
# =========================
def get_ornt_transform_to_RAS(affine: np.ndarray):
    ornt_in = nib.orientations.io_orientation(affine)
    ornt_out = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(ornt_in, ornt_out)
    return transform, ornt_in, ornt_out


def apply_ornt(data: np.ndarray, transform) -> np.ndarray:
    if data.ndim == 3:
        return nib.orientations.apply_orientation(data, transform)
    if data.ndim == 4:
        outs = []
        for c in range(data.shape[-1]):
            outs.append(nib.orientations.apply_orientation(data[..., c], transform))
        return np.stack(outs, axis=-1)
    raise ValueError(f"Unsupported data ndim={data.ndim}, shape={data.shape}")


def reorient_to_RAS_with_affine(data: np.ndarray, nii: nib.Nifti1Image):
    """
    return: data_ras, affine_ras, header_copy, ornt_in, ornt_out
    """
    A0 = nii.affine
    zoom0 = nii.header.get_zooms()[:3]
    Aeff = affine_fix_spacing(A0, zoom0)

    transform, ornt_in, ornt_out = get_ornt_transform_to_RAS(Aeff)
    data_ras = apply_ornt(data, transform)

    orig_shape_3 = data.shape[:3]
    A_ras = Aeff @ nib.orientations.inv_ornt_aff(transform, orig_shape_3)

    hdr = nii.header.copy()
    return data_ras, A_ras, hdr, ornt_in, ornt_out


# =========================
# Pad/crop to fixed shape, no resizing
# =========================
def pad_crop_to_shape_3d_np(vol_3d: np.ndarray, target_shape, pad_value=0.0):
    assert vol_3d.ndim == 3, vol_3d.shape
    D, H, W = vol_3d.shape
    Dt, Ht, Wt = target_shape

    pad_before = [0, 0, 0]
    crop_start = [0, 0, 0]

    if D > Dt:
        crop_start[0] = (D - Dt) // 2
    if H > Ht:
        crop_start[1] = (H - Ht) // 2
    if W > Wt:
        crop_start[2] = (W - Wt) // 2

    d0, h0, w0 = crop_start
    out = vol_3d[
        d0:d0 + min(Dt, D),
        h0:h0 + min(Ht, H),
        w0:w0 + min(Wt, W),
    ]

    Dc, Hc, Wc = out.shape
    pd = max(Dt - Dc, 0)
    ph = max(Ht - Hc, 0)
    pw = max(Wt - Wc, 0)

    pad_before[0] = pd // 2
    pad_before[1] = ph // 2
    pad_before[2] = pw // 2
    pad_after = [pd - pad_before[0], ph - pad_before[1], pw - pad_before[2]]

    if pd > 0 or ph > 0 or pw > 0:
        out = np.pad(
            out,
            pad_width=(
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode="constant",
            constant_values=pad_value,
        )

    return out, np.array(pad_before, dtype=np.int64), np.array(crop_start, dtype=np.int64)


def pad_crop_to_shape_channels_first_np(arr: np.ndarray, target_shape, pad_value=0):
    """arr: (C,D,H,W)"""
    assert arr.ndim == 4, arr.shape
    outs = []
    pad_before = crop_start = None
    for c in range(arr.shape[0]):
        out_c, pb, cs = pad_crop_to_shape_3d_np(arr[c], target_shape, pad_value=pad_value)
        outs.append(out_c)
        pad_before, crop_start = pb, cs
    out = np.stack(outs, axis=0)
    return out, pad_before, crop_start


def update_affine_for_pad_crop(affine: np.ndarray, pad_before: np.ndarray, crop_start: np.ndarray):
    """
    shift = crop_start - pad_before, in voxel index units.
    new_aff[:3,3] = old_aff[:3,3] + old_aff[:3,:3] @ shift
    """
    shift = (crop_start - pad_before).astype(np.float64)
    new_aff = np.array(affine, dtype=np.float64, copy=True)
    new_aff[:3, 3] = affine[:3, 3] + affine[:3, :3] @ shift
    return new_aff


# =========================
# Save NIfTI
# =========================
def save_nifti(arr_np: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, out_path: str, dtype):
    ensure_dir(os.path.dirname(out_path))
    hdr = header.copy()
    hdr.set_data_dtype(dtype)
    hdr.set_data_shape(arr_np.shape)

    z3 = zooms_from_affine(affine)
    oldz = hdr.get_zooms()
    if len(oldz) < arr_np.ndim:
        oldz = oldz + (1.0,) * (arr_np.ndim - len(oldz))
    newz = z3 + oldz[3:arr_np.ndim]
    hdr.set_zooms(newz)

    nii = nib.Nifti1Image(
        arr_np.astype(dtype, copy=False),
        affine=np.array(affine, dtype=np.float64),
        header=hdr,
    )
    nib.save(nii, out_path)


# =========================
# Rigid math
# =========================


def invert_T(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# =========================
# Resampling
# =========================
def rigid_resample_to_fixed(
    mv_data: np.ndarray,
    A_mv: np.ndarray,
    A_fx: np.ndarray,
    T_mv2fx: np.ndarray,
    out_shape,
    order: int = 1,
    cval: float = 0.0,
):
    """
    Output is on fixed grid, with shape=out_shape and affine=A_fx.

    Mapping for scipy.ndimage.affine_transform:
        i_mv = inv(A_mv) @ inv(T_mv2fx) @ A_fx @ i_fx
    """
    A_mv = np.asarray(A_mv, dtype=np.float64)
    A_fx = np.asarray(A_fx, dtype=np.float64)
    T_mv2fx = np.asarray(T_mv2fx, dtype=np.float64)

    T_fx2mv = invert_T(T_mv2fx)
    M = np.linalg.inv(A_mv) @ T_fx2mv @ A_fx

    matrix = M[:3, :3]
    offset = M[:3, 3]

    def _warp(vol3d):
        return ndi.affine_transform(
            vol3d,
            matrix=matrix,
            offset=offset,
            output_shape=out_shape,
            order=order,
            mode="constant",
            cval=float(cval),
            prefilter=(order > 1),
        )

    if mv_data.ndim == 3:
        return _warp(mv_data)
    if mv_data.ndim == 4:
        C = mv_data.shape[3]
        outs = []
        for c in range(C):
            outs.append(_warp(mv_data[..., c]))
        return np.stack(outs, axis=3)
    raise ValueError(f"Unsupported mv_data ndim={mv_data.ndim}, shape={mv_data.shape}")


# =========================
# Dice evaluation and rotation acceptance gate
# =========================
def dice_binary_np(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    vol_sum = a.sum() + b.sum()
    if vol_sum == 0:
        return 1.0
    return float(2.0 * inter / vol_sum)


def compute_label0_dice_after_transform_np(
    moving_onehot_pc: np.ndarray,
    fixed_onehot_pc: np.ndarray,
    A_mv_pc: np.ndarray,
    A_fx_pc: np.ndarray,
    T_mv2fx: np.ndarray,
    target_shape,
    label_ch: int = 0,
) -> float:
    """
    Resample moving label0 to fixed grid using T_mv2fx, then compute label0 Dice.
    """
    mv_label0 = (moving_onehot_pc[label_ch] > 0.5).astype(np.uint8)
    fx_label0 = (fixed_onehot_pc[label_ch] > 0.5).astype(np.uint8)

    mv_label0_4d = mv_label0[..., None]
    mv_label0_on_fx = rigid_resample_to_fixed(
        mv_data=mv_label0_4d,
        A_mv=A_mv_pc,
        A_fx=A_fx_pc,
        T_mv2fx=T_mv2fx,
        out_shape=target_shape,
        order=0,
        cval=0,
    )

    mv_label0_on_fx = mv_label0_on_fx[..., 0] > 0
    return dice_binary_np(mv_label0_on_fx, fx_label0)


# =========================
# Path helpers
# =========================
def collect_subset_files(subset: str):
    mr_image_dir = os.path.join(DATA_ROOT, subset, "mr_images")
    mr_label_dir = os.path.join(DATA_ROOT, subset, "mr_labels")
    us_image_dir = os.path.join(DATA_ROOT, subset, "us_images")
    us_label_dir = os.path.join(DATA_ROOT, subset, "us_labels")

    mr_images = sorted(glob.glob(os.path.join(mr_image_dir, "*.nii.gz")))
    us_images = sorted(glob.glob(os.path.join(us_image_dir, "*.nii.gz")))
    mr_labels = sorted(glob.glob(os.path.join(mr_label_dir, "*.nii.gz")))
    us_labels = sorted(glob.glob(os.path.join(us_label_dir, "*.nii.gz")))

    return mr_images, us_images, mr_labels, us_labels


def prepare_output_dirs(out_dir: str, subset: str):
    subs = [
        "mr_images",
        "us_images",
        "mr_labels",
        "us_labels",
        "mr_labels_maps",
        "us_labels_maps",
        "meta",
    ]
    if SAVE_MR_BEFORE_RIGID:
        subs.append("mr_images_before_rigid")
    for s in subs:
        ensure_dir(os.path.join(out_dir, subset, s))


# =========================
# Main processing
# =========================
def process_subset(subset: str):
    mr_images, us_images, mr_labels, us_labels = collect_subset_files(subset)

    if len(mr_images) == 0 and len(us_images) == 0:
        print(f"[Skip] subset={subset}: no files found")
        return

    if not (len(mr_images) == len(us_images) == len(mr_labels) == len(us_labels)):
        raise RuntimeError(
            f"[Error] subset={subset}: file count mismatch | "
            f"mr_images={len(mr_images)}, us_images={len(us_images)}, "
            f"mr_labels={len(mr_labels)}, us_labels={len(us_labels)}"
        )

    prepare_output_dirs(OUT_DIR, subset)

    print(f"\n{'=' * 24} SUBSET: {subset} | N={len(mr_images)} {'=' * 24}")
    if len(mr_images) > 0:
        print_axcodes(mr_images[0], "MR raw")
        print_axcodes(us_images[0], "US raw")

    all_meta: List[Dict[str, Any]] = []
    accepted_count = 0
    skipped_count = 0

    for k in range(len(mr_images)):
        mr_img_path = mr_images[k]
        us_img_path = us_images[k]
        mr_seg_path = mr_labels[k]
        us_seg_path = us_labels[k]

        base = os.path.basename(us_img_path).replace(".nii.gz", "")
        print("\n==============================")
        print(f"[CASE] subset={subset} index={k:04d} base={base}")

        try:
            # ---- load ----
            mr_nii = nib.load(mr_img_path)
            us_nii = nib.load(us_img_path)
            mr_seg_nii = nib.load(mr_seg_path)
            us_seg_nii = nib.load(us_seg_path)

            mr_img = mr_nii.get_fdata().squeeze()
            us_img = us_nii.get_fdata().squeeze()
            mr_seg = mr_seg_nii.get_fdata()
            us_seg = us_seg_nii.get_fdata()

            # ---- RAS + affine update ----
            if FORCE_RAS:
                mr_img, A_mr, hdr_mr, _, _ = reorient_to_RAS_with_affine(mr_img, mr_nii)
                us_img, A_us, hdr_us, _, _ = reorient_to_RAS_with_affine(us_img, us_nii)
                mr_seg, A_mrseg, hdr_mrseg, _, _ = reorient_to_RAS_with_affine(mr_seg, mr_seg_nii)
                us_seg, A_usseg, hdr_usseg, _, _ = reorient_to_RAS_with_affine(us_seg, us_seg_nii)
            else:
                A_mr = affine_fix_spacing(mr_nii.affine, mr_nii.header.get_zooms()[:3])
                A_us = affine_fix_spacing(us_nii.affine, us_nii.header.get_zooms()[:3])
                hdr_mr = mr_nii.header.copy()
                hdr_us = us_nii.header.copy()
                A_mrseg = affine_fix_spacing(mr_seg_nii.affine, mr_seg_nii.header.get_zooms()[:3])
                A_usseg = affine_fix_spacing(us_seg_nii.affine, us_seg_nii.header.get_zooms()[:3])
                hdr_mrseg = mr_seg_nii.header.copy()
                hdr_usseg = us_seg_nii.header.copy()

            # The same voxel grid is required for image/label preprocessing.
            for modality, image, segmentation, image_aff, label_aff in [
                ("MR", mr_img, mr_seg, A_mr, A_mrseg),
                ("US", us_img, us_seg, A_us, A_usseg),
            ]:
                if image.shape[:3] != segmentation.shape[:3] or not np.allclose(image_aff, label_aff, atol=1e-4):
                    raise ValueError(f"{modality} image and label grids differ")

            # ---- normalize ----
            mr_img = normalize_01(mr_img)
            us_img = normalize_01(us_img)

            # ---- seg -> onehot ----
            mr_onehot = seg_to_onehot_6(mr_seg)
            us_onehot = seg_to_onehot_6(us_seg)

            # ---- pad/crop, no interpolation ----
            mr_img_pc, mr_pb, mr_cs = pad_crop_to_shape_3d_np(mr_img, TARGET_SHAPE, pad_value=0.0)
            us_img_pc, us_pb, us_cs = pad_crop_to_shape_3d_np(us_img, TARGET_SHAPE, pad_value=0.0)

            mr_onehot_pc, mr_lab_pb, mr_lab_cs = pad_crop_to_shape_channels_first_np(
                mr_onehot, TARGET_SHAPE, pad_value=0
            )
            us_onehot_pc, us_lab_pb, us_lab_cs = pad_crop_to_shape_channels_first_np(
                us_onehot, TARGET_SHAPE, pad_value=0
            )

            # For labels, pad/crop should match image pad/crop. If not, warn.
            if not (np.array_equal(mr_pb, mr_lab_pb) and np.array_equal(mr_cs, mr_lab_cs)):
                print("[Warn] MR image and label pad/crop metadata differ.")
            if not (np.array_equal(us_pb, us_lab_pb) and np.array_equal(us_cs, us_lab_cs)):
                print("[Warn] US image and label pad/crop metadata differ.")

            # ---- update affines after pad/crop ----
            A_mr_pc = update_affine_for_pad_crop(A_mr, mr_pb, mr_cs)
            A_us_pc = update_affine_for_pad_crop(A_us, us_pb, us_cs)

            # Persist the exact 3D 0/1 masks passed to the author's NIfTI API.
            mask_dir = Path(OUT_DIR) / subset / "registration_masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            source_mask = mask_dir / f"{base}_MR.nii.gz"
            target_mask = mask_dir / f"{base}_US.nii.gz"
            for mask, affine, header, path in [
                (mr_onehot_pc[LABEL_CH], A_mr_pc, hdr_mrseg, source_mask),
                (us_onehot_pc[LABEL_CH], A_us_pc, hdr_usseg, target_mask),
            ]:
                if not np.any(mask > 0.5):
                    raise ValueError("Channel-0 mask is empty after preprocessing")
                save_nifti((mask > 0.5).astype(np.uint8), affine, header, str(path), np.uint8)
            icp_result = register_masks(source_mask, target_mask)
            T_mr2us = np.asarray(icp_result["T_mv2fx"], dtype=np.float64)
            # Evaluation happens AFTER transform selection and cannot affect acceptance.
            icp_result["best_dice"] = compute_label0_dice_after_transform_np(
                mr_onehot_pc, us_onehot_pc, A_mr_pc, A_us_pc,
                T_mr2us, TARGET_SHAPE, LABEL_CH,
            )

            if not icp_result["accepted"]:
                skipped_count += 1
                print(
                    f"[SKIP] quality gate failed | best_dice={icp_result['best_dice']} | "
                    f"best_rot={icp_result['best_rot_angle_deg']} | reason={icp_result['failure_reason']}"
                )

                meta_failed = {
                    "case": base,
                    "accepted": False,
                    "failure_reason": icp_result["failure_reason"],
                    "rigid_reference": "us_fixed",
                "registration_direction": "MR_to_US",
                    "moving": "MR",
                    "fixed": "US",
                    "target_shape": list(TARGET_SHAPE),
                    "force_ras": bool(FORCE_RAS),
                    "us_pad_before": us_pb.tolist(),
                    "us_crop_start": us_cs.tolist(),
                    "mr_pad_before": mr_pb.tolist(),
                    "mr_crop_start": mr_cs.tolist(),
                    "best_attempt": icp_result["best_attempt"],
                    "best_dice": icp_result["best_dice"],
                    "best_rot_angle_deg": icp_result["best_rot_angle_deg"],
                    "best_T_mr2us_4x4": T_mr2us.tolist(),
                    "best_T_us2mr_4x4": invert_T(T_mr2us).tolist(),
                    "attempts": icp_result["attempts"],
                    "registration_backend": icp_result["backend"],
                    "mesh_points": icp_result["mesh_points"],
                    "icp_params": registration_parameters(),
                }

                with open(
                    os.path.join(OUT_DIR, subset, "meta", f"{base}_failed.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(meta_failed, f, indent=2, ensure_ascii=False)

                all_meta.append(meta_failed)
                continue

            print(f"[Author registration] backend={icp_result['backend']} | "
                  f"rotation={icp_result['best_rot_angle_deg']:.2f} deg | "
                  f"Dice={icp_result['best_dice']:.4f}")

            # =========================
            # Save fixed US on US grid
            # =========================
            us_img_4d = us_img_pc[..., None]
            save_nifti(
                us_img_4d.astype(np.float32, copy=False),
                A_us_pc,
                hdr_us,
                os.path.join(OUT_DIR, subset, "us_images", f"{base}.nii.gz"),
                dtype=np.float32,
            )

            if SAVE_LABEL_6CH_255:
                us_6ch_255 = (np.transpose(us_onehot_pc, (1, 2, 3, 0)) * 255.0).astype(np.uint8)
                save_nifti(
                    us_6ch_255,
                    A_us_pc,
                    hdr_usseg,
                    os.path.join(OUT_DIR, subset, "us_labels", f"{base}.nii.gz"),
                    dtype=np.uint8,
                )

            if SAVE_LABEL_MAP:
                us_lab = onehot_to_label(us_onehot_pc)
                save_nifti(
                    us_lab,
                    A_us_pc,
                    hdr_usseg,
                    os.path.join(OUT_DIR, subset, "us_labels_maps", f"{base}.nii.gz"),
                    dtype=np.uint8,
                )

            # Optional: save MR before rigid for debugging only.
            if SAVE_MR_BEFORE_RIGID:
                mr_img_before_4d = mr_img_pc[..., None]
                save_nifti(
                    mr_img_before_4d.astype(np.float32, copy=False),
                    A_mr_pc,
                    hdr_mr,
                    os.path.join(OUT_DIR, subset, "mr_images_before_rigid", f"{base}.nii.gz"),
                    dtype=np.float32,
                )

            # =========================
            # Resample MR image to US grid
            # =========================
            mr_img_pc_4d = mr_img_pc[..., None]
            mr_rigid_4d = rigid_resample_to_fixed(
                mv_data=mr_img_pc_4d,
                A_mv=A_mr_pc,
                A_fx=A_us_pc,
                T_mv2fx=T_mr2us,
                out_shape=TARGET_SHAPE,
                order=1,
                cval=0.0,
            )

            save_nifti(
                mr_rigid_4d.astype(np.float32, copy=False),
                A_us_pc,
                hdr_us,
                os.path.join(OUT_DIR, subset, "mr_images", f"{base}.nii.gz"),
                dtype=np.float32,
            )

            # Resample MR labels to US grid.
            if SAVE_LABEL_6CH_255:
                mr_6ch_255 = (np.transpose(mr_onehot_pc, (1, 2, 3, 0)) * 255.0).astype(np.uint8)
                mr_6ch_rigid = rigid_resample_to_fixed(
                    mv_data=mr_6ch_255,
                    A_mv=A_mr_pc,
                    A_fx=A_us_pc,
                    T_mv2fx=T_mr2us,
                    out_shape=TARGET_SHAPE,
                    order=0,
                    cval=0,
                ).astype(np.uint8, copy=False)

                save_nifti(
                    mr_6ch_rigid,
                    A_us_pc,
                    hdr_usseg,
                    os.path.join(OUT_DIR, subset, "mr_labels", f"{base}.nii.gz"),
                    dtype=np.uint8,
                )

                if SAVE_LABEL_MAP:
                    mr_lab_rigid = np.argmax((mr_6ch_rigid > 127).astype(np.uint8), axis=3).astype(np.uint8)
                    save_nifti(
                        mr_lab_rigid,
                        A_us_pc,
                        hdr_usseg,
                        os.path.join(OUT_DIR, subset, "mr_labels_maps", f"{base}.nii.gz"),
                        dtype=np.uint8,
                    )

            # Recompute final label0 Dice from saved-style transform for metadata.
            final_label0_dice = compute_label0_dice_after_transform_np(
                moving_onehot_pc=mr_onehot_pc,
                fixed_onehot_pc=us_onehot_pc,
                A_mv_pc=A_mr_pc,
                A_fx_pc=A_us_pc,
                T_mv2fx=T_mr2us,
                target_shape=TARGET_SHAPE,
                label_ch=LABEL_CH,
            )

            meta = {
                "case": base,
                "accepted": True,
                "target_shape": list(TARGET_SHAPE),
                "force_ras": bool(FORCE_RAS),
                "rigid_reference": "us_fixed",
                "registration_direction": "MR_to_US",
                "moving": "MR",
                "fixed": "US",
                "us_pad_before": us_pb.tolist(),
                "us_crop_start": us_cs.tolist(),
                "mr_pad_before": mr_pb.tolist(),
                "mr_crop_start": mr_cs.tolist(),
                # Main transform: x_us = T_mr2us * x_mr
                "T_mr2us_4x4": T_mr2us.tolist(),
                # Inverse for convenience: x_mr = T_us2mr * x_us
                "T_us2mr_4x4": invert_T(T_mr2us).tolist(),
                "label0_dice": float(final_label0_dice),
                "best_dice": icp_result["best_dice"],
                "rot_angle_deg": icp_result["best_rot_angle_deg"],
                "best_attempt": icp_result["best_attempt"],
                "attempts": icp_result["attempts"],
                "registration_backend": icp_result["backend"],
                "mesh_points": icp_result["mesh_points"],
                "icp_params": registration_parameters(),
                "label_channel_used": int(LABEL_CH),
                "note": f"Author pca_icp mesh registration, rigid MR-to-US. Rotation <={ICP_MAX_ROT_DEG} degrees; Dice is evaluation-only.",
            }
            with open(
                os.path.join(OUT_DIR, subset, "meta", f"{base}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            all_meta.append(meta)
            accepted_count += 1
            print(f"[OK] saved fixed US and rigid MR to: {os.path.join(OUT_DIR, subset)}")

        except Exception as e:
            skipped_count += 1
            print(f"[FAIL] case={base} | {repr(e)}")
            meta_failed = {
                "case": base,
                "accepted": False,
                "failure_reason": "exception",
                "registration_direction": "MR_to_US",
                "exception": repr(e),
                "moving": "MR",
                "fixed": "US",
            }
            ensure_dir(os.path.join(OUT_DIR, subset, "meta"))
            with open(
                os.path.join(OUT_DIR, subset, "meta", f"{base}_failed.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(meta_failed, f, indent=2, ensure_ascii=False)
            all_meta.append(meta_failed)

    # subset summary
    summary = {
        "subset": subset,
        "num_total": len(mr_images),
        "num_accepted": accepted_count,
        "num_skipped": skipped_count,
        "accept_rate": accepted_count / max(len(mr_images), 1),
        "out_dir": os.path.join(OUT_DIR, subset),
        "icp_params": registration_parameters(),
    }

    with open(os.path.join(OUT_DIR, subset, "meta", "all_meta.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, subset, "meta", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[Subset Done]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    global DATA_ROOT, OUT_DIR, SUBSETS, ICP_MAX_ROT_DEG
    parser = argparse.ArgumentParser(description="Adapted PCA initialization + upstream rigid ICP for MuReg six-channel NIfTI pairs.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing train/val/test.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New or empty output directory.")
    parser.add_argument("--subsets", nargs="+", choices=["train", "val", "test"], default=SUBSETS)
    parser.add_argument("--registration-config", type=Path, help="JSON fitting settings; omitted fields keep the packaged defaults. Explicit flags override JSON.")
    parser.add_argument('--max-rotation-degrees', type=float, default=30.0, help='Final rotation acceptance limit; does not constrain ICP optimization (default 30).')
    parser.add_argument('--selection-dataset', choices=['unspecified','train','val','train_val','test'], default='unspecified', help='Selection cohort for this direction (metadata only); default unspecified; numeric defaults were selected for historical US-to-MR fits.')
    parser.add_argument('--pca-rotation-gain', type=float, help='Initial rotation-vector gain, 0 to 1; default 0.10')
    parser.add_argument('--smoothing', type=int, help='Mesh smoothing iterations; default 0')
    parser.add_argument('--landmarks', type=int, help='Maximum ICP sampling points; default 7000')
    parser.add_argument('--iterations', type=int, help='Maximum ICP iterations; default 30')
    parser.add_argument('--decimation-reduction', type=float, help='Target mesh reduction fraction, [0,1); default 0.20')
    args = parser.parse_args()
    settings = dict(smoothing=0, landmarks=7000, iterations=30, pca_rotation_gain=0.10, decimation_reduction=0.20)
    if args.registration_config:
        try:
            profile = json.loads(args.registration_config.read_text(encoding='utf-8'))
            if not isinstance(profile, dict):
                raise ValueError('Registration config must be a JSON object')
            settings.update(profile)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    for key in list(settings):
        override = getattr(args, key, None)
        if override is not None:
            settings[key] = override
    try:
        configure_registration(settings)
        configure_rotation_limit(args.max_rotation_degrees)
        configure_selection_dataset(args.selection_dataset)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    ICP_MAX_ROT_DEG = registration_parameters()['max_rot_deg']
    source, output = args.data_root.resolve(), args.out_dir.resolve()
    if source == output or output in source.parents:
        parser.error("Output must not equal the data root or contain it.")
    for split in ["train", "val", "test"]:
        split_root = source / split
        if output == split_root or split_root in output.parents:
            parser.error("Output must not be inside an input split.")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty to avoid mixing runs.")
    DATA_ROOT, OUT_DIR, SUBSETS = str(source), str(output), args.subsets
    for subset in SUBSETS:
        groups = collect_subset_files(subset)
        names = [[Path(p).name for p in group] for group in groups]
        if not names[0] or any(group != names[0] for group in names):
            parser.error(f"Missing or unpaired images/labels in {subset}.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps({
        "data_root": DATA_ROOT, "out_dir": OUT_DIR, "subsets": SUBSETS,
        "max_rotation_degrees": ICP_MAX_ROT_DEG, "acceptance": "rotation_only",
        "method": registration_parameters()["method"], "label_channel": LABEL_CH,
        "registration_direction": "MR_to_US", "moving": "MR", "fixed": "US",
        "output_reference": "US",
        "default_profile_provenance": {"selection_direction": "US_to_MR",
            "selection_dataset": "train_val", "reoptimized_for_MR_to_US": False},
        "upstream_repository": "https://github.com/dalbenzioG/Initial_Registration",
        "upstream_base_commit": "21ee6ceb50bdb1e8d14072cb54ebb1bd2c611029",
        "registration_parameters": registration_parameters(),
        "registration_config_file": str(args.registration_config.resolve()) if args.registration_config else None,
    }, indent=2), encoding="utf-8")
    for subset in SUBSETS:
        process_subset(subset)

    print("\nAll done.")
    print("Saved to:", OUT_DIR)
    rejected = sum(json.loads((output / split / 'meta/summary.json').read_text())['num_skipped'] for split in SUBSETS)
    if rejected:
        print(f"{rejected} case(s) rejected or failed; inspect meta/*_failed.json.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
