from __future__ import annotations

from dataclasses import dataclass
import math

from src.schemas import TrackObject


@dataclass
class _Segment:
	track_id: int
	cls_id: int
	start_frame: int
	end_frame: int
	start_obj: TrackObject
	end_obj: TrackObject


def _estimate_tail_velocity(seq: list[TrackObject], tail_k: int = 4) -> tuple[float, float]:
	if len(seq) < 2:
		return 0.0, 0.0
	pts = seq[-max(2, tail_k) :]
	vxs: list[float] = []
	vys: list[float] = []
	for i in range(1, len(pts)):
		dt = max(1, int(pts[i].frame_id - pts[i - 1].frame_id))
		vxs.append((pts[i].cx - pts[i - 1].cx) / float(dt))
		vys.append((pts[i].cy - pts[i - 1].cy) / float(dt))
	if not vxs:
		return 0.0, 0.0
	vxs.sort()
	vys.sort()
	return vxs[len(vxs) // 2], vys[len(vys) // 2]


def _make_segments(all_tracks_per_frame: list[list[TrackObject]]) -> dict[int, _Segment]:
	by_id: dict[int, list[TrackObject]] = {}
	for frame_tracks in all_tracks_per_frame:
		for t in frame_tracks:
			by_id.setdefault(int(t.track_id), []).append(t)

	out: dict[int, _Segment] = {}
	for tid, seq in by_id.items():
		seq = sorted(seq, key=lambda x: x.frame_id)
		if not seq:
			continue
		out[tid] = _Segment(
			track_id=tid,
			cls_id=int(seq[0].cls_id),
			start_frame=int(seq[0].frame_id),
			end_frame=int(seq[-1].frame_id),
			start_obj=seq[0],
			end_obj=seq[-1],
		)
	return out


def _continuity_score(
	src_seq: list[TrackObject],
	src_end: TrackObject,
	dst_start: TrackObject,
) -> float:
	dt = max(1, int(dst_start.frame_id - src_end.frame_id))
	vx, vy = _estimate_tail_velocity(src_seq)
	pred_x = src_end.cx + vx * dt
	pred_y = src_end.cy + vy * dt

	fw = max(
		float(src_end.frame_w),
		float(dst_start.frame_w),
		float(src_end.bbox_xyxy[2]),
		float(dst_start.bbox_xyxy[2]),
		1.0,
	)
	fh = max(
		float(src_end.frame_h),
		float(dst_start.frame_h),
		float(src_end.bbox_xyxy[3]),
		float(dst_start.bbox_xyxy[3]),
		1.0,
	)
	diag = max(1.0, math.hypot(fw, fh))

	center_dist = math.hypot(pred_x - dst_start.cx, pred_y - dst_start.cy) / diag
	scale_src = max(1e-6, math.sqrt(max(src_end.area, 1.0)))
	scale_dst = max(1e-6, math.sqrt(max(dst_start.area, 1.0)))
	scale_ratio = max(scale_src / scale_dst, scale_dst / scale_src)
	scale_penalty = abs(math.log(scale_ratio))
	return float(center_dist + 0.2 * scale_penalty)


def refine_track_ids(
	all_tracks_per_frame: list[list[TrackObject]],
	max_frame_gap: int = 8,
	max_center_dist: float = 0.06,
	max_size_ratio: float = 2.5,
	min_direction_cos: float = -0.1,
	max_speed_ratio: float = 3.5,
	gap_relax_factor: float = 0.20,
) -> tuple[list[list[TrackObject]], dict[int, int]]:
	"""Refine fragmented track ids.

	Input/output share the same structure as all_tracks_per_frame.
	Returns (refined_tracks_per_frame, id_mapping old->canonical).
	"""
	if not all_tracks_per_frame:
		return all_tracks_per_frame, {}

	by_id: dict[int, list[TrackObject]] = {}
	for frame_tracks in all_tracks_per_frame:
		for t in frame_tracks:
			by_id.setdefault(int(t.track_id), []).append(t)
	for tid in by_id:
		by_id[tid] = sorted(by_id[tid], key=lambda x: x.frame_id)

	segments = _make_segments(all_tracks_per_frame)
	tids_by_start = sorted(segments.keys(), key=lambda tid: segments[tid].start_frame)

	# Keep signature-compatible advanced params but clamp to safe ranges.
	min_direction_cos = max(-1.0, min(1.0, float(min_direction_cos)))
	max_speed_ratio = max(1.0, float(max_speed_ratio))
	gap_relax_factor = max(0.0, float(gap_relax_factor))

	id_map: dict[int, int] = {tid: tid for tid in segments.keys()}
	used_as_parent: set[int] = set()

	for dst_tid in tids_by_start:
		dst = segments[dst_tid]
		best_parent: int | None = None
		best_score = float("inf")

		for src_tid, src in segments.items():
			if src_tid == dst_tid:
				continue
			if src_tid in used_as_parent:
				continue
			if int(src.cls_id) != int(dst.cls_id):
				continue
			if src.end_frame >= dst.start_frame:
				continue

			gap = int(dst.start_frame - src.end_frame)
			if gap > max_frame_gap:
				continue

			src_scale = max(1e-6, math.sqrt(max(src.end_obj.area, 1.0)))
			dst_scale = max(1e-6, math.sqrt(max(dst.start_obj.area, 1.0)))
			ratio = max(src_scale / dst_scale, dst_scale / src_scale)
			if ratio > max_size_ratio:
				continue

			# Simple direction/speed sanity check to reduce false merges.
			svx, svy = _estimate_tail_velocity(by_id[src_tid])
			sn = math.hypot(svx, svy)
			dx = float(dst.start_obj.cx - src.end_obj.cx)
			dy = float(dst.start_obj.cy - src.end_obj.cy)
			dn = math.hypot(dx, dy)
			if sn > 1e-4 and dn > 1e-4:
				cos_sim = (svx * dx + svy * dy) / max(1e-6, sn * dn)
				if cos_sim < min_direction_cos:
					continue
				# speed implied by displacement over gap vs source speed
				implied_speed = dn / max(1.0, float(gap))
				spd_ratio = max(implied_speed / max(1e-6, sn), sn / max(1e-6, implied_speed))
				if spd_ratio > max_speed_ratio:
					continue

			score = _continuity_score(by_id[src_tid], src.end_obj, dst.start_obj)
			allowed = float(max_center_dist) * (1.0 + max(0, gap - 1) * gap_relax_factor)
			if score > allowed:
				continue
			if score < best_score:
				best_score = score
				best_parent = src_tid

		if best_parent is not None:
			id_map[dst_tid] = id_map.get(best_parent, best_parent)
			used_as_parent.add(best_parent)

	refined: list[list[TrackObject]] = []
	for frame_tracks in all_tracks_per_frame:
		new_frame_tracks: list[TrackObject] = []
		for t in frame_tracks:
			old_id = int(t.track_id)
			new_id = int(id_map.get(old_id, old_id))
			if new_id == old_id:
				new_frame_tracks.append(t)
			else:
				new_frame_tracks.append(t.model_copy(update={"track_id": new_id}))
		refined.append(new_frame_tracks)

	return refined, id_map

