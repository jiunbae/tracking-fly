import typing
from functools import partial
from operator import itemgetter

from .utils import metric

import numpy as np


class Tracker:
    def __init__(
            self,
            gt_count: int = 0,
            center: typing.Tuple[int, int] = (400, 400),
            sigma_l: float = .05, sigma_h: float = .5,
            sigma_iou: float = .35, t_min: int = 10,
            init_idx: int = 100,
    ):
        self.gt_count = gt_count
        self.center = center
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min

        self.frame_idx = -1
        self.init_idx = init_idx
        self.tracks_active = []
        self.tracks_finished = []
        self.trusts = []

    def update(self, boxes: np.ndarray):



        # calculate IoUs
        # if self.tracks_finished:
        #     iou_map = np.apply_along_axis(
        #         func1d=lambda box: [metric.iou(box, t['bboxes'][-1]) for t in self.tracks_active],
        #         axis=-1,
        #         arr=bboxes[:, :4],
        #     )
        #     uniques = metric.get_unique(iou_map)

        self.frame_idx += 1
        if not self.frame_idx:
            self.tracks_active = [
                {
                    "bboxes": [np.array(box)],
                    "max_score": score,
                    "start_frame": self.frame_idx,
                    "lost": [-1],
                    "skip": False,
                } for *box, score in boxes
            ]
            return

        elif self.frame_idx < self.init_idx:
            if len(boxes) == self.gt_count:
                self.trusts.append(boxes.copy())

        elif self.frame_idx == self.init_idx:
            if self.gt_count:
                if len(self.tracks_active) != self.gt_count:
                    print(
                        f'Warning! gt count {self.gt_count} is not matched tracking {len(self.tracks_active)}'
                        'Use trust gt for initialization'
                    )
                    if len(boxes) > self.gt_count:
                        max_distance = metric.distance((*self.center, *self.center), (0, 0, 0, 0))
                        distance_score = np.array([
                            -(score + (1 - metric.distance((*self.center, *self.center), box) / max_distance))
                            for *box, score in boxes
                        ])
                        boxes = boxes[distance_score.argsort()][:self.gt_count]
                    else:
                        not_added_trust = []
                        for trust in self.trusts:
                            visit = np.zeros(len(boxes), dtype=bool)
                            for track in self.tracks_active:
                                not_visited_index, *_ = np.where(~visit)
                                best_idx, best_iou, best_box = metric.get_best_match(
                                    track['bboxes'][-1],
                                    trust[not_visited_index, :4],
                                )
                                best_idx = not_visited_index[best_idx]
                                visit[best_idx] = True
                            not_visited_index, *_ = np.where(~visit)
                            not_added_trust.append(not_visited_index)

                        for *box, score in self.trusts[-1][not_added_trust[-1]]:
                            self.tracks_active.append(
                                {
                                    "bboxes": [np.array(box)],
                                    "max_score": score,
                                    "start_frame": self.frame_idx,
                                    "lost": [-1],
                                    "skip": False,
                                }
                            )

        updated_tracks, missing_tracks = [], []
        visit = np.zeros(len(boxes), dtype=bool)
        for track in self.tracks_active:
            best_idx, best_iou, best_box = 0, 0, 0
            if visit.sum() < len(boxes):
                not_visited_index, *_ = np.where(~visit)
                best_idx, best_iou, best_box = metric.get_best_match(
                    track['bboxes'][-1],
                    boxes[not_visited_index, :4],
                )
                best_idx = not_visited_index[best_idx]

                check_idx, check_iou, check_box = metric.get_best_match(
                    best_box,
                    [
                        track['bboxes'][-1]
                        for track in self.tracks_active
                    ],
                )

                if (0. < best_iou < check_iou) and not track['skip']:
                    track['skip'] = True
                    self.tracks_active.append(track)
                    continue
            else:
                pass

            if best_iou > 0.:
                if track['lost'][-1] != -1:
                    track['lost'].append(-1)
                track['bboxes'].append(best_box)
                updated_tracks.append(track)
                visit[best_idx] = True

            else:
                # if track['lost'][-1] == -1 or track['lost'][-1] + self.t_min >= self.frame_idx:
                if track['lost'][-1] == -1:
                    track['lost'][-1] = self.frame_idx

                best_idx, best_iou, best_box = metric.get_best_match(
                    track['bboxes'][-1],
                    boxes[:, :4],
                )

                best_box = track['bboxes'][-1] if best_iou <= 0. else \
                    metric.blend(track['bboxes'][-1], best_box, best_iou)
                track['bboxes'].append(best_box)
                missing_tracks.append(track)
                #
                # elif track['lost'][-1] + self.t_min < self.frame_idx:
                #     if track['start_frame'] + 1 != track['lost'][-1]:
                #         self.tracks_finished.append(track)

                # else:
                #     raise RuntimeError("Not implemented error -1")

        not_visited_index, *_ = np.where(~visit)
        for visit_idx in not_visited_index:
            if len(missing_tracks) <= 0:
                break

            dists = np.array(list(map(
                partial(metric.distance, boxes[visit_idx, :4]),
                [track['bboxes'][-1] for track in missing_tracks],
            )))
            best_idx = dists.argmin()
            if boxes[visit_idx, -1] > self.sigma_l and dists[best_idx] < metric.size(boxes[visit_idx, :4]) * 3:
                missing_tracks[best_idx]['bboxes'][-1] = boxes[visit_idx, :4]
                missing_tracks[best_idx]['lost'].append(-1)
                updated_tracks.append(missing_tracks[best_idx])
                del missing_tracks[best_idx]

        self.tracks_active = updated_tracks + missing_tracks
        for track in self.tracks_active:
            track['skip'] = False

    def release(self) \
            -> typing.List[typing.Dict[str, typing.Any]]:
        return [
            {
                "bboxes": np.array(track['bboxes']),
                "start_frame": track['start_frame'],
                "lost": track['lost'],
            } for track in self.tracks_active
        ]
