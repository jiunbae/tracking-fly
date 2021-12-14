import typing
from pathlib import Path
from functools import cached_property
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as colors
from plotly.subplots import make_subplots
from plotly.figure_factory._2d_density import make_linear_colorscale

from flytracking.analysis import metric, vis


class Analysis:
    def __init__(
        self,
        tracks: np.ndarray,
        base_dir: typing.Union[str, Path],
        cluster_distance_threshold: float,
        cluster_time_threshold: float,
        cluster_outlier_threshold: int,
        interaction_step: int,
        interaction_distance_threshold: float,
        analysis_best_count: int,
        color_density: str,
        width: int = 800,
        height: int = 800,
        skip: int = 0,
        step: int = 1,
        fps: int = 30,
    ):
        """

        Args:
            tracks:
            base_dir:
            cluster_time_threshold: by frame
            width:
            height:
        """
        self.tracks = tracks
        self.centers = metric.center(self.tracks)
        self.base = Path(base_dir)

        self.cluster_distance_threshold = cluster_distance_threshold
        self.cluster_time_threshold = cluster_time_threshold
        self.cluster_outlier_threshold = cluster_outlier_threshold
        self.interaction_step = int(interaction_step * fps / step)
        self.interaction_distance_threshold = interaction_distance_threshold
        self.analysis_best_count = analysis_best_count

        self.color_density = color_density

        self.width = width
        self.height = height
        self.skip = skip
        self.step = step
        self.fps = fps

        self.object_count, self.frame_count, *_ = self.tracks.shape
        self.df = pd.DataFrame(
            np.concatenate((
                self.centers.reshape(-1, 2),
                np.repeat(np.arange(self.object_count), self.frame_count).reshape(-1, 1),
                np.tile(np.arange(self.frame_count), self.object_count).reshape(-1, 1),
            ), axis=-1),
            columns=['x', 'y', 'obj', 'frame'],
        )

        self.arena = (
            self.centers.min(axis=(0, 1)),
            self.centers.max(axis=(0, 1)),
        )

    @cached_property
    def get_distance(self) \
            -> np.ndarray:
        distance = np.zeros((self.object_count, self.object_count, self.frame_count))
        for i in range(self.object_count):
            for j in range(i + 1, self.object_count):
                distance[i, j] = distance[j, i] = np.hypot(*(self.centers[i] - self.centers[j]).T)

        return distance

    def get_cluster(self) \
            -> typing.Iterable[typing.Tuple[int, int]]:
        distance_all = self.get_distance
        distance = distance_all.mean(axis=(0, 1))
        filtered, *_ = np.where(distance < self.cluster_distance_threshold)
        diff = np.diff(filtered)
        flags, *_ = np.where(diff != 1)

        if len(flags) == 0:
            return

        for begin, end in zip(
            [0, *(flags + 1)],
            [*flags, len(filtered)],
        ):
            if end == len(filtered):
                end -= 1
            if filtered[end] - filtered[begin] >= self.cluster_time_threshold:
                begin_frame, end_frame = filtered[begin], filtered[end]

                tar_centers = self.centers[:, begin_frame:end_frame, :]
                object_distances = np.array([
                    np.hypot(*(obj_center - tar_centers.mean(axis=0)).T)
                    for obj_center in tar_centers
                ])
                obj_out_idx, *_ = np.where(
                    object_distances > self.cluster_distance_threshold
                )
                obj_out_idx = ([
                    np.count_nonzero(obj_out_idx == oid)
                    for oid in range(self.object_count)
                ])
                exclude_ids, *_ = np.where(obj_out_idx >= ((end - begin) // 2))
                if len(exclude_ids) < self.cluster_outlier_threshold:
                    yield begin_frame, end_frame, exclude_ids

    def dump_cluster(self):
        distance = self.get_distance
        times = np.around(
            np.linspace(
                *tuple(map(self.to_time, (0, self.frame_count))),
                self.frame_count,
            ),
            2,
        )
        self.draw_distance(distance, times=times)
        self.base.joinpath('distances').mkdir(exist_ok=True, parents=True)
        interaction_map = np.zeros((self.object_count, self.object_count, self.frame_count), dtype=bool)
        distance_dfs = []
        for tid in range(self.object_count):
            target = distance[tid].T
            target_exclude = np.concatenate(
                (
                    target[:, :tid],
                    target[:, tid + 1:],
                ),
                axis=-1,
            )
            frame_ids, target_ids = np.where(target_exclude < self.interaction_distance_threshold)
            interaction_map[tid, target_ids, frame_ids] = 1

            best_sample_idx = (-target_exclude).argsort(axis=-1)[:, :self.analysis_best_count]
            best_samples = target_exclude[np.unravel_index(best_sample_idx, target_exclude.shape)]
            best_columns = [f"top-{bid + 1:02d}" for bid in range(self.analysis_best_count)]

            sample_idx = best_sample_idx.copy()
            sample_idx_idx = np.where(best_sample_idx >= tid)
            sample_idx[sample_idx_idx] = sample_idx[sample_idx_idx] + 1

            distance_df = pd.DataFrame(
                np.concatenate(
                    (
                        np.expand_dims(
                            np.arange(self.frame_count) * self.step + self.skip,
                            axis=-1,
                        ),
                        np.expand_dims(
                            (np.arange(self.frame_count) * self.step + self.skip) / self.fps,
                            axis=-1,
                        ),
                        self.centers[tid],
                        target,
                        np.expand_dims(target.mean(axis=-1), axis=-1),
                        np.expand_dims(target.std(axis=-1), axis=-1),
                        sample_idx,
                        np.expand_dims(best_samples.mean(axis=-1), axis=-1),
                        np.expand_dims(best_samples.std(axis=-1), axis=-1),
                        np.expand_dims(interaction_map[tid].sum(axis=0), axis=-1),
                    ),
                    axis=-1,
                ),
                columns=[
                    "frame",
                    "seconds",
                    "x", "y",
                    *[f"fly-{obj_idx+1:02d}" for obj_idx in range(self.object_count)],
                    "mean", "std",
                    *best_columns,
                    "top-mean", "top-std",
                    "interaction",
                ],
            )
            distance_df.loc[:, best_columns] = 'fly-' + (distance_df.loc[:, best_columns] + 1).astype(int).astype(str)

            distance_df.to_csv(str(self.base.joinpath('distances', f'fly-{tid+1:02d}.csv')))
            distance_dfs.append(distance_df)

        *clusters, exc_ids = zip(*self.get_cluster())
        clusters = tuple(zip(*clusters))
        clusters = np.array(clusters) if len(clusters) else np.empty((0, 2))
        clusters_to_frame = clusters * self.step
        clusters_to_frame_diff = np.diff(clusters_to_frame, axis=-1)
        clusters_to_time = np.around(self.to_time(clusters), 2)
        clusters_to_time_diff = np.diff(clusters_to_time, axis=-1)
        interaction_counts = np.empty(len(clusters), dtype=int)
        interaction_count_by_cluster = np.empty((len(clusters), self.object_count), dtype=int)

        for cid, (begin, end) in enumerate(clusters):
            cluster_distance = distance[:, :, begin:end]

            for oid in range(self.object_count):
                object_distance = cluster_distance[oid]
                object_distance = np.concatenate((
                    object_distance[:oid],
                    object_distance[oid + 1:],
                ))

                interaction_batch = np.array_split(
                    object_distance,
                    object_distance.shape[1] / self.interaction_step,
                    axis=-1
                )

            interaction_count = np.diff(interaction_map[..., begin:end]).sum(axis=-1)
            interaction_counts[cid] = int(interaction_count.sum() / 2)
            interaction_count_by_cluster[cid] = interaction_count.sum(axis=-1)

        df = pd.DataFrame(
            np.concatenate(
                (
                    clusters_to_frame,
                    clusters_to_frame_diff,
                    clusters_to_time,
                    clusters_to_time_diff,
                    np.expand_dims(interaction_counts, axis=-1),
                    interaction_count_by_cluster,
                ),
                axis=1,
            ),
            columns=[
                'begin(frame)', 'end(frame)', 'diff(frame)',
                'begin(time)', 'end(time)', 'diff(time)',
                'interaction counts',
                *[f"fly-{obj_idx+1:02d}" for obj_idx in range(self.object_count)],
            ],
        )
        df.to_csv(str(self.base.joinpath('cluster.csv')), index=True)

        for idx, (begin, end) in enumerate(clusters):
            times = np.around(
                np.linspace(
                    *tuple(map(self.to_time, (begin, end))),
                    end - begin,
                ),
                2,
            )

            df = self.df[(begin <= self.df.frame) & (self.df.frame < end)]
            dist = distance[:, :, begin:end]

            prefix_path = f'clusters/{idx:04d}'
            tar_path = self.base.joinpath(prefix_path)
            tar_path.mkdir(exist_ok=True, parents=True)

            self.draw_distance(dist, times=times, prefix=prefix_path)
            self.draw_distance_heatmap(dist, times=times, prefix=prefix_path)
            self.draw_position_heatmap(df, dist, times=times, prefix=prefix_path)
            self.draw_animated_tracks(df, dist, times=times, prefix=prefix_path)
            self.draw_density(df, dist, times=times, prefix=prefix_path)

            tar_path.joinpath('distances').mkdir(exist_ok=True, parents=True)
            for oid, distance_df in enumerate(distance_dfs):
                df = distance_df[(begin*self.step + self.skip <= distance_df.frame) & (distance_df.frame < end*self.step + self.skip)]
                df.to_csv(str(tar_path.joinpath('distances', f'fly-{oid+1:02d}.csv')))

    def to_time(self, frame):
        return (frame * self.step + self.skip) / self.fps

    def add_arena(self, fig: go.Figure):
        (x0, y0), (x1, y1) = self.arena
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
            line_color="black",
        )

    def draw_images(self, fig: go.Figure, path: str, batch: int = 1, postfix: int = 1):
        path = self.base.joinpath(path)
        # path.mkdir(exist_ok=True, parents=True)
        # for data_idx in range(0, len(fig.data) - postfix, batch):
        #     for batch_idx in range(data_idx, data_idx + batch):
        #         fig.data[batch_idx].visible = True
        #     fig.write_image(str(path.joinpath(f'{data_idx // batch:06d}.jpg')))
        #     for batch_idx in range(data_idx, data_idx + batch):
        #         fig.data[batch_idx].visible = False
        for batch_idx in range(0, batch):
            fig.data[batch_idx].visible = True

    def draw_distance(self, distance: np.ndarray, times: np.ndarray, prefix: str = '.'):
        distances = distance.mean(axis=0)
        distance = distance.mean(axis=(0, 1))
        x = times

        fig = make_subplots(rows=2, cols=1, row_heights=[.95, .05], vertical_spacing=.02)
        fig.update_layout(
            autosize=False,
            xaxis_range=[x[0], x[-1]],
            width=1500,
            height=1000,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=distance,
                text=times,
                mode='markers+lines',
                name=f"fly-all",
            ),
            row=1,
            col=1,
        )
        for obj_idx, dist_value in enumerate(distances):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=dist_value,
                    text=times,
                    mode='markers+lines',
                    name=f"fly-{int(obj_idx)+1:02d}",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Heatmap(
                visible=True,
                z=distance.reshape(1, -1),
                showscale=False,
            ),
            row=2,
            col=1,
        )
        fig.write_html(str(self.base.joinpath(f'{prefix}/distance.html')))

    def draw_distance_heatmap(self, distance: np.ndarray, times: np.ndarray, prefix: str = '.'):
        fig = make_subplots(rows=2, cols=1, row_heights=[.95, .05], vertical_spacing=.02)
        fig.update_layout(
            autosize=False,
            width=1060,
            height=1111,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        for idx in range(distance.shape[-1]):
            fig.add_trace(
                go.Heatmap(
                    visible=False,
                    z=distance[:, :, idx],
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Heatmap(
                visible=True,
                z=distance.mean(axis=(0, 1)).reshape(1, -1),
            ),
            row=2,
            col=1,
        )

        self.draw_images(fig, f'{prefix}-distance-heatmap')
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "frame: "},
                steps=[
                    dict(
                        label=f"{times[idx]}s",
                        method="update",
                        args=[{
                            "visible": [False] * idx + [True] + [False] * (distance.shape[-1] - 1 - idx) + [True],
                        }],
                    ) for idx in range(distance.shape[-1])
                ]
            )
        ]

        fig.update_layout(
            sliders=sliders,
        )
        fig.write_html(str(self.base.joinpath(f'{prefix}/distance-heatmap.html')))

    def draw_position_heatmap(
        self,
        df: pd.DataFrame,
        distance: np.ndarray,
        times: np.ndarray,
        bins: int = 16,
        prefix: str = '.'
    ):
        heatmap = np.zeros((distance.shape[-1], bins, bins))

        fig = make_subplots(rows=2, cols=1, row_heights=[.95, .05], vertical_spacing=.02)
        fig.update_layout(
            autosize=False,
            width=1060,
            height=1111,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        begin = int(df.frame.values[0])

        def heatmap_update(ary: np.ndarray):
            x, y, f = ary
            heatmap[f - begin, x, y] += 1

        np.apply_along_axis(
            heatmap_update,
            1, 
            (
                df.values[:, (0, 1, 3)] / np.array(
                    (self.width / bins, self.height / bins, 1)
                )
            ).astype(int),
        )

        for idx in range(distance.shape[-1]):
            fig.add_trace(
                go.Heatmap(
                    visible=False,
                    z=heatmap[idx],
                    zmin=0,
                    zmax=5,
                ),
                row=1, col=1,
            )

        fig.add_trace(
            go.Heatmap(
                visible=True,
                z=distance.mean(axis=(0, 1)).reshape(1, -1),
            ),
            row=2,
            col=1,
        )

        self.draw_images(fig, f'{prefix}-position-heatmap')
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "frame: "},
                steps=[
                    dict(
                        label=f"{times[idx]}s",
                        method="update",
                        args=[{
                            "visible": [False] * idx + [True] + [False] * (distance.shape[-1] - 1 - idx) + [True],
                        }],
                    ) for idx in range(distance.shape[-1])
                ]
            )
        ]

        fig.update_layout(
            sliders=sliders,
        )
        fig.write_html(str(self.base.joinpath(f'{prefix}/position-heatmap.html')))

    def draw_animated_tracks(
        self,
        df: pd.DataFrame,
        distance: np.ndarray,
        times: np.ndarray,
        prefix: str = '.'
    ):
        fig = make_subplots(
            row_heights=[0.95, 0.05],
            vertical_spacing=0.02,
            rows=2, cols=1,
        )
        fig.update_layout(
            yaxis_range=[0, self.width],
            xaxis_range=[0, self.height],
            autosize=False,
            width=1004,
            height=1050,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        # fig.update_traces(
        #     textposition='outside',
        #     textfont_size=22,
        # )

        for frame_idx, frame in df.groupby(['frame']):
            fig.add_trace(
                go.Scatter(
                    x=frame['x'],
                    y=frame['y'],
                    visible=False,
                    mode='markers+text',
                    marker=dict(
                        size=14,
                    ),
                    name=f'frame{int(frame_idx):02d}',
                    text=frame['obj'] + 1,
                    textposition="top right",
                    textfont=dict(
                        size=22,
                        color="crimson"
                    )
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Heatmap(
                visible=True,
                z=distance.mean(axis=(0, 1)).reshape(1, -1),
            ),
            row=2,
            col=1,
        )

        self.draw_images(fig, f'{prefix}-animated-tracks')
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "frame: "},
                steps=[
                    dict(
                        label=f"{times[idx]}s",
                        method="update",
                        args=[{
                            "visible": [False] * idx + [True] + [False] * (distance.shape[-1] - 1 - idx) + [True],
                        }],
                    ) for idx in range(distance.shape[-1])
                ]
            )
        ]

        self.add_arena(fig)
        fig.update_layout(
            sliders=sliders,
        )
        fig.write_html(str(self.base.joinpath(f'{prefix}/animated-tracks.html')))

    def draw_tracks(self):
        colors = vis.colors()

        fig = go.Figure()
        fig.update_layout(
            yaxis_range=[0, self.width],
            xaxis_range=[0, self.height],
            autosize=False,
            width=1145,
            height=1000,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        for obj_idx, frame in self.df.groupby(['obj']):
            color = colors[int(obj_idx)]
            fig.add_trace(
                go.Scatter(
                    x=frame['x'],
                    y=frame['y'],
                    text=frame['frame'].astype(int),
                    mode="lines+markers",
                    line=dict(color=f'rgb{color}'),
                    marker=dict(color=f'rgb{color}'),
                    name=f"fly-{int(obj_idx)+1:02d}",
                    visible="legendonly",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=frame['x'][:1],
                    y=frame['y'][:1],
                    text='start',
                    mode="markers",
                    name=f"fly-{int(obj_idx)+1:02d}_start",
                    marker=dict(size=16, color=f'rgb{color}'),
                )
            )

        self.add_arena(fig)
        fig.write_html(str(self.base.joinpath('all-tracks.html')))

    def draw_density(
        self,
        df: pd.DataFrame,
        distance: np.ndarray,
        times: np.ndarray,
        ncontours=20,
        hist_color=(0, 0, .5),
        point_color=(0, 0, .5),
        point_size=10,
        point_opacity=.4,
        prefix: str = '.',
    ):
        hist_color = colors.validate_colors(hist_color, "rgb")
        point_color = colors.validate_colors(point_color, "rgb")

        fig = go.Figure()
        fig.update_layout(
            showlegend=False,
            autosize=False,
            height=self.height,
            width=self.width,
            margin=dict(l=20, t=20, b=20, r=20),
            hovermode="closest",
            xaxis_range=[0, self.width],
            yaxis_range=[0, self.height],
            xaxis=dict(domain=[.0, 0.9], showgrid=False, zeroline=False),
            yaxis=dict(domain=[.05, 0.9], showgrid=False, zeroline=False),
            xaxis2=dict(domain=[.9, 1.], showgrid=False, zeroline=False),
            yaxis2=dict(domain=[.9, 1.], showgrid=False, zeroline=False),
            xaxis3=dict(domain=[.0, 1.], showgrid=False, zeroline=False),
            yaxis3=dict(domain=[.0, .05], showgrid=False, zeroline=False),
            bargap=0,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        for _, frame in df.groupby(['frame']):
            fig.add_trace(
                go.Histogram2dContour(
                    visible=False,
                    x=frame['x'],
                    y=frame['y'],
                    name="density",
                    ncontours=ncontours,
                    colorscale=self.color_density,
                    # reversescale=True,
                    showscale=False,
                    contours_showlines=False,
                    hoverlabel=dict(
                        bgcolor='white',
                        bordercolor='black',
                        font=dict(
                            family='Raleway',
                            color='black'
                        )
                    ),
                    histnorm="probability",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    x=frame['x'],
                    y=frame['y'],
                    mode="markers",
                    name="points",
                    marker=dict(
                        color=point_color[0],
                        size=point_size,
                        opacity=point_opacity,
                    ),
                    text=frame['obj'],
                ),
            )
            fig.add_trace(
                go.Histogram(
                    visible=False,
                    x=frame['x'],
                    name="x density",
                    marker=dict(color=hist_color[0]),
                    yaxis="y2",
                ),
            )
            fig.add_trace(
                go.Histogram(
                    visible=False,
                    y=frame['y'],
                    name="y density",
                    marker=dict(color=hist_color[0]),
                    xaxis="x2",
                ),
            )

        dist = distance.mean(axis=(0, 1))
        min_, max_ = dist.min(), dist.max()
        dist = (dist - min_) / (max_ - min_) * self.width
        fig.add_trace(
            go.Heatmap(
                visible=True,
                z=dist.reshape(1, -1),
                xaxis="x3",
                yaxis="y3",
            ),
        )

        self.draw_images(fig, f'{prefix}-density', batch=4)
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "frame: "},
                steps=[
                    dict(
                        label=f"{times[idx]}s",
                        method="update",
                        args=[{
                            "visible": [False] * (idx * 4) + [True] * 4 + [False] * (
                                    (distance.shape[-1] - 1 - idx) * 4) + [True],
                        }],
                    ) for idx in range(distance.shape[-1])
                ]
            )
        ]

        self.add_arena(fig)
        fig.update_layout(
            sliders=sliders,
        )
        fig.write_html(str(self.base.joinpath(f'{prefix}/density.html')))
