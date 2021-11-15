import typing
from pathlib import Path
from functools import cached_property

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
            width: int = 800,
            height: int = 800,
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
        self.width = width
        self.height = height

        self.object_count, self.frame_count, *_ = self.tracks.shape
        self.df = pd.DataFrame(
            np.concatenate((
                self.centers.reshape(-1, 2),
                np.repeat(np.arange(self.object_count), self.frame_count).reshape(-1, 1),
                np.tile(np.arange(self.frame_count), self.object_count).reshape(-1, 1),
            ), axis=-1),
            columns=['x', 'y', 'obj', 'frame'],
        )

    @cached_property
    def get_distance(self) \
            -> np.ndarray:
        distance = np.zeros((self.object_count, self.object_count, self.frame_count))
        for i in range(self.object_count):
            for j in range(i + 1, self.object_count):
                distance[i, j] = distance[j, i] = np.hypot(*(self.centers[i] - self.centers[j]).T)

        return distance

    def draw_distance_heatmap(self):
        fig = make_subplots(rows=2, cols=1, row_heights=[0.95, 0.05], vertical_spacing=0.02)

        distance = self.get_distance

        for idx in range(self.frame_count):
            fig.add_trace(
                go.Heatmap(
                    visible=False,
                    z=distance[..., idx],
                ),
                row=1,
                col=1,
            )
        fig.data[0].visible = True

        fig.add_trace(
            go.Heatmap(
                visible=True,
                z=distance.mean(axis=(0, 1)).reshape(1, -1),
            ),
            row=2,
            col=1,
        )

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "frame: "},
                steps=[
                    dict(
                        method="update",
                        args=[{
                            "visible": [False] * idx + [True] + [False] * (self.frame_count - 1 - idx) + [True],
                        }],
                    ) for idx in range(self.frame_count)
                ]
            )
        ]
        fig.update_layout(
            sliders=sliders,
            autosize=False,
            width=1060,
            height=1111,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.write_html(str(self.base.joinpath('distance-heatmap.html')))

    def draw_position_heatmap(self, bins: int = 16):
        fig = make_subplots(rows=2, cols=1, row_heights=[0.95, 0.05], vertical_spacing=0.02)

        distance = self.get_distance
        heatmap = np.zeros((self.frame_count, bins, bins))

        def heatmap_update(ary: np.ndarray):
            x, y, f = ary
            heatmap[f, x, y] += 1

        np.apply_along_axis(heatmap_update, 1, (
                    self.df.values[:, (0, 1, 3)] / np.array((self.width / bins, self.height / bins, 1))).astype(int))

        for idx in range(self.frame_count):
            fig.add_trace(
                go.Heatmap(
                    visible=False,
                    z=heatmap[idx],
                    zmin=0,
                    zmax=5,
                ),
                row=1, col=1,
            )
        fig.data[0].visible = True

        fig.add_trace(
            go.Heatmap(
                visible=True,
                z=distance.mean(axis=(0, 1)).reshape(1, -1),
            ),
            row=2,
            col=1,
        )

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "frame: "},
                steps=[
                    dict(
                        method="update",
                        args=[{
                            "visible": [False] * idx + [True] + [False] * (self.frame_count - 1 - idx) + [True],
                        }],
                    ) for idx in range(self.frame_count)
                ]
            )
        ]
        fig.update_layout(
            sliders=sliders,
            autosize=False,
            width=1060,
            height=1111,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.write_html(str(self.base.joinpath('position-heatmap.html')))

    def draw_animated_tracks(self):
        fig = make_subplots(
            row_heights=[0.95, 0.05],
            vertical_spacing=0.02,
            rows=2, cols=1,
        )

        distance = self.get_distance

        for frame_idx, frame in self.df.groupby(['frame']):
            fig.add_trace(
                go.Scatter(
                    x=frame['x'],
                    y=frame['y'],
                    text=frame['obj'],
                    visible=False,
                    mode='markers',
                    marker=dict(size=10),
                    name=f'frame{int(frame_idx):02d}',
                ),
                row=1,
                col=1,
            )
        fig.data[0].visible = True
        fig.add_trace(
            go.Heatmap(
                visible=True,
                z=distance.mean(axis=(0, 1)).reshape(1, -1),
            ),
            row=2,
            col=1,
        )

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "frame: "},
                steps=[
                    dict(
                        method="update",
                        args=[{
                            "visible": [False] * idx + [True] + [False] * (self.frame_count - 1 - idx) + [True],
                        }],
                    ) for idx in range(self.frame_count)
                ]
            )
        ]

        fig.update_layout(
            sliders=sliders,
            yaxis_range=[0, self.width],
            xaxis_range=[0, self.height],
            autosize=False,
            width=1004,
            height=1050,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.write_html(str(self.base.joinpath('animated-tracks.html')))

    def draw_tracks(self):
        fig = go.Figure()
        colors = vis.colors()
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
                    name=f"trace{int(obj_idx):02d}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=frame['x'][:1],
                    y=frame['y'][:1],
                    text='start',
                    mode="markers",
                    name=f"trace{int(obj_idx):02d}_start",
                    marker=dict(size=16, color=f'rgb{color}'),
                )
            )

        fig.update_layout(
            yaxis_range=[0, self.width],
            xaxis_range=[0, self.height],
            autosize=False,
            width=1145,
            height=1000,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.write_html(str(self.base.joinpath('all-tracks.html')))

    def draw_density(
        self,
        colorscale="Earth",
        ncontours=20,
        hist_color=(0, 0, .5),
        point_color=(0, 0, .5),
        point_size=10,
        point_opacity=.4,
    ):
        distance = self.get_distance

        colorscale = colors.validate_colors(colorscale, "rgb")
        colorscale = make_linear_colorscale(colorscale)

        hist_color = colors.validate_colors(hist_color, "rgb")
        point_color = colors.validate_colors(point_color, "rgb")

        fig = go.Figure()

        for frame_idx, frame in self.df.groupby(['frame']):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    x=frame['x'],
                    y=frame['y'],
                    mode="markers",
                    name="points",
                    marker=dict(color=point_color[0], size=point_size, opacity=point_opacity),
                ),
            )
            fig.add_trace(
                go.Histogram2dContour(
                    visible=False,
                    x=frame['x'],
                    y=frame['y'],
                    name="density",
                    ncontours=ncontours,
                    colorscale=colorscale,
                    reversescale=True,
                    showscale=False,
                ),
            )
            fig.add_trace(
                go.Histogram(
                    visible=False,
                    x=frame['x'], name="x density", marker=dict(color=hist_color[0]), yaxis="y2"
                ),
            )
            fig.add_trace(
                go.Histogram(
                    visible=False,
                    y=frame['y'], name="y density", marker=dict(color=hist_color[0]), xaxis="x2"
                ),
            )

        for fig_data in fig.data[:4]:
            fig_data.visible = True

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

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "frame: "},
                steps=[
                    dict(
                        method="update",
                        args=[{
                            "visible": [False] * (idx * 4) + [True] * 4 + [False] * (
                                    (self.frame_count - 1 - idx) * 4) + [True],
                        }],
                    ) for idx in range(self.frame_count)
                ]
            )
        ]

        fig.update_layout(
            sliders=sliders,
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
        )
        fig.write_html(str(self.base.joinpath('density.html')))

    def draw_all(self):
        self.draw_distance_heatmap()
        self.draw_position_heatmap()
        self.draw_animated_tracks()
        self.draw_tracks()
        self.draw_density()
