# Flytracking

## Install

- [홈페이지](https://docs.conda.io/en/latest/miniconda.html)에서 `Miniconda3 Windows 64-bit` 설치
- 설치할 때 기본설정으로 하되, Advanced Options에서 Add Miniconda3 to my PATH environment variable를 선택하여 설치하면 된다.
- 설치한 후에 cmd 혹은 Windows terminal등을 열어서 `where conda`할때 경로가 출력되어야합니다.
- 설치한 후에 `python --version` 을 입력해서 파이썬 버전도 확인합니다.

- 데이터셋 준비
    - data
        - video
            - 영상.mp4
            - gt_count.json

- 실행

```
python main.py --data {데이터경로}

```

- 확장자가 .mp4가 아닐경우 `--ext avi`처럼 바꿀수있습니다.
- --no-overwrite을 붙이면 이미 분석된 데이터가 있다면 분석을 실행하지 않고 anaylsis 만 만듭니다.

# 아래와 같은 옵션들을 추가로 쓸 수 있습니다.
    '--data', type=str, "Path to dataset directory"
    '--skip', type=float, "Skip first {skip} frames as seconds"

    '--width', type=int, default=800
    '--height', type=int, default=800

    '--step', type=int, default=2
    '--refine-step', type=int, default=4

    '--cluster-distance-threshold', type=float, default=285,
                        help="cluster minimum distance (800 is 100%, default=285)"
    '--cluster-time-threshold', type=float, default=10.,
                        help="cluster minimum time (seconds)"
    '--cluster-outlier-threshold', type=int, default=5,
                        help="cluster outlier max count"
    '--interaction-step', type=float, default=3.,
                        help="interaction step (seconds)"
    '--interaction-distance-threshold', type=float, default=100,
                        help="interaction distance threshold"
    '--analysis-best-count', type=int, default=3,
                        help="distance report best n-th"

    '--color-density', type=str, default='Reds', choices=[
        'Reds', 'Blues', 'Jet',
        'Oranges', 'Purples', 'Greens',
        'Greys', 'BuGn', 'BuPu',
        'GnBu', 'OrRd', 'PuBu',
        'PuBuGn', 'PuRd', 'RdPu',
        'YlGn', 'YlGnBu', 'YlOrBr',
        'YlOrRd', 'Spectral', 'RdBu',
        'RdGy', 'RdYlBu', 'RdYlGn',
    ], help="density plot color"

    '--batch', type=int, default=2
    '--weights', type=str, default='weights/efficientdet-d4.pth'
    '--compound-coefficient', type=int, default=4
    '--gt', type=str, default='video/gt_count.json'

    '--init-frame', type=int, default=100,
                        help="Tracker initialize frame"

    '--no-dump', action='store_true', default=False,
                        help="Don't dump images"
    '--no-overwrite', action='store_true', default=False,
                        help="Use exist tracking results"
