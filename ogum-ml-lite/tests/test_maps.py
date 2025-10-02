import pandas as pd
from ogum_lite.maps import prepare_segment_heatmap, render_segment_heatmap


def test_prepare_segment_heatmap_extracts_blaine_columns(tmp_path) -> None:
    data = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "fixed_seg1_blaine_n": [2.1, 2.3],
            "fixed_seg2_blaine_n": [2.8, 2.5],
            "fixed_seg1_blaine_mse": [0.01, 0.02],
            "fixed_seg2_blaine_mse": [0.05, 0.03],
        }
    )

    matrix = prepare_segment_heatmap(data, metric="blaine_n")
    assert list(matrix.columns) == ["fixed_seg1", "fixed_seg2"]
    assert list(matrix.index) == ["s1", "s2"]

    png_bytes = render_segment_heatmap(matrix, title="Blaine n heatmap")
    output = tmp_path / "blaine_n.png"
    output.write_bytes(png_bytes)
    assert output.exists()
    assert output.stat().st_size > 0
