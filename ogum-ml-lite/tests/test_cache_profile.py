from pathlib import Path

from app.services import cache, profiling
from ogum_lite.ui.workspace import Workspace


@cache.cache_result(
    "demo",
    inputs=lambda value, **_: [value],
    params=lambda value, **_: {"value": value},
)
@profiling.profile_step("demo")
def _expensive_operation(value: int, *, workspace: Workspace) -> str:
    target = workspace.resolve(f"out_{value}.txt")
    target.write_text(str(value), encoding="utf-8")
    return str(target)


def test_cache_hit_and_stats(tmp_path):
    workspace = Workspace(tmp_path)

    first = _expensive_operation(5, workspace=workspace)
    assert Path(first).exists()
    assert cache.last_cache_hit() is False
    profile = profiling.get_last_profile()
    assert profile is not None
    assert profile["step"] == "demo"

    second = _expensive_operation(5, workspace=workspace)
    assert Path(second).exists()
    assert cache.last_cache_hit() is True

    stats = cache.cache_stats(workspace.resolve(".cache"))
    assert stats["entries"] == 1
    assert stats["tasks"]["demo"]["hits"] >= 2

    third = _expensive_operation(6, workspace=workspace)
    assert Path(third).exists()
    stats_after = cache.cache_stats(workspace.resolve(".cache"))
    assert stats_after["entries"] == 2
