"""Tests for scripts/docs_versions.py: the deploy-order guard that keeps an
older patch from overwriting its minor's docs folder, and the one-time prune
of patch folders into symlink aliases of the kept version of each minor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import docs_versions as dv  # noqa: E402

pytestmark = pytest.mark.unit

# versions.json on gh-pages as of 2026-11-06, newest first.
PUBLISHED_2026_11_06 = [
    "0.22.2", "0.22.1", "0.22.0", "0.21.0", "0.20.2", "0.20.1", "0.20.0",
    "0.19.3", "0.19.2", "0.19.1", "0.19.0", "0.18.4", "0.18.3", "0.18.2",
    "0.18.1", "0.18.0", "0.17.3", "0.17.2", "0.17.1", "0.17.0", "0.16.5",
    "0.16.4", "0.16.3", "0.16.2", "0.16.1", "0.16.0", "0.15.2", "0.15.1",
    "0.15.0", "0.14.0", "0.13.8", "0.13.7", "0.13.6", "0.13.5", "0.13.4",
    "0.13.3", "0.13.2", "0.13.1", "0.13.0",
]  # fmt: skip


def _entries(versions: list[str], latest: str | None = None) -> list[dict]:
    latest = versions[0] if latest is None else latest
    return [
        {"version": v, "title": v, "aliases": ["latest"] if v == latest else []} for v in versions
    ]


def _minor_entry(minor: str, title: str, aliases: list[str] | None = None) -> dict:
    return {"version": minor, "title": title, "aliases": aliases or []}


class TestCheckDeploy:
    def test_newer_patch_over_its_minor_folder(self) -> None:
        dv.check_deploy([_minor_entry("0.22", "0.22.3", ["latest"])], "0.22.4")

    def test_same_version_redeploy_is_allowed(self) -> None:
        dv.check_deploy([_minor_entry("0.22", "0.22.3", ["latest"])], "0.22.3")

    def test_older_patch_than_the_minor_folder_title_is_refused(self) -> None:
        with pytest.raises(dv.DocsVersionsError, match=r"0\.22\.3.*0\.22\.4"):
            dv.check_deploy([_minor_entry("0.22", "0.22.4", ["latest"])], "0.22.3")

    def test_older_patch_than_a_legacy_patch_folder_is_refused(self) -> None:
        with pytest.raises(dv.DocsVersionsError, match=r"0\.22\.1.*0\.22\.2"):
            dv.check_deploy(_entries(PUBLISHED_2026_11_06), "0.22.1")

    def test_newer_patch_than_the_legacy_patch_folders(self) -> None:
        dv.check_deploy(_entries(PUBLISHED_2026_11_06), "0.22.3")

    def test_first_release_of_a_new_minor(self) -> None:
        dv.check_deploy(_entries(PUBLISHED_2026_11_06), "0.23.0")

    def test_empty_versions_json(self) -> None:
        dv.check_deploy([], "0.1.0")

    def test_ignores_other_minors(self) -> None:
        dv.check_deploy([_minor_entry("0.23", "0.23.5", ["latest"])], "0.22.9")

    def test_minor_folder_titled_with_something_else_is_refused(self) -> None:
        with pytest.raises(dv.DocsVersionsError, match="title"):
            dv.check_deploy([_minor_entry("0.22", "0.22", ["latest"])], "0.22.4")

    @pytest.mark.parametrize("bad", ["0.22", "v0.22.3", "0.22.3rc1", ""])
    def test_version_must_be_a_plain_patch_version(self, bad: str) -> None:
        with pytest.raises(dv.DocsVersionsError):
            dv.check_deploy([], bad)


class TestPlanPrune:
    def test_published_tree_keeps_newest_patch_per_minor(self) -> None:
        groups = dv.plan_prune(_entries(PUBLISHED_2026_11_06))

        assert [g.keeper for g in groups] == [
            "0.22.2", "0.20.2", "0.19.3", "0.18.4", "0.17.3", "0.16.5", "0.15.2", "0.13.8",
        ]  # fmt: skip
        assert sum(len(g.pruned) for g in groups) == 29
        by_keeper = {g.keeper: g.pruned for g in groups}
        assert by_keeper["0.22.2"] == ("0.22.0", "0.22.1")
        assert by_keeper["0.13.8"] == tuple(f"0.13.{i}" for i in range(8))
        # minors with a single version (0.21, 0.14) have nothing to prune
        assert "0.21.0" not in by_keeper and "0.14.0" not in by_keeper

    def test_minor_folder_is_the_keeper(self) -> None:
        entries = [_minor_entry("0.22", "0.22.3", ["latest"])] + _entries(
            ["0.22.3", "0.22.2", "0.22.1", "0.22.0", "0.21.0"], latest=""
        )

        groups = dv.plan_prune(entries)

        assert groups == [dv.PruneGroup("0.22", ("0.22.0", "0.22.1", "0.22.2", "0.22.3"))]

    def test_patch_order_is_numeric(self) -> None:
        groups = dv.plan_prune(_entries(["0.13.10", "0.13.9", "0.13.2"]))

        assert groups == [dv.PruneGroup("0.13.10", ("0.13.2", "0.13.9"))]

    def test_pruned_tree_needs_nothing(self) -> None:
        entries = [
            _minor_entry("0.22", "0.22.3", ["latest", "0.22.0", "0.22.1"]),
            {"version": "0.21.0", "title": "0.21.0", "aliases": []},
        ]

        assert dv.plan_prune(entries) == []

    def test_refuses_to_prune_a_version_that_carries_an_alias(self) -> None:
        entries = _entries(["0.22.2", "0.22.1"], latest="0.22.1")

        with pytest.raises(dv.DocsVersionsError, match="0.22.1"):
            dv.plan_prune(entries)

    def test_moves_the_patch_aliases_of_a_pruned_keeper(self) -> None:
        # A prune before the first minor-folder deploy left 0.22.0 to 0.22.2 on 0.22.3.
        entries = [
            _minor_entry("0.22", "0.22.4", ["latest"]),
            _minor_entry("0.22.3", "0.22.3", ["0.22.2", "0.22.0", "0.22.1"]),
        ]

        groups = dv.plan_prune(entries)

        assert groups == [dv.PruneGroup("0.22", ("0.22.3",), ("0.22.0", "0.22.1", "0.22.2"))]
        assert "0.22.0" in dv.format_plan(entries, groups)

    @pytest.mark.parametrize("alias", ["stable", "0.21.0", "0.22"])
    def test_refuses_to_move_an_alias_that_is_not_a_patch_of_its_minor(self, alias: str) -> None:
        entries = [_minor_entry("0.22", "0.22.4"), _minor_entry("0.22.3", "0.22.3", [alias])]

        with pytest.raises(dv.DocsVersionsError, match=alias):
            dv.plan_prune(entries)

    def test_refuses_invalid_entries(self) -> None:
        with pytest.raises(dv.DocsVersionsError):
            dv.plan_prune(_entries(["0.22.2", "dev"]))


@pytest.fixture
def mike_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(dv, "_run", fake_run)
    return calls


class TestApplyPlan:
    def test_one_delete_then_one_symlink_alias_per_minor(self, mike_calls: list) -> None:
        groups = [
            dv.PruneGroup("0.22", ("0.22.0", "0.22.1")),
            dv.PruneGroup("0.13.8", ("0.13.0",)),
        ]

        dv.apply_plan(groups)

        assert mike_calls == [
            ["mike", "delete", "0.22.0", "0.22.1", "0.13.0"],
            ["mike", "alias", "--alias-type", "symlink", "0.22", "0.22.0", "0.22.1"],
            ["mike", "alias", "--alias-type", "symlink", "0.13.8", "0.13.0"],
        ]
        assert not any("--push" in call or "-p" in call for call in mike_calls)

    def test_moved_aliases_follow_the_pruned_names(self, mike_calls: list) -> None:
        dv.apply_plan([dv.PruneGroup("0.22", ("0.22.3",), ("0.22.0", "0.22.1"))])

        assert mike_calls == [
            ["mike", "delete", "0.22.3"],
            ["mike", "alias", "--alias-type", "symlink", "0.22", "0.22.3", "0.22.0", "0.22.1"],
        ]

    def test_empty_plan_runs_nothing(self, mike_calls: list) -> None:
        dv.apply_plan([])

        assert mike_calls == []


class TestVerify:
    GROUPS = [dv.PruneGroup("0.22", ("0.22.0", "0.22.1"))]

    def test_one_entry_per_minor_with_pruned_ids_as_aliases(self) -> None:
        dv.verify(
            [
                _minor_entry("0.22", "0.22.3", ["latest", "0.22.0", "0.22.1"]),
                {"version": "0.21.0", "title": "0.21.0", "aliases": []},
            ],
            self.GROUPS,
        )

    def test_two_entries_for_one_minor_fail(self) -> None:
        with pytest.raises(dv.DocsVersionsError, match="0.22"):
            dv.verify(
                [
                    _minor_entry("0.22", "0.22.3", ["latest", "0.22.0", "0.22.1"]),
                    {"version": "0.22.2", "title": "0.22.2", "aliases": []},
                ],
                self.GROUPS,
            )

    def test_missing_alias_fails(self) -> None:
        with pytest.raises(dv.DocsVersionsError, match="0.22.1"):
            dv.verify([_minor_entry("0.22", "0.22.3", ["latest", "0.22.0"])], self.GROUPS)

    def test_missing_moved_alias_fails(self) -> None:
        groups = [dv.PruneGroup("0.22", ("0.22.3",), ("0.22.0",))]

        with pytest.raises(dv.DocsVersionsError, match="0.22.0"):
            dv.verify([_minor_entry("0.22", "0.22.4", ["latest", "0.22.3"])], groups)


class TestCheckSymlinkAliases:
    GROUPS = [dv.PruneGroup("0.13.8", ("0.13.0", "0.13.1"))]

    @staticmethod
    def _git(tree: dict[str, tuple[str, str]]):
        def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
            name = cmd[-1].split(":")[-1]
            mode, target = tree[name]
            if cmd[1] == "ls-tree":
                return SimpleNamespace(returncode=0, stdout=f"{mode} blob abc\t{name}\n")
            return SimpleNamespace(returncode=0, stdout=target)

        return fake_run

    def test_every_pruned_name_is_a_symlink_to_its_keeper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tree = {"0.13.0": ("120000", "0.13.8"), "0.13.1": ("120000", "0.13.8")}
        monkeypatch.setattr(dv, "_run", self._git(tree))

        dv.check_symlink_aliases(self.GROUPS, "gh-pages")

    @pytest.mark.parametrize(
        "entry",
        [("040000", ""), ("120000", "0.13.7")],
        ids=["still-a-folder", "points-elsewhere"],
    )
    def test_folder_or_wrong_target_fails(
        self, monkeypatch: pytest.MonkeyPatch, entry: tuple[str, str]
    ) -> None:
        tree = {"0.13.0": ("120000", "0.13.8"), "0.13.1": entry}
        monkeypatch.setattr(dv, "_run", self._git(tree))

        with pytest.raises(dv.DocsVersionsError, match="0.13.1"):
            dv.check_symlink_aliases(self.GROUPS, "gh-pages")

    def test_moved_aliases_are_checked_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        groups = [dv.PruneGroup("0.13.8", ("0.13.7",), ("0.13.0",))]
        tree = {"0.13.7": ("120000", "0.13.8"), "0.13.0": ("040000", "")}
        monkeypatch.setattr(dv, "_run", self._git(tree))

        with pytest.raises(dv.DocsVersionsError, match="0.13.0"):
            dv.check_symlink_aliases(groups, "gh-pages")


class TestCli:
    @staticmethod
    def _versions_file(tmp_path: Path, entries: list[dict]) -> Path:
        path = tmp_path / "versions.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def test_check_deploy_exit_codes(self, tmp_path: Path) -> None:
        path = self._versions_file(tmp_path, [_minor_entry("0.22", "0.22.4", ["latest"])])
        runner = CliRunner()

        ok = runner.invoke(dv.cli, ["check-deploy", "0.22.4", "--versions-json", str(path)])
        refused = runner.invoke(dv.cli, ["check-deploy", "0.22.3", "--versions-json", str(path)])

        assert ok.exit_code == 0, ok.output
        assert refused.exit_code == 1
        assert "0.22.4" in refused.output

    def test_prune_without_apply_prints_the_plan_and_runs_nothing(
        self, tmp_path: Path, mike_calls: list
    ) -> None:
        path = self._versions_file(tmp_path, _entries(PUBLISHED_2026_11_06))

        result = CliRunner().invoke(dv.cli, ["prune", "--versions-json", str(path)])

        assert result.exit_code == 0, result.output
        assert mike_calls == []
        assert "0.13.0" in result.output and "0.13.8" in result.output
        assert "29" in result.output and "10" in result.output

    def test_prune_apply_prints_plan_first_then_verifies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = self._versions_file(tmp_path, _entries(["0.13.8", "0.13.7"]))
        after = [{"version": "0.13.8", "title": "0.13.8", "aliases": ["latest", "0.13.7"]}]
        events: list[str] = []

        def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
            events.append(" ".join(cmd))
            if cmd[:2] == ["git", "ls-tree"]:
                return SimpleNamespace(returncode=0, stdout="120000 blob abc\t0.13.7\n")
            if cmd == ["git", "show", "gh-pages:0.13.7"]:
                return SimpleNamespace(returncode=0, stdout="0.13.8")
            stdout = json.dumps(after) if cmd[:2] == ["git", "show"] else ""
            return SimpleNamespace(returncode=0, stdout=stdout)

        monkeypatch.setattr(dv, "_run", fake_run)
        monkeypatch.setattr(dv.click, "echo", lambda msg="", **kw: events.append(f"echo {msg}"))

        result = CliRunner().invoke(dv.cli, ["prune", "--versions-json", str(path), "--apply"])

        assert result.exit_code == 0, result.output
        first_mike = next(i for i, e in enumerate(events) if e.startswith("mike"))
        assert any(e.startswith("echo") and "0.13.7" in e for e in events[:first_mike])
        assert events[first_mike : first_mike + 5] == [
            "mike delete 0.13.7",
            "mike alias --alias-type symlink 0.13.8 0.13.7",
            "git show gh-pages:versions.json",
            "git ls-tree gh-pages -- 0.13.7",
            "git show gh-pages:0.13.7",
        ]

    def test_prune_apply_fails_when_verification_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = self._versions_file(tmp_path, _entries(["0.13.8", "0.13.7"]))
        not_pruned = _entries(["0.13.8", "0.13.7"])
        monkeypatch.setattr(
            dv,
            "_run",
            lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(not_pruned)),
        )

        result = CliRunner().invoke(dv.cli, ["prune", "--versions-json", str(path), "--apply"])

        assert result.exit_code == 1
