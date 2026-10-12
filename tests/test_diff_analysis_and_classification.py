"""Comprehensive unit tests for diff analysis and test classification.

Covers:
- Unified diff parsing (git diff format, empty diffs, syntax variations, binary files, hunks)
- Test file extraction and detection from diffs
- Semantic classification (heuristic, config changes, unclassified changes, doc/CI-only)
- LLM prompt generation and formatting
- LLM error handling and fallback behavior (network errors, invalid JSON, markdown formatting)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ci_test_gate.classifier import (
    HeuristicTestClassifier,
    LLMTestClassifier,
    TestClassifier,
    TestRecommendation,
)
from ci_test_gate.context import (
    AnalysisContext,
    FileContext,
    build_context,
    detect_language,
    detect_test_framework,
    find_test_files,
)
from ci_test_gate.context_builder import ChangeContext, ContextBuilder, is_config_file
from ci_test_gate.diff_parser import DiffParser, FileChange
from ci_test_gate.models import Diff, DiffFile, Language
from ci_test_gate.parser import parse_unified_diff


# ===========================================================================
# 1. Diff Parsing Tests
# ===========================================================================


class TestDiffParserComprehensive:
    """Tests for DiffParser parsing unified diff formats and edge cases."""

    def test_parse_empty_diff(self):
        parser = DiffParser()
        assert parser.parse("") == []
        assert parser.parse("   \n\n\t  ") == []

    def test_parse_diff_with_preamble(self):
        diff_text = """commit 1234567890abcdef
Author: Developer <dev@example.com>
Date:   Mon Jan 1 00:00:00 2026

    Commit message summary

diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -1,2 +1,3 @@
 line1
-line2
+line2_modified
+line3_added
"""
        parser = DiffParser()
        changes = parser.parse(diff_text)
        assert len(changes) == 1
        change = changes[0]
        assert change.path == "module.py"
        assert change.removed_lines == ["line2"]
        assert change.added_lines == ["line2_modified", "line3_added"]

    def test_parse_diff_without_a_b_prefix(self):
        diff_text = """diff --git src/core.py src/core.py
--- src/core.py
+++ src/core.py
@@ -1,1 +1,2 @@
+import sys
 pass
"""
        parser = DiffParser()
        changes = parser.parse(diff_text)
        assert len(changes) == 1
        assert changes[0].path == "src/core.py"
        assert changes[0].added_lines == ["import sys"]

    def test_parse_multiple_hunks_per_file(self):
        diff_text = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -10,3 +10,4 @@
-return 1
+return 10
+return 11
@@ -50,3 +51,3 @@
-return 2
+return 20
"""
        parser = DiffParser()
        changes = parser.parse(diff_text)
        assert len(changes) == 1
        change = changes[0]
        assert len(change.hunks) == 2
        assert len(change.added_lines) == 3
        assert len(change.removed_lines) == 2
        assert "return 10" in change.added_lines
        assert "return 20" in change.added_lines

    def test_parse_hunk_header_variations(self):
        # Single line hunk headers: @@ -1 +1 @@
        diff_text = """diff --git a/single.py b/single.py
--- a/single.py
+++ b/single.py
@@ -1 +1 @@
-old_single
+new_single
"""
        parser = DiffParser()
        changes = parser.parse(diff_text)
        assert len(changes) == 1
        assert changes[0].added_lines == ["new_single"]
        assert changes[0].removed_lines == ["old_single"]

    def test_parse_binary_git_diff_patch(self):
        diff_text = """diff --git a/data.bin b/data.bin
GIT binary patch
literal 1234
...
"""
        parser = DiffParser()
        changes = parser.parse(diff_text)
        assert len(changes) == 1
        assert changes[0].path == "data.bin"
        assert changes[0].is_binary is True
        assert changes[0].added_lines == []

    def test_parse_binary_files_differ_header(self):
        diff_text = """diff --git a/assets/logo.png b/assets/logo.png
Binary files a/assets/logo.png and b/assets/logo.png differ
"""
        parser = DiffParser()
        changes = parser.parse(diff_text)
        assert len(changes) == 1
        assert changes[0].path == "assets/logo.png"
        assert changes[0].is_binary is True

    def test_parse_no_newline_warning_ignored(self):
        diff_text = r"""diff --git a/file.txt b/file.txt
--- a/file.txt
+++ b/file.txt
@@ -1,1 +1,1 @@
-old
\ No newline at end of file
+new
\ No newline at end of file
"""
        parser = DiffParser()
        changes = parser.parse(diff_text)
        assert len(changes) == 1
        assert changes[0].added_lines == ["new"]
        assert changes[0].removed_lines == ["old"]

    def test_parse_large_diff_with_many_files(self):
        parts = []
        for i in range(25):
            parts.append(
                f"diff --git a/src/pkg{i}/file{i}.py b/src/pkg{i}/file{i}.py\n"
                f"--- a/src/pkg{i}/file{i}.py\n"
                f"+++ b/src/pkg{i}/file{i}.py\n"
                f"@@ -1,3 +1,4 @@\n"
                f"-old_val_{i}\n"
                f"+new_val_{i}\n"
                f"+added_{i}\n"
            )
        large_diff = "\n".join(parts)
        parser = DiffParser()
        changes = parser.parse(large_diff)
        assert len(changes) == 25
        assert changes[0].path == "src/pkg0/file0.py"
        assert changes[24].path == "src/pkg24/file24.py"


class TestParserUnifiedDiffModule:
    """Tests for parse_unified_diff in parser.py."""

    def test_parse_empty_unified_diff(self):
        d = parse_unified_diff("")
        assert d.files == []
        assert d.all_paths == []

    def test_parse_single_file_unified_diff(self):
        raw = """diff --git a/main.go b/main.go
--- a/main.go
+++ b/main.go
@@ -1,3 +1,5 @@
 package main
-import "fmt"
+import "log"
+import "os"
"""
        diff = parse_unified_diff(raw)
        assert len(diff.files) == 1
        f = diff.files[0]
        assert f.path == "main.go"
        assert f.additions == 2
        assert f.deletions == 1

    def test_parse_incomplete_unified_diff_header(self):
        # Header without body/match parts
        raw = "diff --git a/only_a"
        diff = parse_unified_diff(raw)
        assert len(diff.files) == 0


# ===========================================================================
# 2. Test File Extraction and Detection Tests
# ===========================================================================


class TestFileExtractionAndDetection:
    """Tests for identifying test files, extensions, and filtering changes."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("tests/test_server.py", True),
            ("tests/server_test.py", True),
            ("src/components/button.test.ts", True),
            ("src/components/button.test.js", True),
            ("src/components/button.spec.ts", True),
            ("src/components/button.spec.js", True),
            ("pkg/service/service_test.go", True),
            ("test/fixtures/mock.py", True),
            ("subproject/test/conftest.py", True),
            ("src/testimony.py", False),
            ("src/protest.py", False),
            ("src/main.py", False),
            ("Makefile", False),
        ],
    )
    def test_file_change_is_test_file(self, path, expected):
        fc = FileChange(path=path)
        assert fc.is_test_file is expected

    def test_get_non_test_and_test_changes(self):
        parser = DiffParser()
        changes = [
            FileChange(path="src/service.py"),
            FileChange(path="tests/test_service.py"),
            FileChange(path="tests/data.bin", is_binary=True),
            FileChange(path="assets/logo.png", is_binary=True),
        ]
        test_changes = parser.get_test_changes(changes)
        non_test_changes = parser.get_non_test_changes(changes)

        assert [c.path for c in test_changes] == ["tests/test_service.py"]
        assert [c.path for c in non_test_changes] == ["src/service.py"]

    def test_changed_paths_and_extension_filtering(self):
        parser = DiffParser()
        changes = [
            FileChange(path="src/index.ts"),
            FileChange(path="src/styles.css"),
            FileChange(path="README.md"),
            FileChange(path="Dockerfile"),  # no extension
        ]
        assert parser.get_changed_paths(changes) == [
            "src/index.ts",
            "src/styles.css",
            "README.md",
            "Dockerfile",
        ]
        assert parser.get_changed_extensions(changes) == {".ts", ".css", ".md"}
        ts_changes = parser.filter_by_extension(changes, ".ts")
        assert len(ts_changes) == 1
        assert ts_changes[0].path == "src/index.ts"


# ===========================================================================
# 3. Semantic Classification Tests
# ===========================================================================


class TestSemanticClassificationLogic:
    """Tests for heuristic and rule-based test selection and categorization."""

    def test_suffix_test_py_matching(self):
        """Covers tf_name.endswith('_test.py') matching."""
        classifier = HeuristicTestClassifier()
        changes = [FileChange(path="src/account_service.py")]
        context = ChangeContext(changed_files=changes)
        test_files = [
            "tests/account_service_test.py",
            "tests/billing_service_test.py",
        ]
        rec = classifier.classify(changes, context, test_files)
        assert "tests/account_service_test.py" in rec.required
        assert "tests/billing_service_test.py" in rec.optional

    def test_unclassified_changes_lead_to_all_optional(self):
        classifier = HeuristicTestClassifier()
        changes = [FileChange(path="src/completely_unrelated.py")]
        context = ChangeContext(changed_files=changes)
        test_files = ["tests/test_auth.py", "tests/test_database.py"]

        rec = classifier.classify(changes, context, test_files)
        assert rec.required == []
        assert set(rec.optional) == {"tests/test_auth.py", "tests/test_database.py"}
        assert "Heuristic match: 0 required test(s)" in rec.reasoning

    def test_ci_only_changes_detection(self):
        classifier = HeuristicTestClassifier()
        changes = [
            FileChange(path=".github/workflows/ci.yml"),
            FileChange(path="actions/lint.yaml"),
        ]
        context = ChangeContext(changed_files=changes)
        test_files = ["tests/test_a.py", "tests/test_b.py"]

        rec = classifier.classify(changes, context, test_files)
        assert rec.required == []
        assert rec.optional == test_files
        assert "CI-only changes detected" in rec.reasoning
        assert rec.estimated_savings_pct == 80

    def test_doc_only_changes_detection(self):
        classifier = HeuristicTestClassifier()
        changes = [
            FileChange(path="docs/guide.md"),
            FileChange(path="RELEASE_NOTES.txt"),
            FileChange(path="index.rst"),
        ]
        context = ChangeContext(changed_files=changes)
        test_files = ["tests/test_a.py"]

        rec = classifier.classify(changes, context, test_files)
        assert rec.required == []
        assert rec.optional == test_files
        assert "Doc-only changes detected" in rec.reasoning
        assert rec.estimated_savings_pct == 80

    def test_mixed_code_and_doc_not_treated_as_doc_only(self):
        classifier = HeuristicTestClassifier()
        changes = [
            FileChange(path="docs/guide.md"),
            FileChange(path="src/router.py"),
        ]
        context = ChangeContext(changed_files=changes)
        test_files = ["tests/test_router.py"]

        rec = classifier.classify(changes, context, test_files)
        assert "tests/test_router.py" in rec.required
        assert "Doc-only changes detected" not in rec.reasoning

    def test_binary_only_changes(self):
        classifier = TestClassifier()
        changes = [
            FileChange(path="image.png", is_binary=True),
            FileChange(path="font.woff2", is_binary=True),
        ]
        context = ChangeContext(changed_files=changes)
        test_files = ["tests/test_ui.py"]

        rec = classifier.classify(changes, context, test_files)
        assert rec.required == []
        assert rec.optional == ["tests/test_ui.py"]
        assert "Binary-only changes" in rec.reasoning

    def test_config_changes_broad_vs_normal(self):
        # Broad mode (default): all non-matched tests recommended
        broad_classifier = TestClassifier(config={"config_changes": "broad"})
        changes = [
            FileChange(path="pyproject.toml"),
            FileChange(path="src/utils.py"),
        ]
        context = ChangeContext(changed_files=changes)
        test_files = ["tests/test_utils.py", "tests/test_other.py"]

        rec_broad = broad_classifier.classify(changes, context, test_files)
        assert "tests/test_utils.py" in rec_broad.required
        assert "tests/test_other.py" in rec_broad.recommended
        assert "Project config changed" in rec_broad.reasoning

        # Normal mode: no broad recommendation
        normal_classifier = TestClassifier(config={"config_changes": "normal"})
        rec_normal = normal_classifier.classify(changes, context, test_files)
        assert "tests/test_utils.py" in rec_normal.required
        assert "tests/test_other.py" not in rec_normal.recommended
        assert "tests/test_other.py" in rec_normal.optional

    def test_classifier_factory_and_invalid_type(self):
        h = TestClassifier.from_config({"classifier_type": "heuristic"})
        assert isinstance(h, HeuristicTestClassifier)

        llm = TestClassifier.from_config({"classifier_type": "llm", "api_key": "test"})
        assert isinstance(llm, LLMTestClassifier)

        with pytest.raises(ValueError, match="Invalid classifier type"):
            TestClassifier.from_config({"classifier_type": "unsupported"})

        invalid = TestClassifier(classifier_type="bogus")
        with pytest.raises(ValueError, match="Invalid classifier type"):
            invalid.classify([], ChangeContext(changed_files=[]))


# ===========================================================================
# 4. Context & LLM Prompt Generation Tests
# ===========================================================================


class TestLLMPromptAndContextGeneration:
    """Tests for prompt context generation and formatting."""

    def test_change_context_to_prompt_context(self):
        ctx = ChangeContext(
            changed_files=[FileChange(path="src/api.py")],
            imports_added={"src/api.py": ["fastapi", "pydantic"]},
            imports_removed={"src/api.py": ["flask"]},
            functions_changed={"src/api.py": ["get_user", "update_user"]},
            test_files_present=["tests/test_api.py"],
            test_runners_detected=["pytest"],
            estimated_scope="medium",
        )
        prompt = ctx.to_prompt_context()

        assert "Scope: medium" in prompt
        assert "Changed files: 1" in prompt
        assert "Test files present: 1" in prompt
        assert "Test runners: pytest" in prompt
        assert "Imports added:" in prompt
        assert "fastapi" in prompt
        assert "Imports removed:" in prompt
        assert "flask" in prompt
        assert "Functions changed:" in prompt
        assert "get_user" in prompt

    def test_analysis_context_summary(self):
        diff = Diff(
            files=[
                DiffFile(path="src/engine.py", additions=15, deletions=3),
                DiffFile(path="src/parser.py", additions=5, deletions=0),
            ]
        )
        ctx = AnalysisContext(
            diff=diff,
            language=Language.PYTHON,
            file_contexts=[],
            test_files_in_repo=[f"tests/test_{i}.py" for i in range(60)],
            test_framework="pytest",
        )
        summary = ctx.summary
        assert "Language: python" in summary
        assert "Test framework: pytest" in summary
        assert "Modified files (2):" in summary
        assert "src/engine.py (+15/-3)" in summary
        assert "Available test files (60):" in summary
        assert "and 10 more" in summary


# ===========================================================================
# 5. LLM Fallback and Error Handling Tests
# ===========================================================================


class TestLLMFallbackAndErrorHandling:
    """Tests for LLM fallback when API is missing, failing, or returning malformed data."""

    def test_fallback_when_no_api_key_configured(self):
        classifier = LLMTestClassifier({"api_key": ""})
        changes = [FileChange(path="src/login.py")]
        context = ChangeContext(changed_files=changes)
        test_files = ["tests/test_login.py"]

        rec = classifier.llm_classify(changes, context, test_files)
        assert "tests/test_login.py" in rec.required

    def test_fallback_on_llm_network_error(self):
        classifier = LLMTestClassifier({"api_key": "sk-dummy"})
        changes = [FileChange(path="src/client.py")]
        context = ChangeContext(changed_files=changes)
        test_files = ["tests/test_client.py"]

        with patch("ci_test_gate.classifier.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.side_effect = RuntimeError("Connection refused")

            rec = classifier.llm_classify(changes, context, test_files)

        assert "tests/test_client.py" in rec.required
        assert "LLM call failed (RuntimeError: Connection refused)" in rec.reasoning
        assert "heuristic fallback" in rec.reasoning

    def test_fallback_on_malformed_json_response(self):
        classifier = LLMTestClassifier({"api_key": "sk-dummy"})
        changes = [FileChange(path="src/handler.py")]
        context = ChangeContext(changed_files=changes)

        with patch("ci_test_gate.classifier.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            choice = MagicMock()
            choice.message.content = "Internal Server Error: Not JSON"
            response = MagicMock(choices=[choice])
            mock_client.chat.completions.create.return_value = response

            rec = classifier.llm_classify(changes, context, ["tests/test_handler.py"])

        assert "JSONDecodeError" in rec.reasoning
        assert "heuristic fallback" in rec.reasoning

    def test_llm_markdown_fences_stripped_cleanly(self):
        payload = json.dumps(
            {
                "required": ["tests/test_fenced.py"],
                "recommended": [],
                "optional": [],
                "reasoning": "Fenced payload parsed successfully",
                "estimated_savings_pct": 50,
            }
        )
        fenced_content = f"```json\n{payload}\n```"

        classifier = LLMTestClassifier({"api_key": "sk-dummy"})
        changes = [FileChange(path="src/fenced.py")]
        context = ChangeContext(changed_files=changes)

        with patch("ci_test_gate.classifier.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            choice = MagicMock()
            choice.message.content = fenced_content
            response = MagicMock(choices=[choice])
            mock_client.chat.completions.create.return_value = response

            rec = classifier.llm_classify(changes, context, ["tests/test_fenced.py"])

        assert rec.required == ["tests/test_fenced.py"]
        assert rec.reasoning == "Fenced payload parsed successfully"
        assert rec.estimated_savings_pct == 50

    def test_llm_empty_json_handles_defaults(self):
        classifier = LLMTestClassifier({"api_key": "sk-dummy"})
        changes = [FileChange(path="src/demo.py")]
        context = ChangeContext(changed_files=changes)

        with patch("ci_test_gate.classifier.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            choice = MagicMock()
            choice.message.content = "{}"
            response = MagicMock(choices=[choice])
            mock_client.chat.completions.create.return_value = response

            rec = classifier.llm_classify(changes, context, ["tests/test_demo.py"])

        assert rec.required == []
        assert rec.recommended == []
        assert rec.optional == []
        assert rec.reasoning == ""
        assert rec.estimated_savings_pct == 0
