from context_eval.env import load_env


class TestLoadEnv:
    def test_parses_allowed_keys_ignoring_comments(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\n"
            "\n"
            "OPENROUTER_API_KEY=sk-test-123\n"
            "SOME_OTHER_KEY=ignored-by-allowlist\n"
            "not-a-line\n"
        )

        loaded = load_env(env_file)

        assert loaded == {"OPENROUTER_API_KEY": "sk-test-123"}
        import os

        assert "SOME_OTHER_KEY" not in os.environ

    def test_real_environment_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
        env_file = tmp_path / ".env"
        env_file.write_text("OPENROUTER_API_KEY=from-file\n")

        loaded = load_env(env_file)

        assert loaded == {}

    def test_missing_file_is_noop(self, tmp_path):
        assert load_env(tmp_path / "nope.env") == {}

    def test_empty_values_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EMPTY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("EMPTY=\n")

        assert load_env(env_file) == {}
