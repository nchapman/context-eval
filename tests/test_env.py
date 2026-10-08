from context_eval.env import load_env


class TestLoadEnv:
    def test_parses_keys_ignoring_comments_and_blanks(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOO", raising=False)
        monkeypatch.delenv("BAR", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\n"
            "\n"
            "FOO=plain\n"
            "BAR = 'quoted value'  \n"
            "not-a-line\n"
        )

        loaded = load_env(env_file)

        assert loaded == {"FOO": "plain", "BAR": "quoted value"}

    def test_real_environment_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOO", "from-shell")
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=from-file\n")

        loaded = load_env(env_file)

        assert loaded == {}
        import os

        assert os.environ["FOO"] == "from-shell"

    def test_missing_file_is_noop(self, tmp_path):
        assert load_env(tmp_path / "nope.env") == {}

    def test_empty_values_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EMPTY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("EMPTY=\n")

        assert load_env(env_file) == {}
