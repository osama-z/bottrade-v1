"""Tier-15 tests: security hardening (OWASP pass).

S1 — the ZMQ PUB/SUB channel carries UNAUTHENTICATED trade commands:
whoever can publish to it drives the executor. Binding it on a
non-loopback interface must therefore be refused unless the operator
explicitly opts in (ZMQ_ALLOW_NONLOCAL, for CURVE/isolated networks).
"""

from pathlib import Path

import pytest

from core.zmq_publisher import SignalPublisher, _require_loopback

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestLoopbackGuard:
    def test_loopback_tcp_is_allowed(self):
        _require_loopback("tcp://127.0.0.1:5555")   # must not raise
        _require_loopback("tcp://localhost:5555")
        _require_loopback("tcp://[::1]:5555")

    def test_ipc_and_inproc_are_allowed(self):
        _require_loopback("ipc:///tmp/neurontrade.sock")
        _require_loopback("inproc://signals")

    def test_all_interfaces_bind_is_refused(self):
        with pytest.raises(ValueError, match="ZMQ_ALLOW_NONLOCAL"):
            _require_loopback("tcp://0.0.0.0:5555")

    def test_specific_public_ip_is_refused(self):
        with pytest.raises(ValueError):
            _require_loopback("tcp://192.168.1.50:5555")

    def test_wildcard_star_is_refused(self):
        with pytest.raises(ValueError):
            _require_loopback("tcp://*:5555")

    def test_publisher_constructor_enforces_the_guard(self):
        """The guard must run on the real bind path, not only in isolation."""
        with pytest.raises(ValueError):
            SignalPublisher(address="tcp://0.0.0.0:5599")

    def test_override_flag_permits_nonlocal(self, monkeypatch):
        """The guard re-imports config.settings at call time, so patching
        the module attribute (not the frozen pydantic instance) is the
        supported way to simulate ZMQ_ALLOW_NONLOCAL=true."""
        import sys
        from types import SimpleNamespace

        # config/__init__ re-exports the settings OBJECT, which shadows
        # the submodule on attribute access — go through sys.modules.
        settings_module = sys.modules["config.settings"]
        monkeypatch.setattr(
            settings_module, "settings",
            SimpleNamespace(zmq_allow_nonlocal=True),
        )
        _require_loopback("tcp://0.0.0.0:5599")   # must not raise


class TestDeployHardening:
    """S2/S3/S4 — pin the deploy-side mitigations so they can't silently
    regress in a later edit of the script or units."""

    def test_env_file_created_with_owner_only_perms(self):
        src = (PROJECT_ROOT / "deploy" / "setup_server.sh").read_text()
        assert "install -m 600 .env.example .env" in src
        assert "chmod 600 .env" in src

    def test_setup_prefers_locked_requirements(self):
        src = (PROJECT_ROOT / "deploy" / "setup_server.sh").read_text()
        assert "requirements.lock" in src

    def test_ci_installs_locked_requirements(self):
        src = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "requirements.lock" in src

    def test_lock_file_exists_and_is_fully_pinned(self):
        lock = PROJECT_ROOT / "requirements.lock"
        assert lock.exists()
        lines = [ln for ln in lock.read_text().splitlines()
                 if ln and not ln.startswith("#")]
        assert len(lines) > 30
        unpinned = [ln for ln in lines if "==" not in ln and " @ " not in ln]
        assert unpinned == [], f"unpinned entries in lock: {unpinned}"

    def test_systemd_units_are_sandboxed(self):
        for unit in ("neurontrade-execution.service",
                     "neurontrade-intelligence.service"):
            src = (PROJECT_ROOT / "deploy" / unit).read_text()
            for directive in ("NoNewPrivileges=true", "PrivateTmp=true",
                              "ProtectSystem=strict", "ProtectHome=read-only",
                              "RestrictSUIDSGID=true"):
                assert directive in src, f"{unit} missing {directive}"


class TestSecretsRedactedInRepr:
    """A failing test once printed the LIVE API keys because pydantic's
    default repr includes every field. Settings.__repr_args__ must redact."""

    def test_repr_hides_secret_values(self):
        from config.settings import Settings
        s = Settings(
            BINANCE_API_KEY="sk-live-SUPERSECRET-KEY",
            GROQ_API_KEY="gsk_another_secret",
            TELEGRAM_BOT_TOKEN="12345:token-value",
            _env_file=None,
        )
        for rendered in (repr(s), str(s)):
            assert "SUPERSECRET" not in rendered
            assert "gsk_another_secret" not in rendered
            assert "token-value" not in rendered
            assert "***REDACTED***" in rendered

    def test_empty_secrets_stay_empty_not_redacted(self):
        from config.settings import Settings
        s = Settings(BINANCE_API_KEY="", _env_file=None)
        assert "***REDACTED***" not in repr(s).split("groq")[0].split("binance_api_key")[1].split(",")[0]

    def test_values_remain_readable_by_code(self):
        """Redaction is display-only — the bot itself still reads keys."""
        from config.settings import Settings
        s = Settings(BINANCE_API_KEY="real-key", _env_file=None)
        assert s.binance_api_key == "real-key"


class TestNoDangerousPrimitives:
    """The scan that found nothing must keep finding nothing."""

    def test_no_eval_exec_pickle_or_shell(self):
        banned = ("eval(", "exec(", "pickle.load", "os.system(", "shell=True")
        offenders: list[str] = []
        for py in PROJECT_ROOT.rglob("*.py"):
            rel = py.relative_to(PROJECT_ROOT)
            parts = rel.parts
            if parts[0] in {".venv", "tests"} or "__pycache__" in parts:
                continue
            src = py.read_text()
            for token in banned:
                if token in src:
                    offenders.append(f"{rel}: {token}")
        assert offenders == [], offenders
