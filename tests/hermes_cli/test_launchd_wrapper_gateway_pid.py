"""A launchd job's wrapper is never the recipient of gateway drain signals."""

import json
import os
import plistlib
from types import SimpleNamespace

import pytest

from gateway import status
from hermes_cli import gateway as gw
from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_cmd_posix_pause as pause


@pytest.fixture
def wrapped_gateway(tmp_path, monkeypatch):
    start_time_probe = status._get_process_start_time
    parent_pid_probe = gw._get_parent_pid
    home = tmp_path / "profiles" / "research"
    home.mkdir(parents=True)
    current = "ai.hermes.gateway"
    label = "ai.hermes.gateway-research"
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    (agents / f"{label}.plist").write_bytes(plistlib.dumps({
        "Label": label, "ProgramArguments": ["/usr/bin/osascript", "-e", "wrapper"],
        "EnvironmentVariables": {"HERMES_HOME": str(home)},
    }))
    record = {"pid": 200, "start_time": 12345, "kind": "hermes-gateway",
              "hermes_home": str(home), "argv": ["hermes", "-p", "research", "gateway", "run"],
              "gateway_state": "running"}
    state = home / "gateway_state.json"
    state.write_text(json.dumps(record))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gw, "get_launchd_label", lambda: current)
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: agents / f"{current}.plist")
    monkeypatch.setattr(gw, "launchd_gateway_labels_for_install", lambda: [label])
    monkeypatch.setattr(gw, "legacy_launchd_labels_for_install", lambda **kw: [])
    monkeypatch.setattr(gw, "_locate_launchd_gateway_service", lambda _: ("gui/502", 100))
    monkeypatch.setattr(gw, "_get_parent_pid", lambda pid: {200: 150, 150: 100, 100: 1}.get(pid))
    monkeypatch.setattr(status, "_pid_exists", lambda pid: pid in {100, 150, 200})
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: {100: 111, 150: 222, 200: 12345}.get(pid))
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: (
        "hermes -p research gateway run" if pid == 200 else "/usr/bin/osascript -e wrapper"))
    from hermes_cli import update_receipt
    monkeypatch.setattr(update_receipt, "_profile_homes", lambda: [("default", tmp_path), ("research", home)])
    monkeypatch.setattr(pause, "_home_of", lambda pid: str(home) if pid == 200 else None)
    return SimpleNamespace(home=home, state=state, record=record, label=label,
                           start_time_probe=start_time_probe, parent_pid_probe=parent_pid_probe)


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("xpc_service_name", [None, "0"], ids=["no-xpc", "zero-xpc"])
@pytest.mark.parametrize("topology", ["wrapper_sibling", "gateway_ancestor", "legacy_ancestor", "external", "legacy_external"])
def test_pause_preserves_the_updaters_own_launchd_job(wrapped_gateway, monkeypatch, xpc_service_name, topology):
    if xpc_service_name is None:
        monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    else:
        monkeypatch.setenv("XPC_SERVICE_NAME", xpc_service_name)
    job, = pause._discover_launchd()
    parents = {200: 150, 150: 100, 100: 1, os.getpid(): 300}
    # The wrapper owns two subtrees: gateway (100 -> 150 -> 200) and updater (100 -> 300 -> us).
    parents[300] = 100 if topology == "wrapper_sibling" else 200
    if topology.startswith("legacy_"):
        job.pop("job_pid")
        job["pid"] = 100
    if topology.endswith("external"):
        parents[300] = 1
    monkeypatch.setattr(gw, "_get_parent_pid", parents.get)

    units, jobs, notices = pause._left_running([], [job])

    assert units == []
    if topology.endswith("external"):
        assert jobs == [job], "an unrelated launchd job must remain eligible for pause"
        assert notices == []
    else:
        assert jobs == [], "bootout of the updater's wrapper would kill the update"
        assert len(notices) == 1 and job["label"] in notices[0]


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("identity", ["state_wrapper", "state_direct", "lock_wrapper", "cross_domain"])
def test_wrapped_gateway_is_drained_and_tracked_instead_of_the_job_pid(wrapped_gateway, monkeypatch, identity, request):
    target = wrapped_gateway
    job_pid = 100
    if identity == "state_direct":
        job_pid = 200
        monkeypatch.setattr(gw, "_locate_launchd_gateway_service", lambda _: ("gui/502", job_pid))
    elif identity == "lock_wrapper":
        import fcntl

        target.state.unlink()
        lock = (target.home / "gateway.lock").open("w+")
        lock.write(json.dumps(target.record))
        lock.flush()
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request.addfinalizer(lock.close)
    signals, waits, kickstarts, reports = [], [], [], []
    monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1", lambda pid, **kw: signals.append(pid) or True)
    monkeypatch.setattr(gw, "_wait_for_launchd_service_pid", lambda label, **kw: waits.append(kw["old_pid"]) or True)
    monkeypatch.setattr(gw, "_launchd_kickstart", lambda *args: kickstarts.append(args))
    from hermes_cli import update_cmd_drain_report
    monkeypatch.setattr(update_cmd_drain_report, "drain_progress_reporter", lambda home, **kw: reports.append(home))
    restarted, failed = [], []
    fleet._restart_macos_launchd_gateways(restarted, failed, 45, require_supervision=False)

    assert signals == [200], "SIGUSR1 must reach the gateway, never osascript"
    assert reports == [target.home]
    assert waits == [job_pid], "launchd replacement checks compare job PIDs"
    assert kickstarts == []
    assert restarted == [target.label] and failed == []
    jobs = pause._discover_launchd()
    assert [(job["pid"], job["home"]) for job in jobs] == [(200, str(target.home))]
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
    listing = f"300 0 {target.label}\n" if identity == "cross_domain" else ""
    monkeypatch.setattr(gw.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=listing))
    protected = gw._get_service_pids(all_profiles=True)
    assert 200 in protected, "the orphan sweep must protect the supervised gateway child"
    if identity == "cross_domain":
        assert {100, 200, 300} <= protected, "domain probes and caller-domain listing must both be retained"
    monkeypatch.setattr(gw, "_launchctl_bootstrap", lambda *a, **kw: None)
    pause._start_job(jobs[0])
    assert waits == [job_pid, job_pid], "pause recovery must also compare launchd job identities"


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("invalid", [
    "reused", "unknown_birth", "dead", "foreign", "unrelated", "missing",
    "missing_plist_home", "denied_birth", "denied_ancestry",
])
def test_unverified_gateway_never_sends_a_drain_signal(wrapped_gateway, monkeypatch, invalid):
    target = wrapped_gateway
    denied_calls = []
    record_changes = {
        "reused": {"start_time": 67890},
        "unknown_birth": {"start_time": None},
        "foreign": {"hermes_home": str(target.home.parent / "other")},
    }
    target.record.update(record_changes.get(invalid, {}))
    if invalid == "dead":
        monkeypatch.setattr(status, "_pid_exists", lambda pid: pid != 200)
    if invalid == "unrelated":
        monkeypatch.setattr(gw, "_get_parent_pid", lambda _: 1)
    if invalid == "missing_plist_home":
        plist = gw.get_launchd_plist_path().with_name(f"{target.label}.plist")
        data = plistlib.loads(plist.read_bytes())
        data["EnvironmentVariables"].pop("HERMES_HOME")
        plist.write_bytes(plistlib.dumps(data))
    if invalid in {"denied_birth", "denied_ancestry"}:
        import psutil
        from hermes_cli import process_identity

        def denied(*args):
            denied_calls.append(args)
            raise psutil.AccessDenied(pid=200)

        if invalid == "denied_birth":
            monkeypatch.setattr(status, "_get_process_start_time", target.start_time_probe)
            monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(pid=pid))
            monkeypatch.setattr(process_identity, "process_birth", denied)
        else:
            monkeypatch.setattr(gw, "_get_parent_pid", target.parent_pid_probe)
            monkeypatch.setattr(psutil, "Process", denied)
    if invalid == "missing":
        target.state.unlink()
    else:
        target.state.write_text(json.dumps(target.record))
    signals, kickstarts = [], []
    monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1", lambda pid, **kw: signals.append(pid) or False)
    monkeypatch.setattr(gw, "_launchd_kickstart", lambda *args: kickstarts.append(args))
    monkeypatch.setattr(gw, "_wait_for_launchd_service_pid", lambda *args, **kw: True)
    restarted, failed = [], []
    fleet._restart_macos_launchd_gateways(restarted, failed, 45, require_supervision=False)
    if invalid in {"denied_birth", "denied_ancestry"}:
        assert denied_calls, "the permission-denied identity probe must actually run"
    assert signals == [], "unverified identities must fall back to the scoped launchd restart"
    assert kickstarts == [(target.label, "gui/502")]
    assert restarted == [target.label] and failed == []
