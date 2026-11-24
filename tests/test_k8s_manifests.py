"""Checks on the Kubernetes files in k8s/, without a cluster.

They pin down the settings the failover and rolling update results depend
on, so a later edit cannot quietly undo them. The live checks (a real kind
cluster, deleting pods under load, rolling updates) are described in the
README; they need Docker and kind and take minutes."""
import os
import re
import stat

import pytest
import yaml

K8S = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "k8s")


def docs(name: str) -> list[dict]:
    with open(os.path.join(K8S, name)) as f:
        return [d for d in yaml.safe_load_all(f) if d]


def find(name: str, kind: str) -> dict:
    return next(d for d in docs(name) if d["kind"] == kind)


def api_container() -> dict:
    return find("app.yaml", "Deployment")["spec"]["template"]["spec"]["containers"][0]


@pytest.mark.parametrize("name", sorted(f for f in os.listdir(K8S) if f.endswith(".yaml")))
def test_every_file_is_valid_yaml_in_our_namespace(name):
    for d in docs(name):
        assert d.get("kind"), name
        if d["kind"] not in ("Namespace", "Cluster"):
            assert d["metadata"]["namespace"] == "filing-analyst", (name, d["kind"])


def test_two_replicas_and_updates_never_drop_below_two():
    spec = find("app.yaml", "Deployment")["spec"]
    assert spec["replicas"] == 2
    assert spec["strategy"]["rollingUpdate"]["maxUnavailable"] == 0


def test_readiness_checks_the_database_and_liveness_does_not():
    c = api_container()
    assert c["readinessProbe"]["httpGet"]["path"] == "/health"
    assert c["livenessProbe"]["httpGet"]["path"] == "/health/live"
    assert c["startupProbe"]["httpGet"]["path"] == "/health/live"


def test_pods_finish_requests_before_stopping():
    c = api_container()
    assert c["lifecycle"]["preStop"]["exec"]["command"][0] == "sleep"


@pytest.mark.parametrize("name, kind", [("app.yaml", "Deployment"), ("ingest-job.yaml", "Job")])
def test_app_pods_are_locked_down(name, kind):
    pod = find(name, kind)["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] != 0
    sc = pod["containers"][0]["securityContext"]
    assert sc["readOnlyRootFilesystem"] is True and sc["allowPrivilegeEscalation"] is False
    assert sc["capabilities"]["drop"] == ["ALL"]


def test_postgres_keeps_its_data_on_a_persistent_volume():
    ss = find("postgres.yaml", "StatefulSet")
    assert ss["spec"]["volumeClaimTemplates"][0]["metadata"]["name"] == "data"
    mounts = ss["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    assert {"name": "data", "mountPath": "/var/lib/postgresql/data"} in mounts


def test_no_secret_values_in_any_file():
    # The password and API key exist only in the Secret that up.sh makes with
    # random values. No file may hold a Secret object or a literal password.
    for name in os.listdir(K8S):
        text = open(os.path.join(K8S, name)).read()
        assert "kind: Secret" not in text, name
        assert not re.search(r"(PASSWORD|API_KEY)\s*[:=]\s*['\"]?[A-Za-z0-9]{8,}", text), name
        assert not re.search(r"postgresql://[^@\s]*:[^@\s]*@", text), name  # no password inside an address
    for name in ("app.yaml", "ingest-job.yaml", "postgres.yaml"):
        assert "secretKeyRef" in open(os.path.join(K8S, name)).read()


def test_the_app_port_is_published_to_this_machine_only():
    mapping = find("kind-config.yaml", "Cluster")["nodes"][0]["extraPortMappings"][0]
    assert mapping["listenAddress"] == "127.0.0.1"
    assert mapping["containerPort"] == find("app.yaml", "Service")["spec"]["ports"][0]["nodePort"]


def test_scripts_are_executable_and_stop_on_errors():
    for name in ("up.sh", "update.sh", "down.sh"):
        path = os.path.join(K8S, name)
        assert os.stat(path).st_mode & stat.S_IXUSR, name
        assert "set -euo pipefail" in open(path).read(), name
