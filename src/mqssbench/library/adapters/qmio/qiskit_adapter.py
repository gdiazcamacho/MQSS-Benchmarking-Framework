# src/mqssbench/library/adapters/qmio/qiskit_adapter.py

import logging
from qiskit import QuantumCircuit, transpile

from ....framework.adapter import DeviceAdapter
from ....framework.adapter_registry import AdapterRegistry
from ....framework.types import (
    ProfilingConfig,
    RunContext,
    ProfilingMetrics,
    ExecutionResult,
)

logger = logging.getLogger(__name__)


@AdapterRegistry.register_adapter
class QmioQiskitAdapter(DeviceAdapter):
    name = "qmio_qiskit"

    def __init__(self, config):
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        self.config = config
        self._backend_name = str(config.get("backend", "QMIO")).strip() or "QMIO"
        self._shots = int(config["shots"]) if config.get("shots") is not None else None

        # Lazy init
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            # Adjust this import if your local QMIO path differs
            from qmiotools.integrations.qiskitqmio import QmioBackend

            # Simplest case: backend object with no credentials needed
            calibration_file = self.config.get("calibration_file", None)
            self._backend = QmioBackend(calibration_file=calibration_file)
        return self._backend

    def get_backend_name(self) -> str:
        return self._backend_name

    @classmethod
    def validate_profiling_config(cls, profiling_config: ProfilingConfig):
        # Start simple: disable custom profiling for QMIO
        if profiling_config is None:
            return
        if profiling_config.enabled:
            raise ValueError(
                f"Profiling is not yet implemented for adapter '{cls.name}'. "
                "Set profiling.enabled: false"
            )

    def execute_circuit(
        self,
        context: RunContext,
        circuit,
        num_qubits=None,
        transpile_mode=True,
    ) -> ExecutionResult:
        # Benchmarks may pass either a built circuit or a callable returning one
        built_circuit = circuit() if callable(circuit) else circuit

        if not isinstance(built_circuit, QuantumCircuit):
            raise TypeError("circuit must be a Qiskit QuantumCircuit")

        backend = self._get_backend()

        if transpile_mode:
            transpiled_circuit = transpile(
                built_circuit,
                backend=backend,
                optimization_level=0,
            )
        else:
            transpiled_circuit = built_circuit

        if self._shots is not None:
            job = backend.run(transpiled_circuit, shots=self._shots)
        else:
            job = backend.run(transpiled_circuit)

        # Be a bit defensive in case QMIO job API differs slightly
        job_id = job.job_id() if callable(getattr(job, "job_id", None)) else str(getattr(job, "job_id", "qmio_job"))
        job_result = job.result()
        counts = job_result.get_counts()

        return ExecutionResult(
            job_id=job_id,
            counts=counts,
            profiling_metrics=ProfilingMetrics(params={}),
            metadata={
                "adapter": self.name,
                "backend": self._backend_name,
                "shots": self._shots,
            },
        )