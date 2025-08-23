import pandas as pd
import numpy as np
from pathlib import Path


class ExecutionDataLoader:
    """
    Reads execution-time and communication-time CSV files and exposes
    the resulting NumPy arrays.

    Parameters
    ----------
    forward_exec_path : str | Path
    backward_exec_path : str | Path
    forward_comm_path : str | Path
    backward_comm_path : str | Path
    """

    def __init__(
        self,
        forward_exec_path: str,
        backward_exec_path: str,
        forward_comm_path: str,
        backward_comm_path: str,
    ) -> None:
        self.forward_exec_path = Path(forward_exec_path)
        self.backward_exec_path = Path(backward_exec_path)
        self.forward_comm_path = Path(forward_comm_path)
        self.backward_comm_path = Path(backward_comm_path)

        # Attributes that will be filled after .load()
        self.n_stages: int | None = None
        self.n_machines: int | None = None
        self.forward_exec: np.ndarray | None = None
        self.backward_exec: np.ndarray | None = None
        self.forward_comm: np.ndarray | None = None
        self.backward_comm: np.ndarray | None = None

    # ---------- internal helpers ---------- #

    @staticmethod
    def _read_execution_csv(path: Path) -> tuple[int, int, np.ndarray]:
        df = pd.read_csv(path)
        return df.shape[1], df.shape[0], df.to_numpy()

    @staticmethod
    def _read_communication_csv(path: Path) -> np.ndarray:
        df = pd.read_csv(path)
        stage_cols = [c for c in df.columns if c.startswith("part")]
        n_stages = len(stage_cols)
        n_clients = df["source_client"].max()  # assumes 1-based contiguous IDs
        comm = np.zeros((n_clients, n_clients, n_stages))

        for _, row in df.iterrows():
            src = int(row["source_client"]) - 1  # zero-based index
            dst = int(row["destination_client"]) - 1
            comm[src, dst, :] = row[stage_cols].to_numpy()

        return comm

    # ---------- public API ---------- #

    def load(self) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads every file exactly once and returns all arrays.

        Returns
        -------
        (n_stages, n_machines,
         forward_exec, backward_exec,
         forward_comm, backward_comm)
        """
        # Execution-time CSVs
        self.n_stages, self.n_machines, self.forward_exec = self._read_execution_csv(
            self.forward_exec_path
        )
        stages2, machines2, self.backward_exec = self._read_execution_csv(
            self.backward_exec_path
        )
        if stages2 != self.n_stages or machines2 != self.n_machines:
            raise ValueError(
                "Mismatch between forward and backward execution CSVs "
                f"({self.forward_exec_path}, {self.backward_exec_path})"
            )

        # Communication-time CSVs
        self.forward_comm = self._read_communication_csv(self.forward_comm_path)
        self.backward_comm = self._read_communication_csv(self.backward_comm_path)

        return (
            self.n_stages,
            self.n_machines,
            self.forward_exec,
            self.backward_exec,
            self.forward_comm,
            self.backward_comm,
        )

    # Optional convenience so the instance itself is “callable”
    def __call__(self):
        if self.forward_exec is None:
            self.load()
        return (
            self.forward_exec,
            self.backward_exec,
            self.forward_comm,
            self.backward_comm,
        )
