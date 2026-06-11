import unittest
from pathlib import Path

from omegaconf import OmegaConf


class HydraConfigTests(unittest.TestCase):
    def test_multirun_subdir_uses_short_job_number_only(self):
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
        raw_cfg = OmegaConf.to_container(cfg, resolve=False)

        self.assertEqual(raw_cfg["hydra"]["sweep"]["subdir"], "${hydra.job.num}")


if __name__ == "__main__":
    unittest.main()
