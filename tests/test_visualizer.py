import unittest
import numpy as np

from src.visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    def test_visualizer(self):
        points = np.array([(1, 2), (2, 3), (3, 4)])

        vis = Visualizer(points, enabled=True, min_weight_visible=0.15, pause_time=5)

        vis.plot_gaussians(
            np.array([[1, 2], [2, 3], [3, 4]]),
            np.array([np.eye(2), [[2, 0], [0, 1]], np.eye(2)]),
            np.array([0.2, 0.8, 0.1]),
        )

        # visual debugging
        # Should plot 3 points, 2 ellipses (last one ignored)
        # One ellipse should be wider along the x-axis


if __name__ == "__main__":
    unittest.main()
