from tests.test_gradient import run_all_tests_gradient
from tests.test_save_load import test_save_load_same_predictions
from tests.test_shapes import run_all_shape_tests


if __name__ == "__main__":
    run_all_tests_gradient()
    test_save_load_same_predictions()
    run_all_shape_tests()