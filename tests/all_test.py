from tests.test_gradient import run_all_tests_gradient
from tests.test_save_load import test_save_load_same_predictions
from tests.test_shapes import run_all_shape_tests
from tests.test_dataloader import all_test_dataloader
from tests.test_layers import test_all_layers

if __name__ == "__main__":
    run_all_tests_gradient()
    test_save_load_same_predictions()
    run_all_shape_tests()
    all_test_dataloader()
    test_all_layers()