import json
import unittest
from ipp import input, image_processing_pipeline, output

class TestPipelineAccuracy(unittest.TestCase):
    def test_accuracy(self):
            
        input_path = "input.json"
        output_path = "output.json"
        expected_path = "expected.json"

        images = input(input_path)
        # Load input data and run pipeline
        results = image_processing_pipeline(images)
        output(results, output_path)

        # Load expected results
        with open(expected_path) as f:
            expected_data = json.load(f)
            
        # Initialize counts for accuracy calculation
        total_detections = 0
        correct_detections = 0
        detection_errors = []
        color_errors = []
        total_colors = 0
        correct_colors = 0

        total_detections_image = 0
        correct_detections_image = 0
        total_colors_image = 0
        correct_colors_image = 0

        # Compare actual results with expected results
        for i, expected_image_result in enumerate(expected_data['results']):
            actual_image_result = results[i]

            total_colors += expected_image_result['num_colors']
            correct_colors += actual_image_result['num_colors']
            color_errors.append(abs(expected_image_result['num_colors'] - actual_image_result['num_colors']) / expected_image_result['num_colors'])

            total_detections += expected_image_result['num_detections']
            correct_detections += actual_image_result['num_detections']
            detection_errors.append(abs(expected_image_result['num_detections'] - actual_image_result['num_detections'])/expected_image_result['num_detections'])

            if expected_image_result['num_colors'] == actual_image_result['num_colors']:
                correct_colors_image += 1
            
            if expected_image_result['num_detections'] == actual_image_result['num_detections']:
                correct_detections_image += 1

            total_detections_image += 1
            total_colors_image += 1


            # TODO: Compare detected objects
            
        # Calculate percentages
        accuracy_detections = (1 - sum(detection_errors) / len(detection_errors)) * 100
        accuracy_colors = (1 - sum(color_errors) / len(color_errors)) * 100

        # Print percentages
        print(f"Accuracy for \033[1m\033[4m{input_path}\033[0m:")
        print(f"  Detections:  \033[1m{accuracy_detections:.2f}%\033[0m ({correct_detections}/{total_detections})")
        print(f"  Colors:      \033[1m{accuracy_colors:.2f}%\033[0m ({correct_colors}/{total_colors})")
        print()
        print(f"Correct Images Detections: {correct_detections_image}/{total_detections_image}")
        print(f"Correct Images Colors: {correct_colors_image}/{total_colors_image}")
        print()


if __name__ == '__main__':
    unittest.main()