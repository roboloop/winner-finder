import json
import os
import unittest

import cv2

import stream


class FrameLengthTestCase(unittest.TestCase):
    def test_detect_length(self):
        input_file = os.path.join(os.path.dirname(__file__), "testdata/detect_length", "input.json")
        with open(input_file) as file:
            data = json.load(file)

        if not isinstance(data, list):
            self.fail("input.json is not array")

        for index, obj in enumerate(data):
            with self.subTest(msg=obj["path"]):
                if "_comment" in obj:
                    self.skipTest(obj["_comment"])
                print(f"start: {obj['path']}")
                image_path = os.path.join(os.path.dirname(__file__), "testdata/detect_length", obj["path"])
                raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                frame = stream.Frame(raw, 0)
                length = frame.detect_length()
                print(f"Length: {obj['length']}. Result: {length}")
                self.assertEqual(obj["length"], length)


if __name__ == "__main__":
    unittest.main()
