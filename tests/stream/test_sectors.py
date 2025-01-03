import json
import os
import unittest

import cv2

import stream


class FrameSectorsTestCase(unittest.TestCase):
    def test_extract_sectors(self):
        input_file = os.path.join(os.path.dirname(__file__), "testdata/extract_sectors", "input.json")
        with open(input_file) as file:
            data = json.load(file)

        if not isinstance(data, list):
            self.fail("input.json is not array")

        for index, obj in enumerate(data):
            with self.subTest(msg=obj["path"]):
                if "_comment" in obj:
                    self.skipTest(obj["_comment"])

                # if obj['path'] != 'a1.png':
                #     self.skipTest('Looking for another')
                print(f"start:{obj['path']}")
                image_path = os.path.join(os.path.dirname(__file__), "testdata/extract_sectors", obj["path"])
                raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                frame = stream.Frame(raw, 0)
                sectors = frame.extract_sectors()
                print(f"Sectors: {obj['sectors']}. Result: {len(sectors)}")
                self.assertEqual(obj["sectors"], len(sectors))


if __name__ == "__main__":
    unittest.main()
