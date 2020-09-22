import os
import flask
import unittest
import tempfile
import requests





class appTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_base(self):
        print('base test')

    def test_request(self):
        resp = requests.post("https://127.0.0.1/predict", files={"file":open('./random01.png', 'rb')}, verify=False)
        print(resp.json())

        resp = requests.post("https://127.0.0.1/predict", files={"file":open('./random06.png', 'rb')}, verify=False)
        print(resp.json())


if __name__ == "__main__":
    unittest.main()
