from django.test import Client
from unittest import TestCase


class GetRequestTest(TestCase):
    def setUp(self) -> None:
        self.client = Client()

    def test_get_index(self):
        # Issue a GET request.
        response = self.client.get('/cube/')
        # Check that the response is 200 OK.
        self.assertEqual(response.status_code, 200)

    def test_get_advance(self):
        # Issue a GET request.
        response = self.client.get('/cube/advance/')
        # Check that the response is 200 OK.
        self.assertEqual(response.status_code, 200)

    def test_get_basic(self):
        # Issue a GET request.
        response = self.client.get('/cube/basic/')
        # Check that the response is 200 OK.
        self.assertEqual(response.status_code, 200)

    def test_get_upload(self):
        # Issue a GET request.
        response = self.client.get('/cube/upload/')
        # Check that the response is 200 OK.
        self.assertEqual(response.status_code, 200)
