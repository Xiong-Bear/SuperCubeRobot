from django.test import Client
from unittest import TestCase


class GetRequestTest(TestCase):
    def setUp(self) -> None:
        self.client = Client()

    def test_get_index(self):
        response = self.client.get('/cube/')
        self.assertEqual(response.status_code, 200)

    def test_get_advance(self):
        response = self.client.get('/cube/advance/')
        self.assertEqual(response.status_code, 200)

    def test_get_basic(self):
        response = self.client.get('/cube/basic/')
        self.assertEqual(response.status_code, 200)

    def test_get_upload(self):
        response = self.client.get('/cube/upload/')
        self.assertEqual(response.status_code, 200)
