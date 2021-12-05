from django.test import SimpleTestCase
from django.urls import reverse, resolve
from cube.views import *


class UrlsTest(SimpleTestCase):

    def test_index_url_resolve(self):
        url = reverse('index')
        self.assertEquals(resolve(url).func, index)

    def test_advance_url_resolve(self):
        url = reverse('advance')
        self.assertEquals(resolve(url).func, advance)

    def test_upload_url_resolve(self):
        url = reverse('upload')
        self.assertEquals(resolve(url).func, upload)

    def test_basic_url_resolve(self):
        url = reverse('basic')
        self.assertEquals(resolve(url).func, basic)

    def test_advance_solve_url_resolve(self):
        url = reverse('advance_solve')
        self.assertEquals(resolve(url).func, solve)

    def test_basic_solve_url_resolve(self):
        url = reverse('basic_solve')
        self.assertEquals(resolve(url).func, basic_solve)

    def test_basic_initState_url_resolve(self):
        url = reverse('basic_initState')
        self.assertEquals(resolve(url).func, basic_initState)

    def test_advance_initState_url_resolve(self):
        url = reverse('advance_initState')
        self.assertEquals(resolve(url).func, initState)
