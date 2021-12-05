import json

import pytest
from django.test import AsyncClient
from unittest import TestCase


class GetRequestTest(TestCase):
    def setUp(self) -> None:
        self.client = AsyncClient()

    @pytest.mark.asyncio
    async def test_post_advance_initState(self):
        response = await self.client.post('/advance/initState/')
        # print(response)
        # with open('cube/static/json/initState.json', 'r') as f:
        #     result = json.load(f)
        self.assertEquals(response.status_code, 200)

    @pytest.mark.asyncio
    async def test_post_basic_initState(self):
        response = await self.client.post('/basic/initState/')
        # print(response)
        # with open('cube/static/json/initState.json', 'r') as f:
        #     result = json.load(f)
        self.assertEquals(response.status_code, 200)

    @pytest.mark.asyncio
    async def test_post_advance_solve(self):
        response = await self.client.post('/advance/solve/')
        # print(response)
        # with open('cube/static/json/initState.json', 'r') as f:
        #     result = json.load(f)
        self.assertEquals(response.status_code, 200)

    @pytest.mark.asyncio
    async def test_post_advance_initState(self):
        response = await self.client.post('/advance/initState/')
        # print(response)
        # with open('cube/static/json/initState.json', 'r') as f:
        #     result = json.load(f)
        self.assertEquals(response.status_code, 200)


