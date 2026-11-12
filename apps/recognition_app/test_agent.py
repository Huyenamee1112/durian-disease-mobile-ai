import json
import os
import unittest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError
import agent_service
from app import create_app
from test_app import FakeRegistry

class AgentTest(unittest.TestCase):
    @patch.dict(os.environ, {'GEMINI_API_KEY': ''})
    def test_missing_key(self):
        self.assertEqual(agent_service.ask_agent('Xin chào', {})['source'], 'unavailable')

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-secret'})
    @patch('agent_service.urlopen')
    def test_history_and_sources(self, call):
        response = MagicMock()
        response.read.return_value = json.dumps({'candidates': [{'content': {'parts': [{'text': 'Kiểm tra độ ẩm [vietgap]'}]}}]}).encode()
        call.return_value.__enter__.return_value = response
        result = agent_service.ask_agent('Tưới thế nào?', {'label': 'Leaf_Healthy', 'image': 'private'}, [{'role': 'user', 'content': 'Cây 3 tuổi'}, {'role': 'assistant', 'content': 'Bạn tưới bao lâu?'}])
        self.assertEqual(result['source'], 'gemini')
        self.assertEqual(result['sources'][0]['id'], 'vietgap')
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(len(payload['contents']), 3)
        self.assertNotIn('private', request.data.decode())
        self.assertNotIn('test-secret', request.full_url)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-secret'})
    @patch('agent_service.urlopen', side_effect=HTTPError('url',429,'secret',{},None))
    def test_quota_safe_error(self, call):
        result = agent_service.ask_agent('Xin chào', {})
        self.assertIn('hạn mức', result['answer'])
        self.assertNotIn('secret', result['answer'])

    def test_invalid_context(self):
        client = create_app(model_registry=FakeRegistry()).test_client()
        for payload in ([1], {'question':'a','diagnosis':[1]}, {'question':'a','history':{}}, {'question':'x'*2001}):
            self.assertEqual(client.post('/api/agent', json=payload).status_code, 400)

if __name__ == '__main__': unittest.main()
