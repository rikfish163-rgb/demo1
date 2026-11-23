import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .decision import call_emrank_ssh,DecisionError


class RemoteTransportTests(unittest.TestCase):
    def test_single_request_key_only_on_stdin_and_tls_preserved(self):
        request={'model':'typesafe/jev-latest','state':{'value':'literal `x` $(y) "z"'}}
        reply={'answers':{}}
        with patch('fr5_jev.decision.shutil.which',return_value='/usr/bin/ssh'), \
             patch('fr5_jev.decision.subprocess.run',return_value=SimpleNamespace(
                 returncode=0,stdout=json.dumps(reply)+'\n__FR5_HTTP_STATUS__:200')) as run:
            result,_=call_emrank_ssh(request,'SYNTHETIC-SECRET')
        self.assertEqual(result,reply);run.assert_called_once()
        args=run.call_args.args[0];self.assertNotIn('SYNTHETIC-SECRET',str(args))
        self.assertIn('4090',args);self.assertNotIn('--insecure',args[-1])
        config=run.call_args.kwargs['input']
        payload=next(x for x in config.splitlines() if x.startswith('data-binary = '))
        self.assertEqual(json.loads(json.loads(payload.split(' = ',1)[1])),request)
        self.assertIn('SYNTHETIC-SECRET',config)
    def test_failure_not_retried_and_upstream_body_not_exposed(self):
        for returncode,body in [(35,'private error'),(0,'private body\n__FR5_HTTP_STATUS__:401'),(0,'invalid')]:
            with self.subTest(returncode=returncode),patch('fr5_jev.decision.shutil.which',return_value='ssh'), \
                 patch('fr5_jev.decision.subprocess.run',return_value=SimpleNamespace(returncode=returncode,stdout=body)) as run:
                with self.assertRaises(DecisionError) as error:call_emrank_ssh({},'synthetic')
                self.assertNotIn('private',str(error.exception));run.assert_called_once()
    def test_newline_key_rejected_without_starting_ssh(self):
        with patch('fr5_jev.decision.subprocess.run') as run:
            with self.assertRaises(DecisionError):call_emrank_ssh({},'bad\nkey')
            run.assert_not_called()
