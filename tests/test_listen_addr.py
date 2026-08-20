"""Bind ollama serve: loopback по умолчанию, LAN только явно."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from modules.ollama_framework.deployment.paths import (
    build_ollama_env,
    resolve_ollama_listen_addr,
)


class ResolveOllamaListenAddrTests(unittest.TestCase):
    def test_default_loopback(self) -> None:
        with patch(
            'modules.ollama_framework.deployment.paths.read_env',
            side_effect=lambda name, default='': {
                'OLLAMA_BASE_URL': 'http://127.0.0.1:11434',
            }.get(name, default),
        ):
            host, port = resolve_ollama_listen_addr()
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 11434)

    def test_base_url_localhost_normalized(self) -> None:
        with patch(
            'modules.ollama_framework.deployment.paths.read_env',
            side_effect=lambda name, default='': {
                'OLLAMA_BASE_URL': 'http://localhost:11434',
            }.get(name, default),
        ):
            host, port = resolve_ollama_listen_addr()
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 11434)

    def test_base_url_lan_does_not_open_serve(self) -> None:
        with patch(
            'modules.ollama_framework.deployment.paths.read_env',
            side_effect=lambda name, default='': {
                'OLLAMA_BASE_URL': 'http://10.0.0.5:11434',
            }.get(name, default),
        ):
            host, port = resolve_ollama_listen_addr()
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 11434)

    def test_explicit_ollama_host_lan(self) -> None:
        with patch(
            'modules.ollama_framework.deployment.paths.read_env',
            side_effect=lambda name, default='': {
                'OLLAMA_HOST': '0.0.0.0:11434',
                'OLLAMA_BASE_URL': 'http://127.0.0.1:11434',
            }.get(name, default),
        ):
            host, port = resolve_ollama_listen_addr()
        self.assertEqual(host, '0.0.0.0')
        self.assertEqual(port, 11434)

    def test_cli_host_override(self) -> None:
        with patch(
            'modules.ollama_framework.deployment.paths.read_env',
            side_effect=lambda name, default='': default,
        ):
            host, port = resolve_ollama_listen_addr(
                override_host='0.0.0.0',
                override_port='11434',
            )
        self.assertEqual(host, '0.0.0.0')
        self.assertEqual(port, 11434)


class BuildOllamaEnvHostTests(unittest.TestCase):
    def test_sets_loopback_when_unset(self) -> None:
        with patch(
            'modules.ollama_framework.deployment.paths.ensure_ollama_models_dir',
            return_value=__import__('pathlib').Path('.'),
        ), patch(
            'modules.ollama_framework.deployment.paths.get_ollama_runtime_home',
            return_value=__import__('pathlib').Path('.'),
        ), patch(
            'modules.ollama_framework.deployment.paths.get_ollama_dir',
            return_value=__import__('pathlib').Path('/missing-ollama-dir'),
        ), patch.dict('os.environ', {}, clear=False):
            env = dict(__import__('os').environ)
            env.pop('OLLAMA_HOST', None)
            with patch('os.environ', env), patch(
                'modules.ollama_framework.deployment.paths.read_env',
                side_effect=lambda name, default='': default,
            ):
                built = build_ollama_env(base={'OLLAMA_HOST': ''})
        self.assertTrue(built['OLLAMA_HOST'].startswith('127.0.0.1:'))

    def test_keeps_explicit_host(self) -> None:
        with patch(
            'modules.ollama_framework.deployment.paths.ensure_ollama_models_dir',
            return_value=__import__('pathlib').Path('.'),
        ), patch(
            'modules.ollama_framework.deployment.paths.get_ollama_runtime_home',
            return_value=__import__('pathlib').Path('.'),
        ), patch(
            'modules.ollama_framework.deployment.paths.get_ollama_dir',
            return_value=__import__('pathlib').Path('/missing-ollama-dir'),
        ):
            built = build_ollama_env(base={'OLLAMA_HOST': '0.0.0.0:11434'})
        self.assertEqual(built['OLLAMA_HOST'], '0.0.0.0:11434')


if __name__ == '__main__':
    unittest.main()
