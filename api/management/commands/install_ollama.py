"""
Django команда для установки Ollama в virtual_env/packages/ollama
"""

import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from modules.ollama_framework.api.paths import (
    ensure_ollama_models_dir,
    get_ollama_dir,
    get_ollama_models_dir,
)

DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_UPDATE_INTERVAL_SEC = 0.5


class Command(BaseCommand):
    help = 'Устанавливает Ollama в virtual_env/packages/ollama, модели — в ollama_models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Переустановить Ollama, даже если он уже установлен',
        )

    def handle(self, *args, **options):
        force = options.get('force', False)

        packages_path = Path(settings.PACKAGES_PATH)
        ollama_path = get_ollama_dir()
        models_path = get_ollama_models_dir()

        if ollama_path.exists() and not force:
            self.stdout.write(self.style.WARNING(
                f'Ollama уже установлен в {ollama_path}\n'
                'Используйте --force для переустановки'
            ))
            return

        packages_path.mkdir(parents=True, exist_ok=True)
        ensure_ollama_models_dir()
        self._migrate_legacy_models_dir(packages_path, models_path)

        if force and ollama_path.exists():
            self.stdout.write('Удаление старой установки...')
            shutil.rmtree(ollama_path, ignore_errors=True)

        self.stdout.write('Установка Ollama в packages...')

        system = platform.system().lower()
        machine = platform.machine().lower()

        try:
            if system == 'windows':
                self._install_windows(ollama_path, machine)
            elif system == 'linux':
                self._install_linux(ollama_path, machine)
            else:
                raise CommandError(f'Неподдерживаемая ОС: {system}')
        except Exception as e:
            raise CommandError(f'Ошибка при установке Ollama: {e}') from e

        models_path_str = str(models_path.absolute())
        self.stdout.write(self.style.SUCCESS(
            f'\nOllama установлен.\n\n'
            f'Бинарник: {ollama_path.absolute()}\n'
            f'Модели: {models_path_str}\n\n'
            f'OLLAMA_MODELS задаётся автоматически при запуске через ergoms.\n'
            f'Запуск сервера: ergoms ollama_framework:start-ollama\n'
        ))

    def _migrate_legacy_models_dir(self, packages_path: Path, models_path: Path) -> None:
        legacy_models = packages_path / 'models'
        if not legacy_models.is_dir():
            return

        if any(models_path.iterdir()):
            self.stdout.write(
                self.style.WARNING(
                    f'Устаревший каталог {legacy_models} не перенесён — '
                    f'в {models_path} уже есть данные'
                )
            )
            return

        for item in legacy_models.iterdir():
            shutil.move(str(item), str(models_path / item.name))

        shutil.rmtree(legacy_models, ignore_errors=True)
        self.stdout.write(
            self.style.WARNING(
                f'Данные из packages/models перенесены в {models_path}'
            )
        )

    def _install_windows(self, ollama_path: Path, machine: str) -> None:
        if 'arm64' in machine or 'aarch64' in machine:
            archive_name = 'ollama-windows-arm64.zip'
        elif '64' in machine or 'x86_64' in machine or 'amd64' in machine:
            archive_name = 'ollama-windows-amd64.zip'
        else:
            raise CommandError(f'Неподдерживаемая архитектура Windows: {machine}')

        url = f'https://github.com/ollama/ollama/releases/latest/download/{archive_name}'
        self._download_and_extract_archive(url, archive_name, ollama_path, 'zip')

    def _install_linux(self, ollama_path: Path, machine: str) -> None:
        if 'x86_64' in machine or 'amd64' in machine:
            archive_name = 'ollama-linux-amd64.tgz'
        elif 'arm64' in machine or 'aarch64' in machine:
            archive_name = 'ollama-linux-arm64.tgz'
        else:
            raise CommandError(f'Неподдерживаемая архитектура Linux: {machine}')

        url = f'https://github.com/ollama/ollama/releases/latest/download/{archive_name}'
        self._download_and_extract_archive(url, archive_name, ollama_path, 'tgz')

    def _download_and_extract_archive(
        self,
        url: str,
        archive_name: str,
        target_dir: Path,
        archive_type: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / archive_name
            self._download_archive(url, archive_path)

            self.stdout.write('Распаковка...')
            target_dir.mkdir(parents=True, exist_ok=True)
            if archive_type == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as archive:
                    archive.extractall(target_dir)
            else:
                with tarfile.open(archive_path, 'r:gz') as archive:
                    archive.extractall(target_dir)

        exe_name = 'ollama.exe' if platform.system().lower() == 'windows' else 'ollama'
        if not (target_dir / exe_name).is_file():
            raise CommandError(
                f'После распаковки не найден {exe_name} в {target_dir}. '
                'Проверьте архив релиза Ollama.'
            )

        self.stdout.write(self.style.SUCCESS(f'Ollama распакован в {target_dir}'))

    def _download_archive(self, url: str, destination: Path) -> None:
        self.stdout.write(f'Скачивание: {url}')

        curl_exe = shutil.which('curl')
        if curl_exe and self._download_with_curl(curl_exe, url, destination):
            self.stdout.write('')
            return

        self._download_with_requests(url, destination)
        self.stdout.write('')

    def _download_with_curl(self, curl_exe: str, url: str, destination: Path) -> bool:
        try:
            result = subprocess.run(
                [
                    curl_exe,
                    '-L',
                    '--fail',
                    '--retry',
                    '3',
                    '--retry-delay',
                    '2',
                    '-o',
                    str(destination),
                    url,
                    '--progress-bar',
                ],
                check=False,
            )
        except OSError:
            return False

        return (
            result.returncode == 0
            and destination.is_file()
            and destination.stat().st_size > 0
        )

    def _download_with_requests(self, url: str, destination: Path) -> None:
        try:
            with requests.Session() as session:
                response = session.get(url, stream=True, timeout=(30, 300))
                response.raise_for_status()
                self._stream_response_to_file(response, destination)
        except requests.RequestException as exc:
            raise CommandError(f'Не удалось скачать архив Ollama: {exc}') from exc

    def _stream_response_to_file(self, response: requests.Response, destination: Path) -> None:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        started_at = time.monotonic()
        last_ui_at = 0.0

        try:
            with destination.open('wb') as output_file:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    output_file.write(chunk)
                    downloaded += len(chunk)

                    now = time.monotonic()
                    is_complete = total_size > 0 and downloaded >= total_size
                    if is_complete or now - last_ui_at >= PROGRESS_UPDATE_INTERVAL_SEC:
                        elapsed = max(now - started_at, 0.001)
                        speed_bps = downloaded / elapsed
                        self._write_download_progress(downloaded, total_size, speed_bps)
                        last_ui_at = now
        except OSError as exc:
            raise CommandError(f'Ошибка записи архива Ollama: {exc}') from exc
        finally:
            response.close()

    def _write_download_progress(
        self,
        downloaded: int,
        total_size: int,
        speed_bps: float,
    ) -> None:
        downloaded_mb = downloaded / (1024 * 1024)
        speed_mbps = speed_bps / (1024 * 1024)

        if total_size > 0:
            total_mb = total_size / (1024 * 1024)
            percent = downloaded / total_size * 100
            message = (
                f'Прогресс: {percent:.1f}% '
                f'({downloaded_mb:.1f} / {total_mb:.1f} МБ) '
                f'— {speed_mbps:.1f} МБ/с'
            )
        else:
            message = f'Скачано: {downloaded_mb:.1f} МБ — {speed_mbps:.1f} МБ/с'

        self.stdout.write(f'\r{message}', ending='')
        self.stdout.flush()
